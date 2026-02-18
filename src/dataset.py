"""
Dataset and tokenization module for Nucleotide Transformer v2 50M.

Bridges CSV data and the augmentation module (src/augmentation.py) with the
NT v2 tokenizer. Tokenization happens in the collator (after augmentation),
not in __getitem__, so the augmentation wrapper operates on raw strings:

    SequenceDataset -> GenomicAugmentationDataset -> NTCollator -> tensors

Usage:
    from src.dataset import SequenceDataset, NTCollator, compute_class_weights
    from src.augmentation import GenomicAugmentationDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    ds = SequenceDataset('data/processed/train.csv')
    aug = GenomicAugmentationDataset(ds)
    tokenizer = AutoTokenizer.from_pretrained(
        'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
        trust_remote_code=True,
    )
    loader = DataLoader(aug, batch_size=32, collate_fn=NTCollator(tokenizer))
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_exploration import FAMILY_ORDER

# Label mappings — order must match FAMILY_ORDER from data_exploration.py
FAMILIES = list(FAMILY_ORDER)
LABEL_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}
IDX_TO_LABEL = {i: fam for i, fam in enumerate(FAMILIES)}


class SequenceDataset(Dataset):
    """PyTorch Dataset that loads chunked CSV and returns raw sequence dicts.

    Each item is ``{'sequence': str, 'label': int, 'id': str}``, compatible
    with ``GenomicAugmentationDataset`` which expects items with ``'sequence'``
    and ``'label'`` keys.

    Args:
        csv_path: Path to a processed CSV with columns
            ``id, parent_sequence_id, chunk_index, sequence, length, family``.
    """

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.ids = df['id'].astype(str).tolist()
        self.sequences = df['sequence'].astype(str).tolist()
        self.labels = [LABEL_TO_IDX[fam] for fam in df['family']]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'id': self.ids[idx],
        }


class NTCollator:
    """Collate function that tokenizes a batch of sequence dicts.

    Designed for the Nucleotide Transformer v2 6-mer tokenizer.  A 5000 bp
    sequence produces ~833 tokens, fitting within the default 1000-token
    context window.

    Args:
        tokenizer: A HuggingFace tokenizer (NT v2).
        max_length: Maximum number of tokens (default 1000).
    """

    def __init__(self, tokenizer, max_length=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sequences = [item['sequence'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        encoding = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
        }


def compute_class_weights(csv_path, device=None):
    """Compute inverse-frequency class weights for cross-entropy loss.

    Args:
        csv_path: Path to a processed CSV with a ``family`` column.
        device: Torch device for the returned tensor (default CPU).

    Returns:
        ``torch.FloatTensor`` of shape ``(7,)`` with one weight per class,
        ordered by :data:`FAMILIES`.
    """
    df = pd.read_csv(csv_path)
    counts = df['family'].value_counts()

    weights = []
    total = len(df)
    n_classes = len(FAMILIES)
    for fam in FAMILIES:
        c = counts.get(fam, 1)
        weights.append(total / (n_classes * c))

    return torch.tensor(weights, dtype=torch.float32, device=device)
