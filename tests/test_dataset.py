"""
Unit tests for the dataset and tokenization module.

Covers SequenceDataset, NTCollator, compute_class_weights, and the end-to-end
pipeline: SequenceDataset -> GenomicAugmentationDataset -> DataLoader -> tensors.
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.dataset import (
    FAMILIES,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    NTCollator,
    SequenceDataset,
    compute_class_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SEQUENCES = [
    ('seq_chunk0', 'seq', 0, 'ATCGATCGATCG' * 50, 600, 'Coronaviridae'),
    ('seq_chunk1', 'seq', 1, 'GCTAGCTAGCTA' * 40, 480, 'Filoviridae'),
    ('seq_chunk2', 'seq2', 0, 'AAACCCGGGTTT' * 30, 360, 'Poxviridae'),
    ('seq_chunk3', 'seq3', 0, 'TTTGGGCCCAAA' * 20, 240, 'Flaviviridae'),
    ('seq_chunk4', 'seq4', 0, 'ACGTACGTACGT' * 25, 300, 'Arenaviridae'),
    ('seq_chunk5', 'seq5', 0, 'TGCATGCATGCA' * 35, 420, 'Orthomyxoviridae'),
    ('seq_chunk6', 'seq6', 0, 'CGATCGATCGAT' * 45, 540, 'Paramyxoviridae'),
]

CSV_HEADER = ['id', 'parent_sequence_id', 'chunk_index', 'sequence', 'length', 'family']


@pytest.fixture
def csv_path(tmp_path):
    """Write a small CSV and return its path."""
    path = tmp_path / 'test_sequences.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in SAMPLE_SEQUENCES:
            writer.writerow(row)
    return str(path)


@pytest.fixture
def dataset(csv_path):
    return SequenceDataset(csv_path)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that behaves like HuggingFace tokenizers."""
    tokenizer = MagicMock()

    def tokenize_fn(sequences, padding=True, truncation=True, max_length=1000,
                    return_tensors='pt'):
        batch_size = len(sequences)
        # Simulate token output: ~1 token per 6 bases (6-mer tokenization)
        seq_lengths = [min(max_length, len(s) // 6 + 2) for s in sequences]
        max_len = max(seq_lengths)

        input_ids = torch.randint(0, 1000, (batch_size, max_len))
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, length in enumerate(seq_lengths):
            attention_mask[i, :length] = 1

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    tokenizer.side_effect = tokenize_fn
    return tokenizer


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_families_length(self):
        assert len(FAMILIES) == 7

    def test_label_to_idx_keys_match_families(self):
        assert set(LABEL_TO_IDX.keys()) == set(FAMILIES)

    def test_idx_to_label_keys_match(self):
        assert set(IDX_TO_LABEL.values()) == set(FAMILIES)

    def test_roundtrip(self):
        for fam in FAMILIES:
            assert IDX_TO_LABEL[LABEL_TO_IDX[fam]] == fam

    def test_indices_are_contiguous(self):
        assert sorted(IDX_TO_LABEL.keys()) == list(range(7))


# ---------------------------------------------------------------------------
# SequenceDataset tests
# ---------------------------------------------------------------------------

class TestSequenceDataset:
    def test_length(self, dataset):
        assert len(dataset) == len(SAMPLE_SEQUENCES)

    def test_item_keys(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {'sequence', 'label', 'id'}

    def test_sequence_is_str(self, dataset):
        for i in range(len(dataset)):
            assert isinstance(dataset[i]['sequence'], str)

    def test_label_dtype_and_range(self, dataset):
        for i in range(len(dataset)):
            label = dataset[i]['label']
            assert isinstance(label, int)
            assert 0 <= label < 7

    def test_id_format(self, dataset):
        item = dataset[0]
        assert isinstance(item['id'], str)
        assert len(item['id']) > 0

    def test_label_mapping_correct(self, dataset):
        # First row is Coronaviridae
        item = dataset[0]
        assert item['label'] == LABEL_TO_IDX['Coronaviridae']

    def test_all_families_mapped(self, dataset):
        labels = {dataset[i]['label'] for i in range(len(dataset))}
        assert len(labels) == 7

    def test_sequence_content_preserved(self, dataset):
        item = dataset[0]
        assert item['sequence'] == SAMPLE_SEQUENCES[0][3]


# ---------------------------------------------------------------------------
# NTCollator tests
# ---------------------------------------------------------------------------

class TestNTCollator:
    def test_output_keys(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer, max_length=1000)
        batch = [dataset[i] for i in range(3)]
        result = collator(batch)
        assert set(result.keys()) == {'input_ids', 'attention_mask', 'labels'}

    def test_output_tensor_shapes(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer, max_length=1000)
        batch = [dataset[i] for i in range(3)]
        result = collator(batch)

        assert result['input_ids'].ndim == 2
        assert result['input_ids'].shape[0] == 3
        assert result['attention_mask'].ndim == 2
        assert result['attention_mask'].shape[0] == 3
        assert result['labels'].shape == (3,)

    def test_labels_dtype(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer, max_length=1000)
        batch = [dataset[i] for i in range(3)]
        result = collator(batch)
        assert result['labels'].dtype == torch.long

    def test_input_ids_dtype(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer, max_length=1000)
        batch = [dataset[i] for i in range(3)]
        result = collator(batch)
        assert result['input_ids'].dtype == torch.long

    def test_tokenizer_receives_sequences(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer, max_length=512)
        batch = [dataset[0], dataset[1]]
        collator(batch)

        call_args = mock_tokenizer.call_args
        sequences_arg = call_args[0][0]
        assert len(sequences_arg) == 2
        assert sequences_arg[0] == dataset[0]['sequence']
        assert call_args[1]['max_length'] == 512

    def test_single_item_batch(self, dataset, mock_tokenizer):
        collator = NTCollator(mock_tokenizer)
        batch = [dataset[0]]
        result = collator(batch)
        assert result['labels'].shape == (1,)


# ---------------------------------------------------------------------------
# compute_class_weights tests
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_shape(self, csv_path):
        weights = compute_class_weights(csv_path)
        assert weights.shape == (7,)

    def test_all_positive(self, csv_path):
        weights = compute_class_weights(csv_path)
        assert (weights > 0).all()

    def test_dtype(self, csv_path):
        weights = compute_class_weights(csv_path)
        assert weights.dtype == torch.float32

    def test_device(self, csv_path):
        weights = compute_class_weights(csv_path)
        assert weights.device == torch.device('cpu')

    def test_uniform_distribution_gives_equal_weights(self, tmp_path):
        """When all classes have equal counts, weights should be equal."""
        path = tmp_path / 'uniform.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for i, fam in enumerate(FAMILIES):
                for j in range(10):
                    writer.writerow([
                        f'{fam}_{j}', f'{fam}_parent', 0,
                        'ACGT' * 100, 400, fam,
                    ])

        weights = compute_class_weights(str(path))
        assert torch.allclose(weights, weights[0].expand_as(weights))

    def test_imbalanced_distribution(self, tmp_path):
        """Rarer classes should receive higher weights."""
        path = tmp_path / 'imbalanced.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            counts = [100, 50, 50, 10, 10, 10, 10]
            for fam, count in zip(FAMILIES, counts):
                for j in range(count):
                    writer.writerow([
                        f'{fam}_{j}', f'{fam}_parent', 0,
                        'ACGT' * 100, 400, fam,
                    ])

        weights = compute_class_weights(str(path))
        # Poxviridae has the most (100), so its weight should be lowest
        assert weights[LABEL_TO_IDX['Poxviridae']] < weights[LABEL_TO_IDX['Filoviridae']]


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Integration test: SequenceDataset -> GenomicAugmentationDataset -> DataLoader."""

    def test_pipeline_produces_valid_tensors(self, dataset, mock_tokenizer):
        from torch.utils.data import DataLoader

        from src.augmentation import GenomicAugmentationDataset

        augmented = GenomicAugmentationDataset(
            dataset,
            reverse_complement=True,
            random_crop=True,
            sequencing_noise='illumina',
            n_masking=True,
            kmer_shuffle=False,
        )

        collator = NTCollator(mock_tokenizer, max_length=1000)
        loader = DataLoader(augmented, batch_size=4, shuffle=False, collate_fn=collator)

        batch = next(iter(loader))

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert batch['input_ids'].ndim == 2
        assert batch['labels'].ndim == 1
        assert batch['labels'].dtype == torch.long
        assert (batch['labels'] >= 0).all()
        assert (batch['labels'] < 7).all()

    def test_augmented_items_have_required_keys(self, dataset):
        from src.augmentation import GenomicAugmentationDataset

        augmented = GenomicAugmentationDataset(dataset)
        item = augmented[0]

        assert 'sequence' in item
        assert 'label' in item
        assert isinstance(item['sequence'], str)
        assert isinstance(item['label'], int)

    def test_full_dataset_iteration(self, dataset, mock_tokenizer):
        """Iterate over the entire dataset to catch any row-level issues."""
        from torch.utils.data import DataLoader

        from src.augmentation import GenomicAugmentationDataset

        augmented = GenomicAugmentationDataset(
            dataset,
            reverse_complement=False,
            random_crop=False,
            n_masking=False,
        )
        collator = NTCollator(mock_tokenizer)
        loader = DataLoader(augmented, batch_size=3, collate_fn=collator)

        total_samples = 0
        for batch in loader:
            total_samples += batch['labels'].shape[0]
            assert batch['input_ids'].shape[0] == batch['labels'].shape[0]

        assert total_samples == len(dataset)

    def test_real_nt_v2_tokenizer(self, dataset):
        """End-to-end with the real NT v2 tokenizer (not mocked)."""
        from torch.utils.data import DataLoader

        from transformers import AutoTokenizer

        from src.augmentation import GenomicAugmentationDataset

        tokenizer = AutoTokenizer.from_pretrained(
            'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
            trust_remote_code=True,
        )

        augmented = GenomicAugmentationDataset(
            dataset,
            reverse_complement=True,
            random_crop=False,
            n_masking=False,
        )

        collator = NTCollator(tokenizer, max_length=1000)
        loader = DataLoader(augmented, batch_size=4, shuffle=False, collate_fn=collator)

        batch = next(iter(loader))

        assert batch['input_ids'].ndim == 2
        assert batch['input_ids'].shape[0] == 4
        assert batch['attention_mask'].shape == batch['input_ids'].shape
        assert batch['labels'].shape == (4,)
        assert batch['labels'].dtype == torch.long
        assert batch['input_ids'].dtype == torch.long
        # Token count should be <= max_length
        assert batch['input_ids'].shape[1] <= 1000
