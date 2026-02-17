"""
Data augmentation for genomic sequences, applied as training-time transforms.

Wraps a base dataset with configurable, on-the-fly augmentations that simulate
real-world variability in metagenomic surveillance data (variable read lengths,
sequencing errors, strand ambiguity, partial coverage).

Usage:
    # As a dataset wrapper
    base_dataset = SequenceDataset('data/processed/train.csv')
    augmented = GenomicAugmentationDataset(
        base_dataset,
        reverse_complement=True,
        random_crop=True,
        sequencing_noise='illumina',
        n_masking=True,
        kmer_shuffle=False,
    )
    loader = DataLoader(augmented, batch_size=32, shuffle=True)

    # CLI preview
    python src/augmentation.py --input data/processed/train.csv --preview 10
"""

import argparse
import random
from pathlib import Path

import numpy as np

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:
    _TorchDataset = None

# Complement mapping for DNA bases
_COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

# Valid DNA bases for substitution (uppercase)
_BASES = ['A', 'C', 'G', 'T']


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def reverse_complement(sequence):
    """
    Return the reverse complement of a DNA sequence.

    DNA is double-stranded, so the virus family is identical regardless of
    which strand is sequenced. This is a free 2x effective data multiplier.
    """
    return sequence.translate(_COMPLEMENT)[::-1]


def random_crop(sequence, min_crop_ratio=0.5):
    """
    Extract a random contiguous subsequence.

    Simulates variable-length metagenomic fragments so the model learns to
    classify from partial information.

    Args:
        sequence: Input DNA sequence string.
        min_crop_ratio: Minimum fraction of the original length to retain.

    Returns:
        A random contiguous substring of the input.
    """
    seq_len = len(sequence)
    if seq_len <= 1:
        return sequence

    min_length = max(1, int(seq_len * min_crop_ratio))
    crop_length = random.randint(min_length, seq_len)
    start = random.randint(0, seq_len - crop_length)
    return sequence[start:start + crop_length]


def sequencing_noise(sequence, mode='illumina'):
    """
    Introduce simulated sequencing errors.

    Args:
        sequence: Input DNA sequence string.
        mode: 'illumina' (~0.1% substitution) or 'nanopore' (~3-5% errors
              with indels).

    Returns:
        Sequence with simulated errors.
    """
    seq = list(sequence.upper())

    if mode == 'illumina':
        error_rate = 0.001
        for i in range(len(seq)):
            if seq[i] in _BASES and random.random() < error_rate:
                seq[i] = random.choice([b for b in _BASES if b != seq[i]])
        return ''.join(seq)

    elif mode == 'nanopore':
        error_rate = 0.04
        result = []
        for base in seq:
            r = random.random()
            if r < error_rate * 0.4:
                # Substitution (~40% of errors)
                if base in _BASES:
                    result.append(random.choice([b for b in _BASES if b != base]))
                else:
                    result.append(base)
            elif r < error_rate * 0.7:
                # Deletion (~30% of errors) — skip this base
                continue
            elif r < error_rate:
                # Insertion (~30% of errors) — add a random base before this one
                result.append(random.choice(_BASES))
                result.append(base)
            else:
                result.append(base)
        return ''.join(result)

    else:
        raise ValueError(f"Unknown noise mode: {mode!r}. Use 'illumina' or 'nanopore'.")


def n_masking(sequence, mask_rate=0.02):
    """
    Randomly replace bases with 'N' (unknown).

    Simulates low-quality regions in real sequencing data. Conceptually
    similar to dropout regularization or image occlusion.

    Args:
        sequence: Input DNA sequence string.
        mask_rate: Fraction of bases to replace with 'N'.

    Returns:
        Sequence with random bases replaced by 'N'.
    """
    seq = list(sequence)
    n_to_mask = int(len(seq) * mask_rate)
    if n_to_mask == 0:
        return sequence

    positions = random.sample(range(len(seq)), n_to_mask)
    for pos in positions:
        seq[pos] = 'N'
    return ''.join(seq)


def kmer_shuffle(sequence, window_size=50, k=3):
    """
    Shuffle k-mers within fixed-size windows.

    Destroys fine-grained positional motifs while preserving local nucleotide
    composition. Forces the model to learn compositional features rather than
    memorizing exact motif positions.

    Args:
        sequence: Input DNA sequence string.
        window_size: Size of each window in bases.
        k: K-mer size for shuffling.

    Returns:
        Sequence with k-mers shuffled within each window.
    """
    result = []
    for start in range(0, len(sequence), window_size):
        window = sequence[start:start + window_size]
        # Split window into k-mers
        kmers = [window[i:i + k] for i in range(0, len(window), k)]
        random.shuffle(kmers)
        result.append(''.join(kmers))
    return ''.join(result)


# ---------------------------------------------------------------------------
# Augmentation dataset wrapper
# ---------------------------------------------------------------------------

# Use torch Dataset as base when available, plain object otherwise
_Base = _TorchDataset if _TorchDataset is not None else object


class GenomicAugmentationDataset(_Base):
    """
    Wraps a base dataset and applies configurable sequence augmentations
    on-the-fly during training.

    The base dataset must return items as dicts with at least a 'sequence' key
    (str) and a 'label' key (int or str). Any additional keys are passed
    through unchanged.

    Each augmentation is independently toggled and applied with its own
    probability on every __getitem__ call, so the same index yields different
    augmentations across epochs.

    Args:
        base_dataset: A torch Dataset returning dict items with 'sequence'.
        reverse_complement: Enable reverse complement (p=0.5).
        random_crop: Enable random subsequence cropping (p=0.3).
        min_crop_ratio: Minimum crop length as fraction of original (0.5).
        sequencing_noise: Noise mode — 'illumina', 'nanopore', or None (p=0.2).
        n_masking: Enable N-masking (p=0.2).
        mask_rate: Fraction of bases to mask (0.02).
        kmer_shuffle: Enable k-mer shuffling within windows (p=0.1).
        kmer_shuffle_window: Window size for k-mer shuffling (50).
        kmer_shuffle_k: K-mer size for shuffling (3).
        p_reverse_complement: Probability for reverse complement.
        p_random_crop: Probability for random crop.
        p_sequencing_noise: Probability for sequencing noise.
        p_n_masking: Probability for N-masking.
        p_kmer_shuffle: Probability for k-mer shuffling.
    """

    def __init__(
        self,
        base_dataset,
        reverse_complement=True,
        random_crop=True,
        min_crop_ratio=0.5,
        sequencing_noise=None,
        n_masking=True,
        mask_rate=0.02,
        kmer_shuffle=False,
        kmer_shuffle_window=50,
        kmer_shuffle_k=3,
        p_reverse_complement=0.5,
        p_random_crop=0.3,
        p_sequencing_noise=0.2,
        p_n_masking=0.2,
        p_kmer_shuffle=0.1,
    ):
        self.base_dataset = base_dataset
        self.do_reverse_complement = reverse_complement
        self.do_random_crop = random_crop
        self.min_crop_ratio = min_crop_ratio
        self.noise_mode = sequencing_noise
        self.do_n_masking = n_masking
        self.mask_rate = mask_rate
        self.do_kmer_shuffle = kmer_shuffle
        self.kmer_shuffle_window = kmer_shuffle_window
        self.kmer_shuffle_k = kmer_shuffle_k
        self.p_reverse_complement = p_reverse_complement
        self.p_random_crop = p_random_crop
        self.p_sequencing_noise = p_sequencing_noise
        self.p_n_masking = p_n_masking
        self.p_kmer_shuffle = p_kmer_shuffle

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        seq = item['sequence']
        applied = []

        # Order: reverse complement -> crop -> noise -> masking -> shuffle
        if self.do_reverse_complement and random.random() < self.p_reverse_complement:
            seq = reverse_complement(seq)
            applied.append('reverse_complement')

        if self.do_random_crop and random.random() < self.p_random_crop:
            seq = random_crop(seq, self.min_crop_ratio)
            applied.append('random_crop')

        if self.noise_mode and random.random() < self.p_sequencing_noise:
            seq = sequencing_noise(seq, self.noise_mode)
            applied.append(f'noise_{self.noise_mode}')

        if self.do_n_masking and random.random() < self.p_n_masking:
            seq = n_masking(seq, self.mask_rate)
            applied.append('n_masking')

        if self.do_kmer_shuffle and random.random() < self.p_kmer_shuffle:
            seq = kmer_shuffle(seq, self.kmer_shuffle_window, self.kmer_shuffle_k)
            applied.append('kmer_shuffle')

        result = dict(item)
        result['sequence'] = seq
        result['augmentations'] = applied
        return result


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------

def _load_preview_dataset(input_path):
    """Load a CSV file as a simple list-of-dicts dataset for preview."""
    import pandas as pd

    df = pd.read_csv(input_path)
    if 'sequence' not in df.columns:
        raise ValueError("CSV must have a 'sequence' column.")

    label_col = 'family' if 'family' in df.columns else 'label'
    if label_col not in df.columns:
        raise ValueError("CSV must have a 'family' or 'label' column.")

    records = []
    for _, row in df.iterrows():
        records.append({
            'sequence': str(row['sequence']),
            'label': str(row[label_col]),
        })
    return records


class _ListDataset(_Base):
    """Minimal Dataset wrapper around a list of dicts."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def preview_augmentations(input_path, n_samples=10, seed=42):
    """
    Print original vs augmented sequence pairs for visual validation.

    Args:
        input_path: Path to a CSV with 'sequence' and 'family'/'label' columns.
        n_samples: Number of samples to preview.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    records = _load_preview_dataset(input_path)
    base = _ListDataset(records)
    augmented = GenomicAugmentationDataset(
        base,
        reverse_complement=True,
        random_crop=True,
        sequencing_noise='illumina',
        n_masking=True,
        kmer_shuffle=True,
    )

    n_samples = min(n_samples, len(base))
    indices = random.sample(range(len(base)), n_samples)

    print(f"=== Augmentation Preview ({n_samples} samples) ===\n")

    for i, idx in enumerate(indices, 1):
        original = base[idx]
        aug = augmented[idx]

        orig_seq = original['sequence']
        aug_seq = aug['sequence']
        transforms = aug['augmentations']

        # Truncate display for readability
        max_display = 80
        orig_display = orig_seq[:max_display] + ('...' if len(orig_seq) > max_display else '')
        aug_display = aug_seq[:max_display] + ('...' if len(aug_seq) > max_display else '')

        print(f"--- Sample {i} (index {idx}) ---")
        print(f"  Label:       {original['label']}")
        print(f"  Original:    {orig_display}  (len={len(orig_seq)})")
        print(f"  Augmented:   {aug_display}  (len={len(aug_seq)})")
        print(f"  Transforms:  {transforms if transforms else ['none']}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preview genomic sequence augmentations."
    )
    parser.add_argument(
        '--input', type=str,
        default='data/processed/train.csv',
        help='Path to sequences CSV (must have sequence + family/label columns)',
    )
    parser.add_argument(
        '--preview', type=int, default=10,
        help='Number of samples to preview',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility',
    )
    args = parser.parse_args()

    preview_augmentations(args.input, n_samples=args.preview, seed=args.seed)
