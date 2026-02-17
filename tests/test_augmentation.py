"""
Unit tests for the genomic sequence augmentation module.

Verifies each transform preserves sequence validity (only valid nucleotides + N,
correct length ranges, label preservation) and that the dataset wrapper applies
augmentations correctly.
"""

import random
import string

import pytest

from src.augmentation import (
    GenomicAugmentationDataset,
    kmer_shuffle,
    n_masking,
    random_crop,
    reverse_complement,
    sequencing_noise,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

VALID_BASES = set('ACGTNacgtn')
VALID_BASES_UPPER = set('ACGTN')

# Realistic genomic sequences for testing
SAMPLE_SEQ = 'ATCGATCGATCGATCGATCGATCGATCGATCG' * 10  # 320 bp
SHORT_SEQ = 'ATCG'
LONG_SEQ = 'ACGTACGTACGT' * 500  # 6000 bp


def make_random_sequence(length=500, seed=42):
    """Generate a random DNA sequence."""
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(length))


class MockDataset:
    """Simple mock dataset for testing the wrapper."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
        }


@pytest.fixture
def mock_dataset():
    seqs = [make_random_sequence(300, seed=i) for i in range(20)]
    labels = [i % 3 for i in range(20)]
    return MockDataset(seqs, labels)


# ---------------------------------------------------------------------------
# Tests: reverse_complement
# ---------------------------------------------------------------------------

class TestReverseComplement:
    def test_basic_complement(self):
        assert reverse_complement('AACCGGTT') == 'AACCGGTT'  # palindrome

    def test_simple_sequence(self):
        assert reverse_complement('AAAA') == 'TTTT'
        assert reverse_complement('CCCC') == 'GGGG'
        assert reverse_complement('ATCG') == 'CGAT'

    def test_preserves_length(self):
        result = reverse_complement(SAMPLE_SEQ)
        assert len(result) == len(SAMPLE_SEQ)

    def test_double_reverse_is_identity(self):
        result = reverse_complement(reverse_complement(SAMPLE_SEQ))
        assert result == SAMPLE_SEQ

    def test_only_valid_bases(self):
        result = reverse_complement(SAMPLE_SEQ)
        assert all(c in VALID_BASES for c in result)

    def test_handles_n_bases(self):
        seq = 'ATCNGATN'
        result = reverse_complement(seq)
        assert result == 'NATCNGAT'
        assert len(result) == len(seq)

    def test_handles_lowercase(self):
        result = reverse_complement('atcg')
        assert result == 'cgat'

    def test_empty_string(self):
        assert reverse_complement('') == ''

    def test_single_base(self):
        assert reverse_complement('A') == 'T'
        assert reverse_complement('C') == 'G'


# ---------------------------------------------------------------------------
# Tests: random_crop
# ---------------------------------------------------------------------------

class TestRandomCrop:
    def test_output_is_substring(self):
        result = random_crop(SAMPLE_SEQ, min_crop_ratio=0.5)
        assert result in SAMPLE_SEQ

    def test_respects_min_crop_ratio(self):
        min_ratio = 0.5
        for _ in range(100):
            result = random_crop(SAMPLE_SEQ, min_crop_ratio=min_ratio)
            assert len(result) >= int(len(SAMPLE_SEQ) * min_ratio)

    def test_never_exceeds_original_length(self):
        for _ in range(100):
            result = random_crop(SAMPLE_SEQ, min_crop_ratio=0.5)
            assert len(result) <= len(SAMPLE_SEQ)

    def test_full_ratio_returns_original(self):
        for _ in range(10):
            result = random_crop(SAMPLE_SEQ, min_crop_ratio=1.0)
            assert result == SAMPLE_SEQ

    def test_only_valid_bases(self):
        result = random_crop(SAMPLE_SEQ, min_crop_ratio=0.5)
        assert all(c in VALID_BASES for c in result)

    def test_short_sequence(self):
        result = random_crop('A', min_crop_ratio=0.5)
        assert result == 'A'

    def test_empty_string(self):
        result = random_crop('', min_crop_ratio=0.5)
        assert result == ''


# ---------------------------------------------------------------------------
# Tests: sequencing_noise (illumina)
# ---------------------------------------------------------------------------

class TestSequencingNoiseIllumina:
    def test_preserves_length(self):
        result = sequencing_noise(SAMPLE_SEQ, mode='illumina')
        assert len(result) == len(SAMPLE_SEQ)

    def test_only_valid_bases(self):
        result = sequencing_noise(SAMPLE_SEQ, mode='illumina')
        assert all(c in VALID_BASES_UPPER for c in result)

    def test_low_error_rate(self):
        """Illumina has ~0.1% error rate — most bases unchanged."""
        seq = make_random_sequence(10000, seed=99)
        result = sequencing_noise(seq, mode='illumina')
        mismatches = sum(a != b for a, b in zip(seq, result))
        # With 0.1% rate on 10k bases, expect ~10 errors. Allow up to 50.
        assert mismatches < 50

    def test_empty_string(self):
        assert sequencing_noise('', mode='illumina') == ''


# ---------------------------------------------------------------------------
# Tests: sequencing_noise (nanopore)
# ---------------------------------------------------------------------------

class TestSequencingNoiseNanopore:
    def test_only_valid_bases(self):
        result = sequencing_noise(SAMPLE_SEQ, mode='nanopore')
        assert all(c in VALID_BASES_UPPER for c in result)

    def test_length_changes_possible(self):
        """Nanopore mode has indels, so length may change."""
        lengths = set()
        for seed in range(50):
            random.seed(seed)
            result = sequencing_noise(LONG_SEQ, mode='nanopore')
            lengths.add(len(result))
        # With insertions and deletions, we should see length variation
        assert len(lengths) > 1

    def test_empty_string(self):
        assert sequencing_noise('', mode='nanopore') == ''

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown noise mode"):
            sequencing_noise(SAMPLE_SEQ, mode='invalid')


# ---------------------------------------------------------------------------
# Tests: n_masking
# ---------------------------------------------------------------------------

class TestNMasking:
    def test_preserves_length(self):
        result = n_masking(SAMPLE_SEQ, mask_rate=0.05)
        assert len(result) == len(SAMPLE_SEQ)

    def test_only_valid_bases(self):
        result = n_masking(SAMPLE_SEQ, mask_rate=0.05)
        assert all(c in VALID_BASES for c in result)

    def test_introduces_n_bases(self):
        """With a 10% mask rate on a long sequence, we should see some N's."""
        seq = make_random_sequence(1000, seed=0)
        result = n_masking(seq, mask_rate=0.10)
        n_count = result.count('N')
        assert n_count > 0

    def test_approximate_mask_rate(self):
        seq = make_random_sequence(10000, seed=1)
        result = n_masking(seq, mask_rate=0.05)
        n_count = result.count('N')
        # Expect ~500 N's; allow a range
        assert 300 < n_count < 700

    def test_zero_rate_no_change(self):
        result = n_masking(SAMPLE_SEQ, mask_rate=0.0)
        assert result == SAMPLE_SEQ

    def test_empty_string(self):
        result = n_masking('', mask_rate=0.05)
        assert result == ''

    def test_short_sequence_zero_masks(self):
        """A very short sequence with low rate may mask 0 bases."""
        result = n_masking('AT', mask_rate=0.01)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: kmer_shuffle
# ---------------------------------------------------------------------------

class TestKmerShuffle:
    def test_preserves_length(self):
        result = kmer_shuffle(SAMPLE_SEQ, window_size=50, k=3)
        assert len(result) == len(SAMPLE_SEQ)

    def test_only_valid_bases(self):
        result = kmer_shuffle(SAMPLE_SEQ, window_size=50, k=3)
        assert all(c in VALID_BASES for c in result)

    def test_preserves_composition(self):
        """K-mer shuffle preserves local nucleotide composition."""
        seq = make_random_sequence(300, seed=5)
        result = kmer_shuffle(seq, window_size=300, k=3)
        # Same bases, just reordered
        assert sorted(result) == sorted(seq)

    def test_empty_string(self):
        result = kmer_shuffle('', window_size=50, k=3)
        assert result == ''

    def test_short_sequence(self):
        result = kmer_shuffle('ACG', window_size=50, k=3)
        assert result == 'ACG'  # single k-mer, nothing to shuffle


# ---------------------------------------------------------------------------
# Tests: GenomicAugmentationDataset
# ---------------------------------------------------------------------------

class TestGenomicAugmentationDataset:
    def test_length_matches_base(self, mock_dataset):
        aug = GenomicAugmentationDataset(mock_dataset)
        assert len(aug) == len(mock_dataset)

    def test_label_preserved(self, mock_dataset):
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=True,
            random_crop=True,
            sequencing_noise='illumina',
            n_masking=True,
            kmer_shuffle=True,
        )
        for i in range(len(mock_dataset)):
            original = mock_dataset[i]
            augmented = aug[i]
            assert augmented['label'] == original['label']

    def test_sequence_is_valid(self, mock_dataset):
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=True,
            random_crop=True,
            sequencing_noise='illumina',
            n_masking=True,
            kmer_shuffle=True,
        )
        for i in range(len(mock_dataset)):
            item = aug[i]
            assert all(c in VALID_BASES_UPPER for c in item['sequence'])

    def test_augmentations_key_present(self, mock_dataset):
        aug = GenomicAugmentationDataset(mock_dataset)
        item = aug[0]
        assert 'augmentations' in item
        assert isinstance(item['augmentations'], list)

    def test_no_augmentation_when_all_disabled(self, mock_dataset):
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=False,
            random_crop=False,
            sequencing_noise=None,
            n_masking=False,
            kmer_shuffle=False,
        )
        for i in range(len(mock_dataset)):
            original = mock_dataset[i]
            augmented = aug[i]
            assert augmented['sequence'] == original['sequence']
            assert augmented['augmentations'] == []

    def test_stochastic_augmentation(self, mock_dataset):
        """Multiple calls to the same index should sometimes differ."""
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=True,
            p_reverse_complement=0.5,
        )
        results = set()
        for _ in range(50):
            item = aug[0]
            results.add(item['sequence'])
        # With p=0.5 over 50 draws, we should see both original and RC
        assert len(results) > 1

    def test_passthrough_extra_keys(self):
        """Extra keys in base items are preserved."""
        class ExtraDataset:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {
                    'sequence': 'ATCGATCG',
                    'label': 0,
                    'id': 'seq_001',
                    'metadata': {'source': 'ncbi'},
                }

        aug = GenomicAugmentationDataset(ExtraDataset(), reverse_complement=False)
        item = aug[0]
        assert item['id'] == 'seq_001'
        assert item['metadata'] == {'source': 'ncbi'}

    def test_compose_multiple_augmentations(self, mock_dataset):
        """Multiple augmentations can be applied to the same sample."""
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=True,
            random_crop=True,
            sequencing_noise='illumina',
            n_masking=True,
            kmer_shuffle=True,
            # Set all probabilities to 1.0 so all are applied
            p_reverse_complement=1.0,
            p_random_crop=1.0,
            p_sequencing_noise=1.0,
            p_n_masking=1.0,
            p_kmer_shuffle=1.0,
        )
        item = aug[0]
        assert 'reverse_complement' in item['augmentations']
        assert 'random_crop' in item['augmentations']
        assert 'noise_illumina' in item['augmentations']
        assert 'n_masking' in item['augmentations']
        assert 'kmer_shuffle' in item['augmentations']

    def test_nanopore_noise_mode(self, mock_dataset):
        aug = GenomicAugmentationDataset(
            mock_dataset,
            reverse_complement=False,
            random_crop=False,
            sequencing_noise='nanopore',
            n_masking=False,
            kmer_shuffle=False,
            p_sequencing_noise=1.0,
        )
        item = aug[0]
        assert 'noise_nanopore' in item['augmentations']
        assert all(c in VALID_BASES_UPPER for c in item['sequence'])
