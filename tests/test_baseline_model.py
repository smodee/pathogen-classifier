"""Tests for the baseline k-mer + XGBoost classifier."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from baseline_model import (
    _build_kmer_vocabulary,
    _compute_class_weights,
    compute_kmer_frequencies,
    extract_features,
    predict_baseline,
    train_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAMILIES = ['Coronaviridae', 'Filoviridae', 'Poxviridae']


def _random_seq(length, rng):
    """Generate a random DNA sequence."""
    bases = list('ACGT')
    return ''.join(rng.choice(bases) for _ in range(length))


@pytest.fixture()
def synthetic_csvs(tmp_path):
    """Create small synthetic train and val CSVs for integration tests."""
    rng = np.random.RandomState(42)
    rows = []
    # 30 train samples: 10 per family (imbalanced not needed for basic test)
    for family in FAMILIES:
        for i in range(10):
            seq = _random_seq(200, rng)
            rows.append({
                'id': f'{family}_{i}_chunk0',
                'parent_sequence_id': f'{family}_{i}',
                'chunk_index': 0,
                'sequence': seq,
                'length': len(seq),
                'family': family,
            })

    train_df = pd.DataFrame(rows)
    train_path = tmp_path / 'train.csv'
    train_df.to_csv(train_path, index=False)

    # 9 val samples: 3 per family
    val_rows = []
    for family in FAMILIES:
        for i in range(3):
            seq = _random_seq(200, rng)
            val_rows.append({
                'id': f'{family}_val{i}_chunk0',
                'parent_sequence_id': f'{family}_val{i}',
                'chunk_index': 0,
                'sequence': seq,
                'length': len(seq),
                'family': family,
            })

    val_df = pd.DataFrame(val_rows)
    val_path = tmp_path / 'val.csv'
    val_df.to_csv(val_path, index=False)

    return train_path, val_path


# ---------------------------------------------------------------------------
# Tests: compute_kmer_frequencies
# ---------------------------------------------------------------------------

class TestComputeKmerFrequencies:
    def test_simple_sequence(self):
        counts = compute_kmer_frequencies('AAAAA', k=3)
        assert counts == {'AAA': 3}

    def test_known_sequence(self):
        counts = compute_kmer_frequencies('ACGTACGT', k=4)
        assert 'ACGT' in counts
        assert counts['ACGT'] == 2

    def test_skips_n_containing_kmers(self):
        counts = compute_kmer_frequencies('ACNGT', k=3)
        # ACN, CNG, NGT all contain N → no k-mers counted
        assert len(counts) == 0

    def test_case_insensitive(self):
        counts_upper = compute_kmer_frequencies('ACGT', k=2)
        counts_lower = compute_kmer_frequencies('acgt', k=2)
        assert counts_upper == counts_lower

    def test_empty_sequence(self):
        counts = compute_kmer_frequencies('', k=3)
        assert counts == {}

    def test_sequence_shorter_than_k(self):
        counts = compute_kmer_frequencies('AC', k=5)
        assert counts == {}


# ---------------------------------------------------------------------------
# Tests: extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_shape(self):
        sequences = ['ACGTACGT', 'TGCATGCA', 'AAAACCCC']
        features = extract_features(sequences, k=3)
        # 4^3 = 64 features
        assert features.shape == (3, 64)

    def test_k5_feature_dimension(self):
        features = extract_features(['ACGT' * 50], k=5)
        assert features.shape == (1, 4**5)
        assert features.shape == (1, 1024)

    def test_normalization_sums_to_one(self):
        seq = 'ACGTACGTACGTACGTACGT'
        features = extract_features([seq], k=3)
        total = features[0].sum()
        assert abs(total - 1.0) < 1e-5, f"Sum of frequencies = {total}, expected ~1.0"

    def test_all_zero_for_n_only_sequence(self):
        features = extract_features(['NNNNNNNN'], k=3)
        assert features[0].sum() == 0.0

    def test_dtype_is_float32(self):
        features = extract_features(['ACGTACGT'], k=2)
        assert features.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: class weights
# ---------------------------------------------------------------------------

class TestClassWeights:
    def test_balanced_classes_equal_weights(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        weights = _compute_class_weights(labels)
        # All classes have equal count → weights should be equal
        assert np.allclose(weights, weights[0])

    def test_imbalanced_minority_gets_higher_weight(self):
        # 8 class-0, 1 class-1, 1 class-2
        labels = np.array([0]*8 + [1, 2])
        weights = _compute_class_weights(labels)
        # Class 0 (majority) should get lower weight than class 1 or 2
        assert weights[0] < weights[8]  # class-0 < class-1
        assert weights[0] < weights[9]  # class-0 < class-2

    def test_weights_formula(self):
        # n=6, 3 classes: [0,0,0, 1,1, 2]
        labels = np.array([0, 0, 0, 1, 1, 2])
        weights = _compute_class_weights(labels)
        # class 0: 6 / (3 * 3) = 0.6667
        # class 1: 6 / (3 * 2) = 1.0
        # class 2: 6 / (3 * 1) = 2.0
        assert abs(weights[0] - 6 / 9) < 1e-5
        assert abs(weights[3] - 1.0) < 1e-5
        assert abs(weights[5] - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: train/predict integration
# ---------------------------------------------------------------------------

class TestTrainPredictIntegration:
    def test_train_creates_model_files(self, synthetic_csvs, tmp_path):
        train_path, val_path = synthetic_csvs
        output_dir = tmp_path / 'model_output'

        train_baseline(
            train_path=str(train_path),
            val_path=str(val_path),
            k=3,  # small k for speed
            output_dir=str(output_dir),
            n_estimators=10,  # few trees for speed
            max_depth=3,
            learning_rate=0.3,
        )

        assert (output_dir / 'xgboost_model.json').exists()
        assert (output_dir / 'label_encoder.json').exists()
        assert (output_dir / 'kmer_params.json').exists()
        assert (output_dir / 'val_metrics.json').exists()

    def test_val_metrics_content(self, synthetic_csvs, tmp_path):
        train_path, val_path = synthetic_csvs
        output_dir = tmp_path / 'model_output2'

        metrics = train_baseline(
            train_path=str(train_path),
            val_path=str(val_path),
            k=3,
            output_dir=str(output_dir),
            n_estimators=10,
            max_depth=3,
        )

        # Check all required metric keys
        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'weighted_f1' in metrics
        assert 'per_class' in metrics

        # Metrics should be valid floats in [0, 1]
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]}"

        # Per-class metrics should exist for each family in the data
        for family in FAMILIES:
            assert family in metrics['per_class']

        # Saved JSON should match returned dict
        with open(output_dir / 'val_metrics.json') as f:
            saved = json.load(f)
        assert saved['accuracy'] == metrics['accuracy']

    def test_label_encoder_content(self, synthetic_csvs, tmp_path):
        train_path, val_path = synthetic_csvs
        output_dir = tmp_path / 'model_output3'

        train_baseline(
            train_path=str(train_path),
            val_path=str(val_path),
            k=3,
            output_dir=str(output_dir),
            n_estimators=10,
            max_depth=3,
        )

        with open(output_dir / 'label_encoder.json') as f:
            label_encoder = json.load(f)

        for family in FAMILIES:
            assert family in label_encoder
        # Indices should be contiguous integers
        assert sorted(label_encoder.values()) == list(range(len(label_encoder)))

    def test_kmer_params_content(self, synthetic_csvs, tmp_path):
        train_path, val_path = synthetic_csvs
        output_dir = tmp_path / 'model_output4'

        train_baseline(
            train_path=str(train_path),
            val_path=str(val_path),
            k=3,
            output_dir=str(output_dir),
            n_estimators=10,
            max_depth=3,
        )

        with open(output_dir / 'kmer_params.json') as f:
            params = json.load(f)

        assert params['k'] == 3
        assert len(params['vocabulary']) == 4**3  # 64 k-mers

    def test_predict_matches_train_output(self, synthetic_csvs, tmp_path):
        train_path, val_path = synthetic_csvs
        output_dir = tmp_path / 'model_output5'

        train_baseline(
            train_path=str(train_path),
            val_path=str(val_path),
            k=3,
            output_dir=str(output_dir),
            n_estimators=10,
            max_depth=3,
        )

        y_true, y_pred, y_prob = predict_baseline(
            test_path=str(val_path),
            model_dir=str(output_dir),
        )

        assert len(y_true) == 9  # 3 families * 3 samples
        assert len(y_pred) == 9
        assert y_prob.shape == (9, 3)
        # Probabilities should sum to ~1 per sample
        for row in y_prob:
            assert abs(row.sum() - 1.0) < 1e-5
