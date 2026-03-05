"""Tests for the Azure ML scoring script (deployment/azure/score.py).

Validates that the scoring script correctly loads the trained XGBoost
baseline model and produces predictions matching the main pipeline.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path so we can import from deployment/azure/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "deployment" / "azure"))

import score  # noqa: E402
from src.baseline_model import (  # noqa: E402
    compute_kmer_frequencies as baseline_kmer_frequencies,
    extract_features as baseline_extract_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"

# Skip all tests if the trained model is not available locally
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "xgboost_model.json").exists(),
    reason="Trained baseline model not found in models/baseline/",
)


@pytest.fixture(autouse=True, scope="module")
def _init_scoring():
    """Initialize the scoring script once for all tests in this module."""
    os.environ["AZUREML_MODEL_DIR"] = str(MODEL_DIR)
    score.init()
    yield
    # Cleanup env var
    os.environ.pop("AZUREML_MODEL_DIR", None)


SAMPLE_SEQUENCE = (
    "ATGGATCCAACATTTCCATTGGGTTCTACTAAAGAGTTTTGTGAACGAACCGC"
    "TGATTTCTTCAATCCGGTGGGTAAAGTTGTCGAGGTAGTTTTATCAAGTATCG"
    "GATGGACTCATCGAGTTAATCTGTCAATGCCTGCTATAGCAGTGTTTGTTAAT"
    "ACAGGCAGCTCCCTTCGCAATCCAATCCTAACTGCGGATGACTCTGTTTGTAA"
)


# ---------------------------------------------------------------------------
# init() tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for the init() function."""

    def test_model_loaded(self):
        """Booster should be loaded after init()."""
        assert score._booster is not None

    def test_labels_loaded(self):
        """Label encoder should have 7 virus families."""
        assert len(score._label_to_idx) == 7
        assert len(score._idx_to_label) == 7
        assert "Coronaviridae" in score._label_to_idx
        assert "Poxviridae" in score._label_to_idx

    def test_kmer_params_loaded(self):
        """K-mer parameters should be loaded correctly."""
        assert score._kmer_k == 5
        assert len(score._kmer_vocab) == 4 ** 5  # 1024
        assert len(score._kmer_to_idx) == 4 ** 5


# ---------------------------------------------------------------------------
# run() tests — valid inputs
# ---------------------------------------------------------------------------


class TestRunValid:
    """Tests for run() with valid inputs."""

    def test_single_sequence(self):
        """run() should return one prediction for one sequence."""
        request = json.dumps({"sequences": [SAMPLE_SEQUENCE]})
        response = json.loads(score.run(request))

        assert "predictions" in response
        assert len(response["predictions"]) == 1

        pred = response["predictions"][0]
        assert "predicted_family" in pred
        assert "confidence" in pred
        assert "probabilities" in pred

        assert pred["predicted_family"] in score._label_to_idx
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_multiple_sequences(self):
        """run() should return one prediction per input sequence."""
        sequences = [SAMPLE_SEQUENCE, SAMPLE_SEQUENCE[:100], SAMPLE_SEQUENCE + "ACGT" * 20]
        request = json.dumps({"sequences": sequences})
        response = json.loads(score.run(request))

        assert len(response["predictions"]) == 3

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to approximately 1.0."""
        request = json.dumps({"sequences": [SAMPLE_SEQUENCE]})
        response = json.loads(score.run(request))

        probs = response["predictions"][0]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_contain_all_families(self):
        """Probabilities should include all 7 families."""
        request = json.dumps({"sequences": [SAMPLE_SEQUENCE]})
        response = json.loads(score.run(request))

        probs = response["predictions"][0]["probabilities"]
        assert len(probs) == 7
        for family in score._label_to_idx:
            assert family in probs

    def test_confidence_matches_max_probability(self):
        """Confidence should equal the probability of the predicted family."""
        request = json.dumps({"sequences": [SAMPLE_SEQUENCE]})
        response = json.loads(score.run(request))

        pred = response["predictions"][0]
        predicted_prob = pred["probabilities"][pred["predicted_family"]]
        assert abs(pred["confidence"] - predicted_prob) < 1e-6

    def test_lowercase_sequence(self):
        """Lowercase DNA sequences should be accepted."""
        request = json.dumps({"sequences": [SAMPLE_SEQUENCE.lower()]})
        response = json.loads(score.run(request))
        assert "predictions" in response
        assert len(response["predictions"]) == 1

    def test_sequence_with_n_bases(self):
        """Sequences with N (ambiguous) bases should be accepted."""
        seq_with_n = "ACGTNNNACGT" * 20
        request = json.dumps({"sequences": [seq_with_n]})
        response = json.loads(score.run(request))
        assert "predictions" in response


# ---------------------------------------------------------------------------
# run() tests — invalid inputs
# ---------------------------------------------------------------------------


class TestRunInvalid:
    """Tests for run() with invalid inputs."""

    def test_missing_sequences_key(self):
        """Missing 'sequences' key should return an error."""
        request = json.dumps({"data": ["ACGT"]})
        response = json.loads(score.run(request))
        assert "error" in response

    def test_empty_sequences_list(self):
        """Empty sequences list should return an error."""
        request = json.dumps({"sequences": []})
        response = json.loads(score.run(request))
        assert "error" in response

    def test_invalid_characters(self):
        """Non-DNA characters should return an error."""
        request = json.dumps({"sequences": ["ACGTXYZ123"]})
        response = json.loads(score.run(request))
        assert "error" in response

    def test_empty_string_sequence(self):
        """Empty string in sequences should return an error."""
        request = json.dumps({"sequences": [""]})
        response = json.loads(score.run(request))
        assert "error" in response

    def test_non_string_sequence(self):
        """Non-string values in sequences should return an error."""
        request = json.dumps({"sequences": [12345]})
        response = json.loads(score.run(request))
        assert "error" in response

    def test_invalid_json(self):
        """Invalid JSON should return an error."""
        response = json.loads(score.run("not valid json"))
        assert "error" in response


# ---------------------------------------------------------------------------
# K-mer extraction consistency
# ---------------------------------------------------------------------------


class TestKmerConsistency:
    """Verify that the inlined k-mer code matches src/baseline_model.py."""

    def test_kmer_frequencies_match(self):
        """Inlined k-mer counting should match baseline_model.py."""
        seq = "ACGTACGTACGTNNNNACGTACGTACGT"
        k = 5

        # Score.py version
        score_counts = score._compute_kmer_frequencies(seq, k)

        # baseline_model.py version
        baseline_counts = baseline_kmer_frequencies(seq, k)

        assert score_counts == baseline_counts

    def test_feature_vectors_match(self):
        """Inlined feature extraction should match baseline_model.py."""
        sequences = [SAMPLE_SEQUENCE, "ACGT" * 50, "GGCCAATTGGCCAATT" * 10]
        k = 5

        # Score.py version
        vocab = score._build_kmer_vocabulary(k)
        kmer_to_idx = {kmer: i for i, kmer in enumerate(vocab)}
        score_features = score._extract_features(sequences, k, vocab, kmer_to_idx)

        # baseline_model.py version
        baseline_features = baseline_extract_features(sequences, k)

        np.testing.assert_array_almost_equal(score_features, baseline_features)

    def test_vocabulary_match(self):
        """K-mer vocabulary should match between score.py and baseline."""
        from src.baseline_model import _build_kmer_vocabulary as baseline_vocab
        assert score._build_kmer_vocabulary(5) == baseline_vocab(5)


# ---------------------------------------------------------------------------
# Validation function tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for the _validate_sequences helper."""

    def test_valid_sequences(self):
        """Valid DNA sequences should pass validation."""
        score._validate_sequences(["ACGT", "acgt", "NNNN"])

    def test_rejects_non_list(self):
        """Non-list input should raise ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            score._validate_sequences("ACGT")

    def test_rejects_empty_list(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            score._validate_sequences([])

    def test_rejects_empty_string(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            score._validate_sequences([""])

    def test_rejects_invalid_chars(self):
        """Invalid characters should raise ValueError."""
        with pytest.raises(ValueError, match="invalid characters"):
            score._validate_sequences(["ACGT123"])

    def test_accepts_iupac_codes(self):
        """IUPAC ambiguity codes should be accepted."""
        score._validate_sequences(["ACGTRYWSMKHBDVN"])

    def test_rejects_non_string(self):
        """Non-string elements should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            score._validate_sequences([42])
