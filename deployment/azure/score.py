"""
Azure ML scoring script for pathogen classification (XGBoost baseline).

Loaded by Azure ML managed online endpoints. The init() function runs once
at container startup; run() is called for each HTTP request.

The k-mer feature extraction is inlined here because Azure ML scoring
containers cannot import from the project's src/ package.
"""

import json
import logging
import os
import re
import time
from itertools import product

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

# Globals populated by init()
_booster = None
_label_to_idx = None
_idx_to_label = None
_kmer_k = None
_kmer_vocab = None
_kmer_to_idx = None


# ============================================================================
# K-mer feature extraction (inlined from src/baseline_model.py)
# ============================================================================

def _build_kmer_vocabulary(k):
    """Return a sorted list of all possible k-mers for the given k."""
    bases = ["A", "C", "G", "T"]
    return sorted(["".join(combo) for combo in product(bases, repeat=k)])


def _compute_kmer_frequencies(sequence, k):
    """Count all k-mer occurrences in a sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence (A, C, G, T, N characters).
    k : int
        Length of k-mers.

    Returns
    -------
    dict
        Mapping from k-mer string to its count in the sequence.
    """
    counts = {}
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if "N" in kmer:
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def _extract_features(sequences, k, vocab, kmer_to_idx):
    """Convert sequences to normalized k-mer frequency vectors.

    Parameters
    ----------
    sequences : list of str
        DNA sequences.
    k : int
        Length of k-mers.
    vocab : list of str
        Ordered list of all possible k-mers.
    kmer_to_idx : dict
        Mapping from k-mer string to feature index.

    Returns
    -------
    np.ndarray
        Array of shape (n_sequences, 4^k) with normalized frequencies.
    """
    n_features = len(vocab)
    features = np.zeros((len(sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        counts = _compute_kmer_frequencies(seq, k)
        for kmer, count in counts.items():
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = count

        total = features[i].sum()
        if total > 0:
            features[i] /= total

    return features


# ============================================================================
# Input validation
# ============================================================================

_VALID_DNA = re.compile(r"^[ACGTNacgtn]+$")


def _validate_sequences(sequences):
    """Validate input sequences.

    Parameters
    ----------
    sequences : list
        Candidate sequences to validate.

    Returns
    -------
    list of str
        Validated sequences.

    Raises
    ------
    ValueError
        If input is invalid.
    """
    if not isinstance(sequences, list):
        raise ValueError("'sequences' must be a list of strings")
    if len(sequences) == 0:
        raise ValueError("'sequences' must contain at least one sequence")
    for i, seq in enumerate(sequences):
        if not isinstance(seq, str) or len(seq) == 0:
            raise ValueError(f"Sequence at index {i} must be a non-empty string")
        if not _VALID_DNA.match(seq):
            raise ValueError(
                f"Sequence at index {i} contains invalid characters. "
                f"Only A, C, G, T, N are allowed."
            )


# ============================================================================
# Azure ML entry points
# ============================================================================


def init():
    """Called once when the endpoint container starts.

    Loads the XGBoost model, label encoder, and k-mer parameters from
    the AZUREML_MODEL_DIR environment variable.
    """
    global _booster, _label_to_idx, _idx_to_label
    global _kmer_k, _kmer_vocab, _kmer_to_idx

    model_dir = os.getenv("AZUREML_MODEL_DIR", "models/baseline")
    logger.info("Loading model from %s", model_dir)

    # Load XGBoost model
    model_path = os.path.join(model_dir, "xgboost_model.json")
    _booster = xgb.Booster()
    _booster.load_model(model_path)
    logger.info("XGBoost model loaded from %s", model_path)

    # Load label encoder
    label_path = os.path.join(model_dir, "label_encoder.json")
    with open(label_path) as f:
        _label_to_idx = json.load(f)
    _idx_to_label = {v: k for k, v in _label_to_idx.items()}
    logger.info("Label encoder loaded: %d classes", len(_label_to_idx))

    # Load k-mer parameters
    kmer_path = os.path.join(model_dir, "kmer_params.json")
    with open(kmer_path) as f:
        kmer_params = json.load(f)
    _kmer_k = kmer_params["k"]
    _kmer_vocab = kmer_params["vocabulary"]
    _kmer_to_idx = {kmer: i for i, kmer in enumerate(_kmer_vocab)}
    logger.info("K-mer params loaded: k=%d, vocab_size=%d", _kmer_k, len(_kmer_vocab))


def run(raw_data):
    """Called for each scoring request.

    Parameters
    ----------
    raw_data : str
        JSON string with key "sequences" containing a list of DNA sequences.

        Example::

            {"sequences": ["ACGTACGT...", "TGCATGCA..."]}

    Returns
    -------
    str
        JSON string with predictions.

        Example::

            {
                "predictions": [
                    {
                        "predicted_family": "Coronaviridae",
                        "confidence": 0.998,
                        "probabilities": {"Poxviridae": 0.001, ...}
                    }
                ]
            }
    """
    start_time = time.time()

    try:
        data = json.loads(raw_data)

        if "sequences" not in data:
            raise ValueError("Request must contain a 'sequences' key")

        sequences = data["sequences"]
        _validate_sequences(sequences)

        # Extract k-mer features
        features = _extract_features(sequences, _kmer_k, _kmer_vocab, _kmer_to_idx)
        dmatrix = xgb.DMatrix(features)

        # Predict
        probabilities = _booster.predict(dmatrix)
        predictions = np.argmax(probabilities, axis=1)

        # Build response
        results = []
        for i in range(len(sequences)):
            pred_idx = int(predictions[i])
            pred_family = _idx_to_label[pred_idx]
            confidence = float(probabilities[i][pred_idx])

            prob_dict = {
                _idx_to_label[j]: float(probabilities[i][j])
                for j in range(probabilities.shape[1])
            }

            results.append({
                "predicted_family": pred_family,
                "confidence": confidence,
                "probabilities": prob_dict,
            })

        elapsed = time.time() - start_time
        logger.info(
            "Scored %d sequences in %.3fs (%.1f seq/s)",
            len(sequences), elapsed, len(sequences) / elapsed,
        )

        return json.dumps({"predictions": results})

    except ValueError as e:
        logger.warning("Validation error: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Scoring error: %s", e, exc_info=True)
        return json.dumps({"error": f"Internal error: {str(e)}"})
