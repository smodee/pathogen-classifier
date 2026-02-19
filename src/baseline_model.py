"""
Baseline classifier using k-mer frequency features + XGBoost.

Establishes a performance floor before investing in transformer training
and validates the full train/evaluate pipeline.

Usage:
    python src/baseline_model.py --train data/processed/train.csv --val data/processed/val.csv
    python src/baseline_model.py --train data/processed/train.csv --val data/processed/val.csv --k 4
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from src.data_exploration import FAMILY_ORDER


def compute_kmer_frequencies(sequence, k=5):
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
        kmer = seq[i:i + k]
        # Skip k-mers containing ambiguous bases
        if 'N' in kmer:
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def _build_kmer_vocabulary(k):
    """Return a sorted list of all possible k-mers for the given k."""
    bases = ['A', 'C', 'G', 'T']
    return sorted([''.join(combo) for combo in product(bases, repeat=k)])


def extract_features(sequences, k=5):
    """Convert sequences to normalized k-mer frequency vectors.

    Parameters
    ----------
    sequences : list of str
        DNA sequences.
    k : int
        Length of k-mers.

    Returns
    -------
    np.ndarray
        Array of shape (n_sequences, 4^k) with normalized frequencies.
    """
    vocab = _build_kmer_vocabulary(k)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(vocab)}
    n_features = len(vocab)

    features = np.zeros((len(sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        counts = compute_kmer_frequencies(seq, k)
        for kmer, count in counts.items():
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = count

        # Normalize to frequencies (sum to ~1.0)
        total = features[i].sum()
        if total > 0:
            features[i] /= total

    return features


def _compute_class_weights(labels):
    """Compute inverse-frequency class weights.

    Parameters
    ----------
    labels : np.ndarray
        Integer-encoded class labels.

    Returns
    -------
    np.ndarray
        Per-sample weights based on inverse class frequency.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)

    # weight_c = n_samples / (n_classes * count_c)
    class_weight = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}

    return np.array([class_weight[label] for label in labels], dtype=np.float32)


def train_baseline(train_path, val_path, k=5, output_dir='models/baseline',
                   n_estimators=300, max_depth=6, learning_rate=0.1,
                   random_state=42):
    """Full training pipeline: load data, extract features, train, evaluate.

    Parameters
    ----------
    train_path : str or Path
        Path to training CSV.
    val_path : str or Path
        Path to validation CSV.
    k : int
        K-mer size.
    output_dir : str or Path
        Directory to save model artifacts.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Boosting learning rate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Validation metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # Encode labels using the full FAMILY_ORDER for consistent ordering
    # with the transformer pipeline (dataset.py). XGBoost handles classes
    # with zero training samples via num_class.
    label_order = list(FAMILY_ORDER)
    label_to_idx = {name: i for i, name in enumerate(label_order)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    y_train = np.array([label_to_idx[f] for f in train_df['family']])
    y_val = np.array([label_to_idx[f] for f in val_df['family']])

    # Extract features
    print(f"Extracting {k}-mer features...")
    vocab = _build_kmer_vocabulary(k)
    X_train = extract_features(train_df['sequence'].tolist(), k)
    X_val = extract_features(val_df['sequence'].tolist(), k)
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Compute sample weights for class imbalance
    sample_weights = _compute_class_weights(y_train)

    # Train XGBoost using the native API so that num_class is respected
    # even when some classes have zero training samples.
    print("Training XGBoost...")
    n_classes = len(label_order)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'seed': random_state,
        'verbosity': 1,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'val')],
        verbose_eval=True,
    )

    # Wrap in XGBClassifier for consistent save/load with predict_baseline
    model = xgb.XGBClassifier()
    model._Booster = booster
    model.n_classes_ = n_classes

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_prob = booster.predict(dval)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = _compute_metrics(y_val, y_pred, y_prob, idx_to_label)

    # Print summary
    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")

    # Save artifacts
    print(f"\nSaving to {output_dir}/...")

    booster.save_model(str(output_dir / 'xgboost_model.json'))

    with open(output_dir / 'label_encoder.json', 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    with open(output_dir / 'kmer_params.json', 'w') as f:
        json.dump({'k': k, 'vocabulary': vocab}, f, indent=2)

    with open(output_dir / 'val_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    return metrics


def _compute_metrics(y_true, y_pred, y_prob, idx_to_label):
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray
        Predicted probabilities of shape (n, n_classes).
    idx_to_label : dict
        Mapping from index to family name.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    labels = list(range(len(target_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    per_class = {}
    for name in target_names:
        if name in report:
            per_class[name] = {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1-score': report[name]['f1-score'],
                'support': report[name]['support'],
            }

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)),
        'per_class': per_class,
    }


def predict_baseline(test_path, model_dir='models/baseline'):
    """Load a saved model and predict on test data.

    Parameters
    ----------
    test_path : str or Path
        Path to test CSV.
    model_dir : str or Path
        Directory containing saved model artifacts.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (y_true, y_pred, y_prob) — true labels, predicted labels,
        and predicted probabilities.
    """
    model_dir = Path(model_dir)

    # Load model (native Booster for consistent handling of num_class)
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'xgboost_model.json'))

    # Load label encoder
    with open(model_dir / 'label_encoder.json') as f:
        label_to_idx = json.load(f)

    # Load k-mer params
    with open(model_dir / 'kmer_params.json') as f:
        kmer_params = json.load(f)
    k = kmer_params['k']

    # Load and process test data
    test_df = pd.read_csv(test_path)
    y_true = np.array([label_to_idx[f] for f in test_df['family']])

    X_test = extract_features(test_df['sequence'].tolist(), k)
    dtest = xgb.DMatrix(X_test)

    y_prob = booster.predict(dtest)
    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(
        description='Train baseline k-mer + XGBoost classifier'
    )
    parser.add_argument(
        '--train', required=True,
        help='Path to training CSV (e.g. data/processed/train.csv)'
    )
    parser.add_argument(
        '--val', required=True,
        help='Path to validation CSV (e.g. data/processed/val.csv)'
    )
    parser.add_argument('--k', type=int, default=5, help='K-mer size (default: 5)')
    parser.add_argument(
        '--output-dir', default='models/baseline',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--n-estimators', type=int, default=300,
        help='Number of boosting rounds (default: 300)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1,
        help='Boosting learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    train_baseline(
        train_path=args.train,
        val_path=args.val,
        k=args.k,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )


if __name__ == '__main__':
    main()
