"""
Unified evaluation pipeline for pathogen classification models.

Produces metrics, confusion matrices, per-class breakdowns, and model
comparison charts.  Works for both baseline (XGBoost / sklearn) and
transformer (Nucleotide Transformer v2) models.

Usage:
    # Evaluate a single model
    python src/evaluate.py --model-type baseline --model-dir models/baseline \
        --test data/processed/test.csv

    # Evaluate transformer model
    python src/evaluate.py --model-type transformer --model-dir models/nt-v2 \
        --test data/processed/test.csv

    # Compare saved metric files
    python src/evaluate.py --compare reports/baseline_metrics.json \
        reports/nt_v2_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from data_exploration import FAMILY_ORDER
except ModuleNotFoundError:
    from src.data_exploration import FAMILY_ORDER

# ---------------------------------------------------------------------------
# Lazy plotting (matches data_exploration.py pattern)
# ---------------------------------------------------------------------------

_plt = None
_sns = None


def _init_plotting():
    """Lazily import matplotlib and seaborn."""
    global _plt, _sns
    if _plt is None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=1.1)
        _plt = plt
        _sns = sns
    return _plt, _sns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(y_true, y_pred, y_prob, label_names):
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: array-like of true integer labels.
        y_pred: array-like of predicted integer labels.
        y_prob: array-like of shape (n_samples, n_classes) with predicted
            probabilities, or ``None`` if unavailable.
        label_names: list of class name strings (index matches label int).

    Returns:
        dict with scalar metrics, per-class breakdown, and confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(label_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }

    # Per-class precision / recall / F1
    per_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
            "f1": float(per_f1[i]),
            "support": int((y_true == i).sum()),
        }
    metrics["per_class"] = per_class

    # Confusion matrix (rows = true, cols = pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["label_names"] = list(label_names)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_matrix(cm, label_names, output_path):
    """Save a normalized confusion-matrix heatmap.

    Args:
        cm: 2-D array (or list-of-lists) confusion matrix.
        label_names: list of class name strings.
        output_path: file path for the saved PNG.
    """
    plt, sns = _init_plotting()
    cm = np.asarray(cm, dtype=float)

    # Row-normalize (per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_class_f1(metrics, label_names, output_path):
    """Save a horizontal bar chart of per-class F1 scores.

    Args:
        metrics: dict returned by :func:`compute_all_metrics`.
        label_names: list of class name strings.
        output_path: file path for the saved PNG.
    """
    plt, sns = _init_plotting()

    f1_scores = [metrics["per_class"][name]["f1"] for name in label_names]
    colors = sns.color_palette("colorblind", n_colors=len(label_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(label_names, f1_scores, color=colors)
    for bar, val in zip(bars, f1_scores):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.invert_yaxis()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_model_comparison(metrics_list, model_names, output_path):
    """Save a grouped bar chart comparing models on key metrics.

    Args:
        metrics_list: list of metric dicts (one per model).
        model_names: list of human-readable model names.
        output_path: file path for the saved PNG.
    """
    plt, sns = _init_plotting()

    key_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(key_metrics))
    width = 0.8 / len(model_names)
    colors = sns.color_palette("colorblind", n_colors=len(model_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, name) in enumerate(zip(metrics_list, model_names)):
        vals = [m[k] for k in key_metrics]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in key_metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model_type, model_dir, test_csv, output_dir):
    """End-to-end evaluation for one model.

    Loads the model and test data, generates predictions, computes metrics,
    and saves outputs (JSON + plots).

    Args:
        model_type: ``"baseline"`` or ``"transformer"``.
        model_dir: path to saved model artefacts.
        test_csv: path to the test split CSV.
        output_dir: directory to write outputs into.

    Returns:
        dict of computed metrics.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(test_csv)
    label_names = [f for f in FAMILY_ORDER if f in test_df["family"].unique()]

    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y_true = np.array([label_to_idx[f] for f in test_df["family"]])

    # Obtain predictions from the appropriate model
    if model_type == "baseline":
        y_pred, y_prob = _predict_baseline(model_dir, test_df, label_names)
    elif model_type == "transformer":
        y_pred, y_prob = _predict_transformer(model_dir, test_df, label_names)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob, label_names)

    # Save JSON
    prefix = model_type
    metrics_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {metrics_path}")

    # Plots
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        label_names,
        output_dir / f"{prefix}_confusion_matrix.png",
    )
    plot_per_class_f1(
        metrics,
        label_names,
        output_dir / f"{prefix}_per_class_f1.png",
    )

    return metrics


def _predict_baseline(model_dir, test_df, label_names):
    """Load a saved baseline model and return predictions."""
    import joblib

    model_path = model_dir / "model.joblib"
    model = joblib.load(model_path)

    X_test = _extract_kmer_features(test_df["sequence"])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return y_pred, y_prob


def _predict_transformer(model_dir, test_df, label_names):
    """Load a saved transformer model and return predictions."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_preds = []
    all_probs = []
    batch_size = 16
    sequences = test_df["sequence"].tolist()

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_probs)


def _extract_kmer_features(sequences, k=4):
    """Convert sequences to k-mer frequency vectors for the baseline model."""
    from itertools import product

    kmers = ["".join(combo) for combo in product("ACGT", repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}

    rows = []
    for seq in sequences:
        seq = seq.upper()
        counts = np.zeros(len(kmers), dtype=np.float32)
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if kmer in kmer_to_idx:
                counts[kmer_to_idx[kmer]] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        rows.append(counts)
    return np.array(rows)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(metric_files, output_dir):
    """Compare multiple models from their saved metric JSON files.

    Produces a comparison CSV and chart.

    Args:
        metric_files: list of paths to ``*_metrics.json`` files.
        output_dir: directory to write outputs into.

    Returns:
        pandas DataFrame with the comparison table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    model_names = []
    for path in metric_files:
        path = Path(path)
        with open(path) as f:
            metrics_list.append(json.load(f))
        model_names.append(path.stem.replace("_metrics", ""))

    # Build comparison table
    key_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    rows = []
    for name, m in zip(model_names, metrics_list):
        row = {"model": name}
        for k in key_metrics:
            row[k] = m[k]
        rows.append(row)
    comparison_df = pd.DataFrame(rows)

    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Chart
    plot_model_comparison(
        metrics_list,
        model_names,
        output_dir / "model_comparison.png",
    )

    return comparison_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Entry-point for the evaluation CLI."""
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Evaluate pathogen classification models."
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "transformer"],
        help="Type of model to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to saved model artefacts.",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Path to test CSV (must contain 'sequence' and 'family' columns).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save outputs (default: reports).",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="JSON",
        help="Metric JSON files to compare (at least 2).",
    )
    args = parser.parse_args()

    if args.compare:
        if len(args.compare) < 2:
            parser.error("--compare requires at least 2 metric JSON files.")
        compare_models(args.compare, args.output_dir)
    elif args.model_type and args.model_dir and args.test:
        evaluate_model(args.model_type, args.model_dir, args.test, args.output_dir)
    else:
        parser.error(
            "Provide --model-type, --model-dir, and --test for evaluation, "
            "or --compare with metric JSON files."
        )


if __name__ == "__main__":
    main()
