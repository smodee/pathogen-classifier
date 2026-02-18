"""
Unit tests for the unified evaluation pipeline.

Verifies metric computation, plot generation, JSON serialization, and the
model comparison workflow using synthetic data (no trained model required).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.evaluate import (
    compare_models,
    compute_all_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_NAMES = ["ClassA", "ClassB", "ClassC"]
NUM_CLASSES = len(LABEL_NAMES)


def _make_predictions(n=200, seed=42, perfect=False, all_same=False):
    """Generate synthetic ground-truth and prediction arrays.

    Args:
        n: number of samples.
        seed: random seed for reproducibility.
        perfect: if True, predictions equal ground truth.
        all_same: if True, predictions are all class 0.

    Returns:
        (y_true, y_pred, y_prob) tuple.
    """
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, NUM_CLASSES, size=n)

    if perfect:
        y_pred = y_true.copy()
    elif all_same:
        y_pred = np.zeros(n, dtype=int)
    else:
        y_pred = rng.randint(0, NUM_CLASSES, size=n)

    # Build plausible probability matrix
    y_prob = rng.dirichlet(np.ones(NUM_CLASSES), size=n).astype(np.float32)
    # For perfect predictions, make the true class dominant
    if perfect:
        y_prob = np.eye(NUM_CLASSES, dtype=np.float32)[y_true]

    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Tests: compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_keys_present(self):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        assert "accuracy" in m
        assert "balanced_accuracy" in m
        assert "macro_f1" in m
        assert "weighted_f1" in m
        assert "per_class" in m
        assert "confusion_matrix" in m
        assert "label_names" in m

    def test_perfect_predictions(self):
        y_true, y_pred, y_prob = _make_predictions(perfect=True)
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        assert m["accuracy"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["weighted_f1"] == pytest.approx(1.0)

        for name in LABEL_NAMES:
            assert m["per_class"][name]["precision"] == pytest.approx(1.0)
            assert m["per_class"][name]["recall"] == pytest.approx(1.0)
            assert m["per_class"][name]["f1"] == pytest.approx(1.0)

    def test_random_predictions_ranges(self):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert 0.0 <= m["weighted_f1"] <= 1.0

        for name in LABEL_NAMES:
            cls = m["per_class"][name]
            assert 0.0 <= cls["precision"] <= 1.0
            assert 0.0 <= cls["recall"] <= 1.0
            assert 0.0 <= cls["f1"] <= 1.0
            assert cls["support"] >= 0

    def test_all_same_predictions(self):
        y_true, y_pred, y_prob = _make_predictions(all_same=True)
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        # Only class 0 should have non-zero recall
        assert m["per_class"][LABEL_NAMES[0]]["recall"] == pytest.approx(1.0)
        for name in LABEL_NAMES[1:]:
            assert m["per_class"][name]["recall"] == pytest.approx(0.0)

        # Balanced accuracy should be low (only 1/3 of classes recalled)
        assert m["balanced_accuracy"] < 0.5

    def test_confusion_matrix_shape(self):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_confusion_matrix_row_sums(self):
        """Row sums of confusion matrix should equal per-class support."""
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)
        cm = np.array(m["confusion_matrix"])

        for i, name in enumerate(LABEL_NAMES):
            assert cm[i].sum() == m["per_class"][name]["support"]

    def test_label_names_preserved(self):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)
        assert m["label_names"] == LABEL_NAMES

    def test_none_y_prob(self):
        """y_prob=None should not cause an error."""
        y_true, y_pred, _ = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, None, LABEL_NAMES)
        assert "accuracy" in m

    def test_metric_types(self):
        """All scalar metrics should be plain floats (JSON-serializable)."""
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        assert isinstance(m["accuracy"], float)
        assert isinstance(m["balanced_accuracy"], float)
        assert isinstance(m["macro_f1"], float)
        assert isinstance(m["weighted_f1"], float)

        for cls in m["per_class"].values():
            assert isinstance(cls["precision"], float)
            assert isinstance(cls["recall"], float)
            assert isinstance(cls["f1"], float)
            assert isinstance(cls["support"], int)


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_metrics_round_trip(self, tmp_path):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(m, f)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == pytest.approx(m["accuracy"])
        assert loaded["label_names"] == m["label_names"]
        assert loaded["confusion_matrix"] == m["confusion_matrix"]


# ---------------------------------------------------------------------------
# Tests: plot generation
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    def test_creates_file(self, tmp_path):
        cm = [[50, 5, 2], [3, 45, 8], [1, 6, 40]]
        output = tmp_path / "cm.png"
        plot_confusion_matrix(cm, LABEL_NAMES, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_handles_zero_row(self, tmp_path):
        """A row of zeros (no true samples) should not cause an error."""
        cm = [[50, 5, 2], [0, 0, 0], [1, 6, 40]]
        output = tmp_path / "cm_zero.png"
        plot_confusion_matrix(cm, LABEL_NAMES, output)
        assert output.exists()
        assert output.stat().st_size > 0


class TestPlotPerClassF1:
    def test_creates_file(self, tmp_path):
        y_true, y_pred, y_prob = _make_predictions()
        m = compute_all_metrics(y_true, y_pred, y_prob, LABEL_NAMES)

        output = tmp_path / "f1.png"
        plot_per_class_f1(m, LABEL_NAMES, output)
        assert output.exists()
        assert output.stat().st_size > 0


class TestPlotModelComparison:
    def test_creates_file(self, tmp_path):
        m1 = {"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.7, "weighted_f1": 0.78}
        m2 = {"accuracy": 0.9, "balanced_accuracy": 0.88, "macro_f1": 0.85, "weighted_f1": 0.89}
        output = tmp_path / "comparison.png"
        plot_model_comparison([m1, m2], ["ModelA", "ModelB"], output)
        assert output.exists()
        assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests: compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def _write_metric_files(self, tmp_path):
        """Write two synthetic metric JSON files and return their paths."""
        m1 = {
            "accuracy": 0.80,
            "balanced_accuracy": 0.75,
            "macro_f1": 0.70,
            "weighted_f1": 0.78,
            "per_class": {
                n: {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 50}
                for n in LABEL_NAMES
            },
            "confusion_matrix": [[40, 5, 5], [5, 35, 10], [3, 7, 40]],
            "label_names": LABEL_NAMES,
        }
        m2 = {
            "accuracy": 0.90,
            "balanced_accuracy": 0.88,
            "macro_f1": 0.85,
            "weighted_f1": 0.89,
            "per_class": {
                n: {"precision": 0.85, "recall": 0.85, "f1": 0.85, "support": 50}
                for n in LABEL_NAMES
            },
            "confusion_matrix": [[45, 3, 2], [2, 43, 5], [1, 4, 45]],
            "label_names": LABEL_NAMES,
        }

        p1 = tmp_path / "baseline_metrics.json"
        p2 = tmp_path / "transformer_metrics.json"
        for p, m in [(p1, m1), (p2, m2)]:
            with open(p, "w") as f:
                json.dump(m, f)
        return [p1, p2]

    def test_produces_csv(self, tmp_path):
        files = self._write_metric_files(tmp_path)
        out = tmp_path / "compare_out"
        compare_models(files, out)

        csv_path = out / "model_comparison.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "model" in df.columns
        assert "macro_f1" in df.columns

    def test_produces_chart(self, tmp_path):
        files = self._write_metric_files(tmp_path)
        out = tmp_path / "compare_out"
        compare_models(files, out)

        chart = out / "model_comparison.png"
        assert chart.exists()
        assert chart.stat().st_size > 0

    def test_returns_dataframe(self, tmp_path):
        files = self._write_metric_files(tmp_path)
        out = tmp_path / "compare_out"
        df = compare_models(files, out)

        assert isinstance(df, pd.DataFrame)
        assert list(df["model"]) == ["baseline", "transformer"]


# Need pandas import for compare_models test
import pandas as pd
