"""
Unit tests for the model export module.

Covers ONNX export, Azure ML packaging, export validation, and CLI
argument parsing using mock models (no real trained model required).
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.dataset import FAMILIES
from src.export_model import (
    _generate_conda_yml,
    _generate_score_py,
    package_for_azure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CSV_HEADER = ['id', 'parent_sequence_id', 'chunk_index', 'sequence', 'length', 'family']

SAMPLE_SEQUENCES = []
for i, fam in enumerate(FAMILIES):
    for j in range(2):
        seq = ('ACGTACGT' * 75)[:600]
        SAMPLE_SEQUENCES.append(
            (f'{fam}_{j}', f'{fam}_parent', j, seq, 600, fam)
        )


def _write_csv(path, rows):
    """Write a CSV with the standard header and given rows."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(row)
    return str(path)


@pytest.fixture
def test_csv(tmp_path):
    return _write_csv(tmp_path / 'test.csv', SAMPLE_SEQUENCES)


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock model directory with label mappings."""
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir()

    # Save label mappings
    label_mappings = {
        "families": list(FAMILIES),
        "label_to_idx": {fam: i for i, fam in enumerate(FAMILIES)},
        "idx_to_label": {str(i): fam for i, fam in enumerate(FAMILIES)},
    }
    with open(model_dir / "label_mappings.json", "w") as f:
        json.dump(label_mappings, f)

    # Create minimal config.json
    config = {"model_type": "esm", "num_labels": 7}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create dummy tokenizer config
    tokenizer_config = {"model_max_length": 1000}
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    return model_dir


# ---------------------------------------------------------------------------
# Tests: score.py and conda.yml generation
# ---------------------------------------------------------------------------


class TestScorePyGeneration:
    """Tests for the Azure ML score.py scaffold."""

    def test_score_py_contains_init_and_run(self):
        content = _generate_score_py(list(FAMILIES))
        assert "def init():" in content
        assert "def run(raw_data):" in content

    def test_score_py_contains_families(self):
        content = _generate_score_py(list(FAMILIES))
        for fam in FAMILIES:
            assert fam in content

    def test_score_py_empty_families(self):
        content = _generate_score_py([])
        assert "FAMILIES = []" in content
        assert "def init():" in content

    def test_score_py_imports(self):
        content = _generate_score_py(list(FAMILIES))
        assert "import torch" in content
        assert "from transformers import" in content
        assert "AutoTokenizer" in content

    def test_score_py_is_valid_python(self):
        content = _generate_score_py(list(FAMILIES))
        # Should compile without syntax errors
        compile(content, "<score.py>", "exec")


class TestCondaYml:
    """Tests for the conda environment file."""

    def test_conda_yml_contains_dependencies(self):
        content = _generate_conda_yml()
        assert "torch" in content
        assert "transformers" in content
        assert "numpy" in content

    def test_conda_yml_name(self):
        content = _generate_conda_yml()
        assert "name: pathogen-classifier" in content


# ---------------------------------------------------------------------------
# Tests: package_for_azure
# ---------------------------------------------------------------------------


class TestPackageForAzure:
    """Tests for Azure ML packaging."""

    def test_creates_output_dir(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)
        assert output_dir.exists()

    def test_creates_score_py(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)
        assert (output_dir / "score.py").exists()

    def test_creates_conda_yml(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)
        assert (output_dir / "conda.yml").exists()

    def test_copies_model_artifacts(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)

        model_out = output_dir / "model"
        assert model_out.exists()
        assert (model_out / "config.json").exists()
        assert (model_out / "label_mappings.json").exists()
        assert (model_out / "tokenizer_config.json").exists()

    def test_label_mappings_in_score_py(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)

        score_content = (output_dir / "score.py").read_text()
        for fam in FAMILIES:
            assert fam in score_content

    def test_score_py_compiles(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        package_for_azure(mock_model_dir, output_dir)

        score_content = (output_dir / "score.py").read_text()
        compile(score_content, "<score.py>", "exec")

    def test_missing_label_mappings(self, tmp_path):
        """Azure packaging should still work without label_mappings.json."""
        model_dir = tmp_path / "model_no_labels"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        output_dir = tmp_path / "azure_output"
        package_for_azure(model_dir, output_dir)
        assert (output_dir / "score.py").exists()

    def test_returns_output_dir(self, mock_model_dir, tmp_path):
        output_dir = tmp_path / "azure_output"
        result = package_for_azure(mock_model_dir, output_dir)
        assert result == output_dir

    def test_copies_bin_files_when_no_safetensors(self, tmp_path):
        """When safetensors not present, .bin files should be copied."""
        model_dir = tmp_path / "model_with_bin"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"fake model weights")

        output_dir = tmp_path / "azure_output"
        package_for_azure(model_dir, output_dir)
        assert (output_dir / "model" / "pytorch_model.bin").exists()


# ---------------------------------------------------------------------------
# Tests: export_onnx (mocked)
# ---------------------------------------------------------------------------


class TestExportOnnx:
    """Tests for ONNX export using mocked models."""

    def test_export_calls_torch_onnx_export(self, tmp_path):
        """Verify export_onnx invokes torch.onnx.export with correct args."""
        from src.export_model import export_onnx

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        mock_onnx = MagicMock()
        output_path = tmp_path / "exports" / "model.onnx"

        with patch("src.export_model.AutoTokenizer") as tok_cls, \
             patch("src.export_model.AutoModelForSequenceClassification") as model_cls, \
             patch("torch.onnx.export") as mock_export:

            tok_cls.from_pretrained.return_value = mock_tokenizer
            model_cls.from_pretrained.return_value = mock_model

            # torch.onnx.export needs to create the file for stat()
            def create_file(*args, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"mock onnx data")
            mock_export.side_effect = create_file

            # Patch onnx import inside export_onnx
            import sys
            mock_onnx_module = MagicMock()
            with patch.dict(sys.modules, {"onnx": mock_onnx_module}):
                result = export_onnx(str(tmp_path / "model"), str(output_path))

        assert result == output_path
        mock_export.assert_called_once()
        # Check dynamic_axes were passed
        call_kwargs = mock_export.call_args
        assert "dynamic_axes" in call_kwargs.kwargs or \
            any(isinstance(a, dict) and "input_ids" in a for a in call_kwargs.args)

    def test_export_creates_parent_dir(self, tmp_path):
        """Export should create parent directories if needed."""
        from src.export_model import export_onnx

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        nested_path = tmp_path / "a" / "b" / "c" / "model.onnx"

        with patch("src.export_model.AutoTokenizer") as tok_cls, \
             patch("src.export_model.AutoModelForSequenceClassification") as model_cls, \
             patch("torch.onnx.export") as mock_export:

            tok_cls.from_pretrained.return_value = mock_tokenizer
            model_cls.from_pretrained.return_value = mock_model

            def create_file(*args, **kwargs):
                nested_path.write_bytes(b"mock onnx data")
            mock_export.side_effect = create_file

            import sys
            with patch.dict(sys.modules, {"onnx": MagicMock()}):
                export_onnx(str(tmp_path / "model"), str(nested_path))

        assert nested_path.parent.exists()

    def test_export_uses_correct_max_length(self, tmp_path):
        """Tokenizer should be called with the specified max_length."""
        from src.export_model import export_onnx

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        output_path = tmp_path / "model.onnx"

        with patch("src.export_model.AutoTokenizer") as tok_cls, \
             patch("src.export_model.AutoModelForSequenceClassification") as model_cls, \
             patch("torch.onnx.export") as mock_export:

            tok_cls.from_pretrained.return_value = mock_tokenizer
            model_cls.from_pretrained.return_value = mock_model

            def create_file(*args, **kwargs):
                output_path.write_bytes(b"mock onnx data")
            mock_export.side_effect = create_file

            import sys
            with patch.dict(sys.modules, {"onnx": MagicMock()}):
                export_onnx(str(tmp_path / "model"), str(output_path),
                            max_length=512)

        # Verify tokenizer was called with max_length=512
        mock_tokenizer.assert_called_once()
        _, kwargs = mock_tokenizer.call_args
        assert kwargs["max_length"] == 512

    def test_export_validates_onnx_structure(self, tmp_path):
        """The exported model should be validated with onnx.checker."""
        from src.export_model import export_onnx

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        output_path = tmp_path / "model.onnx"

        mock_onnx_module = MagicMock()

        with patch("src.export_model.AutoTokenizer") as tok_cls, \
             patch("src.export_model.AutoModelForSequenceClassification") as model_cls, \
             patch("torch.onnx.export") as mock_export:

            tok_cls.from_pretrained.return_value = mock_tokenizer
            model_cls.from_pretrained.return_value = mock_model

            def create_file(*args, **kwargs):
                output_path.write_bytes(b"mock onnx data")
            mock_export.side_effect = create_file

            import sys
            with patch.dict(sys.modules, {"onnx": mock_onnx_module}):
                export_onnx(str(tmp_path / "model"), str(output_path))

        mock_onnx_module.checker.check_model.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: validate_export (mocked)
# ---------------------------------------------------------------------------


class TestValidateExport:
    """Tests for ONNX validation against PyTorch."""

    def _run_validation(self, pt_logits, onnx_logits, test_csv, tmp_path,
                        n_samples=2):
        """Helper to run validate_export with mocked models."""
        from src.export_model import validate_export

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (n_samples, 32)),
            "attention_mask": torch.ones(n_samples, 32, dtype=torch.long),
        }

        mock_pt_model = MagicMock()
        mock_pt_model.eval.return_value = mock_pt_model
        mock_pt_output = MagicMock()
        mock_pt_output.logits = pt_logits
        mock_pt_model.return_value = mock_pt_output

        mock_session = MagicMock()
        mock_session.run.return_value = [onnx_logits]

        mock_ort_module = MagicMock()
        mock_ort_module.InferenceSession.return_value = mock_session

        onnx_path = tmp_path / "model.onnx"

        with patch("src.export_model.AutoTokenizer") as tok_cls, \
             patch("src.export_model.AutoModelForSequenceClassification") as model_cls:

            tok_cls.from_pretrained.return_value = mock_tokenizer
            model_cls.from_pretrained.return_value = mock_pt_model

            import sys
            with patch.dict(sys.modules, {"onnxruntime": mock_ort_module}):
                results = validate_export(
                    str(onnx_path), str(tmp_path), test_csv,
                    n_samples=n_samples,
                )

        return results

    def test_validate_passes_with_matching_output(self, test_csv, tmp_path):
        fixed_logits = torch.randn(2, 7)
        results = self._run_validation(
            pt_logits=fixed_logits,
            onnx_logits=fixed_logits.numpy(),
            test_csv=test_csv,
            tmp_path=tmp_path,
        )
        assert results["passed"] is True
        assert results["max_abs_diff"] == 0.0
        assert results["prediction_matches"] == results["prediction_total"]

    def test_validate_fails_with_divergent_output(self, test_csv, tmp_path):
        pt_logits = torch.zeros(2, 7)
        onnx_logits = np.ones((2, 7), dtype=np.float32)

        results = self._run_validation(
            pt_logits=pt_logits,
            onnx_logits=onnx_logits,
            test_csv=test_csv,
            tmp_path=tmp_path,
        )
        assert results["passed"] is False
        assert results["max_abs_diff"] > 1e-4

    def test_validate_n_samples(self, test_csv, tmp_path):
        n = 5
        fixed_logits = torch.randn(n, 7)
        results = self._run_validation(
            pt_logits=fixed_logits,
            onnx_logits=fixed_logits.numpy(),
            test_csv=test_csv,
            tmp_path=tmp_path,
            n_samples=n,
        )
        assert results["n_samples"] == n
        assert results["prediction_total"] == n

    def test_validate_returns_expected_keys(self, test_csv, tmp_path):
        fixed_logits = torch.randn(2, 7)
        results = self._run_validation(
            pt_logits=fixed_logits,
            onnx_logits=fixed_logits.numpy(),
            test_csv=test_csv,
            tmp_path=tmp_path,
        )
        expected_keys = {
            "passed", "n_samples", "max_abs_diff", "mean_abs_diff",
            "prediction_matches", "prediction_total", "tolerance",
        }
        assert set(results.keys()) == expected_keys

    def test_validate_tolerance_boundary(self, test_csv, tmp_path):
        """Output just barely within tolerance should pass."""
        pt_logits = torch.zeros(2, 7)
        # Add tiny difference just under 1e-4
        onnx_logits = np.full((2, 7), 9e-5, dtype=np.float32)

        results = self._run_validation(
            pt_logits=pt_logits,
            onnx_logits=onnx_logits,
            test_csv=test_csv,
            tmp_path=tmp_path,
        )
        assert results["passed"] is True
