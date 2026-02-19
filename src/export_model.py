"""
Export trained models for Azure ML deployment.

Supports ONNX export with dynamic axes for variable-length input, output
validation against PyTorch, and Azure ML packaging with a score.py scaffold.

Usage:
    # Export transformer model to ONNX
    python src/export_model.py --model-dir models/nt-v2 --format onnx \
        --output models/exports/nt_v2.onnx

    # Export and validate against test data
    python src/export_model.py --model-dir models/nt-v2 --format onnx \
        --output models/exports/nt_v2.onnx --validate --test-csv data/processed/test.csv

    # Package for Azure ML deployment
    python src/export_model.py --model-dir models/nt-v2 --format azure \
        --output models/exports/azure_package
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model_dir, output_path, max_length=1000):
    """Export a trained transformer model to ONNX format.

    Args:
        model_dir: Path to saved HuggingFace model directory.
        output_path: Destination path for the .onnx file.
        max_length: Maximum token length for dynamic axes (default 1000).

    Returns:
        Path to the exported ONNX file.
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )
    model.eval()

    # Create dummy input for tracing
    dummy_sequence = "ACGT" * 50  # 200bp dummy sequence
    dummy_inputs = tokenizer(
        dummy_sequence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]

    # Dynamic axes allow variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate the ONNX model structure
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Opset version: 14")
    print(f"  Dynamic axes: batch_size, sequence_length")

    return output_path


# ---------------------------------------------------------------------------
# Azure ML packaging
# ---------------------------------------------------------------------------

def package_for_azure(model_dir, output_dir):
    """Package a trained model for Azure ML endpoint deployment.

    Copies the model artifacts, tokenizer config, label mappings, and creates
    a score.py scaffold for the Azure ML managed endpoint.

    Args:
        model_dir: Path to saved HuggingFace model directory.
        output_dir: Destination directory for the Azure ML package.

    Returns:
        Path to the output directory.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model artifacts
    model_out = output_dir / "model"
    model_out.mkdir(parents=True, exist_ok=True)

    artifacts = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "label_mappings.json",
    ]
    copied = []
    for name in artifacts:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, model_out / name)
            copied.append(name)

    # Also copy any .bin files if safetensors not present
    if not (model_out / "model.safetensors").exists():
        for bin_file in model_dir.glob("*.bin"):
            shutil.copy2(bin_file, model_out / bin_file.name)
            copied.append(bin_file.name)

    print(f"  Copied artifacts to {model_out}: {', '.join(copied)}")

    # Load label mappings for the score.py scaffold
    label_path = model_dir / "label_mappings.json"
    if label_path.exists():
        with open(label_path) as f:
            label_mappings = json.load(f)
        families = label_mappings.get("families", [])
    else:
        families = []
        print("  Warning: label_mappings.json not found in model_dir")

    # Create score.py scaffold
    score_path = output_dir / "score.py"
    score_content = _generate_score_py(families)
    with open(score_path, "w") as f:
        f.write(score_content)
    print(f"  Created {score_path}")

    # Create conda environment file
    env_path = output_dir / "conda.yml"
    env_content = _generate_conda_yml()
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"  Created {env_path}")

    print(f"\nAzure ML package ready at: {output_dir}")
    return output_dir


def _generate_score_py(families):
    """Generate the score.py scaffold for Azure ML managed endpoints."""
    families_repr = repr(families)
    return f'''"""
Azure ML scoring script for pathogen classification.

This script is loaded by the Azure ML managed endpoint. Customize the
init() and run() functions for your deployment needs.
"""

import json
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

FAMILIES = {families_repr}


def init():
    """Called once when the endpoint starts. Load model and tokenizer."""
    global model, tokenizer, device

    model_dir = os.getenv("AZUREML_MODEL_DIR", "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    logger.info("Loading model from %s", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)


def run(raw_data):
    """Called for each scoring request.

    Args:
        raw_data: JSON string with key "sequences" containing a list of
            DNA sequences.

    Returns:
        JSON string with predictions and probabilities.
    """
    try:
        data = json.loads(raw_data)
        sequences = data["sequences"]

        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1000,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

        results = []
        for i, seq in enumerate(sequences):
            results.append({{
                "predicted_family": FAMILIES[preds[i]] if FAMILIES else str(preds[i]),
                "predicted_label": int(preds[i]),
                "probabilities": {{
                    fam: float(probs[i][j])
                    for j, fam in enumerate(FAMILIES)
                }},
            }})

        return json.dumps({{"predictions": results}})

    except Exception as e:
        logger.error("Scoring error: %s", e)
        return json.dumps({{"error": str(e)}})
'''


def _generate_conda_yml():
    """Generate a minimal conda environment file for Azure ML."""
    return """name: pathogen-classifier
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - torch>=2.10.0
    - transformers>=4.38.0,<5.0.0
    - accelerate>=0.26.0
    - numpy
    - inference-schema
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_export(onnx_path, model_dir, test_csv, n_samples=10):
    """Verify ONNX output matches PyTorch output within tolerance.

    Loads both the ONNX and PyTorch models, runs the same inputs through
    each, and checks that outputs are close.

    Args:
        onnx_path: Path to the exported ONNX model.
        model_dir: Path to the original HuggingFace model directory.
        test_csv: Path to a test CSV with sequence data.
        n_samples: Number of samples to validate (default 10).

    Returns:
        dict with validation results including max absolute difference.
    """
    import onnxruntime as ort
    import pandas as pd

    onnx_path = Path(onnx_path)
    model_dir = Path(model_dir)

    # Load test sequences
    df = pd.read_csv(test_csv)
    sequences = df["sequence"].tolist()[:n_samples]
    print(f"Validating with {len(sequences)} samples...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )

    # PyTorch inference
    print("  Running PyTorch inference...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )
    pt_model.eval()

    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1000,
    )

    with torch.no_grad():
        pt_logits = pt_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits.numpy()

    # ONNX inference
    print("  Running ONNX inference...")
    session = ort.InferenceSession(str(onnx_path))
    onnx_logits = session.run(
        ["logits"],
        {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        },
    )[0]

    # Compare
    abs_diff = np.abs(pt_logits - onnx_logits)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    pt_preds = np.argmax(pt_logits, axis=-1)
    onnx_preds = np.argmax(onnx_logits, axis=-1)
    pred_match = int(np.sum(pt_preds == onnx_preds))

    tolerance = 1e-4
    passed = max_diff < tolerance

    results = {
        "passed": passed,
        "n_samples": len(sequences),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "prediction_matches": pred_match,
        "prediction_total": len(sequences),
        "tolerance": tolerance,
    }

    status = "PASSED" if passed else "FAILED"
    print(f"\n  Validation {status}")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Mean absolute difference: {mean_diff:.2e}")
    print(f"    Prediction matches: {pred_match}/{len(sequences)}")
    print(f"    Tolerance: {tolerance}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Entry-point for the model export CLI."""
    parser = argparse.ArgumentParser(
        description="Export trained models for Azure ML deployment.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to saved model directory (e.g. models/nt-v2).",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "azure"],
        required=True,
        help="Export format: 'onnx' for ONNX model, 'azure' for full Azure ML package.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (file for ONNX, directory for Azure).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum token length for ONNX export (default: 1000).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX export against PyTorch output.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        help="Path to test CSV for validation (required with --validate).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples for validation (default: 10).",
    )
    args = parser.parse_args()

    if args.format == "onnx":
        export_onnx(args.model_dir, args.output, max_length=args.max_length)

        if args.validate:
            if not args.test_csv:
                parser.error("--test-csv is required when using --validate.")
            validate_export(
                args.output,
                args.model_dir,
                args.test_csv,
                n_samples=args.n_samples,
            )

    elif args.format == "azure":
        package_for_azure(args.model_dir, args.output)


if __name__ == "__main__":
    main()
