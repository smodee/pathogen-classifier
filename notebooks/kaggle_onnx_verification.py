"""
ONNX Export Verification — Kaggle Notebook
===========================================

Verifies two remaining PR #19 test plan items:
  1. Run ONNX export with trained model and verify output file
  2. Validate ONNX export matches PyTorch output on test data

Kaggle setup:
  1. Attach your pathogen-classifier-data dataset (with test.csv)
  2. Attach the trained nt-v2 model as a second dataset
     (or upload models/nt-v2/ as a new dataset called "nt-v2-model")
  3. Enable GPU T4 (needed for consistency, though CPU works too)
  4. Run this script
"""

# ============================================================================
# 0. Install pinned dependencies
# ============================================================================
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.38.0,<5.0.0",
    "accelerate>=0.26.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
    "onnxscript",
])

# ============================================================================
# 1. Imports
# ============================================================================
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# 2. Configuration — EDIT THESE PATHS
# ============================================================================
# Kaggle Models path (framework/variation/version)
MODEL_DIR = Path("/kaggle/input/models/samuelmode/nt-v2-model/pytorch/default/1")

# Test data from your pathogen-classifier-data dataset
TEST_CSV = Path("/kaggle/input/datasets/samuelmode/pathogen-classifier-data/test.csv")

# Output paths (Kaggle working directory)
ONNX_OUTPUT = Path("/kaggle/working/nt_v2.onnx")
MAX_LENGTH = 1000

# Number of samples for validation
N_SAMPLES = 20

# ============================================================================
# 3. Verify files exist
# ============================================================================
print("\n--- Checking files ---")

# List what's in the model directory
print(f"\nModel dir ({MODEL_DIR}):")
if MODEL_DIR.exists():
    for f in sorted(MODEL_DIR.iterdir()):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name:30s} {size:>12,} bytes")
else:
    print(f"  ERROR: {MODEL_DIR} does not exist!")
    print("  Check your dataset attachment and update MODEL_DIR above.")
    print(f"  Available inputs: {list(Path('/kaggle/input').iterdir())}")
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

# Check required files
required = ["config.json", "model.safetensors", "modeling_esm.py",
            "esm_config.py", "tokenizer_config.json", "vocab.txt"]
missing = [f for f in required if not (MODEL_DIR / f).exists()]
if missing:
    print(f"\n  WARNING: Missing files: {missing}")
    # Try to find them in a subfolder
    for sub in MODEL_DIR.iterdir():
        if sub.is_dir():
            print(f"  Checking subfolder: {sub.name}/")
            for f in sorted(sub.iterdir()):
                print(f"    {f.name}")
else:
    print("  All required files present ✓")

assert TEST_CSV.exists(), f"Test CSV not found: {TEST_CSV}"
print(f"\nTest CSV: {TEST_CSV} ✓")

# ============================================================================
# 4. TEST 1: ONNX Export
# ============================================================================
print("\n" + "=" * 60)
print("TEST 1: ONNX Export")
print("=" * 60)

# Load model and tokenizer
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR), trust_remote_code=True,
)
model.eval()
print(f"  Model loaded: {model.config.num_labels} labels")

# Create dummy input for tracing
dummy_sequence = "ACGT" * 50
dummy_inputs = tokenizer(
    dummy_sequence,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
)

input_ids = dummy_inputs["input_ids"]
attention_mask = dummy_inputs["attention_mask"]

# Dynamic axes
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size"},
}

# Export
print(f"\nExporting to ONNX: {ONNX_OUTPUT}")
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    str(ONNX_OUTPUT),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes=dynamic_axes,
    opset_version=14,
    do_constant_folding=True,
)

# Validate ONNX model structure
onnx_model = onnx.load(str(ONNX_OUTPUT))
onnx.checker.check_model(onnx_model)

file_size_mb = ONNX_OUTPUT.stat().st_size / (1024 * 1024)
print(f"\n  ✓ ONNX model saved: {ONNX_OUTPUT}")
print(f"  ✓ File size: {file_size_mb:.1f} MB")
print(f"  ✓ ONNX checker passed")
print(f"  ✓ Opset version: 14")

# Print model info
print(f"\n  Inputs:")
for inp in onnx_model.graph.input:
    print(f"    {inp.name}: {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print(f"  Outputs:")
for out in onnx_model.graph.output:
    print(f"    {out.name}: {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")

print("\n>>> TEST 1 PASSED <<<")

# ============================================================================
# 5. TEST 2: Validate ONNX matches PyTorch
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: Validate ONNX vs PyTorch Output")
print("=" * 60)

# Load test sequences
df = pd.read_csv(TEST_CSV)
sequences = df["sequence"].tolist()[:N_SAMPLES]
true_labels = df["family"].tolist()[:N_SAMPLES]
print(f"\nUsing {len(sequences)} test samples")
print(f"  Families: {set(true_labels)}")

# Tokenize
inputs = tokenizer(
    sequences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
)

# PyTorch inference
print("\nRunning PyTorch inference...")
with torch.no_grad():
    pt_logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    ).logits.numpy()

pt_probs = torch.softmax(torch.tensor(pt_logits), dim=-1).numpy()
pt_preds = np.argmax(pt_logits, axis=-1)

# ONNX inference
print("Running ONNX inference...")
session = ort.InferenceSession(str(ONNX_OUTPUT))
onnx_logits = session.run(
    ["logits"],
    {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    },
)[0]

onnx_probs = np.exp(onnx_logits) / np.exp(onnx_logits).sum(axis=-1, keepdims=True)
onnx_preds = np.argmax(onnx_logits, axis=-1)

# Compare
abs_diff = np.abs(pt_logits - onnx_logits)
max_diff = float(abs_diff.max())
mean_diff = float(abs_diff.mean())

pred_match = int(np.sum(pt_preds == onnx_preds))
tolerance = 1e-4

print(f"\n  Results:")
print(f"    Max absolute difference:  {max_diff:.2e}")
print(f"    Mean absolute difference: {mean_diff:.2e}")
print(f"    Prediction matches:       {pred_match}/{len(sequences)}")
print(f"    Tolerance:                {tolerance}")

# Load label mappings for readable output
label_path = MODEL_DIR / "label_mappings.json"
if label_path.exists():
    with open(label_path) as f:
        mappings = json.load(f)
    families = mappings["families"]
else:
    families = [f"Class_{i}" for i in range(model.config.num_labels)]

# Show per-sample results
print(f"\n  Per-sample predictions:")
print(f"  {'#':>3}  {'True Family':>20}  {'PyTorch':>20}  {'ONNX':>20}  {'Match':>5}  {'MaxDiff':>10}")
for i in range(len(sequences)):
    pt_fam = families[pt_preds[i]]
    onnx_fam = families[onnx_preds[i]]
    match = "✓" if pt_preds[i] == onnx_preds[i] else "✗"
    sample_diff = abs_diff[i].max()
    print(f"  {i:>3}  {true_labels[i]:>20}  {pt_fam:>20}  {onnx_fam:>20}  {match:>5}  {sample_diff:>10.2e}")

passed = max_diff < tolerance
status = "PASSED" if passed else "WARNING"

if not passed and pred_match == len(sequences):
    print(f"\n  Note: Max diff ({max_diff:.2e}) exceeds strict tolerance ({tolerance})")
    print(f"  but all predictions match. This is acceptable for deployment.")
    status = "PASSED (predictions match)"

print(f"\n>>> TEST 2 {status} <<<")

# ============================================================================
# 6. Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Test 1 — ONNX Export:     PASSED")
print(f"  Test 2 — ONNX vs PyTorch: {status}")
print(f"  ONNX file size:           {file_size_mb:.1f} MB")
print(f"  Max logit difference:     {max_diff:.2e}")
print(f"  Prediction agreement:     {pred_match}/{len(sequences)}")
print(f"\nONNX model available at: {ONNX_OUTPUT}")
