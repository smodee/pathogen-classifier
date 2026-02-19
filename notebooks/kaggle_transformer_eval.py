"""
Transformer Test-Set Evaluation — Kaggle Notebook
===================================================

Evaluates the trained NT v2 50M model on the held-out test set.
Produces metrics JSON, confusion matrix, and per-class F1 chart
matching the format from src/evaluate.py.

Kaggle setup:
  1. Attach pathogen-classifier-data dataset (with test.csv)
  2. Attach nt-v2-model (your trained model)
  3. Enable GPU T4
  4. Run this script
  5. Download outputs from /kaggle/working/
"""

# ============================================================================
# 0. Install pinned dependencies
# ============================================================================
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.38.0,<5.0.0",
    "accelerate>=0.26.0",
])

# ============================================================================
# 1. Imports
# ============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# 2. Configuration
# ============================================================================
MODEL_DIR = Path("/kaggle/input/models/samuelmode/nt-v2-model/pytorch/default/1")
TEST_CSV = Path("/kaggle/input/datasets/samuelmode/pathogen-classifier-data/test.csv")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = [
    "Poxviridae", "Coronaviridae", "Paramyxoviridae",
    "Filoviridae", "Flaviviridae", "Arenaviridae", "Orthomyxoviridae",
]

# ============================================================================
# 3. Load model and tokenizer
# ============================================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR), trust_remote_code=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"  Model loaded on {device}")

# Get max_length from tokenizer config
max_length = 1000
tokenizer_config = MODEL_DIR / "tokenizer_config.json"
if tokenizer_config.exists():
    with open(tokenizer_config) as f:
        tok_cfg = json.load(f)
    if "model_max_length" in tok_cfg:
        max_length = tok_cfg["model_max_length"]
print(f"  Max length: {max_length}")

# ============================================================================
# 4. Load test data
# ============================================================================
print("\nLoading test data...")
test_df = pd.read_csv(TEST_CSV)
label_names = [f for f in FAMILY_ORDER if f in test_df["family"].unique()]
label_to_idx = {name: i for i, name in enumerate(label_names)}
y_true = np.array([label_to_idx[f] for f in test_df["family"]])
sequences = test_df["sequence"].tolist()
print(f"  Test samples: {len(sequences)}")
print(f"  Families: {label_names}")

# ============================================================================
# 5. Run inference
# ============================================================================
print("\nRunning inference...")
all_preds = []
all_probs = []
batch_size = 16

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i:i + batch_size]
    inputs = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    all_preds.append(preds)
    all_probs.append(probs)

    if (i // batch_size) % 50 == 0:
        print(f"  Processed {min(i + batch_size, len(sequences))}/{len(sequences)}")

y_pred = np.concatenate(all_preds)
y_prob = np.concatenate(all_probs)
print(f"  Inference complete: {len(y_pred)} predictions")

# ============================================================================
# 6. Compute metrics
# ============================================================================
print("\nComputing metrics...")

# Aggregate metrics
accuracy = float(accuracy_score(y_true, y_pred))
balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

# Per-class metrics
labels = list(range(len(label_names)))
report = classification_report(
    y_true, y_pred, labels=labels, target_names=label_names,
    output_dict=True, zero_division=0,
)

per_class = {}
for name in label_names:
    per_class[name] = {
        "precision": report[name]["precision"],
        "recall": report[name]["recall"],
        "f1": report[name]["f1-score"],
        "support": int(report[name]["support"]),
    }

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

metrics = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_acc,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "per_class": per_class,
    "confusion_matrix": cm,
    "label_names": label_names,
}

# ============================================================================
# 7. Save metrics JSON
# ============================================================================
metrics_path = OUTPUT_DIR / "transformer_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# ============================================================================
# 8. Plot confusion matrix
# ============================================================================
cm_array = np.array(cm, dtype=float)
cm_norm = cm_array / cm_array.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("NT v2 50M — Confusion Matrix (Test Set)")
plt.tight_layout()
cm_path = OUTPUT_DIR / "transformer_confusion_matrix.png"
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"  Saved {cm_path}")

# ============================================================================
# 9. Plot per-class F1
# ============================================================================
f1_scores = [per_class[name]["f1"] for name in label_names]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(label_names, f1_scores, color="steelblue")
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 Score")
ax.set_title("NT v2 50M — Per-Class F1 Score (Test Set)")
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center")
plt.tight_layout()
f1_path = OUTPUT_DIR / "transformer_per_class_f1.png"
fig.savefig(f1_path, dpi=150)
plt.close(fig)
print(f"  Saved {f1_path}")

# ============================================================================
# 10. Print summary
# ============================================================================
print("\n" + "=" * 60)
print("TRANSFORMER TEST SET EVALUATION")
print("=" * 60)
print(f"  Accuracy:           {accuracy:.4f}")
print(f"  Balanced accuracy:  {balanced_acc:.4f}")
print(f"  Macro F1:           {macro_f1:.4f}")
print(f"  Weighted F1:        {weighted_f1:.4f}")
print()
print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
print(f"\nOutputs saved to {OUTPUT_DIR}:")
print(f"  - transformer_metrics.json")
print(f"  - transformer_confusion_matrix.png")
print(f"  - transformer_per_class_f1.png")
print(f"\nDownload these files and place them in your local reports/ folder.")
