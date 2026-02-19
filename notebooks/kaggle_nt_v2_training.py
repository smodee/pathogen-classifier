"""
Nucleotide Transformer v2 50M — Fine-tuning for Viral Family Classification
============================================================================

Self-contained Kaggle training script. Bundles all code from:
  - src/dataset.py       (SequenceDataset, NTCollator, compute_class_weights)
  - src/augmentation.py  (GenomicAugmentationDataset + transforms)
  - src/train_transformer.py (WeightedTrainer, freeze_encoder, training loop)

Kaggle setup:
  1. Upload data/processed/ (train.csv, val.csv, test.csv) as a Kaggle Dataset
  2. Create a new Notebook, enable GPU (T4), attach that dataset
  3. Paste this script into a single code cell (or upload as .py and run)
  4. The dataset will be at /kaggle/input/<dataset-name>/

Hardware:  Kaggle free T4 GPU (16 GB VRAM, 30 hrs/week)
Model:     InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
Strategy:  Frozen encoder (only classification head trains, ~266K params)
"""

# ============================================================================
# 0. Install pinned dependencies (Kaggle may have newer transformers)
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
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 2. Configuration — EDIT THESE PATHS
# ============================================================================
# After uploading data/processed/ as a Kaggle Dataset, update this path.
# It will be something like: /kaggle/input/pathogen-classifier-data/
DATA_DIR = Path("/kaggle/input/pathogen-classifier-data")

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV   = DATA_DIR / "val.csv"
TEST_CSV  = DATA_DIR / "test.csv"    # Optional: for final evaluation

OUTPUT_DIR = Path("/kaggle/working/nt-v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
NUM_LABELS = 7
MAX_LENGTH = 1000     # NT v2 context window in tokens

# Training
SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch = 32
LEARNING_RATE = 2e-4              # For frozen encoder
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
UNFREEZE_LAYERS = 0               # 0 = fully frozen encoder

# Augmentation
AUG_ENABLED = True
AUG_REVERSE_COMPLEMENT = True
AUG_RANDOM_CROP = True
AUG_MIN_CROP_RATIO = 0.5
AUG_SEQUENCING_NOISE = "illumina"
AUG_N_MASKING = True
AUG_MASK_RATE = 0.02
AUG_KMER_SHUFFLE = False

# ============================================================================
# 3. Label mappings (must match the order used during data preprocessing)
# ============================================================================
FAMILIES = [
    "Poxviridae", "Coronaviridae", "Paramyxoviridae",
    "Filoviridae", "Flaviviridae", "Arenaviridae", "Orthomyxoviridae",
]
LABEL_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}
IDX_TO_LABEL = {i: fam for i, fam in enumerate(FAMILIES)}

# ============================================================================
# 4. Dataset
# ============================================================================
class SequenceDataset(Dataset):
    """PyTorch Dataset that loads chunked CSV and returns raw sequence dicts."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.ids = df["id"].astype(str).tolist()
        self.sequences = df["sequence"].astype(str).tolist()
        self.labels = [LABEL_TO_IDX[fam] for fam in df["family"]]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "label": self.labels[idx],
            "id": self.ids[idx],
        }


class NTCollator:
    """Collate function that tokenizes a batch of sequence dicts for NT v2."""

    def __init__(self, tokenizer, max_length=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sequences = [item["sequence"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoding = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def compute_class_weights(csv_path, device=None):
    """Compute inverse-frequency class weights for cross-entropy loss."""
    df = pd.read_csv(csv_path)
    counts = df["family"].value_counts()

    weights = []
    total = len(df)
    n_classes = len(FAMILIES)
    for fam in FAMILIES:
        c = counts.get(fam, 1)
        weights.append(total / (n_classes * c))

    return torch.tensor(weights, dtype=torch.float32, device=device)

# ============================================================================
# 5. Data augmentation
# ============================================================================
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_BASES = ["A", "C", "G", "T"]


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


def random_crop(sequence, min_crop_ratio=0.5):
    """Extract a random contiguous subsequence."""
    seq_len = len(sequence)
    if seq_len <= 1:
        return sequence
    min_length = max(1, int(seq_len * min_crop_ratio))
    crop_length = random.randint(min_length, seq_len)
    start = random.randint(0, seq_len - crop_length)
    return sequence[start : start + crop_length]


def sequencing_noise(sequence, mode="illumina"):
    """Introduce simulated sequencing errors."""
    seq = list(sequence.upper())

    if mode == "illumina":
        error_rate = 0.001
        for i in range(len(seq)):
            if seq[i] in _BASES and random.random() < error_rate:
                seq[i] = random.choice([b for b in _BASES if b != seq[i]])
        return "".join(seq)

    elif mode == "nanopore":
        error_rate = 0.04
        result = []
        for base in seq:
            r = random.random()
            if r < error_rate * 0.4:
                if base in _BASES:
                    result.append(random.choice([b for b in _BASES if b != base]))
                else:
                    result.append(base)
            elif r < error_rate * 0.7:
                continue
            elif r < error_rate:
                result.append(random.choice(_BASES))
                result.append(base)
            else:
                result.append(base)
        return "".join(result)

    else:
        raise ValueError(f"Unknown noise mode: {mode!r}")


def n_masking(sequence, mask_rate=0.02):
    """Randomly replace bases with 'N' (unknown)."""
    seq = list(sequence)
    n_to_mask = int(len(seq) * mask_rate)
    if n_to_mask == 0:
        return sequence
    positions = random.sample(range(len(seq)), n_to_mask)
    for pos in positions:
        seq[pos] = "N"
    return "".join(seq)


def kmer_shuffle(sequence, window_size=50, k=3):
    """Shuffle k-mers within fixed-size windows."""
    result = []
    for start in range(0, len(sequence), window_size):
        window = sequence[start : start + window_size]
        kmers = [window[i : i + k] for i in range(0, len(window), k)]
        random.shuffle(kmers)
        result.append("".join(kmers))
    return "".join(result)


class GenomicAugmentationDataset(Dataset):
    """Wraps a base dataset with on-the-fly genomic augmentations."""

    def __init__(
        self,
        base_dataset,
        reverse_complement_flag=True,
        random_crop_flag=True,
        min_crop_ratio=0.5,
        noise_mode=None,
        n_masking_flag=True,
        mask_rate=0.02,
        kmer_shuffle_flag=False,
        p_reverse_complement=0.5,
        p_random_crop=0.3,
        p_sequencing_noise=0.2,
        p_n_masking=0.2,
        p_kmer_shuffle=0.1,
    ):
        self.base_dataset = base_dataset
        self.do_reverse_complement = reverse_complement_flag
        self.do_random_crop = random_crop_flag
        self.min_crop_ratio = min_crop_ratio
        self.noise_mode = noise_mode
        self.do_n_masking = n_masking_flag
        self.mask_rate = mask_rate
        self.do_kmer_shuffle = kmer_shuffle_flag
        self.p_reverse_complement = p_reverse_complement
        self.p_random_crop = p_random_crop
        self.p_sequencing_noise = p_sequencing_noise
        self.p_n_masking = p_n_masking
        self.p_kmer_shuffle = p_kmer_shuffle

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        seq = item["sequence"]
        applied = []

        if self.do_reverse_complement and random.random() < self.p_reverse_complement:
            seq = reverse_complement(seq)
            applied.append("reverse_complement")

        if self.do_random_crop and random.random() < self.p_random_crop:
            seq = random_crop(seq, self.min_crop_ratio)
            applied.append("random_crop")

        if self.noise_mode and random.random() < self.p_sequencing_noise:
            seq = sequencing_noise(seq, self.noise_mode)
            applied.append(f"noise_{self.noise_mode}")

        if self.do_n_masking and random.random() < self.p_n_masking:
            seq = n_masking(seq, self.mask_rate)
            applied.append("n_masking")

        if self.do_kmer_shuffle and random.random() < self.p_kmer_shuffle:
            seq = kmer_shuffle(seq)
            applied.append("kmer_shuffle")

        result = dict(item)
        result["sequence"] = seq
        result["augmentations"] = applied
        return result

# ============================================================================
# 6. Metrics
# ============================================================================
def compute_metrics(eval_pred):
    """Compute classification metrics for Trainer callbacks."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }

# ============================================================================
# 7. Encoder freezing
# ============================================================================
def freeze_encoder(model, unfreeze_layers=0):
    """Freeze the ESM encoder, optionally unfreezing the top N layers."""
    for param in model.esm.parameters():
        param.requires_grad = False

    if unfreeze_layers > 0:
        encoder_layers = model.esm.encoder.layer
        total_layers = len(encoder_layers)
        if unfreeze_layers > total_layers:
            print(f"  Warning: unfreeze_layers {unfreeze_layers} exceeds "
                  f"total layers ({total_layers}). Unfreezing all.")
            unfreeze_layers = total_layers

        for layer in encoder_layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(model.esm.encoder, "emb_layer_norm_after"):
            for param in model.esm.encoder.emb_layer_norm_after.parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


def print_trainable_params(model):
    """Print trainable vs total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

# ============================================================================
# 8. Weighted Trainer
# ============================================================================
class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            reduction=reduction,
        )
        loss = loss_fn(logits, labels)

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss

# ============================================================================
# 9. Training
# ============================================================================
def main():
    # --- Verify data exists ---
    for path in [TRAIN_CSV, VAL_CSV]:
        if not path.exists():
            print(f"ERROR: {path} not found!")
            print(f"  Make sure your Kaggle Dataset is attached and DATA_DIR is correct.")
            print(f"  Current DATA_DIR: {DATA_DIR}")
            if DATA_DIR.exists():
                print(f"  Files in DATA_DIR: {list(DATA_DIR.iterdir())}")
            return
        else:
            print(f"Found: {path}")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = torch.cuda.is_available()
    num_workers = 2 if torch.cuda.is_available() else 0
    lr = LEARNING_RATE if UNFREEZE_LAYERS == 0 else 5e-5

    print(f"\nDevice: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  FP16: {fp16}")
    print(f"  Unfreeze layers: {UNFREEZE_LAYERS}")
    print(f"  Learning rate: {lr}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # --- Datasets ---
    print("Loading datasets...")
    train_dataset = SequenceDataset(TRAIN_CSV)
    val_dataset = SequenceDataset(VAL_CSV)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # --- Augmentation (training only) ---
    if AUG_ENABLED:
        print("  Augmentation: enabled")
        train_dataset = GenomicAugmentationDataset(
            train_dataset,
            reverse_complement_flag=AUG_REVERSE_COMPLEMENT,
            random_crop_flag=AUG_RANDOM_CROP,
            min_crop_ratio=AUG_MIN_CROP_RATIO,
            noise_mode=AUG_SEQUENCING_NOISE,
            n_masking_flag=AUG_N_MASKING,
            mask_rate=AUG_MASK_RATE,
            kmer_shuffle_flag=AUG_KMER_SHUFFLE,
        )
    else:
        print("  Augmentation: disabled")

    # --- Collator ---
    collator = NTCollator(tokenizer, max_length=MAX_LENGTH)

    # --- Model ---
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
    )
    freeze_encoder(model, unfreeze_layers=UNFREEZE_LAYERS)
    print_trainable_params(model)

    # --- Class weights ---
    class_weights = compute_class_weights(TRAIN_CSV)
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        dataloader_num_workers=num_workers,
        report_to="none",
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
            ),
        ],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()

    # --- Save model ---
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # --- Evaluate on validation set ---
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate()

    # --- Detailed classification report ---
    val_preds = trainer.predict(val_dataset)
    y_true = val_preds.label_ids
    y_pred = np.argmax(val_preds.predictions, axis=-1)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=FAMILIES, digits=4))

    # --- Save metrics ---
    results = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": metrics.get("eval_balanced_accuracy", 0),
        "macro_f1": metrics.get("eval_macro_f1", 0),
        "weighted_f1": metrics.get("eval_weighted_f1", 0),
        "per_class": {},
    }
    report = classification_report(y_true, y_pred, target_names=FAMILIES,
                                   output_dict=True, digits=4)
    for fam in FAMILIES:
        results["per_class"][fam] = report[fam]

    metrics_path = OUTPUT_DIR / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # --- Save label mappings ---
    label_path = OUTPUT_DIR / "label_mappings.json"
    with open(label_path, "w") as f:
        json.dump({
            "families": FAMILIES,
            "label_to_idx": LABEL_TO_IDX,
            "idx_to_label": {str(k): v for k, v in IDX_TO_LABEL.items()},
        }, f, indent=2)
    print(f"Saved label mappings to {label_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Macro F1:           {results['macro_f1']:.4f}")
    print(f"  Weighted F1:        {results['weighted_f1']:.4f}")
    print(f"  Balanced accuracy:  {results['balanced_accuracy']:.4f}")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("Download the /kaggle/working/nt-v2/ folder to get your trained model.")

    return results


# ============================================================================
# 10. Run
# ============================================================================
if __name__ == "__main__":
    main()
else:
    # When pasted into a Kaggle notebook cell, call main() directly
    results = main()
