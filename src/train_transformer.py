"""
Fine-tune Nucleotide Transformer v2 50M for viral family classification.

Supports frozen-encoder training (classification head only, ~266K trainable
params) and partial unfreezing of top N encoder layers for better accuracy.

Data pipeline:
    SequenceDataset -> GenomicAugmentationDataset -> NTCollator -> Trainer

Usage:
    # Full training (GPU)
    python src/train_transformer.py --config configs/train_nt.json

    # Quick debug run (CPU)
    python src/train_transformer.py --config configs/train_nt.json --max-train-samples 100

    # Unfreeze top 2 encoder layers
    python src/train_transformer.py --config configs/train_nt.json --unfreeze-layers 2
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.augmentation import GenomicAugmentationDataset
from src.dataset import (
    FAMILIES,
    NTCollator,
    SequenceDataset,
    compute_class_weights,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute classification metrics for Trainer callbacks.

    Args:
        eval_pred: EvalPrediction with .predictions (logits) and .label_ids.

    Returns:
        dict with macro_f1, weighted_f1, balanced_accuracy.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
    }


# ---------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------

def freeze_encoder(model, unfreeze_layers=0):
    """Freeze the ESM encoder, optionally unfreezing the top N layers.

    Args:
        model: EsmForSequenceClassification instance.
        unfreeze_layers: Number of top encoder layers to unfreeze (0 = fully
            frozen encoder, only classification head trains).
    """
    # Freeze all encoder parameters
    for param in model.esm.parameters():
        param.requires_grad = False

    # Unfreeze top N layers if requested
    if unfreeze_layers > 0:
        encoder_layers = model.esm.encoder.layer
        total_layers = len(encoder_layers)
        if unfreeze_layers > total_layers:
            print(f"  Warning: --unfreeze-layers {unfreeze_layers} exceeds "
                  f"total layers ({total_layers}). Unfreezing all.")
            unfreeze_layers = total_layers

        for layer in encoder_layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze post-encoder layer norm
        if hasattr(model.esm.encoder, 'emb_layer_norm_after'):
            for param in model.esm.encoder.emb_layer_norm_after.parameters():
                param.requires_grad = True

    # Classification head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def print_trainable_params(model):
    """Print trainable vs total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss.

    Addresses class imbalance by applying inverse-frequency weights from
    ``compute_class_weights`` to the loss function.

    Args:
        class_weights: torch.FloatTensor of shape (num_classes,).
        **kwargs: Forwarded to Trainer.__init__.
    """

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        # Use sum reduction when gradient accumulation provides item count
        reduction = 'sum' if num_items_in_batch is not None else 'mean'
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            reduction=reduction,
        )
        loss = loss_fn(logits, labels)

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load training configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path, max_train_samples=None, max_val_samples=None,
          unfreeze_layers=None):
    """Fine-tune NT v2 50M for viral family classification.

    Args:
        config_path: Path to JSON config file.
        max_train_samples: Limit training samples (for debugging).
        max_val_samples: Limit validation samples (for debugging).
        unfreeze_layers: Override config's unfreeze_layers (None = use config).

    Returns:
        dict with final evaluation metrics.
    """
    config = load_config(config_path)

    # --- Device detection ---
    use_cuda = torch.cuda.is_available()
    device_params = config['gpu'] if use_cuda else config['cpu']
    device = 'cuda' if use_cuda else 'cpu'

    print(f"Device: {device}")
    print(f"  Batch size: {device_params['batch_size']}")
    print(f"  Gradient accumulation: {device_params['gradient_accumulation_steps']}")
    print(f"  FP16: {device_params['fp16']}")

    n_unfreeze = unfreeze_layers if unfreeze_layers is not None else config.get('unfreeze_layers', 0)
    lr = device_params['lr_unfrozen'] if n_unfreeze > 0 else device_params['lr_frozen']
    print(f"  Unfreeze layers: {n_unfreeze}")
    print(f"  Learning rate: {lr}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=config.get('trust_remote_code', True),
    )

    # --- Datasets ---
    print("Loading datasets...")
    train_dataset = SequenceDataset(config['train_csv'])
    val_dataset = SequenceDataset(config['val_csv'])

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, range(max_train_samples))
    if max_val_samples is not None and max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, range(max_val_samples))

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # --- Augmentation (training set only) ---
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled', True):
        print("  Augmentation: enabled")
        train_dataset = GenomicAugmentationDataset(
            train_dataset,
            reverse_complement=aug_config.get('reverse_complement', True),
            random_crop=aug_config.get('random_crop', True),
            min_crop_ratio=aug_config.get('min_crop_ratio', 0.5),
            sequencing_noise=aug_config.get('sequencing_noise', None),
            n_masking=aug_config.get('n_masking', True),
            mask_rate=aug_config.get('mask_rate', 0.02),
            kmer_shuffle=aug_config.get('kmer_shuffle', False),
        )
    else:
        print("  Augmentation: disabled")

    # --- Collator ---
    collator = NTCollator(tokenizer, max_length=config.get('max_length', 1000))

    # --- Model ---
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config.get('num_labels', len(FAMILIES)),
        trust_remote_code=config.get('trust_remote_code', True),
    )
    freeze_encoder(model, unfreeze_layers=n_unfreeze)
    print_trainable_params(model)

    # --- Class weights ---
    class_weights = compute_class_weights(config['train_csv'])
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # --- Training arguments ---
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get('epochs', 10),
        per_device_train_batch_size=device_params['batch_size'],
        per_device_eval_batch_size=device_params['batch_size'],
        gradient_accumulation_steps=device_params['gradient_accumulation_steps'],
        learning_rate=lr,
        warmup_ratio=config.get('warmup_ratio', 0.1),
        weight_decay=config.get('weight_decay', 0.01),
        fp16=device_params['fp16'],
        eval_strategy=config.get('eval_strategy', 'epoch'),
        save_strategy=config.get('save_strategy', 'epoch'),
        logging_steps=config.get('logging_steps', 10),
        load_best_model_at_end=True,
        metric_for_best_model=config.get('metric_for_best_model', 'eval_macro_f1'),
        greater_is_better=True,
        save_total_limit=2,
        seed=config.get('seed', 42),
        dataloader_num_workers=device_params.get('num_workers', 0),
        report_to='none',
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
                early_stopping_patience=config.get('early_stopping_patience', 3),
            ),
        ],
    )

    print("\nStarting training...")
    trainer.train()

    # --- Save artifacts ---
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Copy custom model code so the saved directory loads standalone
    # with AutoModel.from_pretrained(..., trust_remote_code=True).
    try:
        from transformers.dynamic_module_utils import get_cached_module_file
        for fname in ('modeling_esm.py', 'esm_config.py'):
            try:
                src = get_cached_module_file(config['model_name'], fname)
                shutil.copy2(src, output_dir / fname)
                print(f"  Copied custom code: {fname}")
            except Exception:
                pass
    except ImportError:
        # Fallback: copy from the HuggingFace cache directory directly
        from transformers.utils import cached_file
        for fname in ('modeling_esm.py', 'esm_config.py'):
            try:
                src = cached_file(config['model_name'], fname,
                                  _raise_exceptions_for_missing_entries=False)
                if src is not None:
                    shutil.copy2(src, output_dir / fname)
                    print(f"  Copied custom code: {fname}")
            except Exception:
                pass

    # Evaluate and save metrics
    metrics = trainer.evaluate()
    metrics_path = output_dir / 'eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {metrics_path}")

    # Save label mappings for inference
    label_path = output_dir / 'label_mappings.json'
    with open(label_path, 'w') as f:
        json.dump({
            'families': FAMILIES,
            'label_to_idx': {fam: i for i, fam in enumerate(FAMILIES)},
            'idx_to_label': {str(i): fam for i, fam in enumerate(FAMILIES)},
        }, f, indent=2)
    print(f"  Saved {label_path}")

    # Print final metrics
    print("\nFinal validation metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    print("\nDone.")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Nucleotide Transformer v2 50M for viral family classification.',
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to JSON config file (e.g. configs/train_nt.json)',
    )
    parser.add_argument(
        '--max-train-samples', type=int, default=None,
        help='Limit training samples (for quick debugging)',
    )
    parser.add_argument(
        '--max-val-samples', type=int, default=None,
        help='Limit validation samples (for quick debugging)',
    )
    parser.add_argument(
        '--unfreeze-layers', type=int, default=None,
        help='Number of top encoder layers to unfreeze (overrides config)',
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        unfreeze_layers=args.unfreeze_layers,
    )


if __name__ == '__main__':
    main()
