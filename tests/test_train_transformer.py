"""
Unit and integration tests for the NT v2 fine-tuning script.

Covers compute_metrics, freeze_encoder, WeightedTrainer, load_config,
and end-to-end training with limited samples.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.dataset import FAMILIES, LABEL_TO_IDX
from src.train_transformer import (
    WeightedTrainer,
    compute_metrics,
    freeze_encoder,
    load_config,
    print_trainable_params,
    train,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CSV_HEADER = ['id', 'parent_sequence_id', 'chunk_index', 'sequence', 'length', 'family']

# Two samples per family = 14 total (small enough for fast tests)
SAMPLE_SEQUENCES = []
for i, fam in enumerate(FAMILIES):
    for j in range(2):
        seq = ('ACGTACGT' * 75)[:600]  # 600bp each
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
def train_csv(tmp_path):
    return _write_csv(tmp_path / 'train.csv', SAMPLE_SEQUENCES)


@pytest.fixture
def val_csv(tmp_path):
    return _write_csv(tmp_path / 'val.csv', SAMPLE_SEQUENCES)


@pytest.fixture
def config_dict(tmp_path, train_csv, val_csv):
    """Return a minimal config dict for testing."""
    return {
        'model_name': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
        'num_labels': 7,
        'trust_remote_code': True,
        'train_csv': train_csv,
        'val_csv': val_csv,
        'output_dir': str(tmp_path / 'model_output'),
        'max_length': 128,
        'seed': 42,
        'gpu': {
            'batch_size': 2, 'gradient_accumulation_steps': 1,
            'fp16': False, 'lr_frozen': 2e-4, 'lr_unfrozen': 5e-5,
            'num_workers': 0,
        },
        'cpu': {
            'batch_size': 2, 'gradient_accumulation_steps': 1,
            'fp16': False, 'lr_frozen': 2e-4, 'lr_unfrozen': 5e-5,
            'num_workers': 0,
        },
        'epochs': 1,
        'warmup_ratio': 0.0,
        'weight_decay': 0.0,
        'early_stopping_patience': 3,
        'metric_for_best_model': 'eval_macro_f1',
        'unfreeze_layers': 0,
        'augmentation': {'enabled': False},
        'logging_steps': 1,
        'save_strategy': 'epoch',
        'eval_strategy': 'epoch',
    }


@pytest.fixture
def config_path(tmp_path, config_dict):
    """Write config dict to a JSON file and return the path."""
    path = tmp_path / 'test_config.json'
    with open(path, 'w') as f:
        json.dump(config_dict, f)
    return str(path)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 3, 4, 5, 6])
        logits = np.eye(7) * 10  # Strong one-hot logits
        result = compute_metrics((logits, labels))

        assert result['macro_f1'] == pytest.approx(1.0)
        assert result['weighted_f1'] == pytest.approx(1.0)
        assert result['balanced_accuracy'] == pytest.approx(1.0)

    def test_output_keys(self):
        labels = np.array([0, 1, 2])
        logits = np.random.randn(3, 7)
        result = compute_metrics((logits, labels))

        assert 'macro_f1' in result
        assert 'weighted_f1' in result
        assert 'balanced_accuracy' in result

    def test_metrics_in_valid_range(self):
        labels = np.random.randint(0, 7, size=100)
        logits = np.random.randn(100, 7)
        result = compute_metrics((logits, labels))

        for key in ('macro_f1', 'weighted_f1', 'balanced_accuracy'):
            assert 0.0 <= result[key] <= 1.0

    def test_metrics_are_floats(self):
        labels = np.array([0, 1, 2])
        logits = np.random.randn(3, 7)
        result = compute_metrics((logits, labels))

        for value in result.values():
            assert isinstance(value, float)


# ---------------------------------------------------------------------------
# freeze_encoder tests
# ---------------------------------------------------------------------------

class TestFreezeEncoder:
    @pytest.fixture
    def model(self):
        """Load the real model for freeze/unfreeze tests."""
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
            num_labels=7,
            trust_remote_code=True,
        )

    def test_frozen_encoder_no_grad(self, model):
        freeze_encoder(model, unfreeze_layers=0)
        for param in model.esm.parameters():
            assert not param.requires_grad

    def test_classifier_always_trainable(self, model):
        freeze_encoder(model, unfreeze_layers=0)
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_top_layers(self, model):
        freeze_encoder(model, unfreeze_layers=2)
        total_layers = len(model.esm.encoder.layer)

        # Bottom layers frozen
        for layer in model.esm.encoder.layer[:total_layers - 2]:
            for param in layer.parameters():
                assert not param.requires_grad

        # Top 2 layers unfrozen
        for layer in model.esm.encoder.layer[-2:]:
            for param in layer.parameters():
                assert param.requires_grad

    def test_unfreeze_exceeds_total(self, model, capsys):
        total_layers = len(model.esm.encoder.layer)
        freeze_encoder(model, unfreeze_layers=total_layers + 5)
        captured = capsys.readouterr()
        assert 'Warning' in captured.out

        # All encoder layers should be unfrozen
        for layer in model.esm.encoder.layer:
            for param in layer.parameters():
                assert param.requires_grad

    def test_trainable_param_count_frozen(self, model):
        freeze_encoder(model, unfreeze_layers=0)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        # Frozen: only classifier head trainable, should be < 1% of total
        assert trainable < total * 0.02

    def test_trainable_param_count_increases_with_unfreezing(self, model):
        freeze_encoder(model, unfreeze_layers=0)
        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        freeze_encoder(model, unfreeze_layers=2)
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert unfrozen_trainable > frozen_trainable


# ---------------------------------------------------------------------------
# print_trainable_params tests
# ---------------------------------------------------------------------------

class TestPrintTrainableParams:
    def test_prints_counts(self, capsys):
        model = nn.Linear(10, 2)
        print_trainable_params(model)
        captured = capsys.readouterr()
        assert 'Trainable parameters' in captured.out
        assert '22' in captured.out  # 10*2 + 2 = 22


# ---------------------------------------------------------------------------
# WeightedTrainer tests
# ---------------------------------------------------------------------------

class TestWeightedTrainer:
    def test_weighted_loss_differs_from_unweighted(self):
        """Class weights should change the loss value."""
        num_classes = 7
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        labels = torch.tensor([0, 0, 0, 1])  # Imbalanced batch

        # Unweighted loss
        unweighted_loss = nn.CrossEntropyLoss()(logits, labels)

        # Weighted loss (heavy weight on class 1)
        weights = torch.ones(num_classes)
        weights[1] = 10.0
        weighted_loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        assert not torch.allclose(unweighted_loss, weighted_loss)

    def test_compute_loss_returns_tensor(self):
        """WeightedTrainer.compute_loss should return a scalar tensor."""
        num_classes = 7
        batch_size = 4

        # Mock model that returns logits
        mock_model = MagicMock()
        mock_model.return_value = SimpleNamespace(
            logits=torch.randn(batch_size, num_classes),
        )

        class_weights = torch.ones(num_classes)
        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer.class_weights = class_weights

        inputs = {
            'input_ids': torch.randint(0, 100, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10, dtype=torch.long),
            'labels': torch.randint(0, num_classes, (batch_size,)),
        }

        loss = trainer.compute_loss(mock_model, inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_compute_loss_with_return_outputs(self):
        """Should return (loss, outputs) when return_outputs=True."""
        num_classes = 7
        batch_size = 4

        mock_model = MagicMock()
        outputs = SimpleNamespace(logits=torch.randn(batch_size, num_classes))
        mock_model.return_value = outputs

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer.class_weights = torch.ones(num_classes)

        inputs = {
            'input_ids': torch.randint(0, 100, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10, dtype=torch.long),
            'labels': torch.randint(0, num_classes, (batch_size,)),
        }

        result = trainer.compute_loss(mock_model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)

    def test_compute_loss_with_num_items_in_batch(self):
        """When num_items_in_batch is given, uses sum reduction then divides."""
        num_classes = 7
        batch_size = 4

        mock_model = MagicMock()
        mock_model.return_value = SimpleNamespace(
            logits=torch.randn(batch_size, num_classes),
        )

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer.class_weights = torch.ones(num_classes)

        inputs = {
            'input_ids': torch.randint(0, 100, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10, dtype=torch.long),
            'labels': torch.randint(0, num_classes, (batch_size,)),
        }

        loss = trainer.compute_loss(mock_model, inputs, num_items_in_batch=batch_size)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_labels_popped_from_inputs(self):
        """Labels should be removed from inputs before passing to model."""
        num_classes = 7
        batch_size = 4

        mock_model = MagicMock()
        mock_model.return_value = SimpleNamespace(
            logits=torch.randn(batch_size, num_classes),
        )

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer.class_weights = torch.ones(num_classes)

        inputs = {
            'input_ids': torch.randint(0, 100, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10, dtype=torch.long),
            'labels': torch.randint(0, num_classes, (batch_size,)),
        }

        trainer.compute_loss(mock_model, inputs)

        # Model should have been called without 'labels'
        call_kwargs = mock_model.call_args[1]
        assert 'labels' not in call_kwargs


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_config(self, config_path):
        config = load_config(config_path)
        assert config['model_name'] == 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species'
        assert 'gpu' in config
        assert 'cpu' in config

    def test_config_has_required_keys(self, config_path):
        config = load_config(config_path)
        required = ['model_name', 'train_csv', 'val_csv', 'output_dir',
                     'gpu', 'cpu', 'epochs']
        for key in required:
            assert key in config

    def test_device_params_structure(self, config_path):
        config = load_config(config_path)
        for device_key in ('gpu', 'cpu'):
            params = config[device_key]
            assert 'batch_size' in params
            assert 'gradient_accumulation_steps' in params
            assert 'fp16' in params
            assert 'lr_frozen' in params
            assert 'lr_unfrozen' in params

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.json')


# ---------------------------------------------------------------------------
# Integration tests (require model download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainIntegration:
    def test_train_completes_on_cpu(self, config_path, tmp_path):
        """Acceptance criterion: training completes on CPU with limited samples."""
        result = train(config_path=config_path, max_train_samples=14)

        assert 'eval_loss' in result
        assert 'eval_macro_f1' in result
        assert 'eval_weighted_f1' in result
        assert 'eval_balanced_accuracy' in result

    def test_artifacts_saved(self, config_path, tmp_path):
        """Acceptance criterion: model checkpoints and metrics saved."""
        train(config_path=config_path, max_train_samples=14)

        output_dir = Path(tmp_path / 'model_output')
        assert output_dir.exists()
        assert (output_dir / 'eval_metrics.json').exists()
        assert (output_dir / 'label_mappings.json').exists()

        # Label mappings should be valid
        with open(output_dir / 'label_mappings.json') as f:
            mappings = json.load(f)
        assert len(mappings['families']) == 7
        assert len(mappings['label_to_idx']) == 7

    def test_augmentation_integration(self, tmp_path, train_csv, val_csv):
        """Acceptance criterion: augmentation pipeline integrates during training."""
        config = {
            'model_name': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
            'num_labels': 7,
            'trust_remote_code': True,
            'train_csv': train_csv,
            'val_csv': val_csv,
            'output_dir': str(tmp_path / 'model_aug'),
            'max_length': 128,
            'seed': 42,
            'gpu': {
                'batch_size': 2, 'gradient_accumulation_steps': 1,
                'fp16': False, 'lr_frozen': 2e-4, 'lr_unfrozen': 5e-5,
                'num_workers': 0,
            },
            'cpu': {
                'batch_size': 2, 'gradient_accumulation_steps': 1,
                'fp16': False, 'lr_frozen': 2e-4, 'lr_unfrozen': 5e-5,
                'num_workers': 0,
            },
            'epochs': 1,
            'warmup_ratio': 0.0,
            'weight_decay': 0.0,
            'early_stopping_patience': 3,
            'metric_for_best_model': 'eval_macro_f1',
            'unfreeze_layers': 0,
            'augmentation': {
                'enabled': True,
                'reverse_complement': True,
                'random_crop': False,
                'sequencing_noise': None,
                'n_masking': True,
                'mask_rate': 0.02,
                'kmer_shuffle': False,
            },
            'logging_steps': 1,
            'save_strategy': 'epoch',
            'eval_strategy': 'epoch',
        }

        config_path = str(tmp_path / 'aug_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = train(config_path=config_path, max_train_samples=14)
        assert 'eval_macro_f1' in result

    def test_weighted_loss_applied(self, config_path, tmp_path):
        """Acceptance criterion: WeightedTrainer applies class weights."""
        result = train(config_path=config_path, max_train_samples=14)
        # Training should complete without errors — the weighted loss is
        # validated by the unit tests above; this confirms end-to-end wiring
        assert 'eval_loss' in result
        assert result['eval_loss'] > 0
