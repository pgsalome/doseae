#!/usr/bin/env python3
"""
Main training script for dose autoencoder.
"""

import argparse
import json
import yaml
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import sys
import os
import numpy as np
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

try:
    from utils.clinical_metrics import ClinicalMetricsCalculator
except ImportError:  # pragma: no cover
    ClinicalMetricsCalculator = None

from datasets.loaders import create_data_loaders, create_dataset, collate_medical_batch, _LimitedDataset
from models import get_model
from core.training.trainer import Trainer
try:
    from core.optimization.neptune_optimizer import NeptuneOptimizer
except ImportError:  # pragma: no cover
    NeptuneOptimizer = None


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_wandb_metadata(config: dict, entity_type: str):
    """Enrich wandb configuration with descriptive defaults."""
    wandb_cfg = config.setdefault('wandb', {})
    if not wandb_cfg.get('use_wandb', False):
        return

    model_cfg = config.get('model', {})
    hyper_cfg = config.get('hyperparameters', {})
    training_cfg = config.get('training', {})

    project_default = f"doseae_{entity_type}"
    wandb_cfg.setdefault('project_name', project_default)

    model_type = model_cfg.get('type', 'unknown')
    base_filters = model_cfg.get('base_filters', '')
    latent_dim = model_cfg.get('latent_dim', '')
    learning_rate = hyper_cfg.get('learning_rate', training_cfg.get('learning_rate', 1e-4))
    batch_size = hyper_cfg.get('batch_size', training_cfg.get('batch_size', 1))
    optimizer = training_cfg.get('optimizer', 'adam')
    trial_number = wandb_cfg.get('trial_number')

    if wandb_cfg.get('run_name'):
        run_name = wandb_cfg['run_name']
    else:
        lr_str = f"{learning_rate:.0e}" if isinstance(learning_rate, (int, float)) else str(learning_rate)
        prefix = f"{entity_type}_{model_type}"
        suffix = f"f{base_filters}_l{latent_dim}_lr{lr_str}_bs{batch_size}_{optimizer}"
        if trial_number is not None:
            run_name = f"{prefix}_trial_{int(trial_number):02d}_{suffix}"
        else:
            run_name = f"{prefix}_{suffix}"
        wandb_cfg['run_name'] = run_name

    tags = set(wandb_cfg.get('tags', []))
    tags.add(entity_type)
    tags.add(model_type)
    if base_filters:
        tags.add(f"filters_{base_filters}")
    if latent_dim:
        tags.add(f"latent_{latent_dim}")
    tags.add(f"optimizer_{optimizer}")
    wandb_cfg['tags'] = sorted(tags)
    wandb_cfg.setdefault('group', f"{entity_type}_training")
    wandb_cfg.setdefault('watch', {'log': 'all', 'log_freq': 100})


def _extract_prediction_tensor(outputs: Any) -> Optional[torch.Tensor]:
    """Return the primary prediction tensor from model outputs."""
    if isinstance(outputs, dict):
        for key in ('reconstruction', 'predicted_dose', 'output', 'dose'):
            tensor = outputs.get(key)
            if torch.is_tensor(tensor):
                return tensor
        return None
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item):
                return item
        return None
    if torch.is_tensor(outputs):
        return outputs
    return None


def compute_validation_clinical_metrics(model, val_loader, config: dict, device):
    """Compute clinical metrics on a single validation batch."""
    if ClinicalMetricsCalculator is None:
        return None
    if not config.get('clinical_metrics'):
        return None

    calculator = ClinicalMetricsCalculator(config)
    model.eval()

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return None

    def to_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        return value

    batch = {key: to_device(value) for key, value in batch.items()}

    with torch.no_grad():
        # Determine primary input tensor for the model
        input_tensor = batch.get('input') or batch.get('ct') or batch.get('ct_patches') or batch.get('dose')
        if input_tensor is None:
            return None

        try:
            outputs = model(input_tensor, **batch)
        except TypeError:
            outputs = model(input_tensor)

    predicted = _extract_prediction_tensor(outputs)
    if predicted is None:
        return None

    target = batch.get('dose') or batch.get('target') or batch.get('dose_patches')
    if target is None:
        return None

    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    pred_np = np.squeeze(pred_np)
    target_np = np.squeeze(target_np)

    if pred_np.size == 0 or target_np.size == 0:
        return None

    mask_tensor = batch.get('mask') or batch.get('attention_mask')
    mask_np = None
    if mask_tensor is not None and torch.is_tensor(mask_tensor):
        mask_np = np.squeeze(mask_tensor.detach().cpu().numpy())

    try:
        metrics = calculator.compare_dose_distributions(target_np, pred_np, mask_np)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("Failed to compute clinical metrics: %s", exc)
        return None

    return metrics


def create_model(config: dict, entity_type: str):
    """Create model based on entity type and configuration."""
    model_cfg = config.setdefault('model', {})
    model_type = model_cfg.get('type', '').lower()

    if not model_type:
        model_cfg['type'] = 'resnet_ae'
    return get_model(config)


def train_model(config: dict, entity_type: str, data_dir: str):
    """Train the model."""
    logger = logging.getLogger(__name__)

    # Ensure WandB metadata is prepared even if main() is bypassed
    prepare_wandb_metadata(config, entity_type)

    # Resolve output directories
    output_cfg = config.get('output', {})
    results_path = Path(output_cfg.get('results_dir', './output'))
    model_path = Path(output_cfg.get('model_dir', results_path / 'models'))
    log_path = Path(output_cfg.get('log_dir', results_path / 'logs'))

    for path in {results_path, model_path, log_path}:
        Path(path).mkdir(parents=True, exist_ok=True)

    output_cfg['results_dir'] = str(results_path)
    output_cfg['model_dir'] = str(model_path)
    output_cfg['log_dir'] = str(log_path)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, entity_type, data_dir)
    try:
        logger.info("Train loader batches: %d, batch size: %d", len(train_loader), train_loader.batch_size)
        logger.info("Val loader batches: %d, batch size: %d", len(val_loader), val_loader.batch_size)
    except Exception:
        logger.info("Batch size reporting failed (custom loader).")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, entity_type)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(model, config, entity_type)
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(train_loader, val_loader)

    # Optional additional evaluation with clinical metrics
    try:
        final_val_metrics = trainer.validate_epoch(val_loader)
        training_history['final_val_metrics'] = final_val_metrics
        if final_val_metrics and wandb is not None and wandb.run is not None:
            log_payload = {
                f"val/final_{k}": float(v)
                for k, v in final_val_metrics.items()
                if np.isscalar(v)
            }
            if log_payload:
                wandb.log(log_payload)
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to compute final validation metrics: %s", exc)
        final_val_metrics = None

    clinical_metrics = compute_validation_clinical_metrics(trainer.model, val_loader, config, trainer.device)
    if clinical_metrics:
        scalar_metrics = {
            k: float(v)
            for k, v in clinical_metrics.items()
            if np.isscalar(v)
        }
        training_history['clinical_metrics'] = scalar_metrics
        if wandb is not None and wandb.run is not None:
            log_payload = {
                f"val/clinical_{k}": value
                for k, value in scalar_metrics.items()
            }
            if log_payload:
                wandb.log(log_payload)

    # Evaluate on test set using the best checkpoint
    batch_size = config.get('training', {}).get('batch_size', 1)
    dataset_cfg = config.get('dataset', {})
    num_workers = dataset_cfg.get('num_workers', config.get('data', {}).get('num_workers', 0))
    pin_memory = bool(dataset_cfg.get('pin_memory', False))

    logger.info("Preparing test loader...")
    test_dataset = create_dataset(config, entity_type, 'test', data_dir, transform=None, apply_transforms=False)
    if dataset_cfg.get('test_mode', False):
        limit_batches = int(dataset_cfg.get('n_test_samples', dataset_cfg.get('n_test_batches', 20)))
        if limit_batches > 0:
            limit_samples = limit_batches * batch_size
            logger.info(
                "Test mode enabled: restricting test dataset to %d batches (%d samples)",
                limit_batches,
                limit_samples,
            )
            test_dataset = _LimitedDataset(test_dataset, limit_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=getattr(test_dataset, 'collate_fn', None) or collate_medical_batch,
    )

    if trainer.best_checkpoint_path is not None:
        logger.info(f"Loading best checkpoint from {trainer.best_checkpoint_path}")
        trainer.load_checkpoint(str(trainer.best_checkpoint_path))
    else:
        logger.warning("No best checkpoint recorded; proceeding with current model weights for test evaluation.")

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate_testset(test_loader)
    training_history['test_metrics'] = test_results

    test_metrics_file = results_path / 'test_metrics.json'
    with open(test_metrics_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test metrics saved to {test_metrics_file}")

    if wandb is not None and wandb.run is not None:
        wandb_log = {'test/num_samples': test_results.get('num_samples', 0)}

        table = wandb.Table(columns=["segmentation", "metric", "statistic", "value"])

        def add_rows(segmentation: str, stats: dict):
            if not stats:
                return
            for metric, summary in stats.items():
                for stat_name, value in summary.items():
                    table.add_data(segmentation, metric, stat_name, value)

        add_rows("overall", test_results.get('overall', {}))
        for lobe, metrics in test_results.get('per_lobe', {}).items():
            add_rows(f"lobe:{lobe}", metrics)
        for side, metrics in test_results.get('per_side', {}).items():
            add_rows(f"side:{side}", metrics)

        wandb_log['test/metrics_table'] = table
        wandb.log(wandb_log, commit=True)
        wandb.finish()

    # Save training history
    history_file = results_path / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed. History saved to {history_file}")
    
    return training_history


def optimize_hyperparameters(config: dict, entity_type: str, data_dir: str):
    """Run hyperparameter optimization."""
    logger = logging.getLogger(__name__)

    if NeptuneOptimizer is None:
        raise ImportError(
            "Neptune is not installed. Please install the 'neptune' package to use optimization mode."
        )

    prepare_wandb_metadata(config, entity_type)

    # Create output directory
    output_cfg = config.get('output', {})
    output_path = Path(output_cfg.get('results_dir', './output'))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, entity_type, data_dir)
    
    # Create model factory
    def model_factory(trial_config):
        return create_model(trial_config, entity_type)
    
    # Create optimizer
    logger.info("Creating Neptune optimizer...")
    optimizer = NeptuneOptimizer(config, entity_type)
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    results = optimizer.optimize(model_factory, train_loader, val_loader)
    
    # Save results
    results_file = output_path / 'optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization completed. Results saved to {results_file}")
    
    # Close optimizer
    optimizer.close()
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train dose autoencoder')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--entity', type=str, required=True, choices=['lung', 'hnc'], help='Entity type')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, help='Override results directory defined in config')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize'], default='train', help='Training mode')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set NumExpr thread limit from config (must be done before any NumExpr operations)
    training_cfg = config.get('training', {})
    numexpr_threads = training_cfg.get('numexpr_max_threads')
    if numexpr_threads is not None and numexpr_threads > 0:
        os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr_threads)
        logger.info(f"Set NUMEXPR_MAX_THREADS to {numexpr_threads}")
    else:
        # Default to all available cores if not specified or set to 0/None
        cpu_count = os.cpu_count() or 32
        os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_count)
        logger.info(f"NUMEXPR_MAX_THREADS not specified in config, using all available cores: {cpu_count}")

    # Optionally merge preprocessing config for training/preprocessing parity
    if 'preprocessing_config' in config:
        preproc_path = Path(config['preprocessing_config'])
        if not preproc_path.is_absolute():
            preproc_path = Path(args.config).parent / preproc_path
        preproc_path = preproc_path.resolve()

        if preproc_path.exists():
            logger.info(f"Merging preprocessing settings from {preproc_path}")
            preproc_config = load_config(str(preproc_path))

            for section in ['preprocessing', 'patch_extraction', 'lung']:
                if section in preproc_config:
                    config[section] = preproc_config[section]

            if 'dataset' in preproc_config:
                merged_dataset = preproc_config['dataset'].copy()
                merged_dataset.update(config.get('dataset', {}))
                config['dataset'] = merged_dataset
        else:
            logger.warning(f"Preprocessing config path not found: {preproc_path}")
    
    # Load entity-specific configuration
    entity_config_path = f"configs/entities/{args.entity}_config.yaml"
    if os.path.exists(entity_config_path):
        entity_config = load_config(entity_config_path)
        config.update(entity_config)
        logger.info(f"Loaded entity-specific configuration from {entity_config_path}")

    # Resolve output directories from config and optional CLI override
    output_cfg = config.setdefault('output', {})
    if args.output_dir:
        output_cfg['results_dir'] = args.output_dir

    config_root = Path(args.config).resolve().parent

    def _resolve_path(value: Optional[str], fallback: Optional[Path] = None) -> Path:
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = (config_root / path).resolve()
        elif fallback is not None:
            path = fallback
        else:
            path = (config_root / 'output').resolve()
        return path

    results_path = _resolve_path(output_cfg.get('results_dir'))
    output_cfg['results_dir'] = str(results_path)
    model_path = _resolve_path(output_cfg.get('model_dir'), results_path / 'models')
    output_cfg['model_dir'] = str(model_path)
    log_path = _resolve_path(output_cfg.get('log_dir'), results_path / 'logs')
    output_cfg['log_dir'] = str(log_path)

    for path in {results_path, model_path, log_path}:
        path.mkdir(parents=True, exist_ok=True)

    # Prepare WandB metadata before training starts
    prepare_wandb_metadata(config, args.entity)

    # Run training or optimization
    if args.mode == 'train':
        logger.info("Starting training...")
        training_history = train_model(config, args.entity, args.data_dir)
        logger.info("Training completed successfully")
    elif args.mode == 'optimize':
        logger.info("Starting hyperparameter optimization...")
        results = optimize_hyperparameters(config, args.entity, args.data_dir)
        logger.info("Optimization completed successfully")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
