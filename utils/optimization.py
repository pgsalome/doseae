import torch
import copy
import os
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import get_model
from datasets.loaders import create_data_loaders
from utils.clinical_metrics import ClinicalMetricsCalculator
from utils.clinical_loss import ClinicalLoss
from utils.advanced_loss import AdvancedDoseLoss


def _set_nested(config: dict, path: str, value):
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_training_value(config: dict, key: str, default=None):
    """
    Read a training hyperparameter, preferring the `training` block and falling
    back to the legacy `hyperparameters` block.
    """
    training_cfg = config.get('training', {}) or {}
    if key in training_cfg:
        return training_cfg[key]
    hyper_cfg = config.get('hyperparameters', {}) or {}
    return hyper_cfg.get(key, default)


def _set_training_value(config: dict, key: str, value):
    """
    Write a training hyperparameter, keeping both the `training` and
    `hyperparameters` blocks (if present) in sync so downstream utilities that
    expect either layout keep working.
    """
    training_cfg = config.setdefault('training', {})
    if isinstance(training_cfg, dict):
        training_cfg[key] = value
    hyper_cfg = config.get('hyperparameters')
    if isinstance(hyper_cfg, dict):
        hyper_cfg[key] = value


def _infer_model_channels(config: dict):
    """
    Ensure model.in_channels/output_channels exist by inferring them from the dataset block.
    """
    model_cfg = config.setdefault('model', {})
    dataset_cfg = config.get('dataset', {}) or {}

    input_channels = dataset_cfg.get('input_channels')
    num_channels = None
    if isinstance(input_channels, (list, tuple)):
        num_channels = len(input_channels)
    elif isinstance(input_channels, int):
        num_channels = input_channels
    elif isinstance(input_channels, str):
        num_channels = 1

    if num_channels is None:
        num_channels = dataset_cfg.get('num_input_channels') or 1

    model_cfg.setdefault('in_channels', num_channels)
    model_cfg.setdefault('output_channels', dataset_cfg.get('output_channels', 1) or 1)


_TARGET_KEYS = ("target", "dose", "dose_patches", "reconstruction_target", "image")


def _resolve_target_tensor_from_dict(batch: dict, fallback: torch.Tensor, device) -> torch.Tensor:
    """
    Pick the first available tensor in the batch among standard keys; fallback to the provided tensor.
    """
    for key in _TARGET_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value):
            return value.to(device)
    return fallback


def _extract_prediction_tensor(outputs):
    """
    Return the primary tensor from model outputs (tensor/tuple/dict).
    Mirrors the helper in scripts/train.py so Optuna trials handle dictionary
    outputs.
    """
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ('reconstruction', 'predicted_dose', 'dose', 'output', 'prediction'):
            tensor = outputs.get(key)
            if torch.is_tensor(tensor):
                return tensor
        for value in outputs.values():
            tensor = _extract_prediction_tensor(value)
            if tensor is not None:
                return tensor
        return None
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            tensor = _extract_prediction_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _suggest_value(trial, param_spec: dict):
    name = param_spec['name']
    param_type = str(param_spec.get('type', 'categorical')).lower()

    if param_type in {'int', 'integer'}:
        low = param_spec['low']
        high = param_spec['high']
        log = bool(param_spec.get('log', False))
        step = param_spec.get('step')
        if step is not None:
            return trial.suggest_int(name, low, high, step=step, log=log)
        return trial.suggest_int(name, low, high, log=log)
    if param_type in {'float', 'double'}:
        low = param_spec['low']
        high = param_spec['high']
        log = bool(param_spec.get('log', False))
        step = param_spec.get('step')
        if step is not None:
            return trial.suggest_float(name, low, high, step=step, log=log)
        return trial.suggest_float(name, low, high, log=log)
    if param_type in {'categorical', 'choice'}:
        choices = param_spec['choices']
        return trial.suggest_categorical(name, choices)
    if param_type in {'bool', 'boolean'}:
        return trial.suggest_categorical(name, [True, False])
    raise ValueError(f"Unsupported Optuna parameter type '{param_type}' for '{name}'")


def apply_optuna_parameters(trial, base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault('training', {})
    optuna_cfg = base_config.get('optuna', {})
    param_defs = optuna_cfg.get('parameters') or []

    if not param_defs:
        # Fallback to legacy behaviour
        config = define_model_params(trial, base_config)
        config = define_training_params(trial, config)
        _infer_model_channels(config)
        return config

    for spec in param_defs:
        if 'name' not in spec:
            raise KeyError("Optuna parameter definition missing 'name'")
        suggestion = _suggest_value(trial, spec)
        _set_nested(config, spec['name'], suggestion)

    _infer_model_channels(config)

    return config


def define_model_params(trial, base_config):
    config = copy.deepcopy(base_config)

    # Use the model type that's already set in the config
    model_type = config['model']['type'].lower()
    print(f"Using model type from config: {model_type}")

    # Set parameters based on model type
    if model_type == 'vae':
        # For VAE, use compatible dimensions
        base_filters = trial.suggest_int('base_filters', 16, 64, log=True)
        config['model']['base_filters'] = base_filters

        # Ensure latent_dim is compatible with the encoded size
        encoded_size = 4 * 4 * 4  # Size after 4 layers of strided convolutions
        max_features = base_filters * 8 * encoded_size
        config['model']['latent_dim'] = trial.suggest_int(
            'latent_dim', 32, max_features, log=True)
    elif model_type == 'doseae_resunet':
        # For AdvancedResNetUNetAttention, parameters are set in run_experiments.py
        # This function is called after the parameters are already set
        print("Using AdvancedResNetUNetAttention with parameters from run_experiments.py")
        pass  # Parameters are already set in the config
    else:
        # For other models, use more flexible parameters
        config['model']['latent_dim'] = trial.suggest_int(
            'latent_dim', 32, 512, log=True)
        config['model']['base_filters'] = trial.suggest_int(
            'base_filters', 16, 64, log=True)

    return config


def define_training_params(trial, base_config):
    """
    Define hyperparameters to search over for training.
    Args:
        trial (optuna.Trial): Optuna trial object
        base_config (dict): Base configuration to modify
    Returns:
        dict: Modified configuration
    """
    config = copy.deepcopy(base_config)

    # Learning rate
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    _set_training_value(config, 'learning_rate', lr)

    # Batch size
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    _set_training_value(config, 'batch_size', batch_size)

    # Weight decay
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    _set_training_value(config, 'weight_decay', weight_decay)

    # Optimizer
    config['training']['optimizer'] = trial.suggest_categorical(
        'optimizer', ['adam', 'sgd', 'rmsprop']
    )

    # Scheduler
    config['training']['scheduler'] = trial.suggest_categorical(
        'scheduler', ['reduce_on_plateau', 'cosine_annealing', 'none']
    )

    # Normalization and output activation
    config['preprocessing']['normalize'] = trial.suggest_categorical(
        'normalize', [True, False]
    )
    config['preprocessing']['use_tanh_output'] = trial.suggest_categorical(
        'use_tanh_output', [True, False]
    )

    return config


def objective(trial, base_config, device):
    """Objective function for Optuna optimization."""
    try:
        # Set GPU visibility before any CUDA operations
        gpu_ids = base_config.get('training', {}).get('gpu_ids', [])
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        # Define hyperparameters/architecture from config-driven search space
        config = apply_optuna_parameters(trial, base_config)

        # Make sure we're using the correct model type
        model_type = config['model']['type'].lower()
        print(f"Using model type from config: {model_type}")

        # Create model
        print("Creating model...")
        model = get_model(config)
        print(f"Model created: {type(model).__name__}")
        
        # Configure multi-GPU settings based on config
        gpu_ids = config.get('training', {}).get('gpu_ids', [])
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            print(f"{available_gpus} GPUs detected, but Optuna trials force single-GPU execution to avoid sync issues.")
        print("Using single GPU for training")

        # Move model to device
        model = model.to(device)

        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(config)
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': val_loader,
        }
        
        # Check if data_loaders is None or missing required keys
        if data_loaders is None:
            print("Error: Data loaders are None")
            return float('inf')

        for key in ['train', 'val', 'test']:
            if key not in data_loaders:
                print(f"Error: Missing '{key}' in data_loaders")
                return float('inf')

        # Print the shape of the first batch for debugging
        try:
            first_batch = next(iter(data_loaders['train']))
            if isinstance(first_batch, dict):
                print(f"First batch shape: {first_batch['image'].shape}")
            else:
                print(f"First batch shape: {first_batch[0].shape}")
        except Exception as e:
            print(f"Error getting first batch: {e}")
            return float('inf')

        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = int(_get_training_value(config, 'early_stopping_patience', 5) or 5)
        total_epochs = int(_get_training_value(config, 'epochs', 100) or 100)
        optuna_output_dir = (
            config.get('optuna_output_dir')
            or config.get('optuna', {}).get('output_dir')
            or config.get('output', {}).get('results_dir')
            or "."
        )
        model_dir = os.path.join(optuna_output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"best_model_trial_{trial.number}.pth")

        # Set up optimizer
        optimizer_name = config['training']['optimizer']
        learning_rate = _get_training_value(config, 'learning_rate', 1e-4) or 1e-4
        weight_decay = _get_training_value(config, 'weight_decay', 0.0) or 0.0
        optimizer_params = {
            'lr': learning_rate,
            'weight_decay': weight_decay
        }

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, **optimizer_params)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), **optimizer_params)
        else:
            print(f"Unknown optimizer: {optimizer_name}")
            return float('inf')

        # Set up scheduler
        scheduler = None
        scheduler_name = config['training']['scheduler']
        if scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler_name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs
            )

        # Initialize wandb if enabled
        if config['wandb']['use_wandb']:
            trial_name = f"trial_{trial.number}"
            wandb.init(
                project=config['wandb']['project_name'],
                entity=config['wandb']['entity'],
                name=trial_name,
                config=config
            )

        # Prepare loss function
        loss_fn = None
        loss_type = config.get('loss_function', {}).get('type', 'mse')
        
        if loss_type == 'advanced':
            try:
                loss_fn = AdvancedDoseLoss(config)
                print("Using AdvancedDoseLoss")
            except Exception as e:
                print(f"Warning: AdvancedDoseLoss init failed ({e}); falling back to MSE")
                loss_fn = None
        elif loss_type != 'mse':
            try:
                loss_fn = ClinicalLoss(config)
                print("Using ClinicalLoss")
            except Exception as e:
                print(f"Warning: ClinicalLoss init failed ({e}); falling back to MSE")
                loss_fn = None

        # Prepare clinical metrics calculator and frequency
        clinical_metrics_calculator = ClinicalMetricsCalculator(config)
        gamma_freq = config.get('loss_function', {}).get('gamma_calculation_frequency', 10)
        beta_value = _get_training_value(config, 'beta', 1.0) or 1.0

        # Training loop
        import time
        start_time = time.time()
        
        for epoch in range(total_epochs):
            epoch_start = time.time()
            print(f"\nTrial Epoch {epoch + 1}/{total_epochs}")

            # Initialize loss tracking for this epoch
            train_loss_components = {}
            val_loss_components = {}

            # Training
            model.train()
            train_batches = 0

            # Training progress bar
            train_pbar = tqdm(data_loaders['train'], desc="  Training", leave=False)
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Handle different dataset types
                    if isinstance(batch, dict):  # For DoseAEDataset
                        data = batch["image"].to(device)
                        target_tensor = _resolve_target_tensor_from_dict(batch, data, device)
                        mask_tensor = batch.get("mask")
                        if mask_tensor is not None:
                            mask_tensor = mask_tensor.to(device)
                    else:  # For standard (input, target) dataset
                        data, target_tensor = batch
                        data = data.to(device)
                        if torch.is_tensor(target_tensor):
                            target_tensor = target_tensor.to(device)
                        elif isinstance(target_tensor, dict):
                            target_tensor = _resolve_target_tensor_from_dict(target_tensor, data, device)
                        else:
                            target_tensor = data
                        mask_tensor = None

                    # Training step
                    optimizer.zero_grad()

                    # Forward pass - handle different model types
                    try:
                        if model_type == 'vae':
                            recon, mu, logvar = model(data)
                            
                            # Apply non-negative constraint (ReLU) for dose data
                            recon = torch.relu(recon)
                            
                            if isinstance(loss_fn, (ClinicalLoss, AdvancedDoseLoss)):
                                if isinstance(loss_fn, ClinicalLoss):
                                    losses = loss_fn(recon, target_tensor, mask_tensor, epoch=epoch)
                                else:  # AdvancedDoseLoss
                                    losses = loss_fn(recon, target_tensor, mask_tensor)
                                
                                # Add KL divergence for VAE
                                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                                kl_loss = kl_loss / data.size(0)
                                beta = beta_value
                                losses['kl'] = kl_loss
                                losses['kl_weighted'] = beta * kl_loss
                                losses['total'] = losses['total'] + beta * kl_loss
                                loss = losses['total']
                                
                                # Accumulate
                                for k, v in list(losses.items()):
                                    if k not in train_loss_components:
                                        train_loss_components[k] = 0.0
                                    train_loss_components[k] += v.item()
                                # NEW: ensure early stopping sees a 'total_loss'
                                train_loss_components['total_loss'] = train_loss_components.get('total_loss', 0.0) + losses['total'].item()
                            else:
                                # Default model-provided losses
                                # Handle DataParallel model
                                model_to_use = model.module if hasattr(model, 'module') else model
                                loss_dict = model_to_use.get_losses(
                                    target_tensor, recon, mu, logvar,
                                    beta=beta_value
                                )
                                loss = loss_dict['total_loss']
                                for loss_name, loss_value in list(loss_dict.items()):
                                    if loss_name not in train_loss_components:
                                        train_loss_components[loss_name] = 0.0
                                    train_loss_components[loss_name] += loss_value.item()
                        else:
                            # Regular autoencoder (no VAE)
                            outputs = model(data)

                            recon = _extract_prediction_tensor(outputs)
                            if recon is None:
                                print('Error: model outputs did not contain a tensor prediction')
                                return float('inf')

                            # Apply non-negative constraint (ReLU)
                            recon = torch.relu(recon)

                            if isinstance(loss_fn, (ClinicalLoss, AdvancedDoseLoss)):
                                if isinstance(loss_fn, ClinicalLoss):
                                    losses = loss_fn(recon, target_tensor, mask_tensor, epoch=epoch)
                                else:  # AdvancedDoseLoss
                                    losses = loss_fn(recon, target_tensor, mask_tensor)
                                loss = losses['total']
                                for k, v in list(losses.items()):
                                    if k not in train_loss_components:
                                        train_loss_components[k] = 0.0
                                    train_loss_components[k] += v.item()
                                # NEW: alias for early stopping
                                train_loss_components['total_loss'] = train_loss_components.get('total_loss', 0.0) + losses['total'].item()
                            else:
                                # Calculate MSE loss
                                loss = torch.nn.functional.mse_loss(recon, target_tensor)
                                # Track MSE
                                if 'mse_loss' not in train_loss_components:
                                    train_loss_components['mse_loss'] = 0.0
                                train_loss_components['mse_loss'] += loss.item()
                                # NEW: also track as total_loss for early stopping
                                train_loss_components['total_loss'] = train_loss_components.get('total_loss', 0.0) + loss.item()

                        # Backward and optimize
                        loss.backward()
                        optimizer.step()

                        train_batches += 1

                        # Update progress bar with detailed losses
                        if train_loss_components:
                            loss_str = ', '.join([f'{k}: {v/train_batches:.4f}' for k, v in list(train_loss_components.items())])
                            train_pbar.set_postfix_str(loss_str)

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f"    CUDA OOM in batch {batch_idx}, skipping")
                            # Try to free up memory
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                except Exception as e:
                    print(f"    Error in training batch {batch_idx}: {e}")
                    return float('inf')

            # Validation
            model.eval()
            val_batches = 0

            # Validation progress bar
            val_pbar = tqdm(data_loaders['val'], desc="  Validation", leave=False)
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    try:
                        # Handle different dataset types
                        if isinstance(batch, dict):  # For DoseAEDataset
                            data = batch["image"].to(device)
                            target_tensor = _resolve_target_tensor_from_dict(batch, data, device)
                            mask_tensor = batch.get("mask")
                            if mask_tensor is not None:
                                mask_tensor = mask_tensor.to(device)
                        else:  # For standard (input, target) dataset
                            data, target_tensor = batch
                            data = data.to(device)
                            if torch.is_tensor(target_tensor):
                                target_tensor = target_tensor.to(device)
                            elif isinstance(target_tensor, dict):
                                target_tensor = _resolve_target_tensor_from_dict(target_tensor, data, device)
                            else:
                                target_tensor = data
                            mask_tensor = None

                        # Forward pass - handle different model types
                        if model_type == 'vae':
                            recon, mu, logvar = model(data)
                            
                            # Apply non-negative constraint (ReLU) for dose data
                            recon = torch.relu(recon)
                            
                            if isinstance(loss_fn, (ClinicalLoss, AdvancedDoseLoss)):
                                if isinstance(loss_fn, ClinicalLoss):
                                    losses = loss_fn(recon, target_tensor, mask_tensor, epoch=epoch)
                                else:  # AdvancedDoseLoss
                                    losses = loss_fn(recon, target_tensor, mask_tensor)
                                
                                # Add KL divergence for VAE
                                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                                kl_loss = kl_loss / data.size(0)
                                beta = beta_value
                                losses['kl'] = kl_loss
                                losses['kl_weighted'] = beta * kl_loss
                                losses['total'] = losses['total'] + beta * kl_loss
                                
                                for k, v in list(losses.items()):
                                    if k not in val_loss_components:
                                        val_loss_components[k] = 0.0
                                    val_loss_components[k] += v.item()
                                # NEW: alias for early stopping
                                val_loss_components['total_loss'] = val_loss_components.get('total_loss', 0.0) + losses['total'].item()
                            else:
                                # Handle DataParallel model
                                model_to_use = model.module if hasattr(model, 'module') else model
                                loss_dict = model_to_use.get_losses(
                                    target_tensor, recon, mu, logvar,
                                    beta=beta_value
                                )
                                for loss_name, loss_value in list(loss_dict.items()):
                                    if loss_name not in val_loss_components:
                                        val_loss_components[loss_name] = 0.0
                                    val_loss_components[loss_name] += loss_value.item()
                        else:
                            # Regular autoencoder (no VAE)
                            outputs = model(data)

                            recon = _extract_prediction_tensor(outputs)
                            if recon is None:
                                print('Error: model outputs did not contain a tensor prediction (val)')
                                return float('inf')

                            # Apply non-negative constraint (ReLU)
                            recon = torch.relu(recon)

                            if isinstance(loss_fn, (ClinicalLoss, AdvancedDoseLoss)):
                                if isinstance(loss_fn, ClinicalLoss):
                                    losses = loss_fn(recon, target_tensor, mask_tensor, epoch=epoch)
                                else:  # AdvancedDoseLoss
                                    losses = loss_fn(recon, target_tensor, mask_tensor)
                                for k, v in list(losses.items()):
                                    if k not in val_loss_components:
                                        val_loss_components[k] = 0.0
                                    val_loss_components[k] += v.item()
                                # NEW: alias for early stopping
                                val_loss_components['total_loss'] = val_loss_components.get('total_loss', 0.0) + losses['total'].item()
                            else:
                                # Calculate MSE loss
                                loss = torch.nn.functional.mse_loss(recon, target_tensor)
                                # Track MSE
                                if 'mse_loss' not in val_loss_components:
                                    val_loss_components['mse_loss'] = 0.0
                                val_loss_components['mse_loss'] += loss.item()
                                # NEW: also track as total_loss for early stopping
                                val_loss_components['total_loss'] = val_loss_components.get('total_loss', 0.0) + loss.item()

                        val_batches += 1

                        # Update progress bar with detailed losses
                        if val_loss_components:
                            loss_str = ', '.join([f'{k}: {v/val_batches:.4f}' for k, v in list(val_loss_components.items())])
                            val_pbar.set_postfix_str(loss_str)

                    except Exception as e:
                        print(f"    Error in validation batch {batch_idx}: {e}")
                        return float('inf')

            # Optionally compute clinical metrics once per N epochs on a val batch
            clinical = None
            if (epoch + 1) % gamma_freq == 0:
                try:
                    val_iter_dbg = iter(data_loaders['val'])
                    batch_sample = next(val_iter_dbg)

                    if isinstance(batch_sample, dict):
                        val_data = batch_sample.get("image").to(device)
                        val_target = _resolve_target_tensor_from_dict(batch_sample, val_data, device)
                        val_mask = batch_sample.get("mask")
                        if val_mask is not None:
                            val_mask = val_mask.to(device)
                    else:
                        val_data, val_target = batch_sample
                        val_data = val_data.to(device)
                        if torch.is_tensor(val_target):
                            val_target = val_target.to(device)
                        elif isinstance(val_target, dict):
                            val_target = _resolve_target_tensor_from_dict(val_target, val_data, device)
                        else:
                            val_target = val_data
                        val_mask = None

                    if val_data.ndim == 3:
                        val_data = val_data.unsqueeze(1)
                    if val_target.ndim == 3:
                        val_target = val_target.unsqueeze(1)

                    with torch.no_grad():
                        if model_type == 'vae':
                            val_recon, _, _ = model(val_data)
                        else:
                            out_dbg = model(val_data)
                            val_recon = _extract_prediction_tensor(out_dbg)
                            if val_recon is None:
                                raise ValueError('Unable to extract prediction tensor for clinical metrics')

                    target_np = val_target[0, 0].detach().cpu().numpy()
                    pred_np = val_recon[0, 0].detach().cpu().numpy()
                    mask_np = val_mask[0].detach().cpu().numpy() if val_mask is not None else None

                    clinical = clinical_metrics_calculator.compare_dose_distributions(
                        target_np, pred_np, mask_np
                    )
                except Exception as e:
                    print(f"    Clinical metrics error: {e}")

            # Print detailed epoch summary with timing
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch + 1} Summary (took {epoch_time:.2f}s):")
            if train_loss_components:
                print(f"    Training Losses:")
                for loss_name, loss_value in list(train_loss_components.items()):
                    avg_loss = loss_value / train_batches
                    print(f"      {loss_name}: {avg_loss:.6f}")

            if val_loss_components:
                print(f"    Validation Losses:")
                for loss_name, loss_value in list(val_loss_components.items()):
                    avg_loss = loss_value / val_batches
                    print(f"      {loss_name}: {avg_loss:.6f}")

            # Print clinical metrics summary if available
            if clinical is not None:
                d2 = clinical.get('D2_diff')
                d50 = clinical.get('D50_diff')
                d95 = clinical.get('D95_diff')
                gpr = clinical.get('gamma_pass_rate')
                print(f"    Clinical Metrics:")
                if d2 is not None or d50 is not None or d95 is not None:
                    try:
                        print(f"      D2_diff: {d2:.6f}  D50_diff: {d50:.6f}  D95_diff: {d95:.6f}")
                    except Exception:
                        print(f"      D2_diff: {d2}  D50_diff: {d50}  D95_diff: {d95}")
                if gpr is not None:
                    try:
                        print(f"      Gamma pass rate: {gpr:.2f}%")
                    except Exception:
                        print(f"      Gamma pass rate: {gpr}%")

            # Calculate total losses for early stopping
            avg_train_total = train_loss_components.get('total_loss', 0.0) / train_batches if train_batches > 0 else float('inf')
            avg_val_total = val_loss_components.get('total_loss', 0.0) / val_batches if val_batches > 0 else float('inf')

            # Log to wandb
            if config['wandb']['use_wandb']:
                log_dict = {'epoch': epoch + 1}

                # Add individual training losses
                for loss_name, loss_value in list(train_loss_components.items()):
                    log_dict[f'train/{loss_name}'] = loss_value / train_batches

                # Add individual validation losses
                for loss_name, loss_value in list(val_loss_components.items()):
                    log_dict[f'val/{loss_name}'] = loss_value / val_batches

                # Add clinical scalars if available
                if clinical is not None:
                    if 'gamma_pass_rate' in clinical:
                        log_dict['val/gamma_pass_rate'] = float(clinical['gamma_pass_rate'])
                    for k in ['gamma_mean', 'gamma_max', 'D95_diff', 'D50_diff', 'D2_diff', 'Dmean_diff']:
                        if k in clinical:
                            log_dict[f'val/{k}'] = float(clinical[k])

                wandb.log(log_dict)

                # Track a consistent validation example's middle slice over epochs
                try:
                    val_iter_vis = iter(data_loaders['val'])
                    batch_vis = next(val_iter_vis)

                    # Extract tensors for visualization
                    if isinstance(batch_vis, dict):
                        vis_data = batch_vis.get("image")
                        if vis_data is None:
                            for value in batch_vis.values():
                                if torch.is_tensor(value):
                                    vis_data = value
                                    break
                        if vis_data is None:
                            raise ValueError("Validation batch did not contain a tensor input")
                        vis_data = vis_data.to(device)
                        vis_target = _resolve_target_tensor_from_dict(batch_vis, vis_data, device)
                    else:
                        vis_data, vis_target = batch_vis
                        vis_data = vis_data.to(device)
                        if torch.is_tensor(vis_target):
                            vis_target = vis_target.to(device)
                        elif isinstance(vis_target, dict):
                            vis_target = _resolve_target_tensor_from_dict(vis_target, vis_data, device)
                        else:
                            vis_target = vis_data

                    if not torch.is_tensor(vis_target):
                        vis_target = vis_data

                    # Ensure channel dimension exists
                    if vis_data.ndim == 3:
                        vis_data = vis_data.unsqueeze(1)

                    with torch.no_grad():
                        outputs_vis = model(vis_data)
                    vis_recon = _extract_prediction_tensor(outputs_vis)
                    if vis_recon is None:
                        raise ValueError("Model outputs did not include a tensor prediction")

                    def _collapse_to_volume(tensor: torch.Tensor) -> np.ndarray:
                        arr = tensor.detach().cpu().float().numpy()
                        while arr.ndim > 3:
                            arr = arr[0]
                        return np.squeeze(arr)

                    recon_np = _collapse_to_volume(vis_recon)
                    target_np = _collapse_to_volume(vis_target)

                    if recon_np.ndim == 3 and target_np.ndim == 3:
                        mid = recon_np.shape[0] // 2
                        recon_slice = recon_np[mid]
                        target_slice = target_np[mid]
                    elif recon_np.ndim == 2 and target_np.ndim == 2:
                        recon_slice = recon_np
                        target_slice = target_np
                    else:
                        raise ValueError(f"Unexpected slice shapes: recon {recon_np.shape}, target {target_np.shape}")

                    diff_slice = np.abs(target_slice - recon_slice)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    im1 = axes[0].imshow(target_slice, cmap='viridis')
                    axes[0].set_title('Target (middle)')
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                    im2 = axes[1].imshow(recon_slice, cmap='viridis')
                    axes[1].set_title('Reconstruction')
                    axes[1].axis('off')
                    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                    im3 = axes[2].imshow(diff_slice, cmap='magma')
                    axes[2].set_title('Abs Diff')
                    axes[2].axis('off')
                    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    wandb.log({'track/middle_slice': wandb.Image(fig)}, commit=False)
                    plt.close(fig)
                except Exception as track_e:
                    print(f"    Tracking middle slice logging error: {track_e}")

            # Update scheduler if using ReduceLROnPlateau
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_total)
            elif scheduler is not None:
                scheduler.step()

            # Early stopping
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                patience_counter = 0
                print(f"    Validation loss improved to {best_val_loss:.6f}")
                try:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), model_path)
                    if config.get('wandb', {}).get('use_wandb', False):
                        wandb.save(model_path)
                    print(f"    Saved best model: {model_path}")
                except Exception as e:
                    print(f"    Warning: Could not save model: {e}")
            else:
                patience_counter += 1
                print(f"    No improvement for {patience_counter} epochs")
                if patience_counter >= patience:
                    print(f"    Early stopping triggered after {epoch + 1} epochs")
                    break

        # Clean up
        total_time = time.time() - start_time
        print(f"\nTrial {trial.number} completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        epoch_denom = max(total_epochs, 1)
        print(f"Average time per epoch: {total_time/epoch_denom:.2f}s")
        
        # Ensure a final best model is present even if no improvement was logged
        if not os.path.exists(model_path):
            try:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), model_path)
                if config.get('wandb', {}).get('use_wandb', False):
                    wandb.save(model_path)
                print(f"Saved best model: {model_path}")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")

        if config['wandb']['use_wandb']:
            wandb.finish()

        return best_val_loss

    except Exception as e:
        print(f"Error in objective function: {str(e)}")
        if config['wandb']['use_wandb'] and wandb.run is not None:
            wandb.finish()
        return float('inf')
