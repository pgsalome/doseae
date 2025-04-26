import os
import torch
import optuna
import yaml
import copy

import wandb

from models import get_model
from data.dataset import create_data_loaders


def define_model_params(trial, base_config):
    """
    Define hyperparameters to search over for the model.

    Args:
        trial (optuna.Trial): Optuna trial object
        base_config (dict): Base configuration to modify

    Returns:
        dict: Modified configuration
    """
    config = copy.deepcopy(base_config)

    # Model architecture
    config['model']['type'] = trial.suggest_categorical(
        'model_type', ['vae', 'resnet_ae', 'unet_ae']
    )

    # Latent dimension
    config['model']['latent_dim'] = trial.suggest_int(
        'latent_dim', 32, 512, log=True
    )

    # Base filters
    config['model']['base_filters'] = trial.suggest_int(
        'base_filters', 16, 64, log=True
    )

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
    config['hyperparameters']['learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-5, 1e-2, log=True
    )

    # Batch size
    config['hyperparameters']['batch_size'] = trial.suggest_categorical(
        'batch_size', [4, 8, 16, 32]
    )

    # Weight decay
    config['hyperparameters']['weight_decay'] = trial.suggest_float(
        'weight_decay', 1e-6, 1e-3, log=True
    )

    # Beta for VAE
    if config['model']['type'] == 'vae':
        config['hyperparameters']['beta'] = trial.suggest_float(
            'beta', 0.1, 10.0, log=True
        )

    # Optimizer
    config['training']['optimizer'] = trial.suggest_categorical(
        'optimizer', ['adam', 'sgd', 'rmsprop']
    )

    return config


def objective(trial, base_config, device):
    """Objective function for Optuna optimization."""
    try:
        # Define hyperparameters
        config = define_model_params(trial, base_config)
        config = define_training_params(trial, config)

        # Save model_type for later use
        model_type = config['model']['type'].lower()

        # Create model
        print("Creating model...")
        model = get_model(config)
        print(f"Model created: {type(model).__name__}")
        model = model.to(device)

        # Create data loaders
        print("Creating data loaders...")
        data_loaders = create_data_loaders(config)

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
        patience = config['hyperparameters'].get('early_stopping_patience', 5)

        # Set up optimizer
        optimizer_name = config['training']['optimizer']
        optimizer_params = {
            'lr': config['hyperparameters']['learning_rate'],
            'weight_decay': config['hyperparameters']['weight_decay']
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
                optimizer, T_max=config['hyperparameters'].get('epochs', 100)
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

        # Training loop
        for epoch in range(10):  # Just do a few epochs for quick evaluation
            # Training
            model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(data_loaders['train']):
                try:
                    # Handle different dataset types
                    if isinstance(batch, dict):  # For DoseAEDataset
                        data = batch["image"].to(device)
                    else:  # For standard (input, target) dataset
                        data, _ = batch
                        data = data.to(device)

                    # Training step
                    optimizer.zero_grad()

                    # Forward pass
                    if 'vae' in model_type:
                        recon, mu, logvar = model(data)
                        loss_dict = model.get_losses(
                            data, recon, mu, logvar,
                            beta=config['hyperparameters'].get('beta', 1.0)
                        )
                        loss = loss_dict['total_loss']
                    else:
                        output = model(data)
                        if isinstance(output, tuple):
                            recon = output[0]
                            loss = torch.nn.functional.mse_loss(recon, data)
                        else:
                            loss = torch.nn.functional.mse_loss(output, data)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    if 'CUDA out of memory' in str(e):
                        # Try to free up memory
                        del data
                        if 'recon' in locals(): del recon
                        if 'mu' in locals(): del mu
                        if 'logvar' in locals(): del logvar
                        if 'output' in locals(): del output
                        torch.cuda.empty_cache()
                    return float('inf')

            # Quick validation
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loaders['val']):
                    if batch_idx >= 2:  # Just check a couple batches
                        break

                    try:
                        # Handle different dataset types
                        if isinstance(batch, dict):  # For DoseAEDataset
                            data = batch["image"].to(device)
                        else:  # For standard (input, target) dataset
                            data, _ = batch
                            data = data.to(device)

                        # Forward pass
                        if 'vae' in model_type:
                            recon, mu, logvar = model(data)
                            loss_dict = model.get_losses(
                                data, recon, mu, logvar,
                                beta=config['hyperparameters'].get('beta', 1.0)
                            )
                            loss = loss_dict['total_loss']
                        else:
                            output = model(data)
                            if isinstance(output, tuple):
                                recon = output[0]
                                loss = torch.nn.functional.mse_loss(recon, data)
                            else:
                                loss = torch.nn.functional.mse_loss(output, data)

                        val_loss += loss.item()
                        val_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        return float('inf')

            val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

            # Log to wandb
            if config['wandb']['use_wandb']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss / len(data_loaders['train']),
                    'val_loss': val_loss
                })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Clean up
        if config['wandb']['use_wandb']:
            wandb.finish()

        return best_val_loss

    except Exception as e:
        print(f"Error in objective function: {str(e)}")
        if config['wandb']['use_wandb'] and wandb.run is not None:
            wandb.finish()
        return float('inf')


