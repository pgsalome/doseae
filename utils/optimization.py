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
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.Trial): Optuna trial object
        base_config (dict): Base configuration
        device (torch.device): Device to use

    Returns:
        float: Validation loss to minimize
    """
    # Define hyperparameters to search over
    config = define_model_params(trial, base_config)
    config = define_training_params(trial, config)

    # Create model
    model = get_model(config).to(device)

    # Create data loaders
    data_loaders = create_data_loaders(config)

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

    # Set up scheduler
    scheduler = None
    if config['training']['scheduler'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif config['training']['scheduler'] == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['hyperparameters']['epochs']
        )

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config['hyperparameters']['early_stopping_patience']

    # Set up wandb if enabled
    if config['wandb']['use_wandb']:
        trial_name = f"trial_{trial.number}"
        wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            name=trial_name,
            config=config,
            reinit=True
        )

    # Train for max epochs
    for epoch in range(config['hyperparameters']['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(data_loaders['train']):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass - handle different model types
            if config['model']['type'] == 'vae':
                recon, mu, logvar = model(data)
                loss_dict = model.get_losses(
                    data, recon, mu, logvar,
                    beta=config['hyperparameters']['beta']
                )
            elif config['model']['type'] == 'unet_ae':
                recon, _ = model(data)
                loss_dict = model.get_losses(data, recon)
            else:
                recon = model(data)
                loss_dict = model.get_losses(data, recon)

            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(data_loaders['train'])

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loaders['val']):
                data = data.to(device)

                # Forward pass - handle different model types
                if config['model']['type'] == 'vae':
                    recon, mu, logvar = model(data)
                    loss_dict = model.get_losses(
                        data, recon, mu, logvar,
                        beta=config['hyperparameters']['beta']
                    )
                elif config['model']['type'] == 'unet_ae':
                    recon, _ = model(data)
                    loss_dict = model.get_losses(data, recon)
                else:
                    recon = model(data)
                    loss_dict = model.get_losses(data, recon)

                val_loss += loss_dict['total_loss'].item()

        val_loss /= len(data_loaders['val'])

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log to wandb
        if config['wandb']['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

        # Report intermediate objective value
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Clean up wandb
    if config['wandb']['use_wandb']:
        wandb.finish()

    return best_val_loss


