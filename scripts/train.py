import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
import optuna
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

from models import get_model
from data.dataset import create_data_loaders
from utils.optimization import objective as optuna_objective
from utils.clinical_metrics import ClinicalMetricsCalculator
from utils.clinical_loss import ClinicalLoss
from utils.visualization import find_high_dose_slice

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, save_path):
    """
    Save model to disk.

    Args:
        model (nn.Module): Model to save
        save_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def create_optimizer(model, config):
    """
    Create optimizer based on configuration.

    Args:
        model (nn.Module): Model to optimize
        config (dict): Configuration dictionary

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['hyperparameters']['learning_rate']
    weight_decay = config['hyperparameters']['weight_decay']

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        config (dict): Configuration dictionary

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    scheduler_name = config['training']['scheduler'].lower()

    if scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4
        )
    elif scheduler_name == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['hyperparameters']['epochs'], eta_min=1e-6
        )
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def check_for_nan(tensor, name=""):
    """
    Check if tensor contains NaN values.

    Args:
        tensor (torch.Tensor): Tensor to check
        name (str): Name for error message

    Raises:
        ValueError: If tensor contains NaN values
    """
    if torch.isnan(tensor).any():
        logger.error(f"NaN detected in {name}")
        raise ValueError(f"NaN detected in {name}")


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, normalize_outputs=True):
    """
    Calculate VAE loss (reconstruction + KL divergence).

    Args:
        recon_x (torch.Tensor): Reconstructed output
        x (torch.Tensor): Original input
        mu (torch.Tensor): Mean of the latent distribution
        logvar (torch.Tensor): Log variance of the latent distribution
        beta (float): Weight for the KL divergence term
        normalize_outputs (bool): Whether to normalize the outputs to [0, 1]

    Returns:
        torch.Tensor: Total loss
    """
    if normalize_outputs:
        # Normalize to [0, 1] if outputs are in [-1, 1]
        recon_x = (recon_x + 1) / 2
        x = (x + 1) / 2

    # Binary cross entropy loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD


def log_high_dose_slice(model, val_loader, epoch, device, config):
    """
    Log the slice with highest dose area to wandb.
    """
    if not config.get('logging', {}).get('log_high_dose_slice', True):
        return

    if epoch % config.get('logging', {}).get('log_frequency', 1) != 0:
        return

    model.eval()

    # Get a batch from validation
    batch = next(iter(val_loader))

    # Get data
    if isinstance(batch, dict):
        data = batch["image"].to(device)
    else:
        data, _ = batch
        data = data.to(device)

    # Add channel dimension if needed
    if data.ndim == 3:  # [B, H, W]
        data = data.unsqueeze(1)  # [B, 1, H, W]

    # Get reconstruction
    with torch.no_grad():
        if config['model']['type'] == 'vae':
            recon, _, _ = model(data)
        else:
            recon = model(data)
            if isinstance(recon, tuple):
                recon = recon[0]

    # Find high dose slice
    high_dose_idx = find_high_dose_slice(data)

    # Extract slices
    if data.dim() == 5:  # 3D data
        original_slice = data[0, 0, high_dose_idx].cpu().numpy()
        recon_slice = recon[0, 0, high_dose_idx].cpu().numpy()
    else:  # 2D data
        original_slice = data[0, 0].cpu().numpy()
        recon_slice = recon[0, 0].cpu().numpy()

    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    im1 = axes[0].imshow(original_slice, cmap='jet')
    axes[0].set_title('Original')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Reconstructed
    im2 = axes[1].imshow(recon_slice, cmap='jet')
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference
    diff = np.abs(original_slice - recon_slice)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'High Dose Slice - Epoch {epoch}')
    plt.tight_layout()

    # Log to wandb
    if config['wandb']['use_wandb']:
        wandb.log({
            'high_dose_slice/comparison': wandb.Image(fig),
            'high_dose_slice/original_max': original_slice.max(),
            'high_dose_slice/recon_max': recon_slice.max(),
            'high_dose_slice/max_diff': diff.max(),
            'high_dose_slice/mean_diff': diff.mean(),
            'high_dose_slice/slice_index': high_dose_idx
        })

    plt.close(fig)


def log_images(model, val_loader, epoch, device, config, interval=10):
    """
    Log input and reconstructed images to wandb.
    """
    if (epoch + 1) % interval != 0:
        return

    model.eval()
    with torch.no_grad():
        # Get a batch from the validation loader
        data_iter = iter(val_loader)
        batch = next(data_iter)

        # Handle different dataset types
        if isinstance(batch, dict):  # For DoseAEDataset
            img = batch["image"].to(device)
        else:  # For standard (input, target) dataset
            img, _ = batch
            img = img.to(device)

        # Add channel dimension if needed
        if img.ndim == 3:  # [B, H, W]
            img = img.unsqueeze(1)  # [B, 1, H, W]

        # Forward pass - handle different model types
        outputs = model(img)

        # Extract reconstructed output
        if isinstance(outputs, tuple):
            recon_batch = outputs[0]  # For models that return multiple values (like VAE)
        else:
            recon_batch = outputs

        # Find a non-zero slice to visualize
        for b_idx in range(min(img.size(0), 4)):  # Check first 4 batches at most
            for c_idx in range(img.size(1)):  # Check each channel
                if torch.sum(img[b_idx, c_idx]) > 0:
                    # Extract images
                    input_image = img[b_idx, c_idx].cpu().numpy()
                    reconstructed_image = recon_batch[b_idx, c_idx].cpu().numpy()

                    # Create figure with side-by-side comparison
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    # Plot input image
                    im1 = axes[0].imshow(input_image, cmap='viridis')
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")
                    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                    # Plot reconstructed image
                    im2 = axes[1].imshow(reconstructed_image, cmap='viridis')
                    axes[1].set_title("Reconstructed Image")
                    axes[1].axis("off")
                    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                    # Save and log to wandb
                    plt.tight_layout()
                    plt.savefig(f"reconstruction_epoch_{epoch + 1}.png")
                    plt.close()

                    # Log to wandb if enabled
                    if config['wandb']['use_wandb']:
                        wandb.log(
                            {f"reconstruction_epoch_{epoch + 1}": wandb.Image(f"reconstruction_epoch_{epoch + 1}.png")})

                    # Only log the first non-zero image found
                    return


def train_epoch_with_clinical_loss(model, data_loader, optimizer, loss_fn, device, config, epoch):
    """
    Training epoch with clinical loss function.
    """
    model.train()
    running_losses = {}

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):
        # Get data
        if isinstance(batch, dict):
            data = batch["image"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)
        else:
            data, _ = batch
            data = data.to(device)
            mask = None

        # Add channel dimension if needed
        if data.ndim == 3:  # [B, H, W]
            data = data.unsqueeze(1)  # [B, 1, H, W]

        # Check for NaN in input
        check_for_nan(data, "input data")

        # Forward pass
        if config['model']['type'] == 'vae':
            recon, mu, logvar = model(data)

            # VAE losses
            if isinstance(loss_fn, ClinicalLoss):
                losses = loss_fn(recon, data, mask)

                # Add KL divergence for VAE
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / data.size(0)

                beta = config['hyperparameters'].get('beta', 1.0)
                losses['kl'] = kl_loss
                losses['kl_weighted'] = beta * kl_loss
                losses['total'] = losses['total'] + beta * kl_loss
            else:
                # Standard VAE loss
                loss = vae_loss_function(recon, data, mu, logvar,
                                         beta=config['hyperparameters'].get('beta', 1.0),
                                         normalize_outputs=config['preprocessing'].get('use_tanh_output', True))
                losses = {'total': loss}
        else:
            # Non-VAE models
            recon = model(data)
            if isinstance(recon, tuple):
                recon = recon[0]

            if isinstance(loss_fn, ClinicalLoss):
                losses = loss_fn(recon, data, mask)
            else:
                # Standard MSE loss
                loss = loss_fn(recon, data)
                losses = {'total': loss}

        # Check for NaN in reconstruction and loss
        check_for_nan(recon, "reconstructed output")
        check_for_nan(losses['total'], "loss")

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        if config['training'].get('grad_clip'):
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        optimizer.step()

        # Update running losses
        for loss_name, loss_value in losses.items():
            if loss_name not in running_losses:
                running_losses[loss_name] = 0.0
            running_losses[loss_name] += loss_value.item()

    # Average losses
    avg_losses = {k: v / len(data_loader) for k, v in running_losses.items()}

    return avg_losses


def train_epoch(model, data_loader, optimizer, device, config, accumulation_steps=1):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    grad_clip = config['training'].get('grad_clip', None)
    model_type = config['model']['type']

    # Reset gradients
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):
        # Handle different dataset types
        if isinstance(batch, dict):  # For DoseAEDataset
            data = batch["image"].to(device)
        else:  # For standard (input, target) dataset
            data, _ = batch
            data = data.to(device)

        # Add channel dimension if needed
        if data.ndim == 3:  # [B, H, W]
            data = data.unsqueeze(1)  # [B, 1, H, W]

        # Check for NaN in input
        check_for_nan(data, "input data")

        # Forward pass
        outputs = model(data)

        # Calculate loss based on model type
        if model_type == 'vae':
            recon_batch, mu, logvar = outputs
            loss = vae_loss_function(
                recon_batch,
                data,
                mu,
                logvar,
                beta=config['hyperparameters'].get('beta', 1.0),
                normalize_outputs=config['preprocessing'].get('use_tanh_output', True)
            )
        else:
            # Handle non-VAE models
            if isinstance(outputs, tuple):
                recon_batch = outputs[0]
            else:
                recon_batch = outputs

            # Calculate loss, optionally only on non-zero regions
            if config.get('training', {}).get('non_zero_loss', False):
                # Mask for non-zero regions
                mask = data != 0
                if mask.sum() > 0:  # Make sure there are non-zero elements
                    loss = nn.functional.mse_loss(recon_batch[mask], data[mask])
                else:
                    loss = nn.functional.mse_loss(recon_batch, data)
            else:
                # Regular MSE loss on the entire input
                loss = nn.functional.mse_loss(recon_batch, data)

        # Check for NaN in output and loss
        check_for_nan(recon_batch, "reconstructed output")
        check_for_nan(loss, "loss")

        # Backward pass with gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Apply gradient clipping if specified
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update weights and reset gradients
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    return running_loss / len(data_loader)


def validate(model, data_loader, device, config):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    model_type = config['model']['type']

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # Handle different dataset types
            if isinstance(batch, dict):  # For DoseAEDataset
                data = batch["image"].to(device)
            else:  # For standard (input, target) dataset
                data, _ = batch
                data = data.to(device)

            # Add channel dimension if needed
            if data.ndim == 3:  # [B, H, W]
                data = data.unsqueeze(1)  # [B, 1, H, W]

            # Forward pass
            outputs = model(data)

            # Calculate loss based on model type
            if model_type == 'vae':
                recon_batch, mu, logvar = outputs
                loss = vae_loss_function(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    beta=config['hyperparameters'].get('beta', 1.0),
                    normalize_outputs=config['preprocessing'].get('use_tanh_output', True)
                )
            else:
                # Handle non-VAE models
                if isinstance(outputs, tuple):
                    recon_batch = outputs[0]
                else:
                    recon_batch = outputs

                # Calculate loss, optionally only on non-zero regions
                if config.get('training', {}).get('non_zero_loss', False):
                    # Mask for non-zero regions
                    mask = data != 0
                    if mask.sum() > 0:  # Make sure there are non-zero elements
                        loss = nn.functional.mse_loss(recon_batch[mask], data[mask])
                    else:
                        loss = nn.functional.mse_loss(recon_batch, data)
                else:
                    # Regular MSE loss on the entire input
                    loss = nn.functional.mse_loss(recon_batch, data)

            running_loss += loss.item()

    return running_loss / len(data_loader)


def validate_with_clinical_loss(model, data_loader, loss_fn, device, config):
    """
    Validation with clinical loss calculation.
    """
    model.eval()
    running_losses = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # Get data
            if isinstance(batch, dict):
                data = batch["image"].to(device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)
            else:
                data, _ = batch
                data = data.to(device)
                mask = None

            # Add channel dimension if needed
            if data.ndim == 3:  # [B, H, W]
                data = data.unsqueeze(1)  # [B, 1, H, W]

            # Forward pass
            if config['model']['type'] == 'vae':
                recon, mu, logvar = model(data)

                # Calculate losses
                if isinstance(loss_fn, ClinicalLoss):
                    losses = loss_fn(recon, data, mask)

                    # Add KL divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / data.size(0)

                    beta = config['hyperparameters'].get('beta', 1.0)
                    losses['kl'] = kl_loss
                    losses['kl_weighted'] = beta * kl_loss
                    losses['total'] = losses['total'] + beta * kl_loss
                else:
                    loss = vae_loss_function(recon, data, mu, logvar,
                                             beta=config['hyperparameters'].get('beta', 1.0),
                                             normalize_outputs=config['preprocessing'].get('use_tanh_output', True))
                    losses = {'total': loss}
            else:
                recon = model(data)
                if isinstance(recon, tuple):
                    recon = recon[0]

                if isinstance(loss_fn, ClinicalLoss):
                    losses = loss_fn(recon, data, mask)
                else:
                    loss = loss_fn(recon, data)
                    losses = {'total': loss}

            # Update running losses
            for loss_name, loss_value in losses.items():
                if loss_name not in running_losses:
                    running_losses[loss_name] = 0.0
                running_losses[loss_name] += loss_value.item()

    # Average losses
    avg_losses = {k: v / len(data_loader) for k, v in running_losses.items()}

    return avg_losses


def train(config):
    """
    Train the model with enhanced clinical metrics and logging.
    """
    # Set device
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    # Create data loaders
    data_loaders = create_data_loaders(config)

    # Create model
    model = get_model(config).to(device)
    print(f"Model: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialize loss function
    if config.get('loss_function', {}).get('type', 'mse') != 'mse':
        loss_fn = ClinicalLoss(config)
    else:
        loss_fn = nn.MSELoss()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Initialize wandb if enabled
    if config['wandb']['use_wandb']:
        wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            name=config.get('wandb', {}).get('name', None),
            config=config,
            dir=os.path.join(config['output']['results_dir'], 'wandb')
        )
        wandb.watch(model)

    # Create output directories
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)

    # Training variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = config['hyperparameters']['early_stopping_patience']

    # Training loop
    for epoch in range(config['hyperparameters']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['hyperparameters']['epochs']}")

        # Train with clinical loss if configured
        if isinstance(loss_fn, ClinicalLoss):
            train_losses = train_epoch_with_clinical_loss(
                model, data_loaders['train'], optimizer, loss_fn, device, config, epoch
            )
        else:
            # Traditional training
            train_loss = train_epoch(model, data_loaders['train'], optimizer, device, config)
            train_losses = {'total': train_loss}

        # Validate with clinical loss if configured
        if isinstance(loss_fn, ClinicalLoss):
            val_losses = validate_with_clinical_loss(
                model, data_loaders['val'], loss_fn, device, config
            )
        else:
            # Traditional validation
            val_loss = validate(model, data_loaders['val'], device, config)
            val_losses = {'total': val_loss}

        # Use total loss for tracking
        train_loss = train_losses.get('total', 0.0)
        val_loss = val_losses.get('total', 0.0)

        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Log all losses to wandb
        if config['wandb']['use_wandb']:
            log_dict = {
                'epoch': epoch + 1,
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            # Add all training losses
            for loss_name, loss_value in train_losses.items():
                log_dict[f'train/{loss_name}'] = loss_value

            # Add all validation losses
            for loss_name, loss_value in val_losses.items():
                log_dict[f'val/{loss_name}'] = loss_value

            wandb.log(log_dict)

            # Log high dose slice
            log_high_dose_slice(model, data_loaders['val'], epoch + 1, device, config)

            # Log full reconstructions
            if config.get('logging', {}).get('log_reconstructions', True):
                if (epoch + 1) % config.get('logging', {}).get('reconstruction_frequency', 10) == 0:
                    log_images(model, data_loaders['val'], epoch + 1, device, config)

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            # Save best model
            model_type = config['model']['type']
            save_path = os.path.join(config['output']['model_dir'], f'{model_type}_best_model.pth')
            torch.save(model.state_dict(), save_path)

            print(f"Validation loss improved to {best_val_loss:.6f}, saving model")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Close wandb if enabled
    if config['wandb']['use_wandb']:
        wandb.finish()

    return model, best_val_loss


def hyperparameter_optimization(config):
    """
    Run hyperparameter optimization.
    """
    print("Starting hyperparameter optimization...")

    # Set device
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")

    # Create Optuna study
    sampler = None
    if config['optuna']['sampler'] == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=config['training']['seed'])
    elif config['optuna']['sampler'] == 'random':
        sampler = optuna.samplers.RandomSampler(seed=config['training']['seed'])

    pruner = None
    if config['optuna']['pruner'] == 'median':
        pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction=config['optuna']['direction'],
        sampler=sampler,
        pruner=pruner
    )

    # Run optimization
    study.optimize(
        lambda trial: optuna_objective(trial, config, device),
        n_trials=config['optuna']['n_trials'],
        timeout=config['optuna']['timeout']
    )

    # Print results
    print("Hyperparameter optimization finished!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    with open(os.path.join(config['output']['results_dir'], 'optuna_results.yaml'), 'w') as f:
        yaml.dump({
            'best_trial': study.best_trial.number,
            'best_value': float(study.best_trial.value),
            'best_params': study.best_trial.params
        }, f)

    return study.best_trial.params


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Train an autoencoder for dose distribution")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    parser.add_argument('--optimize', action='store_true', help="Run hyperparameter optimization")
    parser.add_argument('--data-file', type=str, help="Path to a specific .npy file to use for training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Update configuration with command-line arguments
    if args.data_file:
        config['dataset']['use_single_file'] = True
        config['dataset']['data_file'] = args.data_file

    # Run hyperparameter optimization if requested
    if args.optimize:
        best_params = hyperparameter_optimization(config)

        # Update config with best parameters
        for key, value in best_params.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value

    # Train model
    model = train(config)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()