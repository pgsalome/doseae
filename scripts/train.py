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
from data.dataset import create_data_loaders, create_data_loaders_from_numpy, DoseAEDataset
from utils.optimization import objective as optuna_objective

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


def log_images(model, val_loader, epoch, device, config, interval=10):
    """
    Log input and reconstructed images to wandb.

    Args:
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation data loader
        epoch (int): Current epoch
        device (torch.device): Device
        config (dict): Configuration
        interval (int): Interval for logging images
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


def calculate_iqr(values):
    """
    Calculate interquartile range (IQR) for a set of values.

    Args:
        values (numpy.ndarray): Values to calculate IQR for

    Returns:
        tuple: First quartile (Q1), third quartile (Q3), and IQR
    """
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR


def define_bins(zc_values):
    """
    Define bin edges based on IQR.

    Args:
        zc_values (numpy.ndarray): Values to define bins for

    Returns:
        numpy.ndarray: Array of bin edges
    """
    Q1, Q3, IQR = calculate_iqr(zc_values)
    bin_edges = sorted([
        zc_values.min(),
        Q1 - 1.5 * IQR,
        Q1,
        Q3,
        Q3 + 1.5 * IQR,
        zc_values.max()
    ])
    return bin_edges


def assign_bins(zc_values, bin_edges):
    """
    Assign values to bins.

    Args:
        zc_values (numpy.ndarray): Values to assign to bins
        bin_edges (numpy.ndarray): Bin edges

    Returns:
        numpy.ndarray: Bin assignments
    """
    bins = np.digitize(zc_values, bin_edges, right=True)
    return bins


def custom_stratified_split_by_bins(dataset, test_size=0.2, random_state=42):
    """
    Perform a stratified split of a dataset based on the non-zero count.

    Args:
        dataset (Dataset): Dataset to split
        test_size (float): Fraction of dataset to use for validation
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: Train dataset and validation dataset
    """
    if not hasattr(dataset, 'zc_values'):
        raise ValueError("Dataset must have 'zc_values' attribute for stratified split")

    zc_values = np.array(dataset.zc_values)
    bin_edges = define_bins(zc_values)
    bins = assign_bins(zc_values, bin_edges)

    train_indices = []
    val_indices = []

    for bin_value in np.unique(bins):
        indices = np.where(bins == bin_value)[0]
        train_size = int((1 - test_size) * len(indices))
        train_idx, val_idx = train_test_split(
            indices, test_size=len(indices) - train_size, random_state=random_state
        )

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def adjust_learning_rate(scheduler, optimizer, avg_val_loss, epoch):
    """
    Adjust learning rate using scheduler and log the change.

    Args:
        scheduler: Learning rate scheduler
        optimizer: Optimizer
        avg_val_loss: Average validation loss
        epoch: Current epoch
    """
    if scheduler is None:
        return

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch [{epoch + 1}], Learning Rate before scheduler step: {current_lr:.10f}")

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    else:
        scheduler.step()

    new_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch [{epoch + 1}], Learning Rate after scheduler step: {new_lr:.10f}")


def visualize_reconstructions(model, data_loader, device, config, epoch, output_dir):
    """
    Visualize original and reconstructed images.

    Args:
        model (nn.Module): Trained model
        data_loader (DataLoader): Data loader
        device (torch.device): Device
        config (dict): Configuration
        epoch (int): Current epoch
        output_dir (str): Output directory
    """
    model.eval()

    # Get a batch of data
    batch = next(iter(data_loader))

    # Handle different dataset types
    if isinstance(batch, dict):  # For DoseAEDataset
        data = batch["image"].to(device)
    else:  # For standard (input, target) dataset
        data, _ = batch
        data = data.to(device)

    # Add channel dimension if needed
    if data.ndim == 3:  # [B, H, W]
        data = data.unsqueeze(1)  # [B, 1, H, W]

    with torch.no_grad():
        # Forward pass - handle different model types
        outputs = model(data)

        # Extract reconstructed output
        if isinstance(outputs, tuple):
            recon = outputs[0]  # For models that return multiple values (like VAE)
        else:
            recon = outputs

    # Create visualization
    is_2d = config['model'].get('is_2d', False)

    if is_2d:
        # 2D visualization
        fig, axes = plt.subplots(2, min(4, data.size(0)), figsize=(12, 6))

        for i in range(min(4, data.size(0))):
            # Original
            axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[0, i].set_title(f"Original {i + 1}")
            axes[0, i].axis('off')

            # Reconstructed
            axes[1, i].imshow(recon[i, 0].cpu().numpy(), cmap='viridis')
            axes[1, i].set_title(f"Reconstructed {i + 1}")
            axes[1, i].axis('off')
    else:
        # 3D visualization - central slices
        fig, axes = plt.subplots(3, min(4, data.size(0)), figsize=(12, 9))

        for i in range(min(4, data.size(0))):
            # Get central slices
            d, h, w = data.shape[2], data.shape[3], data.shape[4]

            # Original slices
            orig_slice_d = data[i, 0, d // 2, :, :].cpu().numpy()
            orig_slice_h = data[i, 0, :, h // 2, :].cpu().numpy()
            orig_slice_w = data[i, 0, :, :, w // 2].cpu().numpy()

            # Reconstructed slices
            recon_slice_d = recon[i, 0, d // 2, :, :].cpu().numpy()
            recon_slice_h = recon[i, 0, :, h // 2, :].cpu().numpy()
            recon_slice_w = recon[i, 0, :, :, w // 2].cpu().numpy()

            # Plot
            axes[0, i].imshow(orig_slice_d, cmap='viridis')
            axes[0, i].set_title(f"Original D-slice {i + 1}")
            axes[0, i].axis('off')

            axes[1, i].imshow(orig_slice_h, cmap='viridis')
            axes[1, i].set_title(f"Original H-slice {i + 1}")
            axes[1, i].axis('off')

            axes[2, i].imshow(orig_slice_w, cmap='viridis')
            axes[2, i].set_title(f"Original W-slice {i + 1}")
            axes[2, i].axis('off')

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reconstruction_epoch_{epoch}.png'))
    plt.close()

    # Log to wandb if enabled
    if config['wandb']['use_wandb']:
        wandb.log({f"reconstructions_epoch_{epoch}": wandb.Image(
            os.path.join(output_dir, f'reconstruction_epoch_{epoch}.png'))})


def train_epoch(model, data_loader, optimizer, device, config, accumulation_steps=1):
    """
    Train for one epoch.

    Args:
        model (nn.Module): Model to train
        data_loader (DataLoader): Data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device
        config (dict): Configuration
        accumulation_steps (int): Number of steps to accumulate gradients

    Returns:
        float: Average training loss
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
        check_for_nan(re