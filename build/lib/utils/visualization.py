import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect
from torch import nn

from ..models import get_model


def find_high_dose_slice(dose_volume):
    """
    Find the slice with the highest dose area.

    Args:
        dose_volume: Dose tensor of any dimensionality

    Returns:
        Index of the slice with highest dose
    """
    import torch

    # Handle different tensor types
    if isinstance(dose_volume, np.ndarray):
        dose_volume = torch.from_numpy(dose_volume)

    # Handle different dimensions
    if dose_volume.dim() == 5:  # [B, C, D, H, W]
        dose_volume = dose_volume[0, 0]
    elif dose_volume.dim() == 4:  # [B, D, H, W] or [C, D, H, W]
        if dose_volume.shape[0] == 1:  # [C, D, H, W] with C=1
            dose_volume = dose_volume[0]
        else:  # [B, D, H, W]
            dose_volume = dose_volume[0]
    elif dose_volume.dim() == 3:  # [D, H, W]
        pass
    else:
        # 2D data, return 0
        return 0

    # Sum dose in each slice
    slice_doses = torch.sum(dose_volume, dim=(1, 2))

    # Find slice with maximum dose
    max_slice_idx = torch.argmax(slice_doses).item()

    return max_slice_idx
def load_trained_model(model_path, model_type, config=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of the model
        config (dict, optional): Configuration dictionary

    Returns:
        nn.Module: Loaded model in evaluation mode
    """
    if config is None:
        # Default configuration if not provided
        if model_type.lower() == 'vae':
            from ..models.vae_2d import VAE
            model = VAE()
        elif model_type.lower() == 'mlp_autoencoder':
            from ..models.vae_2d import MLPAutoencoder2D
            model = MLPAutoencoder2D()
        elif model_type.lower() == 'conv_autoencoder':
            from ..models.vae_2d import ConvAutoencoder2D
            model = ConvAutoencoder2D()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Create model using the factory function
        model = get_model(config)

    # Load the state dict from checkpoint
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode

    return model


def process_2d_slice(model, input_slice, normalize=True, percentile=None):
    """
    Process a 2D slice through a trained model.

    Args:
        model (nn.Module): Trained model
        input_slice (numpy.ndarray): 2D input slice
        normalize (bool): Whether to normalize the input
        percentile (float, optional): Percentile value for normalization

    Returns:
        numpy.ndarray: Reconstructed output slice
    """
    if input_slice.ndim != 2:
        raise ValueError("Input slice should be a 2D array.")

    # Normalize if requested
    if normalize:
        if percentile is None:
            # Calculate the 95th percentile if not provided
            percentile = np.percentile(input_slice, 95)

        # Normalize to [-1, 1] range for tanh activation
        input_slice_normalized = input_slice / percentile  # Normalize to [0, 1]
        input_slice_normalized = (input_slice_normalized - 0.5) * 2.0  # Normalize to [-1, 1]
    else:
        input_slice_normalized = input_slice

    # Convert to tensor and add batch and channel dimensions
    input_tensor = torch.from_numpy(input_slice_normalized).float().unsqueeze(0).unsqueeze(0)

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Pass through the model
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, "forward") and "mu" in inspect.signature(model.forward).parameters:
            output_tensor, _, _ = model(input_tensor)
        else:
            output_tensor = model(input_tensor)

    # Convert to numpy array
    output_slice = output_tensor.squeeze().cpu().numpy()

    # Denormalize if input was normalized
    if normalize:
        output_slice = (output_slice + 1) / 2.0 * percentile

    return output_slice


def process_3d_volume(model, input_volume, normalize=True, percentile=None):
    """
    Process a 3D volume through a trained model.

    Args:
        model (nn.Module): Trained model
        input_volume (numpy.ndarray): 3D input volume
        normalize (bool): Whether to normalize the input
        percentile (float, optional): Percentile value for normalization

    Returns:
        numpy.ndarray: Reconstructed output volume
    """
    if input_volume.ndim != 3:
        raise ValueError("Input volume should be a 3D array.")

    # Normalize if requested
    if normalize:
        if percentile is None:
            # Calculate the 95th percentile if not provided
            percentile = np.percentile(input_volume, 95)

        # Normalize to [-1, 1] range for tanh activation
        input_normalized = input_volume / percentile  # Normalize to [0, 1]
        input_normalized = (input_normalized - 0.5) * 2.0  # Normalize to [-1, 1]
    else:
        input_normalized = input_volume

    # Convert to tensor and add batch and channel dimensions
    input_tensor = torch.from_numpy(input_normalized).float().unsqueeze(0).unsqueeze(0)

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Pass through the model
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, "forward") and "mu" in inspect.signature(model.forward).parameters:
            output_tensor, _, _ = model(input_tensor)
        elif hasattr(model, "forward") and len(inspect.signature(model.forward).parameters) > 1:
            output_tensor, _ = model(input_tensor)
        else:
            output_tensor = model(input_tensor)

    # Convert to numpy array
    output_volume = output_tensor.squeeze().cpu().numpy()

    # Denormalize if input was normalized
    if normalize:
        output_volume = (output_volume + 1) / 2.0 * percentile

    return output_volume


def visualize_2d_slices(input_slice, output_slice, title=None, save_path=None):
    """
    Visualize original and reconstructed 2D slices.

    Args:
        input_slice (numpy.ndarray): Input 2D slice
        output_slice (numpy.ndarray): Reconstructed 2D slice
        title (str, optional): Title for the figure
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Input Slice")
    plt.imshow(input_slice, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Slice")
    plt.imshow(output_slice, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_3d_volume(input_volume, output_volume, slice_dim=0, slice_indices=None, title=None, save_path=None):
    """
    Visualize original and reconstructed slices from a 3D volume.

    Args:
        input_volume (numpy.ndarray): Input 3D volume
        output_volume (numpy.ndarray): Reconstructed 3D volume
        slice_dim (int): Dimension along which to slice (0, 1, or 2)
        slice_indices (list, optional): List of indices to visualize
        title (str, optional): Title for the figure
        save_path (str, optional): Path to save the figure
    """
    if slice_indices is None:
        # Default to showing central slice
        dim_size = input_volume.shape[slice_dim]
        slice_indices = [dim_size // 2]

    num_slices = len(slice_indices)
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))

    if num_slices == 1:
        axes = axes.reshape(2, 1)

    for i, slice_idx in enumerate(slice_indices):
        # Get slices
        if slice_dim == 0:
            input_slice = input_volume[slice_idx, :, :]
            output_slice = output_volume[slice_idx, :, :]
            slice_name = f"Z={slice_idx}"
        elif slice_dim == 1:
            input_slice = input_volume[:, slice_idx, :]
            output_slice = output_volume[:, slice_idx, :]
            slice_name = f"Y={slice_idx}"
        else:
            input_slice = input_volume[:, :, slice_idx]
            output_slice = output_volume[:, :, slice_idx]
            slice_name = f"X={slice_idx}"

        # Plot input slice
        im1 = axes[0, i].imshow(input_slice, cmap="viridis")
        axes[0, i].set_title(f"Input Slice {slice_name}")
        axes[0, i].axis("off")
        plt.colorbar(im1, ax=axes[0, i])

        # Plot output slice
        im2 = axes[1, i].imshow(output_slice, cmap="viridis")
        axes[1, i].set_title(f"Reconstructed Slice {slice_name}")
        axes[1, i].axis("off")
        plt.colorbar(im2, ax=axes[1, i])

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def calculate_metrics(input_data, output_data):
    """
    Calculate reconstruction metrics.

    Args:
        input_data (numpy.ndarray): Original input data
        output_data (numpy.ndarray): Reconstructed output data

    Returns:
        dict: Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((input_data - output_data) ** 2)

    # Peak Signal-to-Noise Ratio
    data_range = input_data.max() - input_data.min()
    psnr = 20 * np.log10(data_range / np.sqrt(mse)) if mse > 0 else float('inf')

    # Structural Similarity Index (placeholder - would need scikit-image for actual implementation)
    # from skimage.metrics import structural_similarity as ssim
    # ssim_value = ssim(input_data, output_data, data_range=data_range)

    # Mean Absolute Error
    mae = np.mean(np.abs(input_data - output_data))

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Normalized Root Mean Squared Error
    nrmse = rmse / (input_data.max() - input_data.min()) if (input_data.max() - input_data.min()) > 0 else float('inf')

    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
    }


def process_and_visualize(model_path, data_path, model_type, slice_idx=24, config=None, normalize=True, save_dir=None):
    """
    Load a model and data, process the data through the model, and visualize the results.

    Args:
        model_path (str): Path to the model checkpoint
        data_path (str): Path to the input data (.npy file)
        model_type (str): Type of the model
        slice_idx (int, optional): Index of the slice to visualize
        config (dict, optional): Model configuration
        normalize (bool): Whether to normalize the input data
        save_dir (str, optional): Directory to save results
    """
    # Load the model
    model = load_trained_model(model_path, model_type, config)

    # Load the data
    data = np.load(data_path)

    # Handle different data dimensions
    if data.ndim == 3:
        # 3D data with multiple slices
        input_slice = data[slice_idx]
        is_3d = False
    elif data.ndim == 4:
        # 4D data (likely a volume)
        input_volume = data
        is_3d = True
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")

    # Calculate percentile for normalization if needed
    percentile = np.percentile(data, 95) if normalize else None

    # Process the data
    if is_3d:
        output_volume = process_3d_volume(model, input_volume, normalize, percentile)

        # Visualize results
        save_path = os.path.join(save_dir, "volume_reconstruction.png") if save_dir else None
        visualize_3d_volume(
            input_volume, output_volume,
            slice_indices=[input_volume.shape[0] // 4, input_volume.shape[0] // 2, 3 * input_volume.shape[0] // 4],
            title=f"3D Volume Reconstruction - {model_type}",
            save_path=save_path
        )

        # Calculate metrics
        metrics = calculate_metrics(input_volume, output_volume)
    else:
        output_slice = process_2d_slice(model, input_slice, normalize, percentile)

        # Visualize results
        save_path = os.path.join(save_dir, "slice_reconstruction.png") if save_dir else None
        visualize_2d_slices(
            input_slice, output_slice,
            title=f"2D Slice Reconstruction - {model_type}",
            save_path=save_path
        )

        # Calculate metrics
        metrics = calculate_metrics(input_slice, output_slice)

    # Print metrics
    print("Reconstruction Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper()}: {metric_value:.6f}")

    return metrics