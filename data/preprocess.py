import torch
import numpy as np
import random


def normalize_tensor(tensor):
    """
    Normalize a 3D tensor by subtracting mean and dividing by std (per volume).

    Args:
        tensor (torch.Tensor): Input tensor to normalize

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = tensor.mean()
    std = tensor.std() if tensor.std() > 0 else 1.0
    return (tensor - mean) / std


def min_max_normalize(tensor):
    """
    Min-max normalization to [0, 1] range.

    Args:
        tensor (torch.Tensor): Input tensor to normalize

    Returns:
        torch.Tensor: Normalized tensor in [0, 1] range
    """
    min_val = tensor.min()
    max_val = tensor.max()

    if max_val == min_val:
        return torch.zeros_like(tensor)

    return (tensor - min_val) / (max_val - min_val)


def augment_data(tensor, rotate=True, flip=True, adjust_brightness=True):
    """
    Apply data augmentation to a 3D tensor.

    Args:
        tensor (torch.Tensor): Input tensor (C, D, H, W)
        rotate (bool): Whether to apply rotation
        flip (bool): Whether to apply flipping
        adjust_brightness (bool): Whether to adjust brightness

    Returns:
        torch.Tensor: Augmented tensor
    """
    # Convert to numpy for easier manipulation
    data = tensor.numpy()

    # Rotate
    if rotate and random.random() > 0.5:
        # Randomly rotate along one of the axes
        axis = random.choice([1, 2, 3])  # D, H, W axes (not channel axis)
        k = random.choice([1, 2, 3])  # Number of 90-degree rotations
        data = np.rot90(data, k=k, axes=(0, axis))

    # Flip
    if flip and random.random() > 0.5:
        # Randomly flip along one of the axes
        axis = random.choice([1, 2, 3])  # D, H, W axes (not channel axis)
        data = np.flip(data, axis=axis)

    # Adjust brightness
    if adjust_brightness and random.random() > 0.5:
        # Scale intensity by a factor between 0.8 and 1.2
        factor = 0.8 + random.random() * 0.4  # 0.8 to 1.2
        data = data * factor

    return torch.from_numpy(data)


def resize_volume(volume, target_shape):
    """
    Resize a 3D volume to the target shape.

    Args:
        volume (numpy.ndarray): Input 3D volume
        target_shape (tuple): Target shape (D, H, W)

    Returns:
        numpy.ndarray: Resized volume
    """
    from scipy.ndimage import zoom

    # Calculate scale factors
    factors = [t / s for t, s in zip(target_shape, volume.shape)]

    # Apply zoom
    resized = zoom(volume, factors, order=1, mode='nearest')

    return resized


def crop_center(volume, target_shape):
    """
    Crop the center of a 3D volume to match the target shape.

    Args:
        volume (numpy.ndarray): Input 3D volume
        target_shape (tuple): Target shape (D, H, W)

    Returns:
        numpy.ndarray: Center-cropped volume
    """
    d, h, w = volume.shape
    d_t, h_t, w_t = target_shape

    d_start = max(0, (d - d_t) // 2)
    h_start = max(0, (h - h_t) // 2)
    w_start = max(0, (w - w_t) // 2)

    d_end = min(d, d_start + d_t)
    h_end = min(h, h_start + h_t)
    w_end = min(w, w_start + w_t)

    return volume[d_start:d_end, h_start:h_end, w_start:w_end]


def pad_volume(volume, target_shape):
    """
    Pad a 3D volume to match the target shape.

    Args:
        volume (numpy.ndarray): Input 3D volume
        target_shape (tuple): Target shape (D, H, W)

    Returns:
        numpy.ndarray: Padded volume
    """
    d, h, w = volume.shape
    d_t, h_t, w_t = target_shape

    # Calculate padding for each dimension
    pad_d = max(0, d_t - d)
    pad_h = max(0, h_t - h)
    pad_w = max(0, w_t - w)

    # Calculate padding before and after for each dimension
    pad_d_before = pad_d // 2
    pad_d_after = pad_d - pad_d_before

    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before

    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    # Apply padding
    padded = np.pad(
        volume,
        ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant',
        constant_values=0
    )

    return padded


def preprocess_dose_distribution(data, config):
    """
    Preprocess dose distribution data according to the configuration.

    Args:
        data (numpy.ndarray): Input 3D dose distribution data
        config (dict): Configuration dictionary

    Returns:
        torch.Tensor: Preprocessed data as a tensor
    """
    # Resize to target input size if needed
    target_size = tuple(config['dataset']['input_size'])
    if data.shape != target_size:
        data = resize_volume(data, target_size)

    # Convert to tensor
    tensor = torch.from_numpy(data).float()

    # Add channel dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Normalize if specified
    if config['preprocessing']['normalize']:
        tensor = normalize_tensor(tensor)

    return tensor