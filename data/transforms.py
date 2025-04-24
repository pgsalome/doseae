"""
Transformations for dose distribution data.
"""

import torch
import numpy as np
import cv2
from torchvision import transforms


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image = sample["image"]
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a numpy array")

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        return {"image": image, "zc": sample.get("zc", 0)}


class NormalizeTo95PercentDose(object):
    """
    Normalize image to its 95th percentile dose value.
    This is important for dose distribution data where the maximum value
    might be an outlier.
    """

    def __call__(self, sample):
        image = sample["image"]

        # Calculate 95th percentile
        max_dose = np.percentile(image, 95)

        # Normalize only if max_dose is not zero
        if max_dose > 0:
            image = image / max_dose

        return {"image": image, "zc": sample.get("zc", 0)}


class ZscoreNormalization(object):
    """
    Normalize image using Z-score normalization (subtract mean, divide by std).
    """

    def __call__(self, sample):
        if sample is not None:
            image = sample["image"]

            # Calculate mean and std
            mean = image.mean()
            std = image.std()

            # Normalize if std is not zero
            if std > 0:
                image = (image - mean) / std
            else:
                image = image - mean

            return {"image": image, "zc": sample.get("zc", 0)}
        else:
            return None


class MinMaxNormalization(object):
    """
    Normalize image to range [0, 1] using min-max normalization.
    """

    def __call__(self, sample):
        image = sample["image"]

        # Get min and max values
        min_val = image.min()
        max_val = image.max()

        # Normalize if range is not zero
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)

        return {"image": image, "zc": sample.get("zc", 0)}


class Resize2DImage(object):
    """
    Resize 2D image to specified dimensions.

    Args:
        img_px_size (int): The target size (height and width) of the resized image.
    """

    def __init__(self, img_px_size=256):
        self.img_px_size = img_px_size

    def __call__(self, sample):
        image = sample["image"]

        # Resize using OpenCV
        image_n = cv2.resize(
            image,
            (self.img_px_size, self.img_px_size),
            interpolation=cv2.INTER_CUBIC,
        )

        # Add channel dimension if not present
        if image_n.ndim == 2:
            image_n = np.expand_dims(image_n, axis=0)

        return {"image": image_n, "zc": sample.get("zc", 0)}


class RandomRotation(object):
    """
    Randomly rotate the image by a specified range of angles.

    Args:
        max_angle (float): Maximum rotation angle in degrees
    """

    def __init__(self, max_angle=10.0):
        self.max_angle = max_angle

    def __call__(self, sample):
        image = sample["image"]

        # Generate random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Get image dimensions
        if image.ndim == 3:  # [C, H, W]
            h, w = image.shape[1], image.shape[2]
            image = np.transpose(image, (1, 2, 0))  # [H, W, C]
        else:  # [H, W]
            h, w = image.shape

        # Calculate rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        if image.ndim == 3:  # [H, W, C]
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            rotated = np.transpose(rotated, (2, 0, 1))  # [C, H, W]
        else:  # [H, W]
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

        return {"image": rotated, "zc": sample.get("zc", 0)}


class RandomFlip(object):
    """
    Randomly flip the image horizontally and/or vertically.

    Args:
        p_h (float): Probability of horizontal flip
        p_v (float): Probability of vertical flip
    """

    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, sample):
        image = sample["image"]

        # Horizontal flip
        if np.random.random() < self.p_h:
            if image.ndim == 3:  # [C, H, W]
                image = np.flip(image, axis=2)
            else:  # [H, W]
                image = np.flip(image, axis=1)

        # Vertical flip
        if np.random.random() < self.p_v:
            if image.ndim == 3:  # [C, H, W]
                image = np.flip(image, axis=1)
            else:  # [H, W]
                image = np.flip(image, axis=0)

        return {"image": image, "zc": sample.get("zc", 0)}


class RandomBrightness(object):
    """
    Randomly adjust the brightness of the image.

    Args:
        factor_range (tuple): Range of brightness adjustment factor
    """

    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, sample):
        image = sample["image"]

        # Generate random factor
        factor = np.random.uniform(self.factor_range[0], self.factor_range[1])

        # Apply brightness adjustment
        adjusted = image * factor

        return {"image": adjusted, "zc": sample.get("zc", 0)}


def get_transforms(config=None):
    """
    Create a transform composition based on configuration.

    Args:
        config (dict, optional): Configuration dictionary

    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    # If no config is provided, return basic transformation
    if config is None:
        return transforms.Compose([ToTensor()])

    # Create transform list based on config
    transform_list = []

    # Add normalization transforms
    if config.get('preprocessing', {}).get('normalize', True):
        # Choose normalization strategy
        norm_type = config.get('preprocessing', {}).get('norm_type', '95percentile')
        if norm_type == '95percentile':
            transform_list.append(NormalizeTo95PercentDose())
        elif norm_type == 'zscore':
            transform_list.append(ZscoreNormalization())
        elif norm_type == 'minmax':
            transform_list.append(MinMaxNormalization())

    # Add resize transform for 2D data
    if config.get('model', {}).get('is_2d', False):
        transform_list.append(Resize2DImage(config.get('dataset', {}).get('input_size_2d', 256)))

    # Add data augmentation transforms if enabled
    if config.get('preprocessing', {}).get('augment', False):
        # Add random rotation
        if config.get('preprocessing', {}).get('rotate', False):
            transform_list.append(RandomRotation(max_angle=10.0))

        # Add random flip
        if config.get('preprocessing', {}).get('flip', False):
            transform_list.append(RandomFlip(p_h=0.5, p_v=0.5))

        # Add random brightness adjustment
        if config.get('preprocessing', {}).get('adjust_brightness', False):
            transform_list.append(RandomBrightness(factor_range=(0.8, 1.2)))

    # Always add ToTensor as the last transform
    transform_list.append(ToTensor())

    return transforms.Compose(transform_list)