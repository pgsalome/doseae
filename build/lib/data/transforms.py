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
        """Convert ndarrays to Tensors."""
        image, zc = sample['image'], sample['zc']

        # Convert image from numpy array to torch tensor
        image_tensor = torch.from_numpy(image.copy()).float()

        # Ensure image has channel dimension for batch processing
        if image_tensor.ndim == 3:  # [D, H, W]
            image_tensor = image_tensor.unsqueeze(0)  # [1, D, H, W]

        # Convert zc to tensor
        zc_tensor = torch.tensor(zc, dtype=torch.float32)

        return {'image': image_tensor, 'zc': zc_tensor}


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
    Supports both 2D and 3D data.

    Args:
        max_angle (float): Maximum rotation angle in degrees
    """

    def __init__(self, max_angle=10.0):
        self.max_angle = max_angle

    def __call__(self, sample):
        image = sample["image"]

        # Handle 3D data (volumes)
        if isinstance(image, np.ndarray) and image.ndim > 3:
            # For 3D, return original - 3D rotation is complex
            return {"image": image, "zc": sample.get("zc", 0)}

        if isinstance(image, torch.Tensor) and image.ndim > 3:
            # For 3D tensors, return original
            return {"image": image, "zc": sample.get("zc", 0)}

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            is_tensor = True
            device = image.device
            image = image.cpu().numpy()
        else:
            is_tensor = False

        # Generate random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        try:
            # Get image dimensions based on shape
            if image.ndim == 2:  # [H, W]
                h, w = image.shape
                image_to_rotate = image
            elif image.ndim == 3:
                if image.shape[0] in [1, 3]:  # [C, H, W]
                    h, w = image.shape[1], image.shape[2]
                    image_to_rotate = np.transpose(image, (1, 2, 0))  # [H, W, C]
                else:  # Assuming [D, H, W] - handle as 3D volume
                    # Return original for 3D volumes
                    if is_tensor:
                        return {"image": torch.tensor(image, device=device), "zc": sample.get("zc", 0)}
                    return {"image": image, "zc": sample.get("zc", 0)}
            else:
                # Return original for other dimensions
                if is_tensor:
                    return {"image": torch.tensor(image, device=device), "zc": sample.get("zc", 0)}
                return {"image": image, "zc": sample.get("zc", 0)}

            # Calculate rotation matrix
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # [C, H, W]
                rotated = cv2.warpAffine(image_to_rotate, rotation_matrix, (w, h))
                rotated = np.transpose(rotated, (2, 0, 1))  # [C, H, W]
            else:  # [H, W]
                rotated = cv2.warpAffine(image_to_rotate, rotation_matrix, (w, h))

            # Convert back to tensor if input was tensor
            if is_tensor:
                rotated = torch.tensor(rotated, device=device)

            return {"image": rotated, "zc": sample.get("zc", 0)}
        except Exception as e:
            print(f"Error in RandomRotation: {e}")
            # Return original data on error
            if is_tensor:
                return {"image": torch.tensor(image, device=device), "zc": sample.get("zc", 0)}
            return {"image": image, "zc": sample.get("zc", 0)}





class RandomBrightness(object):
    """
    Randomly adjust the brightness of the image.
    Works with any dimensionality of data.

    Args:
        factor_range (tuple): Range of brightness adjustment factor
    """

    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, sample):
        image = sample["image"]

        try:
            # Generate random factor
            factor = np.random.uniform(self.factor_range[0], self.factor_range[1])

            # Apply brightness adjustment (works with any dimensionality)
            adjusted = image * factor

            return {"image": adjusted, "zc": sample.get("zc", 0)}

        except Exception as e:
            print(f"Error in RandomBrightness: {e}")
            return {"image": image, "zc": sample.get("zc", 0)}


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
    if config.get('transforms', {}).get('augment', False):
        # Add random rotation
        if config.get('transforms', {}).get('rotate', False):
            transform_list.append(RandomRotation(max_angle=10.0))

        # Add random flip
        if config.get('transforms', {}).get('flip', False):
            transform_list.append(RandomFlip(p_h=0.5, p_v=0.5))

        # Add random brightness adjustment
        if config.get('transforms', {}).get('adjust_brightness', False):
            transform_list.append(RandomBrightness(factor_range=(0.8, 1.2)))

    # Always add ToTensor as the last transform
    transform_list.append(ToTensor())

    return transforms.Compose(transform_list)


class RandomFlip(object):
    """
    Randomly flip the image horizontally and/or vertically.
    Supports both 2D and 3D data.

    Args:
        p_h (float): Probability of horizontal flip
        p_v (float): Probability of vertical flip
    """

    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, sample):
        image = sample["image"]

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            is_tensor = True
            device = image.device
            image = image.cpu().numpy()
        else:
            is_tensor = False

        try:
            # Make a copy to ensure we don't have negative strides
            image = image.copy()

            # Handle different data dimensions
            if image.ndim == 2:  # [H, W]
                # 2D image
                if np.random.random() < self.p_h:
                    image = np.flip(image, axis=1).copy()  # Horizontal flip
                if np.random.random() < self.p_v:
                    image = np.flip(image, axis=0).copy()  # Vertical flip

            elif image.ndim == 3:
                if image.shape[0] in [1, 3]:  # [C, H, W]
                    # 2D image with channels
                    if np.random.random() < self.p_h:
                        image = np.flip(image, axis=2).copy()  # Horizontal flip
                    if np.random.random() < self.p_v:
                        image = np.flip(image, axis=1).copy()  # Vertical flip
                else:  # [D, H, W]
                    # 3D volume - flip along each axis with different probabilities
                    if np.random.random() < self.p_h:
                        image = np.flip(image, axis=2).copy()  # Depth-wise flip
                    if np.random.random() < self.p_v:
                        image = np.flip(image, axis=1).copy()  # Height-wise flip

            elif image.ndim == 4:  # [C, D, H, W]
                # 3D volume with channels
                if np.random.random() < self.p_h:
                    image = np.flip(image, axis=3).copy()  # Width-wise flip
                if np.random.random() < self.p_v:
                    image = np.flip(image, axis=2).copy()  # Height-wise flip

            elif image.ndim == 5:  # [B, C, D, H, W]
                # Batch of 3D volumes with channels
                if np.random.random() < self.p_h:
                    image = np.flip(image, axis=4).copy()  # Width-wise flip
                if np.random.random() < self.p_v:
                    image = np.flip(image, axis=3).copy()  # Height-wise flip

            # Convert back to tensor if input was tensor
            if is_tensor:
                image = torch.tensor(image, device=device)

            return {"image": image, "zc": sample.get("zc", 0)}

        except Exception as e:
            print(f"Error in RandomFlip: {e}")
            # Return original on error
            if is_tensor:
                return {"image": torch.tensor(image, device=device), "zc": sample.get("zc", 0)}
            return {"image": image, "zc": sample.get("zc", 0)}


def augment_data(tensor):
    """
    Apply random augmentations to a tensor.
    Safely handles both 2D and 3D data.

    Args:
        tensor (torch.Tensor): Input tensor to augment

    Returns:
        torch.Tensor: Augmented tensor
    """
    try:
        # Make a copy to ensure contiguity
        if isinstance(tensor, np.ndarray):
            tensor = tensor.copy()
        elif isinstance(tensor, torch.Tensor):
            # Ensure tensor is contiguous
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

        # Get tensor shape information
        tensor_shape = tensor.shape if hasattr(tensor, 'shape') else None

        if tensor_shape is None:
            return tensor

        # Determine if this is 3D volume data
        is_3d_volume = len(tensor_shape) > 3 or (len(tensor_shape) == 3 and tensor_shape[0] > 3)

        # Create sample for transform functions
        sample = {"image": tensor}

        # Apply transformations based on data dimensionality
        if is_3d_volume:
            # For 3D data, only apply brightness adjustment which works for any dimensionality
            if np.random.random() < 0.5:
                sample = RandomBrightness(factor_range=(0.8, 1.2))(sample)

            # For 3D data also apply flipping which has been updated to support 3D
            if np.random.random() < 0.5:
                sample = RandomFlip(p_h=0.5, p_v=0.5)(sample)
        else:
            # For 2D data, apply all transformations
            if np.random.random() < 0.5:
                sample = RandomRotation(max_angle=10.0)(sample)

            if np.random.random() < 0.5:
                sample = RandomFlip(p_h=0.5, p_v=0.5)(sample)

            if np.random.random() < 0.5:
                sample = RandomBrightness(factor_range=(0.8, 1.2))(sample)

        return sample["image"]

    except Exception as e:
        print(f"Error in augment_data: {e}, returning original tensor")
        return tensor