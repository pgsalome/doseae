import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nrrd
import re
from .preprocess import normalize_tensor, augment_data


def natural_sort(file_list):
    """
    Sort filenames by extracting the numeric part, e.g.,
    'image10.nrrd' will be recognized as 10 instead of 1.
    """

    def extract_num(fname):
        # Extract the digits from the filename
        match = re.search(r'(\d+)', os.path.basename(fname))
        if match:
            return int(match.group(1))
        return 999999  # fallback if no digits found

    return sorted(file_list, key=extract_num)


class DoseDistributionDataset(Dataset):
    """
    Dataset for 3D dose distribution data stored as NRRD files.
    """

    def __init__(self, data_dir, input_size=(64, 64, 64), transform=None, augment=False):
        """
        Args:
            data_dir (str): Directory containing the NRRD files
            input_size (tuple): Desired size for the input data (D, H, W)
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.input_size = input_size
        self.transform = transform
        self.augment = augment

        # Recursively gather all NRRD files
        self.file_paths = natural_sort(
            glob.glob(os.path.join(data_dir, '**', '*.nrrd'), recursive=True)
        )

        if len(self.file_paths) == 0:
            raise ValueError(f"No NRRD files found in {data_dir}")

        print(f"Found {len(self.file_paths)} dose distribution files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load the NRRD file
        data, _ = nrrd.read(file_path)

        # Preprocessing: resize if needed
        if data.shape != self.input_size:
            # Implement resizing logic or use a library
            # This is a placeholder - actual implementation would depend on requirements
            data = self._resize_data(data, self.input_size)

        # Add channel dimension if needed
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)

        # Convert to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Apply transforms if provided
        if self.transform:
            data_tensor = self.transform(data_tensor)

        # Apply augmentation if enabled
        if self.augment:
            data_tensor = augment_data(data_tensor)

        # For autoencoders, input = target
        return data_tensor, data_tensor

    def _resize_data(self, data, target_size):
        """
        Resize 3D data to target size.
        This is a placeholder - actual implementation would depend on requirements.
        """
        # Simple placeholder implementation
        # In practice, you might want to use more sophisticated interpolation methods
        from scipy.ndimage import zoom

        # Calculate zoom factors
        factors = [t / s for t, s in zip(target_size, data.shape)]

        # Apply zoom
        resized_data = zoom(data, factors, order=1)

        return resized_data


class DoseAEDataset(Dataset):
    """
    Dataset class for dose distribution data with zero count (non-zero voxel count) information.
    """

    def __init__(self, image_list, transform=None):
        """
        Args:
            image_list (list): List of images/volumes to use in the dataset
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_list = image_list
        self.zc_values = [np.sum(image != 0) for image in image_list]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        zc = self.zc_values[idx]

        # Create a sample dictionary
        sample = {"image": image, "zc": zc}

        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)

        return sample


class DoseDistribution2DDataset(Dataset):
    """
    Dataset for 2D dose distribution slices.
    """

    def __init__(self, data_dir, input_size=256, transform=None, augment=False, use_patches=False):
        """
        Args:
            data_dir (str): Directory containing the .npy files with slices or patches
            input_size (int): Desired size for the input data (square images)
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to apply data augmentation
            use_patches (bool): Whether the data consists of pre-extracted patches
        """
        self.data_dir = data_dir
        self.input_size = input_size
        self.transform = transform
        self.augment = augment
        self.use_patches = use_patches

        # Gather all .npy files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.npy')))

        if len(self.file_paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        # If using patches, load all patches into memory
        # Otherwise, load file paths for lazy loading of slices
        if use_patches:
            self.patches = []
            for file_path in self.file_paths:
                patches = np.load(file_path)
                self.patches.extend(patches)
            self.patches = np.array(self.patches)
            print(f"Loaded {len(self.patches)} patches from {len(self.file_paths)} files")
        else:
            # Create list of (file_path, slice_idx) tuples for all slices in all files
            self.slices = []
            for file_path in self.file_paths:
                slices = np.load(file_path)
                for i in range(len(slices)):
                    self.slices.append((file_path, i))
            print(f"Found {len(self.slices)} slices in {len(self.file_paths)} files")

    def __len__(self):
        if self.use_patches:
            return len(self.patches)
        else:
            return len(self.slices)

    def __getitem__(self, idx):
        if self.use_patches:
            # Get pre-extracted patch
            data = self.patches[idx]
        else:
            # Lazy loading of slice
            file_path, slice_idx = self.slices[idx]
            slices = np.load(file_path)
            data = slices[slice_idx]

        # Resize if needed
        if self.use_patches:
            # Patches are already sized correctly
            pass
        elif data.shape != (self.input_size, self.input_size):
            # Resize slice if needed
            from scipy.ndimage import zoom

            # Calculate zoom factors
            factors = [self.input_size / s for s in data.shape]

            # Apply zoom
            data = zoom(data, factors, order=1)

        # Convert to torch tensor and add channel dimension
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply transforms if provided
        if self.transform:
            data_tensor = self.transform(data_tensor)

        # Apply augmentation if enabled
        if self.augment:
            data_tensor = augment_data(data_tensor)

        # For autoencoders, input = target
        return data_tensor, data_tensor


class ToTensor:
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, zc = sample['image'], sample['zc']

        # Convert image from numpy array to torch tensor
        image_tensor = torch.from_numpy(image).float()

        # Add channel dimension if it doesn't exist
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Convert zc to tensor
        zc_tensor = torch.tensor(zc, dtype=torch.float32)

        return {'image': image_tensor, 'zc': zc_tensor}


class Normalize:
    """
    Normalize image in sample to [0, 1] or [-1, 1] range.
    """

    def __init__(self, to_range=(-1, 1), percentile=None):
        """
        Args:
            to_range (tuple): Target range for normalization
            percentile (float, optional): Percentile to use for normalization
        """
        self.to_range = to_range
        self.percentile = percentile

    def __call__(self, sample):
        image, zc = sample['image'], sample['zc']

        # Calculate percentile if provided
        if self.percentile is not None:
            max_val = np.percentile(image, self.percentile)
        else:
            max_val = image.max()

        if max_val > 0:
            # Normalize to [0, 1]
            image = image / max_val

            # Scale to target range if needed
            if self.to_range != (0, 1):
                image = image * (self.to_range[1] - self.to_range[0]) + self.to_range[0]

        return {'image': image, 'zc': zc}


def create_data_loaders(config):
    """
    Create train, validation, and test data loaders.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary containing DataLoaders for train, val, and test sets
    """
    # Check if we should use a single file
    if config.get('dataset', {}).get('use_single_file', False):
        data_file = config['dataset']['data_file']
        return create_data_loaders_from_numpy(config, data_file)

    # Check if we're working with 2D or 3D data
    is_2d = config.get('dataset', {}).get('is_2d_data', False)

    if is_2d:
        # Create 2D dataset
        full_dataset = DoseDistribution2DDataset(
            data_dir=config['dataset']['data_dir'],
            input_size=config['dataset']['input_size_2d'],
            augment=config['preprocessing']['augment'],
            use_patches=config.get('dataset', {}).get('use_patches', False)
        )
    else:
        # Create 3D dataset
        full_dataset = DoseDistributionDataset(
            data_dir=config['dataset']['data_dir'],
            input_size=tuple(config['dataset']['input_size']),
            augment=config['preprocessing']['augment']
        )

    # Calculate dataset sizes
    dataset_size = len(full_dataset)
    train_size = int(config['dataset']['train_ratio'] * dataset_size)
    val_size = int(config['dataset']['val_ratio'] * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_data_loaders_from_numpy(config, data_path):
    """
    Create data loaders from a single numpy file.

    Args:
        config (dict): Configuration dictionary
        data_path (str): Path to numpy file containing the data

    Returns:
        dict: Dictionary containing DataLoaders for train, val, and test sets
    """
    # Load all images from numpy file
    images = np.load(data_path)

    # Create custom dataset
    full_dataset = DoseAEDataset(
        image_list=images,
        transform=Transforms.Compose([
            Normalize(
                to_range=(-1, 1) if config['preprocessing'].get('use_tanh_output', True) else (0, 1),
                percentile=95
            ),
            ToTensor()
        ])
    )

    # Calculate dataset sizes
    dataset_size = len(full_dataset)
    train_size = int(config['dataset']['train_ratio'] * dataset_size)
    val_size = int(config['dataset']['val_ratio'] * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }