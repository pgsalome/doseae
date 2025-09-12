import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nrrd
import h5py  # Replaced pickle with h5py
import re

from .transforms import augment_data

def natural_sort(file_list):
    """
    Sort filenames by extracting the numeric part.
    """
    def extract_num(fname):
        match = re.search(r'(\d+)', os.path.basename(fname))
        if match:
            return int(match.group(1))
        return 999999
    return sorted(file_list, key=extract_num)


# --- UNCHANGED: Kept for existing functionality ---
class DoseDistributionDataset(Dataset):
    """
    Dataset for 3D dose distribution data stored as NRRD files.
    """
    def __init__(self, data_dir, input_size=(64, 64, 64), transform=None, augment=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.transform = transform
        self.augment = augment
        self.file_paths = natural_sort(glob.glob(os.path.join(data_dir, '**', '*.nrrd'), recursive=True))
        if len(self.file_paths) == 0:
            raise ValueError(f"No NRRD files found in {data_dir}")
        print(f"Found {len(self.file_paths)} dose distribution files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data, _ = nrrd.read(file_path)
        if data.shape != self.input_size:
            data = self._resize_data(data, self.input_size)
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        if self.transform:
            data_tensor = self.transform(data_tensor)
        if self.augment:
            data_tensor = augment_data(data_tensor)
        return data_tensor, data_tensor

    def _resize_data(self, data, target_size):
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_size, data.shape)]
        return zoom(data, factors, order=1)


# --- UPDATED: This class now handles lazy loading from an HDF5 file ---
class DoseAEDataset(Dataset):
    """
    Dataset class for dose distribution data from an HDF5 file.
    Uses lazy loading for memory efficiency.
    """
    def __init__(self, h5_path, transform=None, indices=None):
        """
        Args:
            h5_path (string): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
            indices (list, optional): List of indices for this dataset split (train/val/test).
        """
        self.h5_path = h5_path
        self.transform = transform

        # Open file once to get total length
        with h5py.File(self.h5_path, 'r') as db:
            # IMPORTANT: Replace 'dose_images' if your HDF5 dataset has a different key
            self._data_key = 'dose_images'
            self.total_samples = len(db[self._data_key])

        # Use provided indices for the split, or all indices if None
        self.indices = indices if indices is not None else list(range(self.total_samples))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Map the split-specific index to the actual index in the H5 file
        file_index = self.indices[idx]

        # Open the file in __getitem__ for compatibility with multi-worker DataLoader
        with h5py.File(self.h5_path, 'r') as db:
            image = db[self._data_key][file_index]

        # Calculate non-zero count on the fly
        zc = np.sum(image != 0)

        # Ensure image has channel dimension
        if image.ndim == 3:  # [D, H, W]
            image = np.expand_dims(image, axis=0)  # Add channel -> [1, D, H, W]

        sample = {"image": image, "zc": zc}

        if self.transform:
            sample = self.transform(sample)

        return sample


# --- UNCHANGED: Kept for existing functionality ---
class DoseDistribution2DDataset(Dataset):
    """
    Dataset for 2D dose distribution slices from .npy files.
    """
    def __init__(self, data_dir, input_size=256, transform=None, augment=False, use_patches=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.transform = transform
        self.augment = augment
        self.use_patches = use_patches
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        if use_patches:
            self.patches = []
            for file_path in self.file_paths:
                self.patches.extend(np.load(file_path))
            self.patches = np.array(self.patches)
        else:
            self.slices = []
            for file_path in self.file_paths:
                # Use mmap_mode for memory-efficient reading of file shapes
                with np.load(file_path, mmap_mode='r') as data:
                    num_slices = len(data)
                for i in range(num_slices):
                    self.slices.append((file_path, i))

    def __len__(self):
        return len(self.patches) if self.use_patches else len(self.slices)

    def __getitem__(self, idx):
        if self.use_patches:
            data = self.patches[idx]
        else:
            file_path, slice_idx = self.slices[idx]
            # Use mmap_mode to avoid loading the whole file for one slice
            data = np.load(file_path, mmap_mode='r')[slice_idx]

        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            data_tensor = self.transform(data_tensor)
        if self.augment:
            data_tensor = augment_data(data_tensor)
        return data_tensor, data_tensor


# --- UNCHANGED: Kept for existing functionality ---
class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, zc = sample['image'], sample['zc']
        return {
            'image': torch.from_numpy(image.copy()).float(), # Use .copy() to be safe
            'zc': torch.tensor(zc, dtype=torch.float32)
        }


# --- UNCHANGED: Kept for existing functionality ---
class Normalize:
    """Normalize image in sample to a given range."""
    def __init__(self, to_range=(-1, 1), percentile=None):
        self.to_range = to_range
        self.percentile = percentile

    def __call__(self, sample):
        image, zc = sample['image'], sample['zc']
        max_val = np.percentile(image, self.percentile) if self.percentile is not None else image.max()
        if max_val > 0:
            image = image / max_val
            if self.to_range != (0, 1):
                image = image * (self.to_range[1] - self.to_range[0]) + self.to_range[0]
        return {'image': image, 'zc': zc}


# --- UPDATED: This function now loads from HDF5 and includes your original logic ---
def create_data_loaders(config):
    """
    Create train, validation, and test data loaders from an HDF5 file.
    """
    try:
        data_h5_path = config.get('dataset', {}).get('data_h5')
        if not data_h5_path or not os.path.exists(data_h5_path):
            raise ValueError("No 'data_h5' path provided in config or file does not exist.")

        print(f"Loading data from HDF5 file: {data_h5_path}")

        with h5py.File(data_h5_path, 'r') as f:
            # IMPORTANT: Replace 'dose_images' if your HDF5 dataset has a different key
            data_key = 'dose_images'
            dataset_size = len(f[data_key])

            # FUNCTIONALITY RESTORED: Dynamically get input shape from the first item
            if dataset_size > 0:
                input_shape = f[data_key][0].shape
                print(f"Data input shape: {input_shape}")
                if len(input_shape) == 3:  # 3D data
                    config['dataset']['input_size'] = list(input_shape)

        # Create a full list of indices
        indices = list(range(dataset_size))

        # FUNCTIONALITY RESTORED: Check if we're in test mode to limit samples
        if config.get('dataset', {}).get('test_mode', False):
            n_test_samples = config.get('dataset', {}).get('n_test_samples', 20)
            if len(indices) > n_test_samples:
                print(f"Test mode: Using {n_test_samples} samples from dataset")
                indices = indices[:n_test_samples]

        # Shuffle indices for splitting
        np.random.seed(config['training']['seed'])
        np.random.shuffle(indices)

        # Calculate split sizes based on the (potentially limited) indices
        current_size = len(indices)
        train_size = int(config['dataset']['train_ratio'] * current_size)
        val_size = int(config['dataset']['val_ratio'] * current_size)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")

        # Create full dataset object (it will manage indices internally)
        full_dataset = DoseAEDataset(data_h5_path, transform=ToTensor())

        # Create subset objects for each split
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # Create data loaders
        batch_size = config['hyperparameters']['batch_size']
        num_workers = config['dataset'].get('num_workers', 4)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print(f"Created data loaders - train: {len(train_loader)} batches, val: {len(val_loader)} batches, test: {len(test_loader)} batches")

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    except Exception as e:
        print(f"Error in create_data_loaders: {str(e)}")
        import traceback
        traceback.print_exc()
        return None