import os
import sys
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import DoseDistributionDataset, DoseAEDataset, DoseDistribution2DDataset
from src.data.transforms import ToTensor, NormalizeTo95PercentDose, ZscoreNormalization


@pytest.fixture
def dummy_2d_data():
    """
    Create a dummy 2D slice dataset.
    """
    # Create a temporary directory for test data
    os.makedirs('test_data', exist_ok=True)

    # Create dummy slices
    slices = np.random.rand(10, 64, 64).astype(np.float32)

    # Save to numpy file
    np.save('test_data/dummy_slices.npy', slices)

    yield 'test_data/dummy_slices.npy'

    # Cleanup
    os.remove('test_data/dummy_slices.npy')
    os.rmdir('test_data')


@pytest.fixture
def dummy_3d_data():
    """
    Create a dummy 3D volume dataset.
    """
    # Create a temporary directory for test data
    os.makedirs('test_data', exist_ok=True)

    # Create dummy volumes
    volumes = np.random.rand(5, 16, 64, 64).astype(np.float32)

    # Save to numpy file
    np.save('test_data/dummy_volumes.npy', volumes)

    yield 'test_data/dummy_volumes.npy'

    # Cleanup
    os.remove('test_data/dummy_volumes.npy')
    os.rmdir('test_data')


def test_dose_ae_dataset(dummy_2d_data):
    """
    Test DoseAEDataset.
    """
    # Load dummy data
    slices = np.load(dummy_2d_data)

    # Create dataset
    dataset = DoseAEDataset(
        image_list=slices,
        transform=ToTensor()
    )

    # Check length
    assert len(dataset) == len(slices)

    # Check item retrieval
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'image' in sample
    assert isinstance(sample['image'], torch.Tensor)
    assert sample['image'].shape == torch.Size([64, 64])

    # Check zero count values
    assert hasattr(dataset, 'zc_values')
    assert len(dataset.zc_values) == len(slices)


def test_dose_ae_dataset_with_normalization(dummy_2d_data):
    """
    Test DoseAEDataset with normalization.
    """
    # Load dummy data
    slices = np.load(dummy_2d_data)

    # Create dataset with normalization
    transform = transforms.Compose([
        NormalizeTo95PercentDose(),
        ZscoreNormalization(),
        ToTensor()
    ])

    dataset = DoseAEDataset(
        image_list=slices,
        transform=transform
    )

    # Get sample
    sample = dataset[0]

    # Check normalization effects
    assert sample['image'].mean().abs() < 1e-5  # Mean should be close to zero
    assert (sample['image'].std() - 1.0).abs() < 1e-5  # Std should be close to 1


def test_dose_distribution_2d_dataset(dummy_2d_data):
    """
    Test DoseDistribution2DDataset.
    """
    # Create test directory
    data_dir = os.path.dirname(dummy_2d_data)

    # Create dataset
    dataset = DoseDistribution2DDataset(
        data_dir=data_dir,
        input_size=64
    )

    # Check length
    assert len(dataset) == 10  # 10 slices in the dummy data

    # Check item retrieval
    data, target = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert data.shape == torch.Size([1, 64, 64])  # [C, H, W]
    assert torch.allclose(data, target)  # Input and target should be the same for autoencoder


def test_dataloader(dummy_2d_data):
    """
    Test DataLoader with dataset.
    """
    # Load dummy data
    slices = np.load(dummy_2d_data)

    # Create dataset
    dataset = DoseAEDataset(
        image_list=slices,
        transform=ToTensor()
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    # Get batch
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert 'image' in batch
    assert batch['image'].shape == torch.Size([4, 64, 64])


if __name__ == "__main__":
    # Add imports needed for direct execution
    from torchvision import transforms

    # Run tests
    pytest.main(['-xvs', __file__])