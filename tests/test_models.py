import pytest
import torch
import yaml
import os
import sys

# Add the 'src' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.models.vae import VAE
from src.models.resnet_ae import ResNetAutoencoder
from src.models.unet_ae import UNetAutoencoder
from src.models.vae_2d import VAE2D, ConvAutoencoder2D, MLPAutoencoder2D, UNet2D


@pytest.fixture
def config_3d():
    """
    Fixture for 3D model configuration.
    """
    return {
        'model': {
            'type': 'vae',
            'is_2d': False,
            'latent_dim': 64,
            'in_channels': 1,
            'base_filters': 16,
            'bottleneck_filters': 128
        },
        'hyperparameters': {
            'beta': 1.0
        },
        'training': {
            'grad_clip': 0.1
        }
    }


@pytest.fixture
def config_2d():
    """
    Fixture for 2D model configuration.
    """
    return {
        'model': {
            'type': 'vae',
            'is_2d': True,
            'latent_dim': 64,
            'in_channels': 1,
            'base_filters': 16,
            'dropout_rate': 0.3
        },
        'hyperparameters': {
            'beta': 1.0,
            'weight_decay': 0.0001
        },
        'training': {
            'grad_clip': 0.1
        },
        'dataset': {
            'input_size_2d': 64
        }
    }


def test_vae_3d(config_3d):
    """
    Test 3D VAE model.
    """
    model = VAE(
        in_channels=config_3d['model']['in_channels'],
        latent_dim=config_3d['model']['latent_dim'],
        base_filters=config_3d['model']['base_filters']
    )

    # Test model creation
    assert isinstance(model, VAE)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, config_3d['model']['in_channels'], 64, 64, 64)

    # Forward pass
    with torch.no_grad():
        output, mu, logvar = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"
    assert mu.shape == (batch_size, config_3d['model'][
        'latent_dim']), f"Expected mu shape {(batch_size, config_3d['model']['latent_dim'])}, got {mu.shape}"
    assert logvar.shape == (batch_size, config_3d['model'][
        'latent_dim']), f"Expected logvar shape {(batch_size, config_3d['model']['latent_dim'])}, got {logvar.shape}"


def test_vae_2d(config_2d):
    """
    Test 2D VAE model.
    """
    input_size = config_2d['dataset']['input_size_2d']

    model = VAE2D(
        input_size=input_size,
        latent_dim=config_2d['model']['latent_dim'],
        dropout_rate=config_2d['model']['dropout_rate'],
        weight_decay=config_2d['hyperparameters']['weight_decay'],
        grad_clip=config_2d['training']['grad_clip']
    )

    # Test model creation
    assert isinstance(model, VAE2D)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, input_size, input_size)

    # Forward pass
    with torch.no_grad():
        output, mu, logvar = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"
    assert mu.shape == (batch_size, config_2d['model'][
        'latent_dim']), f"Expected mu shape {(batch_size, config_2d['model']['latent_dim'])}, got {mu.shape}"
    assert logvar.shape == (batch_size, config_2d['model'][
        'latent_dim']), f"Expected logvar shape {(batch_size, config_2d['model']['latent_dim'])}, got {logvar.shape}"


def test_resnet_ae_3d(config_3d):
    """
    Test 3D ResNet Autoencoder model.
    """
    model = ResNetAutoencoder(
        in_channels=config_3d['model']['in_channels'],
        latent_dim=config_3d['model']['latent_dim'],
        base_filters=config_3d['model']['base_filters']
    )

    # Test model creation
    assert isinstance(model, ResNetAutoencoder)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, config_3d['model']['in_channels'], 64, 64, 64)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"


def test_unet_ae_3d(config_3d):
    """
    Test 3D UNet Autoencoder model.
    """
    model = UNetAutoencoder(
        in_channels=config_3d['model']['in_channels'],
        latent_dim=config_3d['model']['latent_dim'],
        base_filters=config_3d['model']['base_filters']
    )

    # Test model creation
    assert isinstance(model, UNetAutoencoder)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, config_3d['model']['in_channels'], 64, 64, 64)

    # Forward pass
    with torch.no_grad():
        output, latent = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"
    # Latent representation is extracted from bottleneck, but specific shape depends on model implementation


def test_conv_autoencoder_2d(config_2d):
    """
    Test 2D Convolutional Autoencoder model.
    """
    input_size = config_2d['dataset']['input_size_2d']

    model = ConvAutoencoder2D(
        input_size=input_size,
        dropout_rate=config_2d['model']['dropout_rate']
    )

    # Test model creation
    assert isinstance(model, ConvAutoencoder2D)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, input_size, input_size)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"


def test_mlp_autoencoder_2d(config_2d):
    """
    Test 2D MLP Autoencoder model.
    """
    input_size = config_2d['dataset']['input_size_2d']

    model = MLPAutoencoder2D(
        input_size=input_size,
        latent_dim=config_2d['model']['latent_dim'],
        dropout_rate=config_2d['model']['dropout_rate']
    )

    # Test model creation
    assert isinstance(model, MLPAutoencoder2D)

    # Test forward pass with a random tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, input_size, input_size)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"


def test_unet_2d(config_2d):
    """
    Test 2D UNet model.
    """
    model = UNet2D(
        in_channels=config_2d['model']['in_channels'],
        out_channels=config_2d['model']['in_channels'],
        dropout_rate=config_2d['model']['dropout_rate']
    )

    # Test model creation
    assert isinstance(model, UNet2D)

    # Test forward pass with a random tensor
    batch_size = 2
    input_size = config_2d['dataset']['input_size_2d']
    input_tensor = torch.randn(batch_size, config_2d['model']['in_channels'], input_size, input_size)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Check output shapes
    assert output.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, got {output.shape}"


def test_model_factory_3d(config_3d):
    """
    Test model factory function with 3D models.
    """
    # Test VAE
    config_3d['model']['type'] = 'vae'
    model = get_model(config_3d)
    assert isinstance(model, VAE)

    # Test ResNet AE
    config_3d['model']['type'] = 'resnet_ae'
    model = get_model(config_3d)
    assert isinstance(model, ResNetAutoencoder)

    # Tes