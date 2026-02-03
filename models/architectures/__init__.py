"""
Collection of trainable architectures used by the DoseAE project.
"""

from .configurable_autoencoder import (
    ConfigurableAutoencoder,
    create_attention_autoencoder,
    create_dual_input_autoencoder,
    create_multi_channel_autoencoder,
    create_fused_autoencoder,
    create_patches_only_autoencoder,
)
from .doseae_resunet import DoseAEResUNet, create_doseae_resunet
from .vae import VAE
from .vae_2d import VAE2D

__all__ = [
    "ConfigurableAutoencoder",
    "create_attention_autoencoder",
    "create_dual_input_autoencoder",
    "create_multi_channel_autoencoder",
    "create_fused_autoencoder",
    "create_patches_only_autoencoder",
    "DoseAEResUNet",
    "create_doseae_resunet",
    "VAE",
    "VAE2D",
]
