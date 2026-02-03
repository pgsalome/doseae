"""
Model components and building blocks.

This module contains reusable building blocks for constructing
autoencoder models, such as attention mechanisms, ResNet blocks,
and UNet components.
"""

from .attention import (
    ChannelAttentionND,
    HighDoseAttention,
    LobeContextAttention,
    MultiScaleSelfAttention,
    SpatialContextAttention,
)

from .resnet import (
    ResNetBlock3D,
    SEBlock3D,
    ResNet3D,
    ResNetEncoder3D,
    ResNetDecoder3D
)

from .unet import (
    Conv3DBlock,
    UNet3D,
    UNetEncoder,
    UNetDecoder
)

__all__ = [
    # Attention components
    'ChannelAttentionND',
    'HighDoseAttention',
    'LobeContextAttention',
    'MultiScaleSelfAttention',
    'SpatialContextAttention',
    
    # ResNet components
    'ResNetBlock3D',
    'SEBlock3D',
    'ResNet3D',
    'ResNetEncoder3D',
    'ResNetDecoder3D',
    
    # UNet components
    'Conv3DBlock',
    'UNet3D',
    'UNetEncoder',
    'UNetDecoder'
]
