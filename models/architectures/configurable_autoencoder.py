"""
Configurable Autoencoder for Dose Patch Representation Learning.

This single autoencoder can handle all input configurations:
- 1 channel: dose only, CT only, or fused CT+dose
- 2 channels: dose + CT, or dose + fused
- 3 channels: dose + CT + fused

Replaces multiple similar autoencoder files.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.resnet import ResNetBlock3D, SEBlock3D
from ..components.unet import Conv3DBlock
from ..components.attention import (
    ChannelAttentionND,
    HighDoseAttention,
    LobeContextAttention,
    MultiScaleSelfAttention,
    SpatialContextAttention,
)


class Conv2DBlock(nn.Module):
    """2D Convolutional block with batch normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class ResNetBlock2D(nn.Module):
    """2D ResNet block with optional bottleneck."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, use_bottleneck: bool = False,
                 dropout: float = 0.0):
        super().__init__()

        self.use_bottleneck = use_bottleneck
        self.stride = stride

        if use_bottleneck:
            bottleneck_channels = out_channels // 4
            self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(bottleneck_channels)
            self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck_channels)
            self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        if self.use_bottleneck:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        if self.dropout:
            out = self.dropout(out)

        out += residual
        out = self.relu(out)
        return out


class SEBlock2D(nn.Module):
    """2D Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetUNetBackbone(nn.Module):
    """ResNet-UNet encoder/decoder without attention for volumetric inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_filters: int = 32,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.bottleneck_channels = base_filters * 8

        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = nn.Sequential(
            ResNetBlock3D(base_filters, base_filters),
            ResNetBlock3D(base_filters, base_filters),
        )
        self.encoder2 = nn.Sequential(
            ResNetBlock3D(base_filters, base_filters * 2, stride=2),
            ResNetBlock3D(base_filters * 2, base_filters * 2),
        )
        self.encoder3 = nn.Sequential(
            ResNetBlock3D(base_filters * 2, base_filters * 4, stride=2),
            ResNetBlock3D(base_filters * 4, base_filters * 4),
        )
        self.encoder4 = nn.Sequential(
            ResNetBlock3D(base_filters * 4, base_filters * 8, stride=2),
            ResNetBlock3D(base_filters * 8, base_filters * 8),
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters * 4, base_filters * 4),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters * 2, base_filters * 2),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters, base_filters),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(base_filters, base_filters, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters, base_filters),
        )

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.initial_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        d4 = self.decoder4(x4)
        if d4.shape != x3.shape:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='trilinear', align_corners=False)
        d4 = d4 + x3

        d3 = self.decoder3(d4)
        if d3.shape != x2.shape:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='trilinear', align_corners=False)
        d3 = d3 + x2

        d2 = self.decoder2(d3)
        if d2.shape != x1.shape:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='trilinear', align_corners=False)
        d2 = d2 + x1

        d1 = self.decoder1(d2)
        if d1.shape != x0.shape:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='trilinear', align_corners=False)
        d1 = d1 + x0

        output = self.final_conv(d1)
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        latent = self.global_pool(x4).flatten(1)
        return output, latent


class ConfigurableAutoencoder(nn.Module):
    """
    Configurable autoencoder that supports all input channel configurations.
    
    Input modes:
    - 'dose_only': 1 channel (dose patches)
    - 'ct_only': 1 channel (CT patches) 
    - 'dose_ct': 2 channels (dose + CT)
    - 'dose_fused': 2 channels (dose + fused CT+dose)
    - 'dose_ct_fused': 3 channels (dose + CT + fused)
    - 'fused_only': 1 channel (fused CT+dose)
    
    Upsampling methods:
    - bilinear_upsampling=False: Uses transposed convolutions (learnable)
    - bilinear_upsampling=True: Uses bilinear interpolation + 1x1 conv (smoother)
    """
    
    def __init__(self, 
                 input_channels: int = 2,
                 output_channels: int = 1,
                 base_filters: int = 64,
                 feature_dim: int = 256,
                 attention_heads: int = 4,
                 num_patches_per_patient: int = 50,
                 latent_dim: int = 8,
                 input_mode: str = 'dose_ct',
                 fusion_method: str = 'weighted_sum',  # 'weighted_sum', 'concat', 'multiply'
                 architecture: str = 'conv',  # 'conv', 'resnet', 'unet', 'resnet_unet', 'mlp'
                 use_bottleneck: bool = False,  # For ResNet blocks
                 input_size: tuple = (64, 64, 64),  # Spatial dimensions (3D) or (H, W) for 2D
                 bilinear_upsampling: bool = False,  # Use bilinear upsampling instead of transposed conv
                 dropout_mlp: float = 0.1,
                 dropout_attention: float = 0.1,
                 dropout_decoder: float = 0.1,
                 dropout_features: float = 0.1,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_patches_per_patient = num_patches_per_patient
        self.input_mode = input_mode
        self.fusion_method = fusion_method
        self.architecture = architecture
        self.use_bottleneck = use_bottleneck
        self.base_filters = base_filters
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        elif isinstance(input_size, (list, tuple)):
            if len(input_size) >= 3:
                self.input_size = tuple(input_size[:3])
            elif len(input_size) == 2:
                self.input_size = tuple(input_size)
            elif len(input_size) == 1:
                self.input_size = (input_size[0], input_size[0])
            else:
                self.input_size = (64, 64, 64)
        else:
            self.input_size = (64, 64, 64)
        self.bilinear_upsampling = bilinear_upsampling
        self.dropout_features = dropout_features
        self.dropout_mlp = dropout_mlp
        self.dropout_attention = dropout_attention
        self.dropout_decoder = dropout_decoder

        # Determine spatial dimensionality (2D vs 3D)
        self.dim = 3 if len(self.input_size) >= 3 else 2
        if self.dim == 2 and isinstance(self.input_size, tuple) and len(self.input_size) > 2:
            self.input_size = tuple(self.input_size[:2])

        # Module factories based on dimensionality
        if self.dim == 3:
            self.ConvND = nn.Conv3d
            self.ConvTransposeND = nn.ConvTranspose3d
            self.BatchNormND = nn.BatchNorm3d
            self.DropoutND = nn.Dropout3d
            self.MaxPoolND = nn.MaxPool3d
            self.AdaptiveAvgPoolND = nn.AdaptiveAvgPool3d
            self.ConvBlock = Conv3DBlock
            self.ResNetBlock = ResNetBlock3D
            self.SEBlock = SEBlock3D
            self.upsample_mode = 'trilinear'
        else:
            self.ConvND = nn.Conv2d
            self.ConvTransposeND = nn.ConvTranspose2d
            self.BatchNormND = nn.BatchNorm2d
            self.DropoutND = nn.Dropout2d
            self.MaxPoolND = nn.MaxPool2d
            self.AdaptiveAvgPoolND = nn.AdaptiveAvgPool2d
            self.ConvBlock = Conv2DBlock
            self.ResNetBlock = ResNetBlock2D
            self.SEBlock = SEBlock2D
            self.upsample_mode = 'bilinear'
            attention_heads = 0  # Disable 3D-specific attention for 2D inputs

        # Calculate required depth based on input size
        self.encoder_depth = self._calculate_encoder_depth()

        # Input channel mapping based on mode
        self._setup_input_channels()
        
        # Fusion layer for multi-channel inputs
        if input_mode in ['dose_ct', 'dose_fused', 'dose_ct_fused']:
            self._setup_fusion_layer()

        model_cfg = self.config.get('model', {}) if isinstance(self.config, dict) else {}
        self.uses_resnet_unet = self.architecture == 'resnet_unet'
        if self.uses_resnet_unet:
            apply_sigmoid = bool(model_cfg.get('apply_final_sigmoid', True))
            self.resnet_unet_core = ResNetUNetBackbone(
                in_channels=self.input_channels,
                out_channels=self.output_channels,
                base_filters=self.base_filters,
                use_sigmoid=apply_sigmoid,
            )
            self.encoder = None
        else:
            # Encoder
            self.encoder = self._build_encoder()

        # Attention mechanism(s)
        att_cfg = model_cfg.get('attention', {}) if isinstance(model_cfg, dict) else {}
        self.attention_type = 'none'
        self.channel_attention: Optional[nn.Module] = None
        self.sequence_attention: Optional[nn.Module] = None
        self.spatial_context_attention: Optional[nn.Module] = None
        self.lobe_attention: Optional[nn.Module] = None
        self.high_dose_predictor = None
        self.dose_level_classifier = None
        self._latest_attention = None

        if att_cfg.get('enabled', False):
            self.attention_type = str(att_cfg.get('type', 'multi_scale')).lower()

            if self.attention_type == 'channel':
                reduction = int(att_cfg.get('reduction', 16))
                self.channel_attention = ChannelAttentionND(
                    channels=self.input_channels,
                    reduction=reduction,
                    dim=self.dim,
                )
            elif self.attention_type == 'lobe':
                self.lobe_attention = LobeContextAttention(
                    feature_dim,
                    num_lobes=int(att_cfg.get('num_lobes', 6)),
                    num_side_types=int(att_cfg.get('num_side_types', 3)),
                    num_anatomical_sides=int(att_cfg.get('num_anatomical_sides', 3)),
                    lobe_embed_dim=int(att_cfg.get('lobe_embed_dim', 16)),
                    side_embed_dim=int(att_cfg.get('side_embed_dim', 8)),
                    anatomical_embed_dim=int(att_cfg.get('anatomical_embed_dim', 8)),
                    dropout=float(att_cfg.get('dropout', self.dropout_attention)),
                )
            elif self.attention_type == 'spatial':
                coord_dim = int(att_cfg.get('coord_dim', 4 if self.dim == 3 else 3))
                embed_dim = int(att_cfg.get('embed_dim', feature_dim))
                self.spatial_context_attention = SpatialContextAttention(
                    coord_dim,
                    embed_dim,
                    feature_dim,
                    dropout=float(att_cfg.get('dropout', self.dropout_attention)),
                    learnable_bias=bool(att_cfg.get('learnable_bias', True)),
                )
            elif self.attention_type == 'high_dose':
                self.sequence_attention = HighDoseAttention(
                    feature_dim,
                    dose_threshold=float(att_cfg.get('dose_threshold', 0.5)),
                )
            else:
                # Default to multi-scale self-attention.
                self.attention_type = 'multi_scale'
                self.sequence_attention = MultiScaleSelfAttention(
                    feature_dim,
                    num_heads=int(att_cfg.get('num_heads', 4)),
                    dim=self.dim,
                )

        aux_cfg = model_cfg.get('auxiliary', {}) if isinstance(model_cfg, dict) else {}
        if aux_cfg.get('predict_high_dose', False):
            num_classes = aux_cfg.get('high_dose_classes', 2)
            self.high_dose_predictor = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(aux_cfg.get('dropout', 0.3)),
                nn.Linear(latent_dim // 2, num_classes)
            )
        if aux_cfg.get('predict_dose_level', False):
            num_classes = aux_cfg.get('dose_level_classes', 5)
            self.dose_level_classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(aux_cfg.get('dropout', 0.3)),
                nn.Linear(latent_dim // 2, num_classes)
            )
        
        if self.uses_resnet_unet:
            bottleneck_dim = self.resnet_unet_core.bottleneck_channels
            self.latent_projection = nn.Sequential(
                nn.Linear(bottleneck_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_features),
                nn.Linear(feature_dim, latent_dim)
            )
            self.decoder = None
        else:
            # Latent projection
            self.latent_projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_features),
                nn.Linear(feature_dim // 2, latent_dim)
            )
            
            # Decoder
            self.decoder = self._build_decoder()

        self._apply_pretrained_encoder()
    
    def _resize_to_target(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Resize decoder output to match reference spatial dimensions."""
        if not torch.is_tensor(tensor) or not torch.is_tensor(reference):
            return tensor

        spatial_dims = reference.shape[-self.dim:]
        if tensor.shape[-self.dim:] == spatial_dims:
            return tensor

        mode = 'trilinear' if self.dim == 3 else 'bilinear'
        return F.interpolate(tensor, size=spatial_dims, mode=mode, align_corners=False)
        
    def _setup_input_channels(self):
        """Setup input channels based on input mode."""
        if self.input_mode == 'dose_only':
            self.input_channels = 1
        elif self.input_mode == 'ct_only':
            self.input_channels = 1
        elif self.input_mode == 'dose_ct':
            self.input_channels = 2
        elif self.input_mode == 'dose_fused':
            self.input_channels = 2
        elif self.input_mode == 'dose_ct_fused':
            self.input_channels = 3
        elif self.input_mode == 'fused_only':
            self.input_channels = 1
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")
    
    def _setup_fusion_layer(self):
        """Setup fusion layer for multi-channel inputs."""
        if self.fusion_method == 'concat':
            # Concatenate channels, then project to single channel
            self.fusion_layer = nn.Sequential(
                self.ConvND(self.input_channels, 1, kernel_size=1, padding=0),
                nn.ReLU(),
                self.BatchNormND(1)
            )
        elif self.fusion_method == 'weighted_sum':
            # Learnable weighted combination
            self.fusion_weights = nn.Parameter(torch.ones(self.input_channels) / self.input_channels)
        elif self.fusion_method == 'multiply':
            # Element-wise multiplication (for 2 channels)
            if self.input_channels != 2:
                raise ValueError("Multiply fusion only supports 2 channels")
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
    
    def _calculate_encoder_depth(self):
        """Calculate required encoder depth based on input size."""
        # Calculate how many downsampling steps we need to get to ~4x4x4
        # Each downsampling step reduces size by factor of 2
        max_dim = max(self.input_size)
        
        # Calculate number of downsampling steps needed
        # We want to end up around 4x4x4, so log2(max_dim/4) steps
        import math
        depth = int(math.log2(max_dim / 4))
        
        # Ensure minimum depth of 3 and maximum of 6
        depth = max(3, min(depth, 6))
        
        print(f"Input size: {self.input_size} -> Encoder depth: {depth}")
        return depth
    
    def _build_encoder(self):
        """Build the encoder network based on architecture type."""
        if self.architecture == 'conv':
            return self._build_conv_encoder()
        elif self.architecture == 'resnet':
            return self._build_resnet_encoder()
        elif self.architecture == 'unet':
            return self._build_unet_encoder()
        elif self.architecture == 'mlp':
            return self._build_mlp_encoder()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _build_conv_encoder(self):
        """Build standard Conv3D encoder with adaptive depth."""
        layers = []
        
        # First block (always present)
        layers.extend([
            self.ConvND(self.input_channels, self.base_filters,
                        kernel_size=7, stride=2, padding=3, bias=False),
            self.BatchNormND(self.base_filters),
            nn.ReLU(inplace=True)
        ])

        # Add downsampling blocks based on calculated depth
        current_filters = self.base_filters
        for i in range(1, self.encoder_depth):
            next_filters = min(current_filters * 2, self.base_filters * 8)
            layers.extend([
                self.ConvND(current_filters, next_filters,
                            kernel_size=3, stride=2, padding=1, bias=False),
                self.BatchNormND(next_filters),
                nn.ReLU(inplace=True)
            ])
            current_filters = next_filters

        # Global average pooling and final layers
        layers.extend([
            self.AdaptiveAvgPoolND(1),
            nn.Flatten(),
            nn.Linear(current_filters, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_features)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_resnet_encoder(self):
        """Build ResNet-based encoder with adaptive depth."""
        layers = []
        model_cfg = self.config.get('model', {}) if isinstance(self.config, dict) else {}
        legacy_first_block_downsample = bool(model_cfg.get('legacy_resnet_first_block_downsample', False))

        # Initial convolution
        layers.extend([
            self.ConvND(self.input_channels, self.base_filters,
                        kernel_size=7, stride=2, padding=3, bias=False),
            self.BatchNormND(self.base_filters),
            nn.ReLU(inplace=True),
            self.MaxPoolND(kernel_size=3, stride=2, padding=1)
        ])

        # Add ResNet blocks based on calculated depth
        current_filters = self.base_filters
        for i in range(self.encoder_depth):
            if i == 0:
                if legacy_first_block_downsample:
                    next_filters = min(current_filters * 2, self.base_filters * 8)
                    stride = 2
                else:
                    next_filters = current_filters
                    stride = 1
            else:
                next_filters = min(current_filters * 2, self.base_filters * 8)
                stride = 2

            layers.append(self.ResNetBlock(
                current_filters,
                next_filters,
                stride=stride,
                use_bottleneck=self.use_bottleneck,
                dropout=self.dropout_features,
            ))
            current_filters = next_filters

        # SE attention
        layers.append(self.SEBlock(current_filters))

        # Global average pooling and final layers
        layers.extend([
            self.AdaptiveAvgPoolND(1),
            nn.Flatten(),
            nn.Linear(current_filters, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_features)
        ])

        return nn.Sequential(*layers)

    def _build_unet_encoder(self):
        """Build UNet-based encoder."""
        ConvBlock = self.ConvBlock
        MaxPool = self.MaxPoolND
        return nn.Sequential(
            ConvBlock(self.input_channels, self.base_filters, dropout=self.dropout_features),
            MaxPool(2),

            ConvBlock(self.base_filters, self.base_filters * 2, dropout=self.dropout_features),
            MaxPool(2),

            ConvBlock(self.base_filters * 2, self.base_filters * 4, dropout=self.dropout_features),
            MaxPool(2),

            ConvBlock(self.base_filters * 4, self.base_filters * 8, dropout=self.dropout_features),
            MaxPool(2),

            # Bottleneck
            ConvBlock(self.base_filters * 8, self.base_filters * 8, dropout=self.dropout_features),

            # Global average pooling
            self.AdaptiveAvgPoolND(1),
            nn.Flatten(),

            # Feature projection
            nn.Linear(self.base_filters * 8, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_features)
        )

    def _build_mlp_encoder(self):
        """Build MLP-based encoder with adaptive depth."""
        # Calculate input size for MLP
        total_input_size = self.input_channels
        for dim in self.input_size:
            total_input_size *= dim
        
        # Calculate hidden layer sizes based on depth
        hidden_sizes = []
        current_size = total_input_size
        
        # Create hidden layers with decreasing size
        for i in range(self.encoder_depth):
            # Reduce size by factor of 2 each layer, but not below feature_dim
            next_size = max(current_size // 2, self.feature_dim)
            hidden_sizes.append(next_size)
            current_size = next_size
        
        layers = []
        
        # Input layer
        layers.append(nn.Flatten())
        
        # Hidden layers
        prev_size = total_input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_features)
            ])
            prev_size = hidden_size
        
        # Final projection to feature_dim
        if prev_size != self.feature_dim:
            layers.extend([
                nn.Linear(prev_size, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_features)
            ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build the decoder network based on architecture type."""
        if self.architecture == 'mlp':
            return self._build_mlp_decoder()
        else:
            return self._build_conv_decoder()

    def _make_decoder_norm(self, channels: int) -> nn.Module:
        """
        Decoder normalization uses group normalization to avoid running statistics while
        remaining stable across a wide range of batch sizes and multi-GPU usage.
        """
        num_groups = min(32, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        if num_groups <= 0:
            num_groups = 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    
    def _build_conv_decoder(self):
        """Build Conv3D-based decoder with optional bilinear upsampling."""
        if self.bilinear_upsampling:
            return self._build_bilinear_decoder()
        else:
            return self._build_transposed_conv_decoder()
    
    def _build_transposed_conv_decoder(self):
        """Build decoder using transposed convolutions."""
        initial_spatial = (4,) * self.dim
        layers = [
            nn.Linear(self.latent_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder),

            nn.Linear(self.feature_dim, self.base_filters * 8 * int(math.prod(initial_spatial))),
            nn.ReLU(),
            nn.Unflatten(1, (self.base_filters * 8, *initial_spatial))
        ]

        transposed_blocks = [
            (self.base_filters * 8, self.base_filters * 4),
            (self.base_filters * 4, self.base_filters * 2),
            (self.base_filters * 2, self.base_filters)
        ]

        for in_ch, out_ch in transposed_blocks:
            layers.extend([
                self.ConvTransposeND(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                self._make_decoder_norm(out_ch),
                nn.ReLU(inplace=True)
            ])

        layers.extend([
            self.ConvTransposeND(self.base_filters, self.output_channels,
                                 kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        ])

        return nn.Sequential(*layers)
    
    def _build_bilinear_decoder(self):
        """Build decoder using bilinear upsampling + 1x1 convolutions."""
        initial_spatial = (4,) * self.dim
        layers = [
            nn.Linear(self.latent_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder),

            nn.Linear(self.feature_dim, self.base_filters * 8 * int(math.prod(initial_spatial))),
            nn.ReLU(),
            nn.Unflatten(1, (self.base_filters * 8, *initial_spatial))
        ]

        layers.extend([
            self._create_bilinear_upsample_block(self.base_filters * 8, self.base_filters * 4),
            self._create_bilinear_upsample_block(self.base_filters * 4, self.base_filters * 2),
            self._create_bilinear_upsample_block(self.base_filters * 2, self.base_filters),
            self._create_bilinear_upsample_block(self.base_filters, self.output_channels, final=True)
        ])

        return nn.Sequential(*layers)
    
    def _create_bilinear_upsample_block(self, in_channels, out_channels, final=False):
        """Create a bilinear upsampling block."""
        if final:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
                self.ConvND(in_channels, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
                self.ConvND(in_channels, out_channels, kernel_size=1, bias=False),
                self._make_decoder_norm(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def _build_mlp_decoder(self):
        """Build MLP-based decoder."""
        # Calculate output size
        total_output_size = self.output_channels
        for dim in self.input_size:
            total_output_size *= dim
        
        # Calculate hidden layer sizes (reverse of encoder)
        hidden_sizes = []
        current_size = self.feature_dim
        
        # Create hidden layers with increasing size
        for i in range(self.encoder_depth):
            next_size = min(current_size * 2, total_output_size // 2)
            hidden_sizes.append(next_size)
            current_size = next_size
        
        layers = []
        
        # Expand from latent space
        layers.extend([
            nn.Linear(self.latent_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_decoder)
        ])
        
        # Hidden layers
        prev_size = self.feature_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_decoder)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, total_output_size),
            nn.Sigmoid()  # Output in [0, 1] range
        ])
        
        return nn.Sequential(*layers)

    def _copy_tensor_if_match(self, tensor: torch.Tensor, weights: Dict[str, torch.Tensor], key: str) -> int:
        if key not in weights or tensor.shape != weights[key].shape:
            return 0
        tensor.data.copy_(weights[key])
        return 1

    def _copy_batchnorm_if_match(self, bn_module: nn.BatchNorm3d, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        loaded = 0
        loaded += self._copy_tensor_if_match(bn_module.weight, weights, f"{prefix}.weight")
        loaded += self._copy_tensor_if_match(bn_module.bias, weights, f"{prefix}.bias")
        loaded += self._copy_tensor_if_match(bn_module.running_mean, weights, f"{prefix}.running_mean")
        loaded += self._copy_tensor_if_match(bn_module.running_var, weights, f"{prefix}.running_var")
        key = f"{prefix}.num_batches_tracked"
        if key in weights and hasattr(bn_module, "num_batches_tracked"):
            bn_module.num_batches_tracked.copy_(weights[key])
            loaded += 1
        return loaded

    def _load_medicalnet_block(self, block: nn.Module, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        loaded = 0
        if not isinstance(block, ResNetBlock3D):
            return loaded
        loaded += self._copy_tensor_if_match(block.conv1.weight, weights, f"{prefix}.conv1.weight")
        loaded += self._copy_batchnorm_if_match(block.bn1, weights, f"{prefix}.bn1")
        loaded += self._copy_tensor_if_match(block.conv2.weight, weights, f"{prefix}.conv2.weight")
        loaded += self._copy_batchnorm_if_match(block.bn2, weights, f"{prefix}.bn2")

        if hasattr(block, "conv3"):  # Bottleneck case
            loaded += self._copy_tensor_if_match(block.conv3.weight, weights, f"{prefix}.conv3.weight")
            loaded += self._copy_batchnorm_if_match(block.bn3, weights, f"{prefix}.bn3")

        if hasattr(block, "shortcut") and isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) >= 2:
            loaded += self._copy_tensor_if_match(block.shortcut[0].weight, weights, f"{prefix}.downsample.0.weight")
            if hasattr(block.shortcut[0], "bias"):
                loaded += self._copy_tensor_if_match(block.shortcut[0].bias, weights, f"{prefix}.downsample.0.bias")
            if isinstance(block.shortcut[1], nn.BatchNorm3d):
                loaded += self._copy_batchnorm_if_match(block.shortcut[1], weights, f"{prefix}.downsample.1")

        return loaded

    def _load_medicalnet_into_resnet_encoder(self, weights: Dict[str, torch.Tensor]) -> int:
        encoder = getattr(self, "encoder", None)
        if encoder is None or not isinstance(encoder, nn.Sequential):
            return 0

        loaded = 0
        loaded += self._copy_tensor_if_match(encoder[0].weight, weights, "conv1.weight")
        if isinstance(encoder[1], nn.BatchNorm3d):
            loaded += self._copy_batchnorm_if_match(encoder[1], weights, "bn1")

        block_modules = [module for module in encoder if isinstance(module, ResNetBlock3D)]
        stage_names = ["layer1.0", "layer2.0", "layer3.0", "layer4.0"]

        for stage_name, block in zip(stage_names, block_modules):
            loaded += self._load_medicalnet_block(block, weights, stage_name)

        return loaded

    def _load_medicalnet_into_resnet_unet(self, weights: Dict[str, torch.Tensor]) -> int:
        core = getattr(self, "resnet_unet_core", None)
        if core is None:
            return 0

        loaded = 0
        initial_conv = getattr(core, "initial_conv", None)
        if isinstance(initial_conv, nn.Sequential) and len(initial_conv) >= 2:
            loaded += self._copy_tensor_if_match(initial_conv[0].weight, weights, "conv1.weight")
            if isinstance(initial_conv[1], nn.BatchNorm3d):
                loaded += self._copy_batchnorm_if_match(initial_conv[1], weights, "bn1")

        stage_specs = [
            ("encoder1", "layer1"),
            ("encoder2", "layer2"),
            ("encoder3", "layer3"),
            ("encoder4", "layer4"),
        ]

        for stage_attr, layer_name in stage_specs:
            stage = getattr(core, stage_attr, None)
            if stage is None:
                continue
            for block_idx, block in enumerate(stage):
                prefix = f"{layer_name}.{block_idx}"
                loaded += self._load_medicalnet_block(block, weights, prefix)

        return loaded

    def _load_weights_by_shape(self, module: Optional[nn.Module], weights: Dict[str, torch.Tensor]) -> int:
        if module is None:
            return 0

        available = {
            key: tensor for key, tensor in weights.items() if torch.is_tensor(tensor)
        }

        loaded = 0
        for _, param in module.named_parameters(recurse=True):
            if not torch.is_tensor(param) or param.numel() == 0:
                continue
            matched_key = None
            for key, tensor in available.items():
                if tensor.shape == param.shape:
                    param.data.copy_(tensor.to(param.device, dtype=param.dtype))
                    matched_key = key
                    loaded += 1
                    break
            if matched_key:
                del available[matched_key]

        for _, buffer in module.named_buffers(recurse=True):
            if not torch.is_tensor(buffer) or buffer.numel() == 0:
                continue
            matched_key = None
            for key, tensor in available.items():
                if tensor.shape == buffer.shape:
                    buffer.data.copy_(tensor.to(buffer.device, dtype=buffer.dtype))
                    matched_key = key
                    break
            if matched_key:
                del available[matched_key]
                loaded += 1

        return loaded

    def _apply_pretrained_encoder(self) -> None:
        pre_cfg = self.config.get("pretrained_encoder") if isinstance(self.config, dict) else None
        if not pre_cfg or not pre_cfg.get("use_pretrained_ct_encoder", False):
            return

        source = str(pre_cfg.get("source", "medicalnet")).lower()
        logger = logging.getLogger(__name__)
        loaded = 0
        weight_path: Optional[Path] = None

        if source == "medicalnet":
            depth = int(pre_cfg.get("model_depth", 18))
            weight_path_cfg = pre_cfg.get("ct_encoder_path")
            if weight_path_cfg:
                weight_path = Path(weight_path_cfg)
            else:
                filename = f"resnet_{depth}_23dataset.pth" if depth in {10, 18, 34, 50} else f"resnet_{depth}.pth"
                weight_path = Path(__file__).resolve().parents[2] / "external" / "MedicalNet" / "pretrain" / filename

            if not weight_path.exists():
                raise FileNotFoundError(f"MedicalNet checkpoint not found at {weight_path}")

            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            medical_weights = {
                (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
            }

            if self.uses_resnet_unet:
                loaded = self._load_medicalnet_into_resnet_unet(medical_weights)
            elif self.architecture == "resnet":
                loaded = self._load_medicalnet_into_resnet_encoder(medical_weights)
            else:
                logger.warning(
                    "MedicalNet pretraining currently supported only for ResNet-based encoders; skipping."
                )
                return

            logger.info(
                "MedicalNet encoder initialisation loaded %d parameter tensors from %s",
                loaded,
                weight_path,
            )

        elif source == "models_genesis":
            variant = str(pre_cfg.get("ssl_variant", "genesis")).lower()
            candidate_paths: List[Path] = []
            weight_path_cfg = pre_cfg.get("ct_encoder_path")
            if weight_path_cfg:
                candidate_paths.append(Path(weight_path_cfg))
            else:
                base_root = Path(__file__).resolve().parents[2] / "external" / "ModelsGenesis"
                if variant == "genesis":
                    filename = "Genesis_Chest_CT.pt"
                elif variant == "transvw":
                    filename = "TransVW_Chest_CT.pt"
                else:
                    raise ValueError(f"Unsupported SSL variant '{variant}' for Models Genesis weights.")
                candidate_paths.extend([
                    base_root / "pretrained" / filename,
                    base_root / filename,
                    base_root / "weights" / filename,
                ])

            weight_path = next((path for path in candidate_paths if path.exists()), None)
            if weight_path is None:
                raise FileNotFoundError(
                    "Models Genesis checkpoint not found. Checked: "
                    + ", ".join(str(p) for p in candidate_paths)
                )

            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            genesis_weights = {
                (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
            }

            loaded = 0
            if self.architecture in {"unet", "conv"}:
                loaded += self._load_weights_by_shape(getattr(self, "encoder", None), genesis_weights)
                loaded += self._load_weights_by_shape(getattr(self, "decoder", None), genesis_weights)
            else:
                logger.warning(
                    "Models Genesis pretraining currently supported only for 'unet' and 'conv' encoders; skipping."
                )
                return

            logger.info(
                "Models Genesis (%s) encoder initialisation loaded %d parameter tensors from %s",
                variant,
                loaded,
                weight_path,
            )

        else:
            logger.warning(
                "Unsupported pretrained encoder source '%s'; expected 'medicalnet' or 'models_genesis'.",
                source,
            )
            return

        if pre_cfg.get("freeze_ct_encoder", False):
            modules_to_freeze = []
            if self.uses_resnet_unet and hasattr(self, "resnet_unet_core"):
                modules_to_freeze.append(self.resnet_unet_core)
            elif hasattr(self, "encoder"):
                modules_to_freeze.append(self.encoder)
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
    
    def _fuse_inputs(self, inputs):
        """Fuse multiple input channels based on fusion method."""
        if self.input_mode in ['dose_only', 'ct_only', 'fused_only']:
            return inputs[0]  # Single channel input
        
        if self.fusion_method == 'concat':
            # Concatenate along channel dimension
            fused = torch.cat(inputs, dim=1)
            return self.fusion_layer(fused)
        
        elif self.fusion_method == 'weighted_sum':
            # Weighted sum of channels
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * inp for w, inp in zip(weights, inputs))
            return fused
        
        elif self.fusion_method == 'multiply':
            # Element-wise multiplication (for 2 channels)
            if len(inputs) != 2:
                raise ValueError("Multiply fusion requires exactly 2 inputs")
            return inputs[0] * inputs[1]
        
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
    
    def forward(self, inputs, spatial_coords=None, **batch):
        """
        Forward pass.
        
        Args:
            inputs: List of input tensors based on input_mode
            spatial_coords: Spatial coordinates for attention [batch_size, num_patches, 3]
        
        Returns:
            reconstructed: Reconstructed output
            latent: Latent representation
        """
        # Assemble primary input tensor
        fused_input = batch.get('input')
        if fused_input is None:
            if isinstance(inputs, (list, tuple)):
                fused_input = self._fuse_inputs(list(inputs))
            else:
                fused_input = inputs

        if not torch.is_tensor(fused_input):
            raise TypeError("Expected tensor input after fusion.")

        self._latest_attention = None

        # Optional channel attention applied prior to encoding
        if self.channel_attention is not None and self.attention_type == 'channel':
            fused_input, channel_weights = self.channel_attention(fused_input)
            self._latest_attention = channel_weights
        
        backbone_reconstruction: Optional[torch.Tensor] = None
        if self.uses_resnet_unet:
            backbone_reconstruction, features = self.resnet_unet_core(fused_input)
            backbone_reconstruction = self._resize_to_target(backbone_reconstruction, fused_input)
        else:
            # Encode
            features = self.encoder(fused_input)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
        
        # Apply latent attention mechanisms
        if self.sequence_attention is not None and self.attention_type in {'multi_scale', 'high_dose'}:
            tokens = features.unsqueeze(1)
            if self.attention_type == 'high_dose':
                dose_map = batch.get('dose') or batch.get('dose_patches') or batch.get('dose_image')
                attended, weights = self.sequence_attention(tokens, dose_map=None)
            else:
                attended, weights = self.sequence_attention(tokens)
            features = attended.squeeze(1)
            self._latest_attention = weights.squeeze() if weights is not None else None
        elif self.spatial_context_attention is not None and self.attention_type == 'spatial':
            coords = spatial_coords if spatial_coords is not None else batch.get('spatial_coords')
            if coords is not None:
                coords = coords.to(features.device)
                features, weights = self.spatial_context_attention(features, coords)
                self._latest_attention = weights
        elif self.lobe_attention is not None and self.attention_type == 'lobe':
            lobe_index = batch.get('lobe_index')
            if lobe_index is None:
                raise ValueError("Lobe attention requires 'lobe_index' in batch.")
            side_type_index = batch.get('side_type_index')
            anatomical_side_index = batch.get('anatomical_side_index')

            lobe_index = lobe_index.to(features.device)
            side_type_index = side_type_index.to(features.device) if torch.is_tensor(side_type_index) else None
            anatomical_side_index = (
                anatomical_side_index.to(features.device)
                if torch.is_tensor(anatomical_side_index) else None
            )

            features, weights = self.lobe_attention(
                features,
                lobe_index,
                side_type_index=side_type_index,
                anatomical_side_index=anatomical_side_index,
            )
            self._latest_attention = weights
        
        # Project to latent space
        latent = self.latent_projection(features)
        
        # Decode or reuse backbone reconstruction
        if self.uses_resnet_unet:
            reconstructed = backbone_reconstruction.contiguous()
        else:
            reconstructed = self.decoder(latent)
            reconstructed = self._resize_to_target(reconstructed, fused_input)
            reconstructed = reconstructed.contiguous()
        
        outputs = {
            'reconstruction': reconstructed,
            'latent': latent
        }
        
        if self.high_dose_predictor is not None:
            outputs['high_dose_logits'] = self.high_dose_predictor(latent)
        if self.dose_level_classifier is not None:
            outputs['dose_level_logits'] = self.dose_level_classifier(latent)
        if hasattr(self, '_latest_attention'):
            outputs['attention_weights'] = self._latest_attention
        
        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        model_cfg = self.config.get('model', {}) if isinstance(self.config, dict) else {}
        target_keys = list(model_cfg.get('target_keys', [])) if isinstance(model_cfg, dict) else []
        target_keys += ['target', 'dose', 'dose_patches', 'dose_image']

        target_tensor = None
        for key in target_keys:
            tensor = batch.get(key)
            if torch.is_tensor(tensor):
                target_tensor = tensor
                break

        if target_tensor is None:
            raise ValueError('No suitable target tensor found in batch for reconstruction loss.')

        reconstruction = outputs.get('reconstruction')
        if reconstruction is None:
            raise ValueError("Model outputs missing 'reconstruction' entry.")

        losses['reconstruction'] = F.mse_loss(reconstruction, target_tensor)

        if 'high_dose_logits' in outputs:
            aux_cfg = model_cfg.get('auxiliary', {}) if isinstance(model_cfg, dict) else {}
            label_key = aux_cfg.get('high_dose_label_key', 'high_dose_labels')
            labels = batch.get(label_key)
            if torch.is_tensor(labels):
                losses['high_dose'] = F.cross_entropy(outputs['high_dose_logits'], labels)

        if 'dose_level_logits' in outputs:
            aux_cfg = model_cfg.get('auxiliary', {}) if isinstance(model_cfg, dict) else {}
            label_key = aux_cfg.get('dose_level_label_key', 'dose_level_labels')
            labels = batch.get(label_key)
            if torch.is_tensor(labels):
                losses['dose_level'] = F.cross_entropy(outputs['dose_level_logits'], labels)

        losses['total'] = sum(losses.values())
        return losses


# Factory functions for backward compatibility
def create_attention_autoencoder(**kwargs):
    """Create attention autoencoder (1 channel CT)."""
    return ConfigurableAutoencoder(input_mode='ct_only', **kwargs)

def create_dual_input_autoencoder(**kwargs):
    """Create dual input autoencoder (2 channels CT+dose)."""
    return ConfigurableAutoencoder(input_mode='dose_ct', **kwargs)

def create_multi_channel_autoencoder(**kwargs):
    """Create multi-channel autoencoder (configurable channels)."""
    return ConfigurableAutoencoder(**kwargs)

def create_fused_autoencoder(**kwargs):
    """Create fused autoencoder (1 channel fused CT+dose)."""
    return ConfigurableAutoencoder(input_mode='fused_only', **kwargs)

def create_patches_only_autoencoder(**kwargs):
    """Create patches-only autoencoder (1 channel CT patches)."""
    return ConfigurableAutoencoder(input_mode='ct_only', **kwargs)
