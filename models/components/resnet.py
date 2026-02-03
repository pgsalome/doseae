"""
ResNet architecture implementation for 3D medical image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ResNetBlock3D(nn.Module):
    """3D ResNet block with optional bottleneck."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, use_bottleneck: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        
        self.use_bottleneck = use_bottleneck
        self.stride = stride
        
        if use_bottleneck:
            # Bottleneck block
            bottleneck_channels = out_channels // 4
            self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(bottleneck_channels)
            self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm3d(bottleneck_channels)
            self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm3d(out_channels)
        else:
            # Basic block
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
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


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResNet3D(nn.Module):
    """
    3D ResNet architecture for medical image processing.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 64,
                 layers: List[int] = [2, 2, 2, 2],
                 use_bottleneck: bool = False,
                 use_se: bool = False,
                 dropout: float = 0.0):
        """
        Initialize 3D ResNet.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_filters: Number of base filters
            layers: Number of blocks in each layer
            use_bottleneck: Whether to use bottleneck blocks
            use_se: Whether to use SE blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = base_filters
        self.use_se = use_se
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, base_filters, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(base_filters, layers[0], use_bottleneck, dropout)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], use_bottleneck, dropout, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], use_bottleneck, dropout, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], use_bottleneck, dropout, stride=2)
        
        # Global average pooling and final layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_filters * 8, out_channels)
    
    def _make_layer(self, out_channels: int, blocks: int, 
                   use_bottleneck: bool, dropout: float, stride: int = 1) -> nn.Module:
        layers = []
        layers.append(ResNetBlock3D(self.in_channels, out_channels, stride, use_bottleneck, dropout))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock3D(out_channels, out_channels, 1, use_bottleneck, dropout))
        
        layer = nn.Sequential(*layers)
        
        # Add SE block if enabled
        if self.use_se:
            layer = nn.Sequential(layer, SEBlock3D(out_channels))
        
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetEncoder3D(nn.Module):
    """3D ResNet encoder for feature extraction."""
    
    def __init__(self, in_channels: int, base_filters: int = 64,
                 layers: List[int] = [2, 2, 2, 2],
                 use_bottleneck: bool = False,
                 use_se: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        
        self.in_channels = base_filters
        self.use_se = use_se
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, base_filters, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(base_filters, layers[0], use_bottleneck, dropout)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], use_bottleneck, dropout, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], use_bottleneck, dropout, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], use_bottleneck, dropout, stride=2)
    
    def _make_layer(self, out_channels: int, blocks: int, 
                   use_bottleneck: bool, dropout: float, stride: int = 1) -> nn.Module:
        layers = []
        layers.append(ResNetBlock3D(self.in_channels, out_channels, stride, use_bottleneck, dropout))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock3D(out_channels, out_channels, 1, use_bottleneck, dropout))
        
        layer = nn.Sequential(*layers)
        
        # Add SE block if enabled
        if self.use_se:
            layer = nn.Sequential(layer, SEBlock3D(out_channels))
        
        return layer
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections.append(x)  # Store for skip connection
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        skip_connections.append(x)
        x = self.layer2(x)
        skip_connections.append(x)
        x = self.layer3(x)
        skip_connections.append(x)
        x = self.layer4(x)
        
        return x, skip_connections


class ResNetDecoder3D(nn.Module):
    """3D ResNet decoder for feature reconstruction."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 base_filters: int = 64,
                 use_attention: bool = False):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose3d(in_channels, base_filters * 4, 2, 2)
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, 2)
        self.upconv3 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, 2)
        self.upconv4 = nn.ConvTranspose3d(base_filters, base_filters // 2, 2, 2)
        
        # ResNet blocks for upsampling
        self.resblock1 = ResNetBlock3D(base_filters * 4, base_filters * 4)
        self.resblock2 = ResNetBlock3D(base_filters * 2, base_filters * 2)
        self.resblock3 = ResNetBlock3D(base_filters, base_filters)
        self.resblock4 = ResNetBlock3D(base_filters // 2, base_filters // 2)
        
        # Final output layer
        self.final_conv = nn.Conv3d(base_filters // 2, out_channels, 1)
        
        # Attention modules for skip connections (optional)
        if use_attention:
            self.attention_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(base_filters * 4 * 2, base_filters * 4, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Conv3d(base_filters * 2 * 2, base_filters * 2, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Conv3d(base_filters * 2, base_filters, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Conv3d(base_filters, base_filters // 2, 1),
                    nn.Sigmoid()
                )
            ])
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Upsampling with skip connections
        x = self.upconv1(x)
        if x.shape != skip_connections[0].shape:
            x = F.interpolate(x, size=skip_connections[0].shape[2:], mode='trilinear', align_corners=False)
        
        if self.use_attention:
            concat_features = torch.cat([x, skip_connections[0]], dim=1)
            attention_weights = self.attention_modules[0](concat_features)
            skip = skip_connections[0] * attention_weights
        else:
            skip = skip_connections[0]
        
        x = torch.cat([x, skip], dim=1)
        x = self.resblock1(x)
        
        x = self.upconv2(x)
        if x.shape != skip_connections[1].shape:
            x = F.interpolate(x, size=skip_connections[1].shape[2:], mode='trilinear', align_corners=False)
        
        if self.use_attention:
            concat_features = torch.cat([x, skip_connections[1]], dim=1)
            attention_weights = self.attention_modules[1](concat_features)
            skip = skip_connections[1] * attention_weights
        else:
            skip = skip_connections[1]
        
        x = torch.cat([x, skip], dim=1)
        x = self.resblock2(x)
        
        x = self.upconv3(x)
        if x.shape != skip_connections[2].shape:
            x = F.interpolate(x, size=skip_connections[2].shape[2:], mode='trilinear', align_corners=False)
        
        if self.use_attention:
            concat_features = torch.cat([x, skip_connections[2]], dim=1)
            attention_weights = self.attention_modules[2](concat_features)
            skip = skip_connections[2] * attention_weights
        else:
            skip = skip_connections[2]
        
        x = torch.cat([x, skip], dim=1)
        x = self.resblock3(x)
        
        x = self.upconv4(x)
        if x.shape != skip_connections[3].shape:
            x = F.interpolate(x, size=skip_connections[3].shape[2:], mode='trilinear', align_corners=False)
        
        if self.use_attention:
            concat_features = torch.cat([x, skip_connections[3]], dim=1)
            attention_weights = self.attention_modules[3](concat_features)
            skip = skip_connections[3] * attention_weights
        else:
            skip = skip_connections[3]
        
        x = torch.cat([x, skip], dim=1)
        x = self.resblock4(x)
        
        # Final output
        x = self.final_conv(x)
        return x
