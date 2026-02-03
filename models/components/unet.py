"""
UNet architecture implementation for 3D medical image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class UNet3D(nn.Module):
    """
    3D UNet architecture for medical image processing.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_filters: int = 32,
                 depth: int = 4,
                 dropout: float = 0.0,
                 use_attention: bool = False):
        """
        Initialize 3D UNet.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_filters: Number of base filters
            depth: Network depth
            dropout: Dropout rate
            use_attention: Whether to use attention in skip connections
        """
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoder_blocks.append(Conv3DBlock(in_ch, out_ch, dropout=dropout))
            if i < depth - 1:  # No pooling after last encoder block
                self.pool_layers.append(nn.MaxPool3d(2))
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = Conv3DBlock(in_ch, bottleneck_ch, dropout=dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upconv_layers.append(nn.ConvTranspose3d(bottleneck_ch, out_ch, 2, 2))
            self.decoder_blocks.append(Conv3DBlock(bottleneck_ch, out_ch, dropout=dropout))
            bottleneck_ch = out_ch
        
        # Final output layer
        self.final_conv = nn.Conv3d(base_filters, out_channels, 1)
        
        # Attention modules for skip connections (optional)
        if use_attention:
            self.attention_modules = nn.ModuleList()
            for i in range(depth - 1):
                ch = base_filters * (2 ** i)
                self.attention_modules.append(
                    nn.Sequential(
                        nn.Conv3d(ch * 2, ch, 1),
                        nn.Sigmoid()
                    )
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        for i, (encoder_block, pool_layer) in enumerate(zip(self.encoder_blocks, self.pool_layers + [None])):
            x = encoder_block(x)
            skip_connections.append(x)
            if pool_layer is not None:
                x = pool_layer(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            x = upconv(x)
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Apply attention if enabled
            if self.use_attention and i < len(self.attention_modules):
                concat_features = torch.cat([x, skip], dim=1)
                attention_weights = self.attention_modules[i](concat_features)
                skip = skip * attention_weights
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Final output
        x = self.final_conv(x)
        return x


class UNetEncoder(nn.Module):
    """UNet encoder for feature extraction."""
    
    def __init__(self, in_channels: int, base_filters: int = 32, 
                 depth: int = 4, dropout: float = 0.0):
        super().__init__()
        
        self.depth = depth
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoder_blocks.append(Conv3DBlock(in_ch, out_ch, dropout=dropout))
            if i < depth - 1:
                self.pool_layers.append(nn.MaxPool3d(2))
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = Conv3DBlock(in_ch, bottleneck_ch, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        for i, (encoder_block, pool_layer) in enumerate(zip(self.encoder_blocks, self.pool_layers + [None])):
            x = encoder_block(x)
            skip_connections.append(x)
            if pool_layer is not None:
                x = pool_layer(x)
        
        x = self.bottleneck(x)
        return x, skip_connections


class UNetDecoder(nn.Module):
    """UNet decoder for feature reconstruction."""
    
    def __init__(self, out_channels: int, base_filters: int = 32, 
                 depth: int = 4, dropout: float = 0.0, use_attention: bool = False):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        bottleneck_ch = base_filters * (2 ** depth)
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upconv_layers.append(nn.ConvTranspose3d(bottleneck_ch, out_ch, 2, 2))
            self.decoder_blocks.append(Conv3DBlock(bottleneck_ch, out_ch, dropout=dropout))
            bottleneck_ch = out_ch
        
        # Final output layer
        self.final_conv = nn.Conv3d(base_filters, out_channels, 1)
        
        # Attention modules for skip connections (optional)
        if use_attention:
            self.attention_modules = nn.ModuleList()
            for i in range(depth - 1):
                ch = base_filters * (2 ** i)
                self.attention_modules.append(
                    nn.Sequential(
                        nn.Conv3d(ch * 2, ch, 1),
                        nn.Sigmoid()
                    )
                )
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            x = upconv(x)
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Apply attention if enabled
            if self.use_attention and i < len(self.attention_modules):
                concat_features = torch.cat([x, skip], dim=1)
                attention_weights = self.attention_modules[i](concat_features)
                skip = skip * attention_weights
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Final output
        x = self.final_conv(x)
        return x
