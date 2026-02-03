"""
Advanced ResNet-UNet with Multi-Scale Attention for CT+Dose prediction.
Optimized for accuracy, training speed, and interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialEmbedding(nn.Module):
    """Spatial coordinate embedding with positional encoding."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(3, d_model)
        
        # Add positional encoding for better spatial understanding
        self.pos_encoding = nn.Parameter(torch.randn(1, d_model) * 0.1)
    
    def forward(self, spatial_coords):
        embedded = self.embedding(spatial_coords)
        return embedded + self.pos_encoding


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def _make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    if groups <= 0:
        groups = 1
    return nn.GroupNorm(groups, channels)


class ResNetBlock3D(nn.Module):
    """3D ResNet block with SE attention."""
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_group_norm(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    _make_group_norm(out_channels)
                )
        
        # SE attention
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += residual
        out = F.relu(out)
        return out


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for different anatomical regions."""
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        
        # Multi-scale feature extraction
        self.scale_conv1 = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)
        self.scale_conv2 = nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.scale_conv3 = nn.Conv3d(feature_dim, feature_dim, kernel_size=5, padding=2)
        
    def forward(self, x):
        batch_size, num_patches, feature_dim = x.shape
        
        # Multi-scale feature extraction (simplified approach)
        # Instead of spatial reshaping, use 1D convolutions for multi-scale
        x_flat = x.view(-1, feature_dim, 1, 1, 1)  # [batch*patches, feature_dim, 1, 1, 1]
        
        # Apply multi-scale convolutions
        scale1 = self.scale_conv1(x_flat)
        scale2 = self.scale_conv2(x_flat)
        scale3 = self.scale_conv3(x_flat)
        
        multi_scale = (scale1 + scale2 + scale3) / 3
        x_enhanced = multi_scale.view(batch_size, num_patches, feature_dim)
        
        # Multi-head self-attention
        q = self.q_linear(x_enhanced).view(batch_size, num_patches, self.num_heads, self.head_dim)
        k = self.k_linear(x_enhanced).view(batch_size, num_patches, self.num_heads, self.head_dim)
        v = self.v_linear(x_enhanced).view(batch_size, num_patches, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, patches, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)  # [batch, heads, patches, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_patches, feature_dim)
        
        # Output projection
        output = self.out_linear(attended)
        
        return output, attention_weights.mean(dim=1)  # Average across heads


class ResNetUNet3D(nn.Module):
    """ResNet-UNet with SE attention and multi-scale features."""
    
    def __init__(self, n_channels=1, n_classes=1, base_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.initial_conv = nn.Sequential(
            nn.Conv3d(n_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            _make_group_norm(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = nn.Sequential(
            ResNetBlock3D(base_filters, base_filters),
            ResNetBlock3D(base_filters, base_filters)
        )
        
        self.encoder2 = nn.Sequential(
            ResNetBlock3D(base_filters, base_filters * 2, stride=2),
            ResNetBlock3D(base_filters * 2, base_filters * 2)
        )
        
        self.encoder3 = nn.Sequential(
            ResNetBlock3D(base_filters * 2, base_filters * 4, stride=2),
            ResNetBlock3D(base_filters * 4, base_filters * 4)
        )
        
        self.encoder4 = nn.Sequential(
            ResNetBlock3D(base_filters * 4, base_filters * 8, stride=2),
            ResNetBlock3D(base_filters * 8, base_filters * 8)
        )
        
        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters * 4, base_filters * 4)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters * 2, base_filters * 2)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters, base_filters)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(base_filters, base_filters, kernel_size=2, stride=2),
            ResNetBlock3D(base_filters, base_filters)
        )
        
        self.final_conv = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x0 = self.initial_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Decoder with skip connections (handle size mismatches)
        d4 = self.decoder4(x4)
        # Handle size mismatch for skip connection
        if d4.shape != x3.shape:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='trilinear', align_corners=False)
        d4 = d4 + x3  # Skip connection
        
        d3 = self.decoder3(d4)
        if d3.shape != x2.shape:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='trilinear', align_corners=False)
        d3 = d3 + x2  # Skip connection
        
        d2 = self.decoder2(d3)
        if d2.shape != x1.shape:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='trilinear', align_corners=False)
        d2 = d2 + x1  # Skip connection
        
        d1 = self.decoder1(d2)
        if d1.shape != x0.shape:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='trilinear', align_corners=False)
        d1 = d1 + x0  # Skip connection
        
        output = self.final_conv(d1)
        return output


class DoseAEResUNet(nn.Module):
    """
    DoseAE ResNet-UNet with Multi-Scale Attention for CT+Dose prediction.
    
    Features:
    - ResNet blocks with SE attention
    - Multi-scale attention between patches
    - Spatial coordinate embedding
    - High-dose region prediction
    - Interpretable attention maps
    """
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 base_filters=64,
                 feature_dim=256,
                 attention_heads=4,
                 num_patches_per_patient=50,
                 latent_dim=8,  # NEW: Ultra-compressed latent space
                 projection_sizes=[256, 128, 64, 32, 16, 8, 4]):  # NEW: Configurable projection sizes
        super().__init__()
        
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_patches_per_patient = num_patches_per_patient
        
        # ResNet-UNet for patch encoding/decoding
        self.resnet_unet_encoder = ResNetUNet3D(
            n_channels=input_channels, 
            n_classes=feature_dim, 
            base_filters=base_filters
        )
        
        # Multi-head projection layers for different 1D vector sizes
        # Each head projects from the rich bottleneck to a specific dimension
        self.projection_sizes = projection_sizes
        self.projection_heads = nn.ModuleDict({
            str(size): nn.Linear(feature_dim, size) for size in projection_sizes
        })
        
        self.resnet_unet_decoder = ResNetUNet3D(
            n_channels=feature_dim, 
            n_classes=output_channels, 
            base_filters=base_filters
        )
        
        # Spatial coordinate embedding
        self.spatial_embedding = SpatialEmbedding(feature_dim)
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(
            feature_dim=feature_dim,
            num_heads=attention_heads
        )
        
        # High-dose region predictor
        self.high_dose_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Dose level classifier (for interpretability)
        self.dose_level_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # 4 dose levels: none, low, medium, high
            nn.Softmax(dim=-1)
        )
        
    def forward(self, ct_patches, spatial_coords, is_high_dose=None):
        """
        Forward pass with interpretable outputs.
        
        Returns:
            predicted_dose: [batch_size, num_patches, channels, depth, height, width]
            attention_weights: [batch_size, num_patches, num_patches] (interpretable)
            high_dose_pred: [batch_size, num_patches, 1]
            dose_levels: [batch_size, num_patches, 4] (interpretable)
        """
        batch_size, num_patches = ct_patches.shape[:2]
        
        # Encode all patches using ResNet-UNet
        patch_features = []
        rich_bottlenecks = []  # Store rich bottleneck features for reconstruction
        multi_head_features = {}  # Store features from all projection heads
        
        for i in range(num_patches):
            patch = ct_patches[:, i]  # [batch_size, channels, depth, height, width]
            features = self.resnet_unet_encoder(patch)  # [batch_size, feature_dim, depth, height, width]
            
            # Store the rich bottleneck (before global pooling)
            rich_bottlenecks.append(features)  # [batch_size, feature_dim, depth, height, width]
            
            # Global average pooling to get feature vector for attention
            features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # [batch_size, feature_dim, 1, 1, 1]
            features = features.view(batch_size, self.feature_dim)  # [batch_size, feature_dim]
            
            # Generate multiple 1D vectors using different projection heads
            for head_size, head in self.projection_heads.items():
                if head_size not in multi_head_features:
                    multi_head_features[head_size] = []
                projected_vector = head(features)  # [batch_size, head_size]
                multi_head_features[head_size].append(projected_vector)
            
            patch_features.append(features)
        
        patch_features = torch.stack(patch_features, dim=1)  # [batch_size, num_patches, feature_dim]
        
        # Stack multi-head features
        for head_size in multi_head_features:
            multi_head_features[head_size] = torch.stack(multi_head_features[head_size], dim=1)  # [batch_size, num_patches, head_size]
        
        # Add spatial coordinate embedding
        spatial_embedded = self.spatial_embedding(spatial_coords)  # [batch_size, num_patches, feature_dim]
        
        # Combine patch features with spatial embedding
        enhanced_features = patch_features + spatial_embedded
        
        # Apply multi-scale attention
        attended_features, attention_weights = self.multi_scale_attention(enhanced_features)
        
        # Decode to dose patches using ResNet-UNet decoder
        # Use the rich bottlenecks directly for reconstruction (no compression/decompression)
        predicted_dose_patches = []
        for i in range(num_patches):
            # Use the rich bottleneck directly for reconstruction
            rich_bottleneck = rich_bottlenecks[i]  # [batch_size, feature_dim, depth, height, width]
            
            # Apply attention to the rich bottleneck
            # Reshape attended features to match bottleneck spatial dimensions
            attended_feature = attended_features[:, i]  # [batch_size, feature_dim]
            attended_feature_spatial = attended_feature.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            attended_feature_spatial = attended_feature_spatial.expand_as(rich_bottleneck)
            
            # Combine rich bottleneck with attended features
            enhanced_bottleneck = rich_bottleneck + attended_feature_spatial
            
            # Decode directly from the rich bottleneck
            dose_patch = self.resnet_unet_decoder(enhanced_bottleneck)
            predicted_dose_patches.append(dose_patch)
        
        predicted_dose = torch.stack(predicted_dose_patches, dim=1)
        predicted_dose = torch.sigmoid(predicted_dose)
        
        # Predict high-dose regions
        high_dose_pred = None
        if is_high_dose is not None:
            high_dose_pred = self.high_dose_predictor(attended_features)
        
        # Predict dose levels for interpretability
        dose_levels = self.dose_level_classifier(attended_features)
        
        return predicted_dose, attention_weights, high_dose_pred, dose_levels, multi_head_features
    
    def get_attention_visualization(self, ct_patches, spatial_coords):
        """Get attention weights for visualization and interpretability."""
        with torch.no_grad():
            _, attention_weights, _, dose_levels = self.forward(ct_patches, spatial_coords)
            return {
                'attention_weights': attention_weights,
                'dose_levels': dose_levels,
                'attention_map': attention_weights.mean(dim=0)  # Average across batch
            }


def create_doseae_resunet(config):
    """Create the advanced ResNet-UNet attention model."""
    model_config = config.get('model', {})
    
    model = DoseAEResUNet(
        input_channels=1,
        output_channels=1,
        base_filters=model_config.get('base_filters', 64),
        feature_dim=model_config.get('feature_dim', 256),
        attention_heads=model_config.get('attention_heads', 4),
        num_patches_per_patient=model_config.get('num_patches_per_patient', 50),
        latent_dim=model_config.get('latent_dim', 8),  # NEW: Ultra-compressed latent space
        projection_sizes=model_config.get('projection_sizes', [256, 128, 64, 32, 16, 8, 4])  # NEW: Configurable projection sizes
    )
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    config = {
        'model': {
            'base_filters': 64,
            'feature_dim': 256,
            'attention_heads': 4,
            'num_patches_per_patient': 50,
            'projection_sizes': [256, 128, 64, 32, 16, 8, 4]
        }
    }
    
    model = create_advanced_model(config)
    print(f"Advanced ResNet-UNet Attention Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 2
    num_patches = 10
    patch_size = (64, 64, 64)
    
    ct_patches = torch.randn(batch_size, num_patches, 1, *patch_size)
    spatial_coords = torch.randn(batch_size, num_patches, 3)
    is_high_dose = torch.randint(0, 2, (batch_size, num_patches)).float()
    
    predicted_dose, attention_weights, high_dose_pred, dose_levels, multi_head_features = model(
        ct_patches, spatial_coords, is_high_dose
    )
    
    print(f"Input shape: {ct_patches.shape}")
    print(f"Output dose shape: {predicted_dose.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"High-dose prediction shape: {high_dose_pred.shape}")
    print(f"Dose levels shape: {dose_levels.shape}")
    print(f"Multi-head features:")
    for size, features in multi_head_features.items():
        print(f"  {size}D features shape: {features.shape}")
    print("✅ Advanced ResNet-UNet Attention Model test passed!")
    
    # Test interpretability
    viz_data = model.get_attention_visualization(ct_patches, spatial_coords)
    print(f"Attention map shape: {viz_data['attention_map'].shape}")
    print(f"Dose levels shape: {viz_data['dose_levels'].shape}")
    print("✅ Interpretability features working!")
