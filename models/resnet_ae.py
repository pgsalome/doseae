import torch
import torch.nn as nn
from .base_ae import BaseAutoencoder, ResidualBlock3D, Conv3DBlock, DeconvBlock


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for 3D dose distribution.
    """

    def __init__(self, in_channels=1, base_filters=32, latent_dim=128):
        """
        Initialize the ResNet encoder.

        Args:
            in_channels (int): Number of input channels
            base_filters (int): Number of base filters
            latent_dim (int): Dimension of the latent space
        """
        super(ResNetEncoder, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, blocks=2, stride=2)

        # Assuming input size is 64x64x64
        # After initial: 32x32x32
        # After layer1: 32x32x32
        # After layer2: 16x16x16
        # After layer3: 8x8x8
        # After layer4: 4x4x4
        self.flatten_size = base_filters * 8 * 4 * 4 * 4

        # Final projection to latent space
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Create a layer of residual blocks.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            blocks (int): Number of residual blocks
            stride (int): Stride for the first block

        Returns:
            nn.Sequential: Sequential layer of residual blocks
        """
        layers = []

        # First block may have a stride
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded latent vector
        """
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = torch.mean(x, dim=(2, 3, 4))

        # Project to latent space
        x = self.fc(x)

        return x


class ResNetDecoder(nn.Module):
    """
    ResNet-style decoder for 3D dose distribution.
    """

    def __init__(self, latent_dim=128, base_filters=32, out_channels=1):
        """
        Initialize the ResNet decoder.

        Args:
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters
            out_channels (int): Number of output channels
        """
        super(ResNetDecoder, self).__init__()

        # Initial projection from latent space
        self.fc = nn.Linear(latent_dim, base_filters * 8 * 4 * 4 * 4)

        # Reshape to 3D volume
        self.unflatten = nn.Unflatten(1, (base_filters * 8, 4, 4, 4))

        # Upsampling blocks with residual connections
        self.up1 = nn.Sequential(
            DeconvBlock(base_filters * 8, base_filters * 4),
            ResidualBlock3D(base_filters * 4, base_filters * 4)
        )  # 8x8x8

        self.up2 = nn.Sequential(
            DeconvBlock(base_filters * 4, base_filters * 2),
            ResidualBlock3D(base_filters * 2, base_filters * 2)
        )  # 16x16x16

        self.up3 = nn.Sequential(
            DeconvBlock(base_filters * 2, base_filters),
            ResidualBlock3D(base_filters, base_filters)
        )  # 32x32x32

        self.up4 = nn.Sequential(
            DeconvBlock(base_filters, base_filters // 2),
            ResidualBlock3D(base_filters // 2, base_filters // 2)
        )  # 64x64x64

        # Final convolution to output channels
        self.final = nn.Sequential(
            nn.Conv3d(base_filters // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Latent vector

        Returns:
            torch.Tensor: Reconstructed output
        """
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)

        return x


class ResNetAutoencoder(BaseAutoencoder):
    """
    ResNet-style autoencoder for 3D dose distribution.
    """

    def __init__(self, in_channels=1, latent_dim=128, base_filters=32):
        """
        Initialize the ResNet autoencoder.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters
        """
        super(ResNetAutoencoder, self).__init__(in_channels, latent_dim)

        # Encoder and decoder
        self.encoder = ResNetEncoder(in_channels, base_filters, latent_dim)
        self.decoder = ResNetDecoder(latent_dim, base_filters, in_channels)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        z = self.encoder(x)
        return self.decoder(z)

    def get_losses(self, x, x_recon):
        """
        Calculate reconstruction loss.

        Args:
            x (torch.Tensor): Original input
            x_recon (torch.Tensor): Reconstructed input

        Returns:
            dict: Dictionary of loss components
        """
        # MSE loss
        mse_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Add L1 loss for better preservation of sharp edges
        l1_loss = nn.functional.l1_loss(x_recon, x, reduction='mean')

        # Combined loss
        total_loss = mse_loss + 0.1 * l1_loss

        return {
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'total_loss': total_loss
        }