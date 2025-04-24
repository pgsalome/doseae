import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder, Conv3DBlock


class DoubleConv(nn.Module):
    """
    Double convolution block used in UNet.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetBottleneck(nn.Module):
    """
    Bottleneck for UNet with latent space projection.
    """

    def __init__(self, in_channels, bottleneck_channels, latent_dim):
        super(UNetBottleneck, self).__init__()
        self.conv = DoubleConv(in_channels, bottleneck_channels)

        # Calculate size of flattened bottleneck
        # Assuming input size is 64x64x64, at bottleneck it's 4x4x4
        self.flatten_size = bottleneck_channels * 4 * 4 * 4

        # Projection to and from latent space
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        # Reshape parameters
        self.bottleneck_channels = bottleneck_channels

    def forward(self, x):
        """
        Forward pass through the bottleneck with latent space projection.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Output tensor and latent vector
        """
        # Convolution
        x = self.conv(x)

        # Remember shape for reshaping after latent projection
        batch_size = x.size(0)

        # Flatten and project to latent space
        flat = x.view(batch_size, -1)
        latent = self.fc_encode(flat)

        # Project back and reshape
        flat_recon = self.fc_decode(latent)
        x_recon = flat_recon.view(batch_size, self.bottleneck_channels, 4, 4, 4)

        return x_recon, latent


class UNetAutoencoder(BaseAutoencoder):
    """
    UNet-style autoencoder for 3D dose distribution.
    """

    def __init__(self, in_channels=1, latent_dim=128, base_filters=32):
        """
        Initialize the UNet autoencoder.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters
        """
        super(UNetAutoencoder, self).__init__(in_channels, latent_dim)

        # Encoder path
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck with latent space
        self.bottleneck = UNetBottleneck(base_filters * 8, base_filters * 16, latent_dim)

        # Decoder path
        self.upconv4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        # Output layer
        self.outconv = nn.Conv3d(base_filters, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        Encode input through UNet encoder and bottleneck.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Encoder features and latent vector
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck, latent = self.bottleneck(self.pool4(enc4))

        return (enc1, enc2, enc3, enc4, bottleneck), latent

    def decode(self, features):
        """
        Decode from encoder features through UNet decoder.

        Args:
            features (tuple): Encoder features (enc1, enc2, enc3, enc4, bottleneck)

        Returns:
            torch.Tensor: Decoded output
        """
        # Unpack features
        enc1, enc2, enc3, enc4, bottleneck = features

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((enc4, dec4), dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((enc3, dec3), dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((enc2, dec2), dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((enc1, dec1), dim=1))

        # Output
        output = self.outconv(dec1)
        output = self.sigmoid(output)

        return output

    def forward(self, x):
        """
        Forward pass through the UNet autoencoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Reconstructed output and latent vector
        """
        features, latent = self.encode(x)
        output = self.decode(features)

        return output, latent

    def get_latent_representation(self, x):
        """
        Get the latent representation for an input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Latent representation
        """
        with torch.no_grad():
            _, latent = self.encode(x)
            return latent

    def get_losses(self, x, x_recon):
        """
        Calculate reconstruction losses.

        Args:
            x (torch.Tensor): Original input
            x_recon (torch.Tensor): Reconstructed input

        Returns:
            dict: Dictionary of loss components
        """
        # MSE loss
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')

        # SSIM loss for structural similarity (important for dose distributions)
        # Placeholder for actual SSIM implementation
        # Would need a 3D SSIM implementation which is not trivial

        # Total loss
        total_loss = mse_loss

        return {
            'mse_loss': mse_loss,
            'total_loss': total_loss
        }