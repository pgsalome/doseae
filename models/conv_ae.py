import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder, Conv3DBlock, DeconvBlock


class ConvAutoencoder3D(BaseAutoencoder):
    """
    3D Convolutional Autoencoder for dose distribution volumes.
    """

    def __init__(self, in_channels=1, base_filters=32, latent_dim=128, input_size=(64, 64, 64), dropout_rate=0.3):
        """
        Initialize the 3D Convolutional Autoencoder.

        Args:
            in_channels (int): Number of input channels
            base_filters (int): Number of base filters
            latent_dim (int): Dimension of the latent space
            input_size (tuple): Size of the input data (D, H, W)
            dropout_rate (float): Dropout rate for regularization
        """
        super(ConvAutoencoder3D, self).__init__(in_channels, latent_dim)

        self.input_size = input_size

        # Calculate encoded feature map size after 3 downsampling operations
        self.encoded_size = (
            input_size[0] // 8,
            input_size[1] // 8,
            input_size[2] // 8
        )

        # Calculate the flattened size for the bottleneck
        self.flattened_size = base_filters * 8 * self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2]

        # Encoder
        self.encoder = nn.Sequential(
            # Input: [batch, in_channels, D, H, W]
            Conv3DBlock(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, base_filters, D/2, H/2, W/2]
            nn.Dropout3d(dropout_rate),

            Conv3DBlock(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, base_filters*2, D/4, H/4, W/4]
            nn.Dropout3d(dropout_rate),

            Conv3DBlock(base_filters * 2, base_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, base_filters*4, D/8, H/8, W/8]
            nn.Dropout3d(dropout_rate),

            Conv3DBlock(base_filters * 4, base_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.Dropout3d(dropout_rate)
        )

        # Bottleneck fully connected layers
        self.bottleneck_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim)
        )

        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Input: [batch, base_filters*8, D/8, H/8, W/8]
            DeconvBlock(base_filters * 8, base_filters * 4, kernel_size=2, stride=2, padding=0),
            nn.Dropout3d(dropout_rate),

            DeconvBlock(base_filters * 4, base_filters * 2, kernel_size=2, stride=2, padding=0),
            nn.Dropout3d(dropout_rate),

            DeconvBlock(base_filters * 2, base_filters, kernel_size=2, stride=2, padding=0),
            nn.Dropout3d(dropout_rate),

            nn.Conv3d(base_filters, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode input.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            torch.Tensor: Encoded representation
        """
        features = self.encoder(x)
        return self.bottleneck_encoder(features)

    def decode(self, z):
        """
        Decode from encoded representation.

        Args:
            z (torch.Tensor): Encoded representation

        Returns:
            torch.Tensor: Decoded output
        """
        features = self.bottleneck_decoder(z)
        features = features.view(-1, 8 * self.base_filters, *self.encoded_size)
        return self.decoder(features)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            torch.Tensor: Reconstructed output
        """
        z = self.encode(x)
        return self.decode(z)

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
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')

        return {
            'mse_loss': mse_loss,
            'total_loss': mse_loss
        }