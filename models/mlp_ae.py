import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder


class MLPAutoencoder3D(BaseAutoencoder):
    """
    3D MLP-based Autoencoder for dose distribution volumes.
    """

    def __init__(self, in_channels=1, latent_dim=256, input_size=(64, 64, 64), dropout_rate=0.3):
        """
        Initialize the 3D MLP Autoencoder.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            input_size (tuple): Size of the input data (D, H, W)
            dropout_rate (float): Dropout rate for regularization
        """
        super(MLPAutoencoder3D, self).__init__(in_channels, latent_dim)

        self.input_size = input_size
        # Calculate flattened input size
        self.flattened_size = in_channels * input_size[0] * input_size[1] * input_size[2]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, self.flattened_size),
            nn.Tanh()
        )

    def encode(self, x):
        """
        Encode input.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode from encoded representation.

        Args:
            z (torch.Tensor): Encoded representation

        Returns:
            torch.Tensor: Decoded output
        """
        x = self.decoder(z)
        return x.view(-1, self.in_channels, *self.input_size)

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