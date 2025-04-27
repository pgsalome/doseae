import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder


class MLPAutoencoder3D(BaseAutoencoder):
    """
    Memory-optimized 3D MLP-based Autoencoder for dose distribution volumes.
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

        # Use smaller intermediate layer sizes to reduce memory consumption
        encoder_dims = [
            self.flattened_size,  # Input size
            min(2048, self.flattened_size // 2),  # First hidden layer
            min(1024, self.flattened_size // 4),  # Second hidden layer
            min(512, self.flattened_size // 8),  # Third hidden layer
            latent_dim  # Latent layer
        ]

        # Build encoder layers dynamically
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # Don't add activation after the latent layer
                encoder_layers.append(nn.ReLU(True))
                encoder_layers.append(nn.Dropout(dropout_rate))

        # Encoder
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers dynamically (reverse of encoder)
        decoder_dims = list(reversed(encoder_dims))
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # Don't add activation after the output layer
                decoder_layers.append(nn.ReLU(True))
                decoder_layers.append(nn.Dropout(dropout_rate))

        # Add final Tanh activation
        decoder_layers.append(nn.Tanh())

        # Decoder
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Encode input.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            torch.Tensor: Encoded representation
        """
        # Flatten the input
        x = x.view(-1, self.flattened_size)
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
        # Use smaller batch sizes for the forward pass
        batch_size = x.size(0)
        max_batch = 2  # Process at most 2 samples at a time

        if batch_size <= max_batch or not self.training:
            # Process normally for inference or small batches
            z = self.encode(x)
            return self.decode(z)
        else:
            # Split the batch for training
            outputs = []
            for i in range(0, batch_size, max_batch):
                end = min(i + max_batch, batch_size)
                mini_batch = x[i:end]
                z = self.encode(mini_batch)
                output = self.decode(z)
                outputs.append(output)

            return torch.cat(outputs, dim=0)

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