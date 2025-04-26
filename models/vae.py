import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder, Conv3DBlock, DeconvBlock


class VAE(BaseAutoencoder):
    """
    Variational Autoencoder (VAE) for 3D dose distribution.
    """

    def __init__(self, in_channels=1, latent_dim=128, base_filters=32, input_size=(64, 64, 64)):
        """
        Initialize the VAE.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters
            input_size (tuple): Input dimensions (D, H, W)
        """
        super(VAE, self).__init__(in_channels, latent_dim)

        self.input_size = input_size

        # Calculate downsampling
        # After 4 layers of stride 2 convolution, the size is reduced by a factor of 16
        self.encoded_size = (
            input_size[0] // 16,
            input_size[1] // 16,
            input_size[2] // 16
        )

        # Encoder
        self.encoder = nn.Sequential(
            # Input: [batch, in_channels, D, H, W]
            Conv3DBlock(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters, D/2, H/2, W/2]
            Conv3DBlock(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters*2, D/4, H/4, W/4]
            Conv3DBlock(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters*4, D/8, H/8, W/8]
            Conv3DBlock(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters*8, D/16, H/16, W/16]
        )

        # Calculate the size of the flattened feature maps
        self.flatten_size = base_filters * 8 * self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2]

        # Mean and log variance projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder input
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_filters * 8, self.encoded_size[0], self.encoded_size[1], self.encoded_size[2])),
            # [batch, base_filters*8, D/16, H/16, W/16]
            DeconvBlock(base_filters * 8, base_filters * 4, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters*4, D/8, H/8, W/8]
            DeconvBlock(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters*2, D/4, H/4, W/4]
            DeconvBlock(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            # [batch, base_filters, D/2, H/2, W/2]
            DeconvBlock(base_filters, in_channels, kernel_size=4, stride=2, padding=1, activation=nn.Sigmoid()),
            # [batch, in_channels, D, H, W]
        )