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

    def encode(self, x):
        """
        Encode input into latent space parameters.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        # Pass through encoder
        h = self.encoder(x)

        # Flatten
        h_flat = h.view(h.size(0), -1)

        # Check that the flattened size matches what we expect
        if h_flat.size(1) != self.flatten_size:
            raise ValueError(f"Expected encoder output size {self.flatten_size}, got {h_flat.size(1)}")

        # Get mean and log variance
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution

        Returns:
            torch.Tensor: Sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode from latent space to output space.

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Decoded output
        """
        # Project to flattened size
        h = self.fc_decoder(z)

        # Pass through decoder
        recon_x = self.decoder(h)

        return recon_x

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]

        Returns:
            tuple: Reconstructed output, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def get_losses(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Calculate VAE loss components.

        Args:
            x (torch.Tensor): Original input
            x_recon (torch.Tensor): Reconstructed input
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            beta (float): Weight for the KL divergence term

        Returns:
            dict: Dictionary of loss components
        """
        # Check for size mismatch and resize if necessary
        if x.shape != x_recon.shape:
            print(f"Input shape {x.shape} and reconstructed shape {x_recon.shape} don't match. Resizing...")
            import torch.nn.functional as F
            x_recon = F.interpolate(x_recon, size=x.shape[2:], mode='trilinear', align_corners=False)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

    def reconstruct(self, x):
        """
        Reconstruct the input (without sampling, for deterministic output).

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = mu  # Use mean instead of sampling
            return self.decode(z)