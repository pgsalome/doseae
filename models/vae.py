import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder, Conv3DBlock, DeconvBlock


class VAE(BaseAutoencoder):
    """
    Variational Autoencoder (VAE) for 3D dose distribution.
    """

    def __init__(self, in_channels=1, latent_dim=128, base_filters=32):
        """
        Initialize the VAE.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters (will be multiplied in deeper layers)
        """
        super(VAE, self).__init__(in_channels, latent_dim)

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
        # Assuming input of size [batch, in_channels, 64, 64, 64]
        self.flatten_size = base_filters * 8 * (64 // 16) * (64 // 16) * (64 // 16)

        # Mean and log variance projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder input
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_filters * 8, 64 // 16, 64 // 16, 64 // 16)),
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
        Encode input into latent space parameters (mean and log variance).

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        # Encode through convolutional layers
        h = self.encoder(x)

        # Flatten
        h_flat = h.view(h.size(0), -1)

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
        # Project latent vector to decoder input dimension
        h = self.fc_decoder(z)

        # Decode through transposed convolutional layers
        return self.decoder(h)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Reconstructed output, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def get_losses(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Calculate VAE loss components (reconstruction + KL divergence).

        Args:
            x (torch.Tensor): Original input
            x_recon (torch.Tensor): Reconstructed input
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            beta (float): Weight for the KL divergence term

        Returns:
            dict: Dictionary of loss components
        """
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

    def sample(self, num_samples=1, device='cuda'):
        """
        Sample from the latent space and decode.

        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to use

        Returns:
            torch.Tensor: Generated samples
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Decode
            samples = self.decode(z)

            return samples

    def reconstruct(self, x):
        """
        Reconstruct the input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z)