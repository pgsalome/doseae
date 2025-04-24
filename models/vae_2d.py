import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder


class VAE2D(BaseAutoencoder):
    """
    2D Variational Autoencoder for dose distribution images.
    This implementation is based on the original VAE from the previous repository.
    """

    def __init__(self, input_size=256, latent_dim=1024, dropout_rate=0.3, weight_decay=0.0001, grad_clip=0.1):
        """
        Initialize the 2D VAE.

        Args:
            input_size (int): Size of the input (assuming square images)
            latent_dim (int): Dimension of the latent space
            dropout_rate (float): Dropout rate for regularization
            weight_decay (float): Weight decay for regularization
            grad_clip (float): Gradient clipping threshold
        """
        super(VAE2D, self).__init__(in_channels=1, latent_dim=latent_dim)

        self.input_size = input_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size * input_size, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate),
        )

        # Mean and log variance
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, input_size * input_size),
            nn.Tanh(),  # Using Tanh as in the original implementation
        )

    def encode(self, x):
        """
        Encode input into latent space parameters.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        # Flatten the input
        x = x.view(-1, self.input_size * self.input_size)

        # Encode
        h = self.encoder(x)

        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

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
        # Decode
        x_hat = self.decoder(z)

        # Apply gradient clipping
        x_hat = torch.clamp(x_hat, -self.grad_clip, self.grad_clip)

        # Reshape to image
        return x_hat.view(-1, 1, self.input_size, self.input_size)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            tuple: Reconstructed output, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar

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
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        # Add weight decay regularization if specified
        if self.weight_decay > 0:
            l2_reg = torch.tensor(0.0, device=x.device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            total_loss += self.weight_decay * l2_reg

        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

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


class ConvAutoencoder2D(BaseAutoencoder):
    """
    2D Convolutional Autoencoder for dose distribution images.
    """

    def __init__(self, input_size=256, dropout_rate=0.3):
        """
        Initialize the 2D Convolutional Autoencoder.

        Args:
            input_size (int): Size of the input (assuming square images)
            dropout_rate (float): Dropout rate for regularization
        """
        super(ConvAutoencoder2D, self).__init__(in_channels=1, latent_dim=128 * 32 * 32)

        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 32, 128, 128]
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 64, 64]
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 32, 32]
            nn.Dropout(dropout_rate),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Use Sigmoid for output pixel values between 0 and 1
        )

    def encode(self, x):
        """
        Encode input.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

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
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

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


class MLPAutoencoder2D(BaseAutoencoder):
    """
    2D MLP-based Autoencoder for dose distribution images.
    """

    def __init__(self, input_size=256, latent_dim=256, dropout_rate=0.3):
        """
        Initialize the 2D MLP Autoencoder.

        Args:
            input_size (int): Size of the input (assuming square images)
            latent_dim (int): Dimension of the latent space
            dropout_rate (float): Dropout rate for regularization
        """
        super(MLPAutoencoder2D, self).__init__(in_channels=1, latent_dim=latent_dim)

        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size * input_size, 4096),
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
            nn.Linear(512, latent_dim),
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
            nn.Linear(4096, input_size * input_size),
            nn.Tanh(),
        )

    def encode(self, x):
        """
        Encode input.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Encoded representation
        """
        x = x.view(-1, self.input_size * self.input_size)
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
        return x.view(-1, 1, self.input_size, self.input_size)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

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


class UNet2D(BaseAutoencoder):
    """
    2D UNet model for dose distribution images.
    """

    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.3):
        """
        Initialize the 2D UNet.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            dropout_rate (float): Dropout rate for regularization
        """
        super(UNet2D, self).__init__(in_channels=in_channels, latent_dim=0)

        self.enc1 = self.conv_block(in_channels, 64, dropout_rate)
        self.enc2 = self.conv_block(64, 128, dropout_rate)
        self.enc3 = self.conv_block(128, 256, dropout_rate)
        self.enc4 = self.conv_block(256, 512, dropout_rate)
        self.enc5 = self.conv_block(512, 1024, dropout_rate)

        self.upconv5 = self.upconv_block(1024, 512, dropout_rate)
        self.dec5 = self.conv_block(1024, 512, dropout_rate)

        self.upconv4 = self.upconv_block(512, 256, dropout_rate)
        self.dec4 = self.conv_block(512, 256, dropout_rate)

        self.upconv3 = self.upconv_block(256, 128, dropout_rate)
        self.dec3 = self.conv_block(256, 128, dropout_rate)

        self.upconv2 = self.upconv_block(128, 64, dropout_rate)
        self.dec2 = self.conv_block(128, 64, dropout_rate)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate):
        """
        Create a convolutional block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            dropout_rate (float): Dropout rate

        Returns:
            nn.Sequential: Convolutional block
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        return block

    def upconv_block(self, in_channels, out_channels, dropout_rate):
        """
        Create an up-convolutional block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            dropout_rate (float): Dropout rate

        Returns:
            nn.Sequential: Up-convolutional block
        """
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        return block

    def forward(self, x):
        """
        Forward pass through the UNet.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        enc5 = self.enc5(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec5 = self.upconv5(enc5)
        dec5 = torch.cat((dec5, enc4), dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)

        out = self.conv_last(dec2)

        return out

    def encode(self, x):
        """
        This model doesn't have a typical encoder-decoder structure for latent space.
        This method is included for compatibility with the BaseAutoencoder class.
        """
        raise NotImplementedError("UNet doesn't have a typical encoder-decoder structure for latent space")

    def decode(self, z):
        """
        This model doesn't have a typical encoder-decoder structure for latent space.
        This method is included for compatibility with the BaseAutoencoder class.
        """
        raise NotImplementedError("UNet doesn't have a typical encoder-decoder structure for latent space")

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