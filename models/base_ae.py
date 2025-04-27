import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    """
    Base class for all autoencoder models with common functionality.
    """

    def __init__(self, in_channels=1, latent_dim=128):
        """
        Initialize the base autoencoder.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
        """
        super(BaseAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # These will be implemented by the specific autoencoder architectures
        self.encoder = None
        self.decoder = None

    def encode(self, x):
        """
        Encode input into latent representation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation in latent space
        """
        if self.encoder is None:
            raise NotImplementedError("Encoder has not been implemented!")
        return self.encoder(x)

    def decode(self, z):
        """
        Decode from latent space to output space.

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Decoded output
        """
        if self.decoder is None:
            raise NotImplementedError("Decoder has not been implemented!")
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        z = self.encode(x)
        return self.decode(z)

    def get_latent_representation(self, x):
        """
        Get the latent representation for an input without decoding.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Latent representation
        """
        with torch.no_grad():
            return self.encode(x)

    def reconstruct(self, x):
        """
        Reconstruct the input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        with torch.no_grad():
            return self.forward(x)

    def get_losses(self, x, x_recon):
        """
        Calculate reconstruction loss.

        Args:
            x (torch.Tensor): Original input
            x_recon (torch.Tensor): Reconstructed input

        Returns:
            dict: Dictionary of loss components
        """
        # Default is MSE loss
        mse_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        return {'mse_loss': mse_loss, 'total_loss': mse_loss}

    def count_parameters(self):
        """
        Count the total number of trainable parameters.

        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Conv3DBlock(nn.Module):
    """
    3D Convolutional block with batch normalization and activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, activation=nn.ReLU()):
        """
        Initialize the 3D convolutional block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            bias (bool): Whether to use bias
            activation (nn.Module): Activation function
        """
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after convolution, batch norm, and activation
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x




class ResidualBlock3D(nn.Module):
    """
    3D Residual block for ResNet-like architectures.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the 3D residual block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
        """
        super(ResidualBlock3D, self).__init__()

        # Main path
        self.conv1 = Conv3DBlock(in_channels, out_channels, stride=stride)
        self.conv2 = Conv3DBlock(out_channels, out_channels, activation=None)

        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        # Activation after addition
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)

        # Shortcut path
        shortcut = self.shortcut(x)

        # Add and activate
        out += shortcut
        out = self.activation(out)

        return out

class DeconvBlock(nn.Module):
    """
    3D transposed convolutional block with batch normalization and activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                 output_padding=0, bias=False, activation=nn.ReLU()):
        """
        Initialize the 3D transposed convolutional block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            output_padding (int): Output padding to control output size
            bias (bool): Whether to use bias
            activation (nn.Module): Activation function
        """
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after transposed convolution, batch norm, and activation
        """
        x = self.deconv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x