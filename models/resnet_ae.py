import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ae import BaseAutoencoder, ResidualBlock3D, Conv3DBlock, DeconvBlock


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for 3D dose distribution.
    """

    def __init__(self, in_channels=1, base_filters=32, latent_dim=128, input_size=(64, 64, 64)):
        """
        Initialize the ResNet encoder.

        Args:
            in_channels (int): Number of input channels
            base_filters (int): Number of base filters
            latent_dim (int): Dimension of the latent space
            input_size (tuple): Input dimensions (D, H, W)
        """
        super(ResNetEncoder, self).__init__()

        # Store input size and base filters
        self.input_size = input_size
        self.base_filters = base_filters
        self.latent_dim = latent_dim

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

        # We'll initialize the flatten_size and fc in the first forward pass
        self.flatten_size = None
        self.fc = None
        self.is_initialized = False

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
        input_shape = x.shape  # Store input shape for decoder

        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Store encoded feature shape for decoder
        encoded_feature_shape = x.shape[2:]

        # Flatten the output
        flattened = x.view(x.size(0), -1)

        # Initialize the fc layer if this is the first forward pass
        if not self.is_initialized:
            self.flatten_size = flattened.size(1)
            print(f"Initializing fc layer with input size: {self.flatten_size}, output size: {self.latent_dim}")
            self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
            self.is_initialized = True

        # Project to latent space
        z = self.fc(flattened)

        return z, encoded_feature_shape, input_shape


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

        self.latent_dim = latent_dim
        self.base_filters = base_filters

        # We'll initialize these in the first forward pass
        self.fc = None
        self.encoded_shape = None
        self.input_shape = None
        self.is_initialized = False

        # Prepare upsampling blocks that don't depend on dimensions
        self.up1 = nn.Sequential(
            DeconvBlock(base_filters * 8, base_filters * 4, kernel_size=2, stride=2, padding=0, output_padding=0),
            ResidualBlock3D(base_filters * 4, base_filters * 4)
        )

        self.up2 = nn.Sequential(
            DeconvBlock(base_filters * 4, base_filters * 2, kernel_size=2, stride=2, padding=0, output_padding=0),
            ResidualBlock3D(base_filters * 2, base_filters * 2)
        )

        self.up3 = nn.Sequential(
            DeconvBlock(base_filters * 2, base_filters, kernel_size=2, stride=2, padding=0, output_padding=0),
            ResidualBlock3D(base_filters, base_filters)
        )

        self.up4 = nn.Sequential(
            DeconvBlock(base_filters, base_filters // 2, kernel_size=2, stride=2, padding=0, output_padding=0),
            ResidualBlock3D(base_filters // 2, base_filters // 2)
        )

        # Final convolution to output channels
        self.final = nn.Sequential(
            nn.Conv3d(base_filters // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, encoded_shape=None, input_shape=None):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vector
            encoded_shape (tuple, optional): Spatial dimensions of the encoded features
            input_shape (tuple, optional): Original input shape for final interpolation

        Returns:
            torch.Tensor: Reconstructed output
        """
        # Initialize on first forward pass
        if not self.is_initialized:
            if encoded_shape is None:
                # Default small encoded shape if not provided
                encoded_shape = (1, 1, 1)
                print(f"Warning: Using default encoded shape: {encoded_shape}")

            self.encoded_shape = encoded_shape
            flatten_size = self.base_filters * 8 * encoded_shape[0] * encoded_shape[1] * encoded_shape[2]
            print(f"Initializing decoder fc layer with input size: {self.latent_dim}, output size: {flatten_size}")
            self.fc = nn.Linear(self.latent_dim, flatten_size).to(z.device)
            self.is_initialized = True

            if input_shape is not None:
                self.input_shape = input_shape

        # Project from latent space
        x = self.fc(z)

        # Reshape to 3D volume
        x = x.view(-1, self.base_filters * 8, *self.encoded_shape)

        # Upsample
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)

        # Ensure output matches input shape exactly using interpolation
        if self.input_shape is not None and x.shape != self.input_shape:
            x = F.interpolate(x, size=self.input_shape[2:], mode='trilinear', align_corners=False)

        return x


class ResNetAutoencoder(BaseAutoencoder):
    """
    ResNet-style autoencoder for 3D dose distribution.
    """

    def __init__(self, in_channels=1, latent_dim=128, base_filters=32, input_size=(64, 64, 64)):
        """
        Initialize the ResNet autoencoder.

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            base_filters (int): Number of base filters
            input_size (tuple): Input dimensions (D, H, W)
        """
        super(ResNetAutoencoder, self).__init__(in_channels, latent_dim)

        # Store parameters
        self.input_size = input_size
        self.base_filters = base_filters

        # Initialize encoder and decoder
        self.encoder = ResNetEncoder(in_channels, base_filters, latent_dim, input_size)
        self.decoder = ResNetDecoder(latent_dim, base_filters, in_channels)

        # This will store the encoded and input shapes
        self.encoded_shape = None
        self.orig_input_shape = None

    def encode(self, x):
        """
        Encode input into latent space.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded latent representation
        """
        # Get latent vector and shapes
        z, encoded_shape, input_shape = self.encoder(x)
        # Save shapes for decoder
        self.encoded_shape = encoded_shape
        self.orig_input_shape = input_shape
        return z

    def decode(self, z):
        """
        Decode from latent space.

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Decoded output
        """
        return self.decoder(z, self.encoded_shape, self.orig_input_shape)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed output
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon