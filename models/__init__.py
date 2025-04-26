from .vae import VAE
from .resnet_ae import ResNetAutoencoder
from .unet_ae import UNetAutoencoder
from .vae_2d import VAE2D, ConvAutoencoder2D, MLPAutoencoder2D, UNet2D


def get_model(config):
    """
    Factory function to create an autoencoder model based on configuration.
    """
    model_type = config['model']['type'].lower()
    print(f"Creating model of type: {model_type}")

    # Common parameters
    in_channels = config['model']['in_channels']
    latent_dim = config['model']['latent_dim']
    base_filters = config['model']['base_filters']

    # Get input dimensions from dataset config
    input_size = config.get('dataset', {}).get('input_size', [64, 64, 64])
    if isinstance(input_size, list):
        input_size = tuple(input_size)

    print(f"Using input size: {input_size}")

    # Create the appropriate model
    if model_type == 'vae':
        return VAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )
    elif model_type == 'resnet_ae':
        return ResNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )
    elif model_type == 'unet_ae':
        # Use UNetAutoencoder regardless of the model_type to ensure consistent behavior
        return UNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )
    else:
        # Default to UNetAutoencoder for unknown model types
        print(f"Unknown model type: {model_type}, defaulting to UNetAutoencoder")
        return UNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )