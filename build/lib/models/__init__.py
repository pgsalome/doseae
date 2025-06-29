from .vae import VAE
from .resnet_ae import ResNetAutoencoder
from .unet_ae import UNetAutoencoder
from .vae_2d import VAE2D, ConvAutoencoder2D, MLPAutoencoder2D, UNet2D
from .mlp_ae import MLPAutoencoder3D
from .conv_ae import ConvAutoencoder3D


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

    # Check if model is 2D
    is_2d = config.get('model', {}).get('is_2d', False)

    # Create the appropriate model
    if model_type == 'vae':
        if is_2d:
            input_size_2d = config.get('dataset', {}).get('input_size_2d', 256)
            return VAE2D(
                input_size=input_size_2d,
                latent_dim=latent_dim,
                dropout_rate=config['model'].get('dropout_rate', 0.3),
                weight_decay=config['hyperparameters'].get('weight_decay', 0.0001),
                grad_clip=config['training'].get('grad_clip', 0.1)
            )
        else:
            return VAE(
                in_channels=in_channels,
                latent_dim=latent_dim,
                base_filters=base_filters,
                input_size=input_size
            )
    elif model_type == 'resnet_ae':
        return ResNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters,
            input_size=input_size
        )
    elif model_type == 'unet_ae':
        return UNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )
    elif model_type == 'mlp_autoencoder':
        if is_2d:
            input_size_2d = config.get('dataset', {}).get('input_size_2d', 256)
            return MLPAutoencoder2D(
                input_size=input_size_2d,
                latent_dim=latent_dim,
                dropout_rate=config['model'].get('dropout_rate', 0.3)
            )
        else:
            return MLPAutoencoder3D(
                in_channels=in_channels,
                latent_dim=latent_dim,
                input_size=input_size,
                dropout_rate=config['model'].get('dropout_rate', 0.3)
            )
    elif model_type == 'conv_autoencoder':
        if is_2d:
            input_size_2d = config.get('dataset', {}).get('input_size_2d', 256)
            return ConvAutoencoder2D(
                input_size=input_size_2d,
                dropout_rate=config['model'].get('dropout_rate', 0.3)
            )
        else:
            return ConvAutoencoder3D(
                in_channels=in_channels,
                base_filters=base_filters,
                latent_dim=latent_dim,
                input_size=input_size,
                dropout_rate=config['model'].get('dropout_rate', 0.3)
            )
    else:
        # Default to UNetAutoencoder for unknown model types
        print(f"Unknown model type: {model_type}, defaulting to UNetAutoencoder")
        return UNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )