from .vae import VAE
from .resnet_ae import ResNetAutoencoder
from .unet_ae import UNetAutoencoder
from .vae_2d import VAE2D, ConvAutoencoder2D, MLPAutoencoder2D, UNet2D


def get_model(config):
    """
    Factory function to create an autoencoder model based on configuration.

    Args:
        config (dict): Configuration dictionary with model parameters

    Returns:
        nn.Module: Instantiated model
    """
    model_type = config['model']['type'].lower()

    # Common parameters
    in_channels = config['model']['in_channels']
    latent_dim = config['model']['latent_dim']
    base_filters = config['model']['base_filters']

    # Check if it's a 2D or 3D model
    is_2d = config.get('model', {}).get('is_2d', False)

    # Create the appropriate model
    if model_type == 'vae':
        if is_2d:
            input_size = config.get('dataset', {}).get('input_size_2d', 256)
            dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)
            weight_decay = config.get('hyperparameters', {}).get('weight_decay', 0.0001)
            grad_clip = config.get('training', {}).get('grad_clip', 0.1)

            return VAE2D(
                input_size=input_size,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                grad_clip=grad_clip
            )
        else:
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
        return UNetAutoencoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_filters=base_filters
        )
    elif model_type == 'conv_autoencoder' and is_2d:
        input_size = config.get('dataset', {}).get('input_size_2d', 256)
        dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)

        return ConvAutoencoder2D(
            input_size=input_size,
            dropout_rate=dropout_rate
        )
    elif model_type == 'mlp_autoencoder' and is_2d:
        input_size = config.get('dataset', {}).get('input_size_2d', 256)
        dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)

        return MLPAutoencoder2D(
            input_size=input_size,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate
        )
    elif model_type == 'unet' and is_2d:
        dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)

        return UNet2D(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")