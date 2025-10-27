from .architectures import (
    ConfigurableAutoencoder,
    DoseAEResUNet,
    VAE,
    VAE2D,
    create_attention_autoencoder,
    create_doseae_resunet,
    create_dual_input_autoencoder,
    create_fused_autoencoder,
    create_multi_channel_autoencoder,
    create_patches_only_autoencoder,
)


def get_model(config):
    """
    Factory function to create an autoencoder model based on configuration.
    """
    model_type = config['model']['type'].lower()
    print(f"Creating model of type: {model_type}")
    print(f"Model config: {config['model']}")

    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    # Common parameters
    in_channels = model_cfg['in_channels']
    latent_dim = model_cfg['latent_dim']
    base_filters = model_cfg['base_filters']

    # Determine dimensionality and input size
    is_2d = model_cfg.get('is_2d', False)
    if is_2d:
        input_size_cfg = dataset_cfg.get('input_size_2d', dataset_cfg.get('input_size', 256))
        if isinstance(input_size_cfg, int):
            input_size = (input_size_cfg, input_size_cfg)
        elif isinstance(input_size_cfg, (list, tuple)):
            input_size = tuple(input_size_cfg)
            if len(input_size) == 1:
                input_size = (input_size[0], input_size[0])
            elif len(input_size) > 2:
                input_size = input_size[:2]
        else:
            input_size = (256, 256)
    else:
        input_size_cfg = dataset_cfg.get('input_size_3d', dataset_cfg.get('input_size', [64, 64, 64]))
        if isinstance(input_size_cfg, int):
            input_size = (input_size_cfg, input_size_cfg, input_size_cfg)
        elif isinstance(input_size_cfg, (list, tuple)):
            input_size = tuple(input_size_cfg)
            if len(input_size) < 3:
                input_size = tuple(list(input_size) + [input_size[-1]] * (3 - len(input_size)))
            elif len(input_size) > 3:
                input_size = input_size[:3]
        else:
            input_size = (64, 64, 64)

    print(f"Using input size: {input_size}")

    # Create the appropriate model
    if model_type == 'vae':
        if is_2d:
            return VAE2D(
                input_size=input_size[0],
                latent_dim=latent_dim,
                dropout_rate=model_cfg.get('dropout_rate', 0.3),
                weight_decay=config.get('hyperparameters', {}).get('weight_decay', 0.0001),
                grad_clip=config.get('training', {}).get('grad_clip', 0.1)
            )
        else:
            return VAE(
                in_channels=in_channels,
                latent_dim=latent_dim,
                base_filters=base_filters,
                input_size=input_size
            )
    elif model_type == 'resnet_ae':
        # Use ConfigurableAutoencoder with ResNet architecture
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic ResNet
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='resnet',
            use_bottleneck=config['model'].get('use_bottleneck', False),
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'unet_ae':
        # Use ConfigurableAutoencoder with UNet architecture
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic UNet
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='unet',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'mlp_autoencoder':
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=model_cfg.get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=model_cfg.get('feature_dim', 256),
            attention_heads=0,
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='mlp',
            use_bottleneck=False,
            input_size=input_size,
            dropout_mlp=model_cfg.get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=model_cfg.get('dropout_decoder', 0.1),
            dropout_features=model_cfg.get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'conv_autoencoder':
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=model_cfg.get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=model_cfg.get('feature_dim', 256),
            attention_heads=0,
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            dropout_mlp=model_cfg.get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=model_cfg.get('dropout_decoder', 0.1),
            dropout_features=model_cfg.get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'doseae_resunet':
        # Use the DoseAE ResNet-UNet with attention
        return create_doseae_resunet(config)
    elif model_type == 'advanced_resnet_unet':
        # Use ConfigurableAutoencoder with ResNet architecture
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic ResNet-UNet
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='resnet',
            use_bottleneck=config['model'].get('use_bottleneck', False),
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'dual_input_attention_autoencoder':
        # Use ConfigurableAutoencoder with attention and dual input
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=config['model'].get('attention_heads', 4),
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=config['model'].get('dropout_attention', 0.1),
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'full_image_autoencoder':
        # Use ConfigurableAutoencoder for full images
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic full image
            latent_dim=latent_dim,
            input_mode='ct_only',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'patches_only_autoencoder':
        # Use ConfigurableAutoencoder for patches only
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic patches
            latent_dim=latent_dim,
            input_mode='ct_only',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'fused_ct_dose_autoencoder':
        # Use ConfigurableAutoencoder with fused input
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for basic fused
            latent_dim=latent_dim,
            input_mode='fused_only',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'multi_channel_autoencoder':
        # Use ConfigurableAutoencoder with multi-channel input
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=config['model'].get('attention_heads', 4),
            latent_dim=latent_dim,
            input_mode=config['model'].get('input_mode', 'dose_ct'),
            fusion_method=config['model'].get('fusion_method', 'weighted_sum'),
            architecture=config['model'].get('architecture', 'conv'),
            use_bottleneck=config['model'].get('use_bottleneck', False),
            input_size=input_size,
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=config['model'].get('dropout_attention', 0.1),
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'simple_3d_autoencoder':
        # Use ConfigurableAutoencoder with simple 3D ResNet
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention for simple 3D
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='resnet',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    elif model_type == 'configurable_autoencoder':
        # Use the configurable autoencoder that can handle all input configurations
        input_mode = config['model'].get('input_mode', 'dose_ct')
        fusion_method = config['model'].get('fusion_method', 'weighted_sum')
        architecture = config['model'].get('architecture', 'conv')  # 'conv', 'resnet', 'unet', 'mlp'
        use_bottleneck = config['model'].get('use_bottleneck', False)
        input_size = config.get('dataset', {}).get('input_size', [64, 64, 64])
        if isinstance(input_size, list):
            input_size = tuple(input_size)
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=config['model'].get('attention_heads', 4),
            num_patches_per_patient=config['model'].get('num_patches_per_patient', 50),
            latent_dim=latent_dim,
            input_mode=input_mode,
            fusion_method=fusion_method,
            architecture=architecture,
            use_bottleneck=use_bottleneck,
            input_size=input_size,
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=config['model'].get('dropout_attention', 0.1),
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
    else:
        # Default to ConfigurableAutoencoder for unknown model types
        print(f"Unknown model type: {model_type}, defaulting to ConfigurableAutoencoder")
        return ConfigurableAutoencoder(
            input_channels=in_channels,
            output_channels=config['model'].get('output_channels', 1),
            base_filters=base_filters,
            feature_dim=config['model'].get('feature_dim', 256),
            attention_heads=0,  # No attention by default
            latent_dim=latent_dim,
            input_mode='dose_ct',
            fusion_method='weighted_sum',
            architecture='conv',
            use_bottleneck=False,
            input_size=input_size,
            bilinear_upsampling=config['model'].get('bilinear_upsampling', False),
            dropout_mlp=config['model'].get('dropout_mlp', 0.1),
            dropout_attention=0.0,
            dropout_decoder=config['model'].get('dropout_decoder', 0.1),
            dropout_features=config['model'].get('dropout_features', 0.1),
            config=config
        )
