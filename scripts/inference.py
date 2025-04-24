import os
import argparse
import yaml
import torch
import numpy as np

from utils.visualization import (
    load_trained_model,
    process_2d_slice,
    process_3d_volume,
    visualize_2d_slices,
    visualize_3d_volume,
    calculate_metrics,
    process_and_visualize
)


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main function for inference.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with trained autoencoder models")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--data', type=str, required=True, help="Path to the input data file (.npy)")
    parser.add_argument('--config', type=str, default=None, help="Path to the configuration file")
    parser.add_argument('--model_type', type=str, default='vae',
                        help="Model type (vae, mlp_autoencoder, conv_autoencoder)")
    parser.add_argument('--slice_idx', type=int, default=24, help="Index of the slice to visualize (for 3D data)")
    parser.add_argument('--normalize', action='store_true', help="Normalize the input data")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save visualization results")

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process and visualize
    metrics = process_and_visualize(
        model_path=args.model,
        data_path=args.data,
        model_type=args.model_type,
        slice_idx=args.slice_idx,
        config=config,
        normalize=args.normalize,
        save_dir=args.output_dir
    )

    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()