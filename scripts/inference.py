import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.visualization import (
    load_trained_model,
    process_2d_slice,
    process_3d_volume,
    visualize_2d_slices,
    visualize_3d_volume,
    calculate_metrics,
    process_and_visualize,
    find_high_dose_slice
)
from utils.clinical_metrics import ClinicalMetricsCalculator
from data.dataset import create_data_loaders
import wandb


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


def create_dvh_figure(dose_ref, dose_pred, metrics_calculator, title="DVH Comparison"):
    """Create DVH comparison figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate DVHs
    dose_axis_ref, dvh_ref = metrics_calculator.calculate_dvh(dose_ref)
    dose_axis_pred, dvh_pred = metrics_calculator.calculate_dvh(dose_pred)

    # Plot
    ax.plot(dose_axis_ref, dvh_ref, 'b-', linewidth=2, label='Original')
    ax.plot(dose_axis_pred, dvh_pred, 'r--', linewidth=2, label='Reconstructed')

    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max(dose_ref.max(), dose_pred.max()) * 1.1)
    ax.set_ylim(0, 105)

    return fig


def create_gamma_figure(gamma_map, pass_rate, title="Gamma Analysis"):
    """Create gamma map visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Take central slice for 3D data
    if gamma_map.ndim == 3:
        slice_idx = gamma_map.shape[0] // 2
        gamma_slice = gamma_map[slice_idx]
    else:
        gamma_slice = gamma_map

    # Plot
    im = ax.imshow(gamma_slice, cmap='RdYlBu_r', vmin=0, vmax=2)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gamma Index')

    ax.set_title(f'{title} (Pass Rate: {pass_rate:.1f}%)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    return fig


def comprehensive_evaluation(model_path, data_loader, config, output_dir, use_wandb=True):
    """
    Comprehensive evaluation including clinical metrics.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if use_wandb and config['wandb']['use_wandb']:
        wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            name=f"evaluation_{config['model']['type']}_{wandb.util.generate_id()}",
            config=config,
            tags=['evaluation', 'clinical_metrics']
        )

    # Set device
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(model_path, config['model']['type'], config)
    model = model.to(device)
    model.eval()

    # Initialize metrics calculator
    metrics_calculator = ClinicalMetricsCalculator(config)

    # Storage for results
    all_results = []

    # Evaluation loop
    print("Starting evaluation...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 10:  # Limit to first 10 batches for inference
            break

        # Get data
        if isinstance(batch, dict):
            data = batch["image"].to(device)
            mask = batch.get("mask", None)
        else:
            data, _ = batch
            data = data.to(device)
            mask = None

        # Get reconstruction
        with torch.no_grad():
            if config['model']['type'] == 'vae':
                recon, _, _ = model(data)
            else:
                recon = model(data)
                if isinstance(recon, tuple):
                    recon = recon[0]

        # Convert to numpy for clinical metrics
        data_np = data[0, 0].cpu().numpy()
        recon_np = recon[0, 0].cpu().numpy()
        mask_np = mask[0].cpu().numpy() if mask is not None else None

        # Calculate metrics
        basic_metrics = calculate_metrics(data_np, recon_np)

        # Calculate clinical metrics
        clinical_metrics = metrics_calculator.compare_dose_distributions(
            data_np, recon_np, mask_np
        )

        # Combine results
        result = {
            'batch_idx': batch_idx,
            **basic_metrics,
            **clinical_metrics
        }
        all_results.append(result)

        # Create visualizations
        # High dose slice visualization
        high_dose_idx = find_high_dose_slice(data)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if data.dim() == 5:  # 3D data
            original_slice = data[0, 0, high_dose_idx].cpu().numpy()
            recon_slice = recon[0, 0, high_dose_idx].cpu().numpy()
        else:
            original_slice = data[0, 0].cpu().numpy()
            recon_slice = recon[0, 0].cpu().numpy()

        # Original
        im1 = axes[0].imshow(original_slice, cmap='jet')
        axes[0].set_title('Original')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Reconstructed
        im2 = axes[1].imshow(recon_slice, cmap='jet')
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Difference
        diff = np.abs(original_slice - recon_slice)
        im3 = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f'High Dose Slice - Sample {batch_idx}')
        plt.tight_layout()

        # Save figure
        fig_path = output_dir / f'high_dose_comparison_{batch_idx}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        if use_wandb and config['wandb']['use_wandb']:
            wandb.log({f'evaluation/high_dose_slice_{batch_idx}': wandb.Image(fig)})

        plt.close(fig)

        # DVH comparison
        dvh_fig = create_dvh_figure(data_np, recon_np, metrics_calculator,
                                    title=f"DVH Comparison - Sample {batch_idx}")

        dvh_path = output_dir / f'dvh_comparison_{batch_idx}.png'
        plt.savefig(dvh_path, dpi=300, bbox_inches='tight')

        if use_wandb and config['wandb']['use_wandb']:
            wandb.log({f'evaluation/dvh_{batch_idx}': wandb.Image(dvh_fig)})

        plt.close(dvh_fig)

        # Gamma map
        if 'gamma_map' in clinical_metrics:
            gamma_fig = create_gamma_figure(
                clinical_metrics['gamma_map'],
                clinical_metrics['gamma_pass_rate'],
                title=f"Gamma Analysis - Sample {batch_idx}"
            )

            gamma_path = output_dir / f'gamma_map_{batch_idx}.png'
            plt.savefig(gamma_path, dpi=300, bbox_inches='tight')

            if use_wandb and config['wandb']['use_wandb']:
                wandb.log({f'evaluation/gamma_{batch_idx}': wandb.Image(gamma_fig)})

            plt.close(gamma_fig)

    # Aggregate results
    print("Aggregating results...")
    avg_metrics = {}
    for key in all_results[0].keys():
        if key != 'batch_idx' and isinstance(all_results[0][key], (int, float)):
            values = [r[key] for r in all_results]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    # Save results
    results_path = output_dir / 'evaluation_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(avg_metrics, f)

    # Create summary figure
    metrics_to_plot = ['mse', 'mae', 'psnr']
    if 'gamma_pass_rate' in all_results[0]:
        metrics_to_plot.append('gamma_pass_rate')
    if 'D95_diff' in all_results[0]:
        metrics_to_plot.extend(['D95_diff', 'D50_diff'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axes):
            break
        values = [r[metric] for r in all_results if metric in r]
        axes[i].hist(values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(metric)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(np.mean(values), color='red', linestyle='--',
                        label=f'Mean: {np.mean(values):.3f}')
        axes[i].legend()

    plt.suptitle('Distribution of Evaluation Metrics')
    plt.tight_layout()

    summary_fig_path = output_dir / 'metrics_distribution.png'
    plt.savefig(summary_fig_path, dpi=300, bbox_inches='tight')

    if use_wandb and config['wandb']['use_wandb']:
        wandb.log({'evaluation/metrics_distribution': wandb.Image(fig)})

    plt.close(fig)

    # Log average metrics
    if use_wandb and config['wandb']['use_wandb']:
        for metric, stats in avg_metrics.items():
            wandb.log({f'evaluation/avg_{metric}': stats['mean']})

    print(f"Evaluation completed. Results saved to {results_path}")

    if use_wandb and config['wandb']['use_wandb']:
        wandb.finish()

    return avg_metrics


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
    parser.add_argument('--comprehensive', action='store_true',
                        help="Run comprehensive evaluation with clinical metrics")

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.comprehensive and config is not None:
        # Run comprehensive evaluation
        data_loaders = create_data_loaders(config)
        test_loader = data_loaders['test']

        comprehensive_evaluation(
            args.model,
            test_loader,
            config,
            args.output_dir,
            use_wandb=True
        )
    else:
        # Run basic inference
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