#!/usr/bin/env python3
"""
Main entry point for DoseAE - Radiation Therapy Dose Distribution Autoencoder.

This script provides a simple interface to your existing functionality:
- Direct training with config
- Neptune hyperparameter optimization
- Data preprocessing
"""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description='DoseAE - Radiation Therapy Dose Distribution Autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  python scripts/main.py preprocess --config config/new_pipeline_config.yaml --splits data/full_splits_auto.json --output ./processed_data --experiment_type patch

  # Train a model directly
  python scripts/main.py train --config config/new_pipeline_config.yaml --entity lung --data_dir ./processed_data --output_dir ./output

  # Run Neptune optimization
  python scripts/main.py optimize --base_config config/config.yaml

  # Run multi-experiments
  python scripts/main.py experiments --experiment 1 --config_dir configs/experiments
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--config', required=True, help='Path to experiment config file')
    preprocess_parser.add_argument('--splits', required=True, help='Path to data splits JSON file')
    preprocess_parser.add_argument('--output', required=True, help='Output directory')
    preprocess_parser.add_argument('--n_patients', type=int, default=5, help='Number of patients to process')
    preprocess_parser.add_argument('--experiment_type', choices=['image', 'patch'], required=True,
                                   help='Type of experiment: image or patch')
    preprocess_parser.add_argument('--workers', type=int, default=1, help='Number of workers for parallel processing')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model directly')
    train_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    train_parser.add_argument('--entity', type=str, required=True, choices=['lung', 'hnc'], help='Entity type')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    train_parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    train_parser.add_argument('--mode', type=str, choices=['train', 'optimize'], default='train', help='Training mode')
    train_parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    # Optimization command (Neptune)
    optimize_parser = subparsers.add_parser('optimize', help='Run Neptune hyperparameter optimization')
    optimize_parser.add_argument('--base_config', default='config/config.yaml', type=str, 
                                 help='Path to base config file')
    
    # Experiments command
    experiments_parser = subparsers.add_parser('experiments', help='Run multi-experiments')
    experiments_parser.add_argument('--experiment', type=str, required=True,
                                    choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'],
                                    help='Experiment to run (1-8, or all)')
    experiments_parser.add_argument('--config_dir', type=str, default='configs/experiments',
                                    help='Directory containing experiment configs')
    experiments_parser.add_argument('--trial_number', type=int, default=None,
                                    help='Trial number for wandb naming')
    experiments_parser.add_argument('--test_mode', action='store_true',
                                    help='Run in test mode with reduced epochs and patients')
    experiments_parser.add_argument('--optuna', action='store_true',
                                    help='Run with Optuna optimization')
    experiments_parser.add_argument('--dataset_root', type=str,
                                    help='Root directory of the dataset')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate command by calling the existing scripts
    if args.command == 'preprocess':
        cmd = [
            'python', 'scripts/preprocess_with_new_pipeline.py',
            '--config', args.config,
            '--splits', args.splits,
            '--output', args.output,
            '--n_patients', str(args.n_patients),
            '--experiment_type', args.experiment_type,
            '--workers', str(args.workers)
        ]
    elif args.command == 'train':
        cmd = [
            'python', 'scripts/train.py',
            '--config', args.config,
            '--entity', args.entity,
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--mode', args.mode,
            '--log_level', args.log_level
        ]
    elif args.command == 'optimize':
        cmd = [
            'python', 'scripts/optimize.py',
            '--base_config', args.base_config
        ]
    elif args.command == 'experiments':
        cmd = [
            'python', 'scripts/train_three_experiments.py',
            '--experiment', args.experiment,
            '--config_dir', args.config_dir
        ]
        if args.trial_number:
            cmd.extend(['--trial_number', str(args.trial_number)])
        if args.test_mode:
            cmd.append('--test_mode')
        if args.optuna:
            cmd.append('--optuna')
        if args.dataset_root:
            cmd.extend(['--dataset_root', args.dataset_root])
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
