import os
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
import glob

# Import Optuna for Bayesian optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import Trial

# Import wandb for handling runs
import wandb

# Import from your project
from utils.optimization import define_model_params, define_training_params, objective
from scripts.train import train


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def load_base_config(config_path: str) -> Dict[str, Any]:
    """
    Load the base configuration file.

    Args:
        config_path: Path to the base configuration file

    Returns:
        Base configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def update_config_with_params(base_config: Dict[str, Any], param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with parameters.

    Args:
        base_config: Base configuration dictionary
        param_dict: Dictionary of parameters (with nested paths as keys)

    Returns:
        Updated configuration dictionary
    """
    config = copy.deepcopy(base_config)

    for param_name, param_value in param_dict.items():
        # Convert period-separated path to nested dictionary keys
        keys = param_name.split(".")

        # Navigate to the right level in the config
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = param_value

    return config


def define_study_params(trial: Trial, high_dim: bool = False) -> Dict[str, Any]:
    """
    Define the parameters for a trial in the Optuna study.

    Args:
        trial: Optuna trial object
        high_dim: Whether to use high-dimensional search space

    Returns:
        Dictionary of parameters
    """
    params = {}

    # Model architecture parameters
    params["model.type"] = trial.suggest_categorical(
        "model.type", ["vae", "resnet_ae", "unet_ae", "conv_autoencoder", "mlp_autoencoder"]
    )

    params["model.latent_dim"] = trial.suggest_int(
        "model.latent_dim", 32, 512, log=True
    )

    params["model.base_filters"] = trial.suggest_int(
        "model.base_filters", 16, 64, log=True
    )

    # Training hyperparameters
    params["hyperparameters.learning_rate"] = trial.suggest_float(
        "hyperparameters.learning_rate", 1e-5, 1e-2, log=True
    )

    params["hyperparameters.batch_size"] = trial.suggest_categorical(
        "hyperparameters.batch_size", [4, 8, 16, 32, 64]
    )

    params["hyperparameters.weight_decay"] = trial.suggest_float(
        "hyperparameters.weight_decay", 1e-6, 1e-3, log=True
    )

    params["training.optimizer"] = trial.suggest_categorical(
        "training.optimizer", ["adam", "sgd", "rmsprop"]
    )

    params["training.scheduler"] = trial.suggest_categorical(
        "training.scheduler", ["reduce_on_plateau", "cosine_annealing", "none"]
    )

    # VAE-specific parameters
    if params["model.type"] == "vae":
        params["hyperparameters.beta"] = trial.suggest_float(
            "hyperparameters.beta", 0.1, 10.0, log=True
        )

    # Preprocessing parameters
    params["preprocessing.normalize"] = trial.suggest_categorical(
        "preprocessing.normalize", [True, False]
    )

    params["preprocessing.use_tanh_output"] = trial.suggest_categorical(
        "preprocessing.use_tanh_output", [True, False]
    )

    # Additional parameters for high-dimensional search
    if high_dim:
        params["model.dropout_rate"] = trial.suggest_float(
            "model.dropout_rate", 0.1, 0.5
        )

        params["model.is_2d"] = trial.suggest_categorical(
            "model.is_2d", [True, False]
        )

        params["training.non_zero_loss"] = trial.suggest_categorical(
            "training.non_zero_loss", [True, False]
        )

        params["training.grad_clip"] = trial.suggest_float(
            "training.grad_clip", 0.01, 1.0, log=True
        )

        params["hyperparameters.early_stopping_patience"] = trial.suggest_int(
            "hyperparameters.early_stopping_patience", 5, 20
        )

    return params


def extract_trial_number_from_filename(filename):
    """Extract trial number from config filename."""
    match = re.search(r'trial(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return -1


def extract_suggestions_from_config(config_path):
    """Extract parameter suggestions from a configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    suggestions = {}

    # Extract model parameters
    suggestions["model.type"] = config.get("model", {}).get("type", "vae")
    suggestions["model.latent_dim"] = config.get("model", {}).get("latent_dim", 128)
    suggestions["model.base_filters"] = config.get("model", {}).get("base_filters", 32)

    # Extract hyperparameters
    suggestions["hyperparameters.learning_rate"] = config.get("hyperparameters", {}).get("learning_rate", 1e-4)
    suggestions["hyperparameters.batch_size"] = config.get("hyperparameters", {}).get("batch_size", 16)
    suggestions["hyperparameters.weight_decay"] = config.get("hyperparameters", {}).get("weight_decay", 1e-4)

    # Extract training parameters
    suggestions["training.optimizer"] = config.get("training", {}).get("optimizer", "adam")
    suggestions["training.scheduler"] = config.get("training", {}).get("scheduler", "reduce_on_plateau")

    # Extract beta for VAE if applicable
    if suggestions["model.type"] == "vae":
        suggestions["hyperparameters.beta"] = config.get("hyperparameters", {}).get("beta", 1.0)

    # Extract preprocessing parameters
    suggestions["preprocessing.normalize"] = config.get("preprocessing", {}).get("normalize", True)
    suggestions["preprocessing.use_tanh_output"] = config.get("preprocessing", {}).get("use_tanh_output", True)

    return suggestions


def find_completed_trials(output_dir: str) -> tuple:
    """
    Find all completed trials in the output directory.

    Args:
        output_dir: Directory containing configuration files

    Returns:
        Tuple of (max_trial_number, config_files_dict)
    """
    # Find all config files
    config_files = glob.glob(os.path.join(output_dir, "config_*_trial*.json"))

    # Extract trial numbers and create mapping
    config_files_dict = {}
    max_trial_number = -1

    for config_file in config_files:
        trial_number = extract_trial_number_from_filename(config_file)
        if trial_number >= 0:
            config_files_dict[trial_number] = config_file
            max_trial_number = max(max_trial_number, trial_number)

    return max_trial_number, config_files_dict


def load_previous_trials(results_file, trial_numbers):
    """
    Load previous trial results from CSV file.

    Args:
        results_file: Path to CSV with previous results
        trial_numbers: List of trial numbers to load

    Returns:
        Dictionary mapping trial numbers to results
    """
    if not os.path.exists(results_file):
        return {}

    results_df = pd.read_csv(results_file)

    # Filter for the specified trial numbers and convert to dictionary
    trial_results = {}
    for _, row in results_df.iterrows():
        if 'trial_number' in row and row['trial_number'] in trial_numbers:
            trial_results[row['trial_number']] = row.to_dict()

    return trial_results


def ensure_dir(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)


def run_custom_bayesian_optimization(
        base_config: Dict[str, Any],
        output_dir: str,
        start_trial: int = 0,
        n_trials: int = 20,
        random_state: int = 42,
        results_file: str = "experiment_results.csv",
        high_dim: bool = False
):
    """
    Run Bayesian optimization for hyperparameter tuning.

    Args:
        base_config: Base configuration dictionary
        output_dir: Directory to save experiment configurations
        start_trial: Trial number to start from
        n_trials: Total number of trials to run
        random_state: Random seed
        results_file: Path to save experiment results
        high_dim: Whether to use high-dimensional search space
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Ensure directory for results file exists
    ensure_dir(os.path.dirname(results_file))

    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if we need to load previous results
    resuming = start_trial > 0

    if resuming:
        print(f"Resuming from trial {start_trial} to {n_trials}.")
        # Load previous trials to guide the new trials (not strictly necessary)
        previous_results = []
        if os.path.exists(results_file):
            try:
                previous_df = pd.read_csv(results_file)
                previous_results = previous_df.to_dict('records')

                # Find best result so far
                best_val_loss = min([r.get('val_loss', float('inf')) for r in previous_results], default=float('inf'))
                print(
                    f"Loaded {len(previous_results)} previous results. Best validation loss so far: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error loading previous results: {e}")
                previous_results = []
    else:
        print(f"Starting fresh optimization with {n_trials} trials.")
        previous_results = []

    # List to store new results
    new_results = []

    # Create a sampler for parameter exploration
    sampler = optuna.samplers.TPESampler(seed=random_state)

    # Create a study object (but we won't use study.optimize directly)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_config = None
    best_params = None

    # Load previous best score if available
    if previous_results:
        for result in previous_results:
            val_loss = result.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Extract parameters from this result
                params = {k: v for k, v in result.items()
                          if k.startswith('model.') or k.startswith('hyperparameters.') or k.startswith(
                        'training.') or k.startswith('preprocessing.')}
                best_params = params

    # For each trial to run
    trials_to_run = n_trials - start_trial
    print(f"Will run {trials_to_run} additional trials (from {start_trial} to {n_trials - 1}).")

    if trials_to_run <= 0:
        print("No additional trials needed.")
        return best_config, best_val_loss, new_results

    # Set up device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for trial_num in range(start_trial, n_trials):
        # Create a trial object
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

        print(f"\nTrial {trial_num}/{n_trials - 1}")

        # Get parameters for this trial
        params = define_study_params(trial, high_dim)
        print(f"Parameters: {params}")

        # Create a new config with the parameters
        config = update_config_with_params(base_config, params)

        # Set trial number in the config
        config["trial_num"] = trial_num

        # Modify Wandb configuration for this trial
        if "wandb" in config and config["wandb"].get("use_wandb", False):
            # Terminate any existing wandb run
            if wandb.run is not None:
                wandb.finish()

            # Create a unique run name for this trial
            config["wandb"]["name"] = f"trial_{trial_num}"

            # Add group to associate all trials together
            group_prefix = "resumed_opt" if resuming else "opt"
            config["wandb"]["group"] = f"{group_prefix}_{timestamp}"

            # Add trial-specific tags
            if "tags" not in config["wandb"]:
                config["wandb"]["tags"] = []

            config["wandb"]["tags"].extend([f"optimization", f"trial_{trial_num}"])

            # Make sure wandb doesn't try to resume the previous run
            os.environ["WANDB_RESUME"] = "never"

        # Save the configuration
        config_path = os.path.join(output_dir, f"config_{timestamp}_trial{trial_num}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        try:
            # Run training
            val_loss = objective(trial, config, device)

            # Record the results
            experiment_result = {
                "timestamp": timestamp,
                "trial_number": trial_num,
                "val_loss": val_loss,
            }

            # Add parameters to results
            for param_name, param_value in params.items():
                experiment_result[param_name] = param_value

            # Add to results list
            new_results.append(experiment_result)

            # Save results to CSV
            if os.path.exists(results_file):
                # Load existing results and append new ones
                existing_df = pd.read_csv(results_file)
                new_df = pd.DataFrame([experiment_result])
                results_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Create new results file
                results_df = pd.DataFrame([experiment_result])

            results_df.to_csv(results_file, index=False)

            # Update best score and config
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                best_params = params

                # Save the best configuration separately
                best_config_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
                with open(best_config_path, "w") as f:
                    json.dump(best_config, f, indent=2, cls=NumpyEncoder)

                print(f"New best validation loss: {val_loss:.4f}")

            # Tell Optuna about this result for future sampling
            study.tell(trial, val_loss)

            print(f"Trial {trial_num} completed with validation loss: {val_loss:.4f}")

        except Exception as e:
            print(f"Error in trial {trial_num}: {str(e)}")

            # Tell Optuna this trial failed
            study.tell(trial, float('inf'))  # Severe penalty

        finally:
            # Ensure wandb run is finished
            if wandb.run is not None:
                wandb.finish()

    # Create a comprehensive report
    report = {
        "timestamp": timestamp,
        "best_validation_loss": float(best_val_loss),
        "best_parameters": best_params,
        "total_trials": n_trials,
        "resumed_from_trial": start_trial if resuming else None,
        "runtime_info": {
            "start_time": timestamp,
            "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "n_trials": n_trials,
            "random_state": random_state
        }
    }

    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    print(f"Optimization report saved to {report_path}")

    # Generate visualization if enough trials completed
    if len(study.trials) > 2:
        try:
            # Plot optimization history
            fig = plot_optimization_history(study)
            fig.update_layout(width=1000, height=600)
            history_path = os.path.join(output_dir, f"optimization_history_{timestamp}.png")
            fig.write_image(history_path)
            print(f"Optimization history plot saved to {history_path}")

            # Plot parameter importances
            fig = plot_param_importances(study)
            fig.update_layout(width=1000, height=600)
            importances_path = os.path.join(output_dir, f"param_importances_{timestamp}.png")
            fig.write_image(importances_path)
            print(f"Parameter importances plot saved to {importances_path}")

        except Exception as e:
            print(f"Warning: Could not generate visualization plots: {e}")

    return best_config, best_val_loss, new_results


def find_last_completed_trial(output_dir: str) -> int:
    """
    Find the highest trial number in the output directory.

    Args:
        output_dir: Directory containing configuration files

    Returns:
        Highest trial number found, or -1 if none found
    """
    max_trial, _ = find_completed_trials(output_dir)
    return max_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Dose Distribution Autoencoder experiments with Bayesian optimization")
    parser.add_argument("--base_config", default="./config/config.json", type=str, help="Path to base config file")
    parser.add_argument("--output_dir", type=str, default="./config/bayesian_opt",
                        help="Directory to save experiment configs")
    parser.add_argument("--results_file", type=str, default="./results/bayesian_opt_results.csv",
                        help="Path to save experiment results")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start_trial", type=int, default=None,
                        help="Trial number to start from (if None, auto-detect)")
    parser.add_argument("--high_dim", action="store_true", help="Use high-dimensional search space")
    args = parser.parse_args()

    # Load base configuration
    base_config = load_base_config(args.base_config)

    # Determine start trial
    if args.start_trial is not None:
        start_trial = args.start_trial
    else:
        # Auto-detect the last completed trial
        last_trial = find_last_completed_trial(args.output_dir)
        start_trial = last_trial + 1 if last_trial >= 0 else 0

    print(f"Starting from trial {start_trial}")

    # Run custom Bayesian optimization
    best_config, best_val_loss, results = run_custom_bayesian_optimization(
        base_config=base_config,
        output_dir=args.output_dir,
        start_trial=start_trial,
        n_trials=args.n_trials,
        random_state=args.random_seed,
        results_file=args.results_file,
        high_dim=args.high_dim
    )

    print("\nBayesian optimization completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to {args.results_file}")