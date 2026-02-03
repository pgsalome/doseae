import os
import json
import yaml
import copy
import argparse
import numpy as np
from datetime import datetime
import glob
import pandas as pd
from pathlib import Path

# Import Optuna for Bayesian optimization
import optuna
from optuna.trial import Trial

# Import wandb for experiment tracking
import wandb

from utils.optimization import objective, apply_optuna_parameters

# Resolve project root and default configs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "config" / "training_patches_05mm.yaml"

# Set wandb directory
os.environ["WANDB_DIR"] = "/data/pgsal/wandb_doseae"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Optuna-based hyperparameter optimization for DoseAE configs"
    )
    parser.add_argument(
        "--base_config",
        default=str(DEFAULT_BASE_CONFIG),
        help="Path to base YAML/JSON configuration file (default: training_patches_05mm.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        default="./optuna_runs",
        help="Directory where per-trial configurations are stored",
    )
    parser.add_argument(
        "--results_file",
        default="./optuna_results.csv",
        help="CSV file to append trial results",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of trials to run (overrides config)",
    )
    parser.add_argument(
        "--start_trial",
        type=int,
        default=None,
        help="Trial index to resume from. Auto-detect if omitted.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for the TPE sampler (overrides config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device string (e.g., 'cuda', 'cpu'). Defaults to auto-detect.",
    )
    return parser.parse_args()


def ensure_dir(directory):
    """Ensure directory exists."""
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)


def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    if config_path.endswith('.json'):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


def update_config_with_params(base_config, param_dict):
    """Update configuration with parameters."""
    config = copy.deepcopy(base_config)
    for param_name, param_value in param_dict.items():
        keys = param_name.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = param_value
    return config


def _get_nested(config: dict, path: str):
    keys = path.split('.')
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def define_study_params(trial):
    """Define the parameters for a trial in the Optuna study."""
    params = {}

    # Fixed architecture: DualInputAttentionAutoencoder
    params["model.type"] = "dual_input_attention_autoencoder"
    params["model.in_channels"] = 1
    params["model.output_channels"] = 1

    # CRITICAL PARAMETERS ONLY (6 parameters for 20 trials)
    
    # 1. Model capacity - most important
    params["model.base_filters"] = trial.suggest_int(
        "model.base_filters", 32, 128, log=True  # Base filters for ResNet blocks
    )

    # 2. Latent dimension - multi-level compression for medical data
    params["model.latent_dim"] = trial.suggest_categorical(
        "model.latent_dim", [8, 16, 32, 64, 128, 256]  # Multi-level compression options
    )

    # 3. Attention mechanism - key for this architecture
    params["model.attention_heads"] = trial.suggest_categorical(
        "model.attention_heads", [2, 4, 8]  # Number of attention heads
    )

    # 4. Learning rate - critical for training
    params["hyperparameters.learning_rate"] = trial.suggest_float(
        "hyperparameters.learning_rate", 1e-4, 1e-2, log=True
    )

    # 5. Batch size - affects training stability
    params["hyperparameters.batch_size"] = trial.suggest_categorical(
        "hyperparameters.batch_size", [1, 2, 4]  # Small batch sizes for attention model
    )

    # 6. Optimizer - different optimization strategies
    params["training.optimizer"] = trial.suggest_categorical(
        "training.optimizer", ["adam", "sgd"]  # Most effective optimizers
    )

    return params


def find_last_completed_trial(output_dir):
    """Find the highest trial number in the output directory."""
    config_files = glob.glob(os.path.join(output_dir, "config_*_trial*.json"))
    max_trial = -1

    for config_file in config_files:
        import re
        match = re.search(r'trial(\d+)\.json$', config_file)
        if match:
            trial_number = int(match.group(1))
            if trial_number > max_trial:
                max_trial = trial_number

    return max_trial


def run_optimization(
    base_config,
    output_dir,
    start_trial=0,
    n_trials=20,
    random_seed=42,
    results_file=None,
    device_override=None,
):
    """Run Bayesian optimization for hyperparameter tuning."""
    ensure_dir(output_dir)
    if results_file:
        ensure_dir(os.path.dirname(results_file))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_config = None
    best_params = None
    new_results = []

    # Calculate trials to run
    trials_to_run = n_trials - start_trial
    print(f"Will run {trials_to_run} additional trials (from {start_trial} to {n_trials - 1}).")

    if trials_to_run <= 0:
        print("No additional trials needed.")
        return best_config, best_val_loss, new_results

    # Get device
    import torch
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run trials
    for trial_num in range(start_trial, n_trials):
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        print(f"\nTrial {trial_num}/{n_trials - 1}")

        config = apply_optuna_parameters(trial, base_config)
        config["trial_num"] = trial_num
        config["optuna_output_dir"] = output_dir

        # Set up wandb
        if "wandb" in config and config["wandb"].get("use_wandb", False):
            if wandb.run is not None:
                wandb.finish()

            config["wandb"]["name"] = f"trial_{trial_num}"
            config["wandb"]["group"] = f"opt_{timestamp}"
            config["wandb"]["tags"] = ["optimization", f"trial_{trial_num}"]

        # Save config
        config_path = os.path.join(output_dir, f"config_{timestamp}_trial{trial_num}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        # Snapshot parameter values for logging
        params = {}
        optuna_defs = base_config.get('optuna', {}).get('parameters', [])
        for spec in optuna_defs:
            name = spec.get('name')
            if not name:
                continue
            params[name] = _get_nested(config, name)

        try:
            # Run objective function
            val_loss = objective(trial, config, device)

            # Record results
            result = {
                "timestamp": timestamp,
                "trial_number": trial_num,
                "val_loss": val_loss
            }

            # Add parameters to results
            for param_name, param_value in params.items():
                result[param_name] = param_value

            # Add to results
            new_results.append(result)

            # Save results
            if results_file:
                if os.path.exists(results_file):
                    existing_df = pd.read_csv(results_file)
                    new_df = pd.DataFrame([result])
                    results_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    results_df = pd.DataFrame([result])

                results_df.to_csv(results_file, index=False)

            # Update best score
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                best_params = params

                # Save best config
                best_config_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
                with open(best_config_path, "w") as f:
                    json.dump(best_config, f, indent=2, cls=NumpyEncoder)

                print(f"New best validation loss: {val_loss:.4f}")

            # Tell Optuna
            study.tell(trial, val_loss)
            print(f"Trial {trial_num} completed with validation loss: {val_loss:.4f}")

        except Exception as e:
            print(f"Error in trial {trial_num}: {str(e)}")
            study.tell(trial, float('inf'))

        finally:
            if wandb.run is not None:
                wandb.finish()

    # Create report
    report = {
        "timestamp": timestamp,
        "best_validation_loss": float(best_val_loss),
        "best_parameters": best_params,
        "total_trials": n_trials,
        "resumed_from_trial": start_trial if start_trial > 0 else None
    }

    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    return best_config, best_val_loss, new_results


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
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


if __name__ == "__main__":
    args = parse_args()

    base_config = load_config(args.base_config)

    optuna_cfg = base_config.get("optuna", {})
    training_cfg = base_config.get("training", {})

    n_trials = args.n_trials or int(optuna_cfg.get("n_trials", 100))
    random_seed = args.random_seed or int(optuna_cfg.get("seed", training_cfg.get("seed", 42)))

    output_dir = args.output_dir or optuna_cfg.get("configs_dir", "./optuna_runs")
    results_file = args.results_file or optuna_cfg.get("results_file", "./optuna_results.csv")

    ensure_dir(output_dir)
    if results_file:
        ensure_dir(os.path.dirname(results_file))

    if args.start_trial is not None:
        start_trial = args.start_trial
    else:
        last_trial = find_last_completed_trial(output_dir)
        start_trial = last_trial + 1 if last_trial >= 0 else 0

    print(f"Starting from trial {start_trial} (total trials target: {n_trials})")

    best_config, best_val_loss, results = run_optimization(
        base_config=base_config,
        output_dir=output_dir,
        start_trial=start_trial,
        n_trials=n_trials,
        random_seed=random_seed,
        results_file=results_file,
        device_override=args.device,
    )

    print("\nBayesian optimization completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to {results_file}")
