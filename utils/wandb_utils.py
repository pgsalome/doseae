"""
Utilities for working with Weights & Biases (wandb).
"""

import os
import argparse
import yaml
import wandb
from tqdm import tqdm


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


def download_wandb_run_files(entity, project, run_id, output_dir, exclude_pattern=None):
    """
    Download files from a wandb run.

    Args:
        entity (str): wandb entity (username or team name)
        project (str): wandb project name
        run_id (str): wandb run ID
        output_dir (str): Directory to save the downloaded files
        exclude_pattern (str, optional): Pattern to exclude files
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get the run
    run = api.run(f"{entity}/{project}/{run_id}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download files
    print(f"Downloading files from run {run_id}...")
    for file in tqdm(run.files()):
        if exclude_pattern and exclude_pattern in file.name:
            continue
        file.download(root=output_dir)

    print(f"Files downloaded successfully to {output_dir}")


def list_wandb_runs(entity, project):
    """
    List all runs in a wandb project.

    Args:
        entity (str): wandb entity (username or team name)
        project (str): wandb project name
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Print run information
    print(f"Runs in {entity}/{project}:")
    print(f"{'Run ID':<12} {'Name':<30} {'Created At':<25} {'State':<10}")
    print("-" * 80)

    for run in runs:
        print(f"{run.id:<12} {run.name:<30} {run.created_at:<25} {run.state:<10}")

    print(f"\nTotal runs: {len(runs)}")


def initialize_wandb(config, resume=False, run_id=None):
    """
    Initialize wandb for tracking an experiment.

    Args:
        config (dict): Configuration dictionary
        resume (bool): Whether to resume a previous run
        run_id (str, optional): Run ID to resume

    Returns:
        wandb.Run: Initialized wandb run
    """
    # Check if wandb is enabled
    if not config['wandb'].get('use_wandb', False):
        print("Weights & Biases tracking is disabled in the configuration.")
        return None

    # Initialize wandb
    if resume and run_id:
        run = wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            id=run_id,
            resume="must",
            config=config
        )
    else:
        run = wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            config=config
        )

    return run


def log_model_to_wandb(model_path, config):
    """
    Log a trained model to wandb.

    Args:
        model_path (str): Path to the model file
        config (dict): Configuration dictionary

    Returns:
        wandb.Artifact: Created artifact
    """
    # Check if wandb is enabled
    if not config['wandb'].get('use_wandb', False):
        print("Weights & Biases tracking is disabled in the configuration.")
        return None

    # Initialize wandb if not already initialized
    if wandb.run is None:
        run = initialize_wandb(config)
    else:
        run = wandb.run

    # Create an artifact
    model_artifact = wandb.Artifact(
        name=f"{config['model']['type']}_model",
        type="model",
        description=f"Trained {config['model']['type']} model"
    )

    # Add the model file to the artifact
    model_artifact.add_file(model_path)

    # Log the artifact
    run.log_artifact(model_artifact)

    return model_artifact


def download_model_from_wandb(entity, project, model_artifact_name, output_dir):
    """
    Download a model artifact from wandb.

    Args:
        entity (str): wandb entity (username or team name)
        project (str): wandb project name
        model_artifact_name (str): Name of the model artifact (e.g., "vae_model:v0")
        output_dir (str): Directory to save the downloaded model

    Returns:
        str: Path to the downloaded model file
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get the artifact
    artifact = api.artifact(f"{entity}/{project}/{model_artifact_name}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the artifact
    artifact_dir = artifact.download(root=output_dir)

    # Return the path to the model file
    model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
    if not model_files:
        raise ValueError(f"No model files found in the artifact {model_artifact_name}")

    model_path = os.path.join(artifact_dir, model_files[0])
    print(f"Model downloaded to {model_path}")

    return model_path


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description="Utilities for working with Weights & Biases")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List runs command
    list_parser = subparsers.add_parser("list", help="List all runs in a project")
    list_parser.add_argument("--entity", type=str, required=True, help="wandb entity (username or team)")
    list_parser.add_argument("--project", type=str, required=True, help="wandb project name")

    # Download run files command
    download_parser = subparsers.add_parser("download", help="Download files from a wandb run")
    download_parser.add_argument("--entity", type=str, required=True, help="wandb entity (username or team)")
    download_parser.add_argument("--project", type=str, required=True, help="wandb project name")
    download_parser.add_argument("--run-id", type=str, required=True, help="wandb run ID")
    download_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save files")
    download_parser.add_argument("--exclude", type=str, help="Pattern to exclude files")

    # Download model command
    model_parser = subparsers.add_parser("download-model", help="Download a model artifact from wandb")
    model_parser.add_argument("--entity", type=str, required=True, help="wandb entity (username or team)")
    model_parser.add_argument("--project", type=str, required=True, help="wandb project name")
    model_parser.add_argument("--artifact", type=str, required=True, help="Model artifact name (e.g., 'vae_model:v0')")
    model_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save model")

    args = parser.parse_args()

    if args.command == "list":
        list_wandb_runs(args.entity, args.project)
    elif args.command == "download":
        download_wandb_run_files(args.entity, args.project, args.run_id, args.output_dir, args.exclude)
    elif args.command == "download-model":
        download_model_from_wandb(args.entity, args.project, args.artifact, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()