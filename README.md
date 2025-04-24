# Radiation Therapy Dose Distribution Autoencoder

This repository contains implementations of various autoencoder architectures for modeling and compressing radiation therapy dose distributions. The code supports both 2D and 3D dose distributions, informative patch extraction, and multiple state-of-the-art autoencoder architectures.

## Features

- Multiple autoencoder architectures:
  - Variational Autoencoder (VAE)
  - ResNet Autoencoder
  - UNet Autoencoder
  - Convolutional Autoencoder
  - MLP (Fully Connected) Autoencoder
- Support for both 2D and 3D dose distributions
- Informative patch extraction from dose distributions
- Flexible data loading and preprocessing pipeline
- Specialized normalization techniques for dose distributions
- Hyperparameter optimization with Optuna
- Experiment tracking with Weights & Biases
- Comprehensive data analysis and visualization tools

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── __init__.py          # Model factory
│   │   ├── base_ae.py           # Base autoencoder class
│   │   ├── vae.py               # 3D Variational Autoencoder
│   │   ├── vae_2d.py            # 2D Autoencoder models
│   │   ├── resnet_ae.py         # ResNet Autoencoder
│   │   └── unet_ae.py           # UNet Autoencoder
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset and DataLoader creation
│   │   ├── preprocess.py        # Data preprocessing utilities
│   │   └── transforms.py        # Data transformation classes
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py     # Visualization utilities
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── optimization.py      # Optuna hyperparameter optimization
│   │   ├── utils.py             # General utility functions
│   │   ├── patch_extraction.py  # Informative patch extraction 
│   │   ├── wandb_utils.py       # Weights & Biases utilities
│   │   └── extract_patches.py   # Command-line patch extraction script
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.yaml          # Default configuration
│   ├── train.py                 # Main training script
│   └── inference.py             # Inference script
├── notebooks/
│   ├── data_exploration.ipynb   # Data exploration examples
│   └── model_evaluation.ipynb   # Model evaluation examples
├── tests/
│   ├── __init__.py
│   ├── test_models.py           # Tests for models
│   ├── test_data.py             # Tests for data loading
│   └── test_utils.py            # Tests for utilities
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions workflow
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dose-distribution-autoencoder.git
cd dose-distribution-autoencoder
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

### Extracting Slices from NRRD Files

To extract 2D slices from 3D NRRD files:

```bash
python -m src.utils.utils extract_slices --input /path/to/file.nrrd --output /path/to/output.npy --mode all
```

### Extracting Informative Patches

To extract informative patches from slices:

```bash
python -m src.utils.extract_patches --input-dir /path/to/slices --output-dir /path/to/patches
```

### Analyzing Dataset Statistics

```bash
python -m src.utils.utils analyze --input-dir /path/to/data --output-csv stats.csv
```

## Usage

### Configuration

The project uses YAML configuration files to define all model and training parameters. You can modify the default configuration in `src/config/config.yaml` or create your own configuration files.

Key configuration sections include:
- `dataset`: Data loading and splitting parameters
- `preprocessing`: Data preprocessing options
- `model`: Model architecture and parameters
- `hyperparameters`: Training hyperparameters
- `training`: Training process configuration
- `wandb`: Weights & Biases integration
- `optuna`: Hyperparameter optimization settings
- `output`: Output directories for models and results
- `patch_extraction`: Patch extraction parameters

### Training a Model

To train a model with the default configuration:

```bash
python -m src.train --config src/config/config.yaml
```

To train a model on a specific .npy file:

```bash
python -m src.train --config src/config/config.yaml --data-file /path/to/data.npy
```

To run hyperparameter optimization:

```bash
python -m src.train --config src/config/config.yaml --optimize
```

### Inference

To run inference on a trained model:

```bash
python -m src.inference --model /path/to/model.pth --data /path/to/data.npy --config src/config/config.yaml
```

### Using Weights & Biases

This project integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking. To use it:

1. Install wandb: `pip install wandb`
2. Login to your wandb account: `wandb login`
3. Set `use_wandb: true` in your configuration file
4. Configure your project and entity name in the configuration file

To download runs from Weights & Biases:

```bash
python -m src.utils.wandb_utils download --entity your_entity --project your_project --run-id run_id --output-dir ./runs
```

### Hyperparameter Optimization

The project uses [Optuna](https://optuna.org/) for hyperparameter optimization. To use it:

1. Set `use_optuna: true` in your configuration file
2. Configure the optimization parameters in the `optuna` section
3. Run training with the `--optimize` flag

## Model Architectures

### Variational Autoencoder (VAE)

The VAE implementation includes both 2D and 3D versions with configurable latent space dimensions and network architectures.

### ResNet Autoencoder

The ResNet autoencoder uses residual blocks to improve gradient flow during training, making it suitable for deeper architectures.

### UNet Autoencoder

The UNet autoencoder uses skip connections to preserve spatial information, which is important for dose distribution modeling.

### Additional Architectures

- Convolutional Autoencoder: A simple convolutional architecture
- MLP Autoencoder: A fully connected architecture for smaller datasets

## Data Preprocessing

The project includes comprehensive preprocessing options for dose distribution data:

- 95th percentile normalization
- Z-score normalization
- Min-max normalization
- Data augmentation (rotation, flipping, brightness adjustment)
- Informative patch extraction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon previous work in radiation therapy dose distribution modeling. Special thanks to the contributors of the original codebase and the research community working on this important healthcare application.