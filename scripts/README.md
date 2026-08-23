# Scripts Directory

This directory contains the main executable scripts for the DoseAE project.

## Quick Start

Use `main.py` as the primary interface:

```bash
# Get help
python scripts/main.py --help

# Preprocess data
python scripts/main.py preprocess --config config/new_pipeline_config.yaml --splits data/full_splits_auto.json --output ./processed_data --experiment_type patch

# Train a model directly
python scripts/main.py train --config config/new_pipeline_config.yaml --entity lung --data_dir ./processed_data --output_dir ./output

# Run Neptune optimization
python scripts/main.py optimize --base_config config/config.yaml

# Run multi-experiments
python scripts/main.py experiments --experiment all --config_dir configs/experiments
```

## Available Scripts

### 1. `main.py` - Unified Entry Point
Simple interface that calls your existing scripts:
- `preprocess` → calls `preprocess.py`
- `train` → calls `train.py`
- `optimize` → calls `optimize.py` (Neptune optimization)
- `experiments` → calls `train_three_experiments.py`

### 2. `train.py` - Direct Training
Train models directly with a config file. Supports both lung and HNC entities.

**Usage:**
```bash
python scripts/train.py --config config.yaml --entity lung --data_dir ./data --output_dir ./output
```

**Features:**
- Entity-based training (lung/HNC)
- Uses shared models from `/models/` directory
- Entity-specific datasets from `/entities/` directory
- Clinical loss calculations (gamma index, DVH)

### 3. `optimize.py` - Neptune Hyperparameter Optimization
Run Bayesian optimization with Neptune tracking.

**Usage:**
```bash
python scripts/optimize.py --base_config config/config.yaml
```

**Features:**
- Optuna-based hyperparameter optimization
- Neptune experiment tracking
- Supports all model architectures
- Configurable number of trials

### 4. `preprocess.py` - Data Preprocessing
Preprocess medical imaging data for both image and patch experiments.

**Usage:**
```bash
python scripts/preprocess.py --config config.yaml --splits data_splits.json --output ./processed --experiment_type patch
```

### 5. `train_three_experiments.py` - Multi-Experiment Training
Run multiple experiments with different model configurations.

**Usage:**
```bash
python scripts/train_three_experiments.py --experiment 1 --config_dir configs/experiments
```

## Project Structure

```
doseae/
├── scripts/                     # Main executable scripts
│   ├── main.py                  # Unified entry point
│   ├── train.py                 # Direct training
│   ├── optimize.py              # Neptune optimization
│   ├── preprocess.py
│   └── train_three_experiments.py
├── models/                      # Shared model architectures
│   ├── __init__.py              # get_model() factory
│   ├── vae.py, resnet_ae.py, etc.
│   └── clinical/                # Clinical losses (gamma, DVH)
├── entities/                    # Entity-specific data handling
│   ├── lung/                    # Lung datasets & preprocessors
│   ├── hnc/                     # HNC datasets & preprocessors
│   └── base/                    # Base classes
└── utils/                       # Shared utilities
```

## Key Design Principles

1. **Shared Models**: All model architectures are in `/models/` and work for all entities
2. **Entity-Specific Data**: Each entity (lung, HNC) has its own datasets and preprocessors
3. **Two Training Interfaces**: 
   - Direct training with config
   - Neptune hyperparameter optimization
4. **Clinical Focus**: Built-in support for radiation therapy metrics (gamma index, DVH)

## Configuration

Your config files should specify:
- `model.type`: Which model architecture to use
- `entity`: Which entity (lung/HNC) for data handling
- `dataset`: Data loading parameters
- `training`: Training parameters
- `clinical_metrics`: Clinical evaluation settings

## Adding New Entities

To add a new entity (e.g., prostate):
1. Create `/entities/prostate/` directory
2. Add entity-specific dataset and preprocessor classes
3. Update the entity choices in `train.py`
4. Models remain the same - they're shared!
