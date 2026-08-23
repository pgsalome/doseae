# DoseAE – Radiation Therapy Dose Autoencoders

DoseAE is a research toolkit for learning generative and representation models of radiotherapy dose distributions.  The repository provides a modern preprocessing pipeline that consolidates raw planning data into HDF5 caches, flexible multi‑modal datasets (dose, CT, fused channels), configurable autoencoder architectures, and an Optuna‑driven optimisation workflow – all orchestrated through a single training script.

---

## Key Features

- **Unified preprocessing**: `scripts/preprocess_with_new_pipeline.py` extracts CT volumes, dose grids, metadata, spatial coordinates, lobes, and cached patches into reusable HDF5 files.
- **Multi-channel inputs**: build tensors with any combination of dose, CT, and fused channels; attention blocks can consume spatial, lobe, or dose metadata.
- **Configurable architectures**: the `models/architectures` package contains a configurable autoencoder (conv/resnet/unet/mlp variants), DoseAE ResUNet, and both 2D/3D VAEs.
- **Clinical losses & metrics**: optional gamma, DVH, and auxiliary targets integrate directly into the training loop.
- **Optuna search space in config**: define tunable parameters once under `optuna.parameters`, then run Bayesian optimisation with `scripts/train.py --mode optimize`.
- **WandB logging**: automatic run naming, tagging, and visualisation hooks.

---

## Repository Layout

```
├── config/                     # YAML configurations
│   ├── new_pipeline_config.yaml
│   └── training_patches_05mm.yaml
├── datasets/                   # Dataset helpers & transforms
│   ├── __init__.py
│   ├── loaders.py              # Shared dataset/loader utilities
│   ├── transforms.py           # Legacy transform classes
│   └── cached_dataset.py       # Backwards compatible shim
├── entities/                   # Entity-specific logic (lung, hnc, …)
│   └── lung/datasets/dataset.py
├── models/
│   ├── architectures/          # Configurable autoencoders & DoseAE ResUNet
│   ├── components/             # Shared blocks (attention, resnet, unet)
│   └── __init__.py             # Model factory
├── core/
│   ├── training/trainer.py     # Main training engine
│   ├── evaluation/             # Clinical/standard metrics
│   └── optimization/           # Neptune legacy optimiser
├── scripts/
│   ├── preprocess_with_new_pipeline.py
│   ├── train.py                # Main entry (train / optimise)
│   ├── inference.py            # Evaluation utilities
│   ├── train_three_experiments.py (legacy)
│   └── run_experiments.py (legacy)
├── utils/
│   ├── clinical_loss.py, clinical_metrics.py, optimization.py, …
├── README.md
└── requirements.txt
```

Legacy code from the original project remains under `legacy_*` scripts and unused packages; avoid `data/datasets` – all dataset helpers now live in `datasets/`.

---

## Installation

```bash
git clone https://github.com/your-org/doseae.git
cd doseae

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to use WandB or Neptune ensure the corresponding environment variables / logins are configured.

---

## Data Preparation

1. **Prepare split definitions & mapping**  
   - `config/new_pipeline_config.yaml` references `data/full_splits_auto.json` and `data/ct_dose_file_mapping.json`.  Update these to match your institution identifiers and file layout.

2. **Run preprocessing**  
   ```bash
   python scripts/preprocess_with_new_pipeline.py \
       --config config/new_pipeline_config.yaml \
       --splits data/full_splits_auto.json \
       --output /data/NSCLC-Cetuximab_AE_cache \
       --n_patients 8 \
       --experiment_type patch \
       --workers 4
   ```
   This creates `/processed_patches/{split}.h5` and `/processed_images/{split}.h5` directories containing the cached tensors and metadata consumed during training.

---

### Synthetic Dose Augmentation (OpenTPS)

DoseAE can synthesise additional treatment plans per patient by wrapping the [OpenTPS](https://gitlab.com/openmcsquare/opentps) photon planning toolkit.  This is optional and disabled by default.

1. Install OpenTPS (either `pip install opentps` or clone into `external/OpenTPS` – the preprocessing script assumes the latter).
2. Ensure `psutil` is available (`pip install psutil`) and that Collapse Cone convolution resources in OpenTPS can be executed on your system.
3. Enable augmentation in the preprocessing config:
   ```yaml
   augmentation:
     enable_synthetic_dose: true
     synthetic_doses_per_patient: 3
     opentps:
       core_path: "external/OpenTPS/opentps_core"   # set if different
       min_beams: 3
       max_beams: 5
       target_isodose_range: [0.75, 0.95]
       prescription_scale_range: [0.9, 1.1]
   ```
4. Run `scripts/preprocess.py` as usual.  For every patient the pipeline will derive a high-dose target region from the normalised plan, randomise beam angles within the provided ranges, optimise a new fluence map with OpenTPS, and append the resulting dose volumes to the same split (IDs are suffixed with `_synthetic_##`).

All caches, full-image exports, and HDF5 consolidation automatically include the augmented samples.  If augmentation cannot be initialised (e.g., OpenTPS missing), the preprocessing script logs a warning and continues with the original cohort.

To generate clinically perturbed plans directly (without the rest of preprocessing), use:

```bash
python scripts/run_opentps_perturbations.py \
    --config config/single_patient_synth.yaml \
    --mapping data/ct_dose_file_mapping.json \
    --patient_id 0617697905 \
    --n_samples 3 \
    --output_dir outputs/perturbations
```

---

## Training

The entire training pipeline is handled by `scripts/train.py`.  The YAML config fully specifies dataset paths, model choices, clinical losses, logging, and optimisation ranges.

### Configuration essentials (`config/training_patches_05mm.yaml`)

- `dataset`: references the preprocessed HDF5 files; toggles between patches vs. full images, 2D slice mode, transform usage, and channel fusion.
- `model`: chooses the architecture type (`resnet_ae`, `unet_ae`, `conv_autoencoder`, `mlp_autoencoder`, `vae`, `doseae_resunet`) and sets latent dimensions, attention config, auxiliary heads, etc.
- `hyperparameters` & `training`: standard optimisation parameters (lr, batch size, scheduler, loss mode, grad clip, accumulation steps).
- `clinical_metrics` / `loss_function`: enable gamma/DVH losses and diagnostics.
- `optuna.parameters`: declarative search space for optimisation mode.
- `output`: where checkpoints, logs, and training history are stored (defaults to config paths but can be overridden on the CLI).

### Run a standard training job

```bash
python scripts/train.py \
    --config config/training_patches_05mm.yaml \
    --entity lung \
    --data_dir /data/NSCLC-Cetuximab_AE_cache \
    --mode train \
    --log_level INFO
```

Key behaviours:
- Output directories are resolved from the config (and created automatically); use `--output_dir` to override `output.results_dir`.
- Datasets are constructed through `datasets.loaders.create_data_loaders`, so no more imports from the removed `data.datasets` package.
- The trainer automatically merges legacy `hyperparameters` values into the `training` block and handles attention/clinical loss wiring.
- Clinical metrics (gamma, DVH) are computed periodically if enabled and logged to WandB.

### Train with WandB

Ensure `wandb.use_wandb: true` and set `project_name`, `entity`, and optional tags in the config.  The script will initialise WandB before training begins and log reconstruction samples/metrics.

---

## Hyperparameter Optimisation

Optuna parameters are now declared directly in the config under `optuna.parameters`.  Each entry describes the dot-path into the config, the sampler type, and ranges/choices:

```yaml
optuna:
  use_optuna: true
  n_trials: 50
  timeout: 86400
  parameters:
    - name: model.latent_dim
      type: int
      low: 32
      high: 256
      log: true
    - name: training.loss
      type: categorical
      choices: ['mse', 'combined', 'clinical']
    - name: loss_function.weights.gamma
      type: float
      low: 0.0
      high: 0.5
```

Run optimisation with:

```bash
python scripts/train.py \
    --config config/training_patches_05mm.yaml \
    --entity lung \
    --data_dir /data/NSCLC-Cetuximab_AE_cache \
    --mode optimize
```

`utils/optimization.py` consumes the parameter list and applies suggestions to a copied config before each trial; no code changes are needed to expand the search space – simply edit the YAML.

> **Neptune legacy support**: `core/optimization/neptune_optimizer.py` remains for backwards compatibility.  The new Optuna flow does not require Neptune.

---

## Inference & Evaluation

Use `scripts/inference.py` to generate clinical metrics, DVH plots, and gamma maps for a trained checkpoint:

```bash
python scripts/inference.py \
    --config config/training_patches_05mm.yaml \
    --model /path/to/checkpoint.pth \
    --data_dir /data/NSCLC-Cetuximab_AE_cache \
    --output_dir ./evaluation \
    --use_wandb
```

The script relies on `datasets.loaders.create_data_loaders`, so the same HDF5 caches and config are required.

---

## Manuscript Reproduction

Rebuild the packaged manuscript figures and tables from the exported
patient-level predictions and analysis caches:

```bash
./.venv/bin/python scripts/reproduce_manuscript.py --strict
```

The default workflow is cache-only and does not retrain models, rerun DoseAE
inference, or extract embeddings. Inspect all commands and dependencies without
writing outputs with:

```bash
./.venv/bin/python scripts/reproduce_manuscript.py --dry-run --strict
```

Source-derived Table 1 data and supplementary imaging require private local
inputs and are therefore opt-in:

```bash
./.venv/bin/python scripts/reproduce_manuscript.py \
    --include-source-derived \
    --include-supplementary \
    --strict
```

The workflow writes a SHA-256 output manifest to
`oliver_paper/manifests/reproducibility_manifest.json`.

---

## Notes & Legacy Artifacts

- `scripts/train_three_experiments.py` and `scripts/run_experiments.py` are preserved for reference; they now import the compatibility shim in `datasets/cached_dataset.py`.
- The original `data/datasets` package was removed to avoid clashes with the new dataset helpers.  Update any downstream notebooks or scripts to import from `datasets.*` instead.
- Pretrained encoder support is currently a stub (`pretrained_encoder` block); loading ImageNet / MedImageNet weights will require further integration.

---

## Contributing

1. Fork and clone the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Make your edits and ensure `python -m compileall` (or your linter/tests) succeed.
4. Open a pull request describing your changes; include sample commands/config snippets where relevant.

---

## Citation

If you use DoseAE in academic work, please cite the repository and (if applicable) the associated publication once available.  A proper BibTeX entry will be added when the manuscript is released.
