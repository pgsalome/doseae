#!/bin/bash
# Helper script to launch the preprocessing pipeline on ERISTwo.
# Adjust the paths below to match your project layout and Python module.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (edit these to match your cluster environment)
# ---------------------------------------------------------------------------
PROJECT_ROOT="/PHShome/ps1059/projects/git/doseae"
DATA_SPLITS="${PROJECT_ROOT}/data/full_splits_auto_cluster.json"
CONFIG_FILE="${PROJECT_ROOT}/config/new_pipeline_config.yaml"
OUTPUT_DIR="/PHShome/ps1059/scratch/NSCLC-Cetuximab_AE_cache"
ENV_DIR="/PHShome/ps1059/.venvs/doseae"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
module load python/3.9.13

if [[ ! -d "${ENV_DIR}" ]]; then
    python -m venv "${ENV_DIR}"
    source "${ENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "${ENV_DIR}/bin/activate"
fi

# ---------------------------------------------------------------------------
# Run preprocessing (adjust workers/args for your job resources)
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

python scripts/preprocess.py \
    --config "${CONFIG_FILE}" \
    --splits "${DATA_SPLITS}" \
    --output "${OUTPUT_DIR}" \
    --experiment_type patch \
    --n_patients 100000 \
    --workers 4
