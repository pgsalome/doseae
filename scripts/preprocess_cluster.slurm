#!/bin/bash
#SBATCH --job-name=doseae_preproc
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

set -euo pipefail

cd /PHShome/ps1059/projects/git/doseae
mkdir -p logs

./scripts/run_preprocessing_cluster.sh

