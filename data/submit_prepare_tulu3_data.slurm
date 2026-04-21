#!/bin/bash
#SBATCH --job-name=tulu3-data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="/users/${USER}/interpretable-routing"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/data/tulu3_subsets"
cd "${PROJECT_ROOT}"

python data/prepare_tulu3_subset.py \
  --output data/tulu3_subsets/tulu3_reasoning_small.jsonl \
  --algebra 1000 \
  --instruction-following 1000 \
  --gsm 0 \
  --seed 13 \
  --sleep 1.0
