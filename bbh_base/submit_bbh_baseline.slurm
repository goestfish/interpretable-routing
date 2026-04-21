#!/bin/bash
#SBATCH --job-name=olmoe-bbh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

CODE_ROOT="/users/${USER}/interpretable-routing/oscar_bbh_baseline"
SCRATCH_ROOT="/oscar/scratch/${USER}/interpretable-routing"
MODEL_DIR="${SCRATCH_ROOT}/models/allenai__OLMoE-1B-7B-0125-Instruct"
HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
BBH_CACHE="${SCRATCH_ROOT}/datasets/bbh_official"
OUTPUT_DIR="${CODE_ROOT}/results/bbh_baseline"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${CODE_ROOT}/logs"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

cd "${CODE_ROOT}"

python run_bbh_baseline.py \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${BBH_CACHE}" \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 256
