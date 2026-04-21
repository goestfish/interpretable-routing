#!/bin/bash
#SBATCH --job-name=olmoe-ztrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="/users/${USER}/interpretable-routing"
SCRATCH_ROOT="/oscar/scratch/${USER}/interpretable-routing"
MODEL_DIR="${SCRATCH_ROOT}/models/allenai__OLMoE-1B-7B-0125-Instruct"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/tulu3_reasoning/tulu3_reasoning_30k.jsonl}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-z_router_cross_shared_2x2_block6_9_k8_tau1.0_bal0.001_perturb0.001_30k}"
OUTPUT_DIR="${PROJECT_ROOT}/z_router/checkpoints/${CHECKPOINT_NAME}"
START_LAYER="${START_LAYER:-6}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
NUM_Z="${NUM_Z:-8}"
SHARING="${SHARING:-cross_layer_shared}"
SHARING_GROUP_SIZE="${SHARING_GROUP_SIZE:-2}"
U_SHARING="${U_SHARING:-per_layer}"
ALPHA_INIT="${ALPHA_INIT:-0.001}"
LAMBDA_BALANCE="${LAMBDA_BALANCE:-0.001}"
LAMBDA_PERTURB="${LAMBDA_PERTURB:-0.001}"
TAU="${TAU:-1.0}"
SOFT_Z="${SOFT_Z:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
ROUTER_LR="${ROUTER_LR:-1e-6}"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}/z_router"

SOFT_Z_ARG=()
if [[ "${SOFT_Z}" == "1" || "${SOFT_Z}" == "true" || "${SOFT_Z}" == "True" ]]; then
  SOFT_Z_ARG=(--soft-z)
fi
SHARING_GROUP_ARG=()
if [[ -n "${SHARING_GROUP_SIZE}" ]]; then
  SHARING_GROUP_ARG=(--sharing-group-size "${SHARING_GROUP_SIZE}")
fi

python train_z_router.py \
  --model-dir "${MODEL_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --start-layer "${START_LAYER}" \
  --block-size "${BLOCK_SIZE}" \
  --num-z "${NUM_Z}" \
  --sharing "${SHARING}" \
  "${SHARING_GROUP_ARG[@]}" \
  --u-sharing "${U_SHARING}" \
  --alpha-init "${ALPHA_INIT}" \
  --lambda-balance "${LAMBDA_BALANCE}" \
  --lambda-perturb "${LAMBDA_PERTURB}" \
  --tau "${TAU}" \
  "${SOFT_Z_ARG[@]}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --router-lr "${ROUTER_LR}"
