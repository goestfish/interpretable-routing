#!/bin/bash
#SBATCH --job-name=olmoe-bbh-z
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="/users/${USER}/interpretable-routing"
CODE_ROOT="${PROJECT_ROOT}/bbh_base"
SCRATCH_ROOT="/oscar/scratch/${USER}/interpretable-routing"
MODEL_DIR="${SCRATCH_ROOT}/models/allenai__OLMoE-1B-7B-0125-Instruct"
HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
BBH_CACHE="${SCRATCH_ROOT}/datasets/bbh_official"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-z_router_cross_shared_2x2_block6_9_k8_tau1.0_bal0.001_perturb0.001_30k}"
CHECKPOINT_DIR="${PROJECT_ROOT}/z_router/checkpoints/${CHECKPOINT_NAME}"
EXPERIMENT_DIR="${CODE_ROOT}/results/${CHECKPOINT_NAME}"
OUTPUT_DIR="${EXPERIMENT_DIR}/bbh_eval"
TASKS="${TASKS:-boolean_expressions,formal_fallacies,logical_deduction_five_objects,tracking_shuffled_objects_seven_objects,multistep_arithmetic_two,reasoning_about_colored_objects,web_of_lies,word_sorting}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LIMIT_PER_TASK="${LIMIT_PER_TASK:-}"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONNOUSERSITE=1
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${PROJECT_ROOT}/logs" "${EXPERIMENT_DIR}" "${OUTPUT_DIR}" "${HF_HOME}"
cd "${CODE_ROOT}"

LIMIT_ARG=()
if [[ -n "${LIMIT_PER_TASK}" ]]; then
  LIMIT_ARG=(--limit-per-task "${LIMIT_PER_TASK}")
fi
TASKS_ARG=()
if [[ "${TASKS}" != "all" ]]; then
  TASKS_ARG=(--tasks "${TASKS}")
fi

python run_bbh_z_router.py \
  --model-dir "${MODEL_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${BBH_CACHE}" \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${TASKS_ARG[@]}" \
  "${LIMIT_ARG[@]}"
