#!/bin/bash
#SBATCH --job-name=tulu3-30k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="/users/${USER}/interpretable-routing"
SCRATCH_ROOT="/oscar/scratch/${USER}/interpretable-routing"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/data/tulu3_reasoning/tulu3_reasoning_30k.jsonl}"
MATH="${MATH:-8000}"
ALGEBRA="${ALGEBRA:-4000}"
MATH_GRADE="${MATH_GRADE:-3000}"
INSTRUCTION_FOLLOWING="${INSTRUCTION_FOLLOWING:-9000}"
CODE="${CODE:-6000}"
SEED="${SEED:-13}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-10000}"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/.cache/huggingface/datasets"
mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT}")" "${HF_DATASETS_CACHE}"
cd "${PROJECT_ROOT}"

python data/prepare_tulu3_reasoning_30k.py \
  --output "${OUTPUT}" \
  --math "${MATH}" \
  --algebra "${ALGEBRA}" \
  --math-grade "${MATH_GRADE}" \
  --instruction-following "${INSTRUCTION_FOLLOWING}" \
  --code "${CODE}" \
  --seed "${SEED}" \
  --shuffle-buffer "${SHUFFLE_BUFFER}"
