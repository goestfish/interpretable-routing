#!/bin/bash
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/oscar/scratch/${USER}/interpretable-routing}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.cache/huggingface}"
MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0125-Instruct}"
MODEL_DIR="${MODEL_DIR:-${SCRATCH_ROOT}/models/allenai__OLMoE-1B-7B-0125-Instruct}"
ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"

mkdir -p "${SCRATCH_ROOT}" "${HF_HOME}" "${MODEL_DIR}"
module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true
conda activate "${ENV_NAME}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export MODEL_ID
export MODEL_DIR

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
model_dir = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN")

snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)

print(f"Model downloaded to: {model_dir}")
PY

echo "Done."
