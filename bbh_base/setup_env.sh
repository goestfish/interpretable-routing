#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-olmoe-bbh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

module purge
module load "${ANACONDA_MODULE}"
eval "$(conda shell.bash hook)" || true

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists."
else
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch
python -m pip install --upgrade \
  accelerate \
  huggingface_hub \
  requests \
  sentencepiece \
  "protobuf<7" \
  tqdm
python -m pip install --upgrade git+https://github.com/huggingface/transformers.git

