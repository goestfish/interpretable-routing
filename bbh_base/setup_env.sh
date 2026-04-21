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
  echo "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION} ..."
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch
python -m pip install --upgrade \
  accelerate \
  "datasets>=2.18.0" \
  "fsspec[http]" \
  dill \
  huggingface_hub \
  multiprocess \
  pandas \
  pyarrow \
  requests \
  sentencepiece \
  protobuf \
  tqdm \
  xxhash

python -m pip install --upgrade git+https://github.com/huggingface/transformers.git

python - <<'PY'
import accelerate, datasets, dill, fsspec, multiprocess, pandas, pyarrow, requests, torch, transformers, xxhash
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("requests", requests.__version__)
PY

echo "Environment ready: ${ENV_NAME}"
echo "Activate with:"
echo "module load ${ANACONDA_MODULE}"
echo "eval \"\$(conda shell.bash hook)\""
echo "conda activate ${ENV_NAME}"
