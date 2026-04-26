#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
  torch==2.3.0 \
  torchvision==0.18.0 \
  torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

"${PYTHON}" -m pip install seaborn thop timm einops opencv-python pillow pycocotools requests gdown

(
  cd selective_scan
  "../${PYTHON}" -m pip install .
)

"${PYTHON}" -m pip install --no-build-isolation --no-deps -v -e .

"${PYTHON}" tools/check_visdrone_vid_runtime.py
