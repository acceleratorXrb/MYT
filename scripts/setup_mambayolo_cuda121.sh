#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.0}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

wheel_tag="$("${PYTHON}" - <<'PY'
import platform
import sys

machine = platform.machine().lower()
if machine == "x86_64":
    platform_tag = "linux_x86_64"
elif machine in {"aarch64", "arm64"}:
    platform_tag = "linux_aarch64"
else:
    raise SystemExit(f"unsupported machine for prebuilt CUDA wheels: {machine}")

major, minor = sys.version_info[:2]
if (major, minor) not in {(3, 10), (3, 11), (3, 12)}:
    raise SystemExit(f"unsupported Python version for pinned CUDA wheels: {major}.{minor}")

py_tag = f"cp{major}{minor}"
print(f"{py_tag}-{py_tag}-{platform_tag}")
PY
)"

torch_url="${PYTORCH_WHEEL_BASE}/torch-${TORCH_VERSION}+cu121-${wheel_tag}.whl"
torchvision_url="${PYTORCH_WHEEL_BASE}/torchvision-${TORCHVISION_VERSION}+cu121-${wheel_tag}.whl"
torchaudio_url="${PYTORCH_WHEEL_BASE}/torchaudio-${TORCHAUDIO_VERSION}+cu121-${wheel_tag}.whl"

"${PYTHON}" -m pip install --upgrade pip -i "${PIP_INDEX_URL}"
if "${PYTHON}" - <<PY
import importlib

expected = {
    "torch": "${TORCH_VERSION}",
    "torchvision": "${TORCHVISION_VERSION}",
    "torchaudio": "${TORCHAUDIO_VERSION}",
}
for name, version in expected.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        raise SystemExit(1)
    if getattr(module, "__version__", "").split("+", 1)[0] != version:
        raise SystemExit(1)
print("Pinned torch packages already installed, skipping reinstall")
PY
then
  :
else
  "${PYTHON}" -m pip install "${torch_url}" "${torchvision_url}" "${torchaudio_url}" -i "${PIP_INDEX_URL}"
fi

"${PYTHON}" -m pip install \
  seaborn \
  thop \
  timm \
  einops \
  opencv-python \
  pillow \
  pycocotools \
  requests \
  gdown \
  -i "${PIP_INDEX_URL}"

(
  cd selective_scan
  "../${PYTHON}" -m pip install --no-build-isolation --no-deps -v . -i "${PIP_INDEX_URL}"
)

"${PYTHON}" -m pip install --no-build-isolation --no-deps -v -e . -i "${PIP_INDEX_URL}"

"${PYTHON}" tools/check_visdrone_vid_runtime.py
