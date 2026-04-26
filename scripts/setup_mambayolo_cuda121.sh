#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.0}"

detect_cuda_tag() {
  if [[ -n "${TORCH_CUDA_TAG:-}" ]]; then
    printf '%s\n' "${TORCH_CUDA_TAG}"
    return
  fi

  local nvcc_version
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_version="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n1)"
    case "${nvcc_version}" in
      11.8) printf '%s\n' "cu118"; return ;;
      12.1) printf '%s\n' "cu121"; return ;;
    esac
  fi

  if [[ -n "${CUDA_HOME:-}" && -f "${CUDA_HOME}/version.txt" ]]; then
    if grep -q 'CUDA Version 11.8' "${CUDA_HOME}/version.txt"; then
      printf '%s\n' "cu118"
      return
    fi
    if grep -q 'CUDA Version 12.1' "${CUDA_HOME}/version.txt"; then
      printf '%s\n' "cu121"
      return
    fi
  fi

  printf '%s\n' "cu121"
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
TORCH_CUDA_TAG="$(detect_cuda_tag)"
PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/${TORCH_CUDA_TAG}}"

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

torch_url="${PYTORCH_WHEEL_BASE}/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}-${wheel_tag}.whl"
torchvision_url="${PYTORCH_WHEEL_BASE}/torchvision-${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}-${wheel_tag}.whl"
torchaudio_url="${PYTORCH_WHEEL_BASE}/torchaudio-${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG}-${wheel_tag}.whl"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}"
if "${PYTHON}" - <<PY
import importlib

expected = {
    "torch": "${TORCH_VERSION}",
    "torchvision": "${TORCHVISION_VERSION}",
    "torchaudio": "${TORCHAUDIO_VERSION}",
}
expected_cuda_tag = "${TORCH_CUDA_TAG}"
cuda_tag_map = {
    "11.8": "cu118",
    "12.1": "cu121",
}

def installed_cuda_tag(module):
    cuda_version = getattr(getattr(module, "version", None), "cuda", None)
    return cuda_tag_map.get(cuda_version, None)

for name, version in expected.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        raise SystemExit(1)
    if getattr(module, "__version__", "").split("+", 1)[0] != version:
        raise SystemExit(1)
    if name == "torch" and installed_cuda_tag(module) != expected_cuda_tag:
        raise SystemExit(1)
print("Pinned torch packages already installed, skipping reinstall")
PY
then
  :
else
  "${PYTHON}" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  "${PYTHON}" -m pip install "${torch_url}" "${torchvision_url}" "${torchaudio_url}" -i "${PIP_INDEX_URL}"
fi

"${PYTHON}" -m pip install \
  matplotlib \
  pandas \
  pyyaml \
  scipy \
  tqdm \
  psutil \
  py-cpuinfo \
  seaborn \
  ultralytics-thop>=0.2.5 \
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
