#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${VENV_DIR}/bin/python"

DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
PROJECT="${PROJECT:-output_dir/visdrone_vid}"
NAME="${NAME:-mambayolo_t}"
DATA_YAML="${DATA_YAML:-${PROJECT}/VisDrone-VID.local.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml}"

TOOLKIT_DIR="${TOOLKIT_DIR:-third_party/VisDrone2018-VID-toolkit}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/datasets/VisDrone-VID/raw}"
YOLO_ROOT="${YOLO_ROOT:-${ROOT_DIR}/datasets/VisDrone-VID}"
ZIP_ROOT="${ZIP_ROOT:-${ROOT_DIR}/datasets/VisDrone-VID}"

INSTALL_SYSTEM="${INSTALL_SYSTEM:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
START_TRAIN="${START_TRAIN:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mambayolo_matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/mambayolo_ultralytics}"

log() {
  printf '[setup] %s\n' "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

install_system_packages() {
  require_cmd git
  require_cmd "${PYTHON_BIN}"

  if [[ "${INSTALL_SYSTEM}" != "1" ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "Skipping apt packages because apt-get is unavailable"
    return
  fi

  local -a apt_prefix=()
  if [[ "${EUID}" -ne 0 ]]; then
    require_cmd sudo
    apt_prefix=(sudo)
  fi

  log "Installing system packages"
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y \
    aria2 \
    build-essential \
    git \
    octave \
    python3-dev \
    python3-venv \
    unzip
}

clone_toolkit() {
  if [[ -d "${TOOLKIT_DIR}/.git" || -f "${TOOLKIT_DIR}/calcAccuracy.m" ]]; then
    log "Official toolkit already present at ${TOOLKIT_DIR}"
    return
  fi

  mkdir -p "$(dirname "${TOOLKIT_DIR}")"
  log "Cloning official VisDrone-VID toolkit"
  git clone https://github.com/VisDrone/VisDrone2018-VID-toolkit.git "${TOOLKIT_DIR}"
}

ensure_python_env() {
  log "Preparing Python environment"
  PYTHON_BIN="${PYTHON_BIN}" VENV_DIR="${VENV_DIR}" bash scripts/setup_mambayolo_cuda121.sh
}

download_raw_zips_if_needed() {
  mkdir -p "${ZIP_ROOT}"

  local missing=0
  for split in train val test-dev; do
    local zip_path="${ZIP_ROOT}/VisDrone2019-VID-${split}.zip"
    if [[ ! -f "${zip_path}" || ! -s "${zip_path}" ]]; then
      missing=1
      break
    fi
  done

  if [[ "${missing}" != "1" ]]; then
    return
  fi

  if [[ "${DOWNLOAD_DATA}" != "1" ]]; then
    printf 'VisDrone zip archives are missing under %s and DOWNLOAD_DATA=%s\n' "${ZIP_ROOT}" "${DOWNLOAD_DATA}" >&2
    exit 1
  fi

  log "Downloading official VisDrone-VID zip archives"
  "${PYTHON}" tools/download_visdrone_vid_zips.py --out "${ZIP_ROOT}"
}

restore_raw_splits_from_zips() {
  mkdir -p "${RAW_ROOT}"

  for split in train val test-dev; do
    local split_dir="${RAW_ROOT}/VisDrone2019-VID-${split}"
    local zip_path="${ZIP_ROOT}/VisDrone2019-VID-${split}.zip"
    if [[ ! -f "${zip_path}" ]]; then
      printf 'Missing archive: %s\n' "${zip_path}" >&2
      exit 1
    fi
    local expected_files
    expected_files="$(unzip -Z1 "${zip_path}" | grep -v '/$' | wc -l | tr -d ' ')"
    local actual_files=0
    if [[ -d "${split_dir}" ]]; then
      actual_files="$(find "${split_dir}" -type f | wc -l | tr -d ' ')"
    fi

    if [[ "${actual_files}" == "${expected_files}" && -d "${split_dir}/annotations" && -d "${split_dir}/sequences" ]]; then
      continue
    fi
    if [[ -d "${split_dir}" ]]; then
      log "Removing incomplete split ${split_dir} (${actual_files}/${expected_files} files)"
      rm -rf "${split_dir}"
    fi
    log "Extracting ${zip_path} -> ${RAW_ROOT}"
    unzip -q "${zip_path}" -d "${RAW_ROOT}"
  done
}

count_files() {
  find "$1" -type f 2>/dev/null | wc -l | tr -d ' '
}

raw_frame_count() {
  local split_dir="${RAW_ROOT}/VisDrone2019-VID-$1"
  if [[ ! -d "${split_dir}/sequences" ]]; then
    printf '0\n'
    return 1
  fi
  find "${split_dir}/sequences" -type f | wc -l | tr -d ' '
}

raw_split_complete() {
  local split_dir="${RAW_ROOT}/VisDrone2019-VID-$1"
  [[ -d "${split_dir}/annotations" && -d "${split_dir}/sequences" ]] || return 1
  [[ "$(find "${split_dir}/annotations" -type f 2>/dev/null | wc -l | tr -d ' ')" != "0" ]] || return 1
  [[ "$(find "${split_dir}/sequences" -type f 2>/dev/null | wc -l | tr -d ' ')" != "0" ]] || return 1
}

all_raw_splits_complete() {
  local split
  for split in train val test-dev; do
    raw_split_complete "${split}" || return 1
  done
}

expected_yolo_test_dir() {
  if [[ -d "${YOLO_ROOT}/images/test" && -d "${YOLO_ROOT}/labels/test" ]]; then
    printf '%s\n' "test"
  else
    printf '%s\n' "test-dev"
  fi
}

is_yolo_layout_complete() {
  local split yolo_split expected actual_images actual_labels
  for split in train val test-dev; do
    expected="$(raw_frame_count "${split}")" || return 1
    if [[ "${split}" == "test-dev" ]]; then
      yolo_split="$(expected_yolo_test_dir)"
    else
      yolo_split="${split}"
    fi
    actual_images="$(count_files "${YOLO_ROOT}/images/${yolo_split}")"
    actual_labels="$(count_files "${YOLO_ROOT}/labels/${yolo_split}")"
    if [[ "${actual_images}" != "${expected}" || "${actual_labels}" != "${expected}" ]]; then
      return 1
    fi
  done
  return 0
}

write_dataset_yaml() {
  local test_dir
  test_dir="$(expected_yolo_test_dir)"
  local dataset_root
  dataset_root="$(cd "${YOLO_ROOT}" && pwd)"
  cat > "${DATA_YAML}" <<EOF
# Auto-generated VisDrone2019-VID dataset config
path: ${dataset_root}
train: images/train
val: images/val
test: images/${test_dir}

task: vid

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
EOF
}

ensure_dataset_yaml() {
  mkdir -p "${PROJECT}" "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}"

  if ! all_raw_splits_complete; then
    printf 'VisDrone raw splits are incomplete under %s; refusing to rebuild images/labels and preserve existing YOLO layout.\n' "${RAW_ROOT}" >&2
    return 1
  fi

  if is_yolo_layout_complete; then
    write_dataset_yaml
    return
  fi

  log "Rebuilding VisDrone-VID YOLO layout from raw splits"
  rm -rf "${YOLO_ROOT}/images" "${YOLO_ROOT}/labels"
  if [[ ! -d "${RAW_ROOT}/VisDrone2019-VID-train" || ! -d "${RAW_ROOT}/VisDrone2019-VID-val" || ! -d "${RAW_ROOT}/VisDrone2019-VID-test-dev" ]]; then
    printf 'VisDrone raw splits are missing under %s\n' "${RAW_ROOT}" >&2
    exit 1
  fi

  "${PYTHON}" tools/prepare_visdrone_vid_yolo.py \
    --src "${RAW_ROOT}" \
    --out "${YOLO_ROOT}" \
    --yaml "${DATA_YAML}" \
    --overwrite

  write_dataset_yaml
}

run_checks() {
  log "Running project checks"
  "${PYTHON}" tools/check_visdrone_vid_runtime.py
}

start_training() {
  if [[ "${START_TRAIN}" != "1" ]]; then
    log "Setup finished. Training not started because START_TRAIN=${START_TRAIN}"
    return
  fi

  log "Starting formal training"
  "${PYTHON}" mbyolo_train.py \
    --task train \
    --data "${DATA_YAML}" \
    --config "${MODEL_CONFIG}" \
    --imgsz "${IMGSZ}" \
    --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --amp \
    --project "${PROJECT}" \
    --name "${NAME}"
}

install_system_packages
clone_toolkit
ensure_python_env
download_raw_zips_if_needed
restore_raw_splits_from_zips
ensure_dataset_yaml
run_checks
start_training
