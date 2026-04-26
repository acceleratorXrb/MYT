#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/mnt/datasets/VisDrone2019-VID-raw}"
YOLO_ROOT="${YOLO_ROOT:-datasets/VisDrone-VID}"
PROJECT="${PROJECT:-output_dir/visdrone_vid}"
NAME="${NAME:-mambayolo_t}"
DATA_YAML="${DATA_YAML:-${PROJECT}/VisDrone-VID.local.yaml}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-300}"

mkdir -p "${PROJECT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mambayolo_matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/mambayolo_ultralytics}"
mkdir -p "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}"

if [[ -d "${YOLO_ROOT}/images/train" && -d "${YOLO_ROOT}/images/val" && -d "${YOLO_ROOT}/labels/train" && -d "${YOLO_ROOT}/labels/val" ]]; then
  TEST_DIR="images/test-dev"
  if [[ -d "${YOLO_ROOT}/images/test" ]]; then
    TEST_DIR="images/test"
  fi
  DATASET_ROOT="$(cd "${YOLO_ROOT}" && pwd)"
  cat > "${DATA_YAML}" <<EOF
# Auto-generated VisDrone2019-VID dataset config
path: ${DATASET_ROOT}
train: images/train
val: images/val
test: ${TEST_DIR}

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
else
  TEST_DIR="images/test-dev"
  "${PYTHON}" tools/prepare_visdrone_vid_yolo.py \
    --src "${RAW_ROOT}" \
    --out "${YOLO_ROOT}" \
    --yaml "${DATA_YAML}" \
    --overwrite
fi

"${PYTHON}" mbyolo_train.py \
  --task train \
  --data "${DATA_YAML}" \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --amp \
  --project "${PROJECT}" \
  --name "${NAME}"

BEST="${PROJECT}/${NAME}/weights/best.pt"

"${PYTHON}" mbyolo_train.py \
  --task val \
  --weights "${BEST}" \
  --data "${DATA_YAML}" \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --name "${NAME}_val"

"${PYTHON}" mbyolo_train.py \
  --task test \
  --weights "${BEST}" \
  --data "${DATA_YAML}" \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --name "${NAME}_testdev"

"${PYTHON}" tools/export_visdrone_vid_results.py \
  --weights "${BEST}" \
  --source "${YOLO_ROOT}/${TEST_DIR}" \
  --out "${PROJECT}/${NAME}_visdrone_vid_results" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device "${DEVICE}"
