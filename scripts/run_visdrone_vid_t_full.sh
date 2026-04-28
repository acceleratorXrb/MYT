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
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-300}"
VAL_PERIOD="${VAL_PERIOD:-1}"
MODEL_CONFIG="${MODEL_CONFIG:-ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml}"
EXTRA_EVAL_PERIOD="${EXTRA_EVAL_PERIOD:-0}"
EXTRA_EVAL_OFFICIAL_ROOT="${EXTRA_EVAL_OFFICIAL_ROOT:-${RAW_ROOT}/VisDrone2019-VID-val}"
EXTRA_EVAL_TOOLKIT="${EXTRA_EVAL_TOOLKIT:-third_party/VisDrone2018-VID-toolkit}"
EXTRA_EVAL_TRACKER="${EXTRA_EVAL_TRACKER:-ultralytics/cfg/trackers/bytetrack.yaml}"
EXTRA_EVAL_BATCH="${EXTRA_EVAL_BATCH:-16}"
EXTRA_EVAL_CONF="${EXTRA_EVAL_CONF:-0.001}"
EXTRA_EVAL_TRACK_CONF="${EXTRA_EVAL_TRACK_CONF:-0.1}"
EXTRA_EVAL_IOU="${EXTRA_EVAL_IOU:-0.7}"

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
else
  TEST_DIR="images/test-dev"
  "${PYTHON}" tools/prepare_visdrone_vid_yolo.py \
    --src "${RAW_ROOT}" \
    --out "${YOLO_ROOT}" \
    --yaml "${DATA_YAML}" \
    --overwrite
fi

TRAIN_ARGS=(
  --task train
  --data "${DATA_YAML}"
  --config "${MODEL_CONFIG}"
  --imgsz "${IMGSZ}"
  --batch_size "${BATCH}"
  --epochs "${EPOCHS}"
  --val_period "${VAL_PERIOD}"
  --workers "${WORKERS}"
  --device "${DEVICE}"
  --amp
  --project "${PROJECT}"
  --name "${NAME}"
)

if [[ "${EXTRA_EVAL_PERIOD}" -gt 0 ]]; then
  TRAIN_ARGS+=(
    --extra_eval_period "${EXTRA_EVAL_PERIOD}"
    --extra_eval_official_root "${EXTRA_EVAL_OFFICIAL_ROOT}"
    --extra_eval_toolkit "${EXTRA_EVAL_TOOLKIT}"
    --extra_eval_tracker "${EXTRA_EVAL_TRACKER}"
    --extra_eval_batch "${EXTRA_EVAL_BATCH}"
    --extra_eval_conf "${EXTRA_EVAL_CONF}"
    --extra_eval_track_conf "${EXTRA_EVAL_TRACK_CONF}"
    --extra_eval_iou "${EXTRA_EVAL_IOU}"
  )
fi

"${PYTHON}" mbyolo_train.py "${TRAIN_ARGS[@]}"

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
