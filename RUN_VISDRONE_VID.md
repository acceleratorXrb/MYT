# Run Mamba-YOLO on VisDrone2019-VID

This repository is image-detection based. For VisDrone2019-VID, first flatten the video sequences into an Ultralytics YOLO detection dataset while preserving sequence subfolders.

## 1. Prepare Data

Download VisDrone2019-VID train, val, and test-dev from the official VisDrone dataset page, then place them under one root, for example:

```text
/mnt/datasets/VisDrone2019-VID-raw/
  VisDrone2019-VID-train/
    sequences/
    annotations/
  VisDrone2019-VID-val/
    sequences/
    annotations/
  VisDrone2019-VID-test-dev/
    sequences/
    annotations/
```

Convert to YOLO layout:

```bash
python tools/prepare_visdrone_vid_yolo.py \
  --src /mnt/datasets/VisDrone2019-VID-raw \
  --out /mnt/datasets/VisDrone2019-VID-YOLO \
  --yaml output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --overwrite
```

The script creates this layout:

```text
/mnt/datasets/VisDrone2019-VID-YOLO/
  images/train/<sequence>/<frame>.jpg
  labels/train/<sequence>/<frame>.txt
  images/val/<sequence>/<frame>.jpg
  labels/val/<sequence>/<frame>.txt
  images/test-dev/<sequence>/<frame>.jpg
  labels/test-dev/<sequence>/<frame>.txt
```

By default images are symlinked instead of copied. Add `--copy` if your training environment cannot follow symlinks.

Use the generated `output_dir/visdrone_vid/VisDrone-VID.local.yaml` for training if your converted dataset root is not `/mnt/datasets/VisDrone2019-VID-YOLO`.

## 2. Install

The official environment is Python 3.11, PyTorch 2.3.0, CUDA 12.x:

```bash
pip install torch==2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

`selective_scan` builds CUDA extensions, so a CUDA-capable machine is required for real training.

The same setup is wrapped in:

```bash
bash scripts/setup_mambayolo_cuda121.sh
```

For a fresh machine after cloning this GitHub repository, the one-click entry is:

```bash
bash setup.sh
```

This setup script will:

- install system packages with `apt-get` when available
- clone `third_party/VisDrone2018-VID-toolkit`
- create `.venv` and install Python/CUDA dependencies
- download official `VisDrone2019-VID-{train,val,test-dev}.zip` archives when they are missing
- restore raw official splits from `datasets/VisDrone-VID/*.zip` when those zips exist
- build the YOLO-format dataset when needed
- run runtime checks
- stop at a ready-to-train state by default

The Python dependency bootstrap now uses domestic mirrors by default:

- `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- `PYTORCH_WHEEL_BASE=https://mirrors.aliyun.com/pytorch-wheels/cu121`

You can override them if your machine needs a different mirror.

Useful overrides on a new server:

```bash
RAW_ROOT=/path/to/VisDrone2019-VID-raw DEVICE=0 BATCH=4 WORKERS=4 EPOCHS=100 \
bash setup.sh
```

The default behavior is to restore the environment and data without immediately
starting training:

```bash
bash setup.sh
```

If you want to forbid automatic dataset downloads and only use local archives:

```bash
DOWNLOAD_DATA=0 START_TRAIN=0 bash setup.sh
```

If you want the setup script to launch formal training at the end:

```bash
START_TRAIN=1 bash setup.sh
```

## 3. Train

Start with a small batch and the T model:

Before training, verify the project-local dataset, Python environment, CUDA visibility, and model construction:

```bash
.venv/bin/python tools/check_visdrone_vid_runtime.py
```

To verify that the converted data is backed by the original VisDrone2019-VID split archives:

```bash
.venv/bin/python tools/verify_visdrone_vid_source.py
```

If an archive is missing, download the official split archives into the project:

```bash
.venv/bin/python tools/download_visdrone_vid_zips.py
```

```bash
python mbyolo_train.py \
  --task train \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --epochs 100 \
  --workers 8 \
  --device 0 \
  --amp \
  --project output_dir/visdrone_vid \
  --name mambayolo_t
```

## 4. Validate

```bash
python mbyolo_train.py \
  --task val \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --workers 8 \
  --device 0 \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_val
```

## 5. Test-dev and Official Result Files

Run test-dev validation with the trained T-model weights:

```bash
python mbyolo_train.py \
  --task test \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --workers 8 \
  --device 0 \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_testdev
```

For the most conservative local evaluation path, use the official split root
and run export + official evaluation as one command:

```bash
python tools/run_visdrone_vid_official_eval.py \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --official-root datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --toolkit third_party/VisDrone2018-VID-toolkit \
  --results output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --out output_dir/visdrone_vid/mambayolo_t_official_eval \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

If you want the two stages separately for debugging, first export per-sequence txt
files for the official VisDrone-VID toolkit:

```bash
python tools/export_visdrone_vid_results.py \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --source datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --out output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Run the official VisDrone-VID MATLAB toolkit on those exported results:

```bash
python tools/eval_visdrone_vid_official.py \
  --toolkit third_party/VisDrone2018-VID-toolkit \
  --official-root datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --results output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --out output_dir/visdrone_vid/mambayolo_t_official_eval
```

This official-evaluation wrapper is intentionally conservative: it does not
reimplement VisDrone AP/AR in Python. It requires the official toolkit files
(`findSeqList.m`, `saveAnnoRes.m`, `displaySeq.m`, `calcAccuracy.m`) and a
MATLAB-compatible runtime (`matlab` or `octave`) on `PATH`.

The official root must be the original VisDrone-VID split directory with
`annotations/` and `sequences/`; do not point it at the converted YOLO dataset.
The wrapper also checks that every official sequence has a corresponding result
TXT before it tries to launch the toolkit.

## 6. Tracking ID Metrics

The tracking path does not use external ReID weights. It uses the VisDrone
sequence annotations preserved by `tools/prepare_visdrone_vid_yolo.py` under
`tracks/<split>/` plus the Mamba-YOLO detector outputs.

Train with the tracking config entry point:

```bash
python mbyolo_train.py \
  --task train_track \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-Track.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --epochs 100 \
  --workers 8 \
  --device 0 \
  --amp \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_track
```

Export per-sequence tracking result files with non-negative predicted IDs:

```bash
python mbyolo_train.py \
  --task track_export \
  --weights output_dir/visdrone_vid/mambayolo_t_track/weights/best.pt \
  --official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
  --out output_dir/visdrone_vid/mambayolo_t_track_val_tracks \
  --tracker ultralytics/cfg/trackers/mambayolo_visdrone_track.yaml \
  --imgsz 640 \
  --device 0
```

Evaluate local ID metrics on the val split:

```bash
python mbyolo_train.py \
  --task mot_eval \
  --official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
  --results output_dir/visdrone_vid/mambayolo_t_track_val_tracks \
  --out output_dir/visdrone_vid/mambayolo_t_track_val_mot.json
```

The local MOT evaluator reports `IDF1`, `IDP`, `IDR`, `ID Switches`, and
`Frag`. Test-dev has no public GT in this workflow, so use `track_export` for
submission files and val for measurable ID metrics.

The full T-model pipeline is wrapped in:

```bash
bash scripts/run_visdrone_vid_t_full.sh
```

The default dataset path is project-local: `datasets/VisDrone-VID`.
