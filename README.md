# Mamba-YOLO-T-VID for VisDrone Video Detection

This repository adapts the official Mamba-YOLO codebase for VisDrone-VID video
object detection. The current research constraint is:

- Keep the official Mamba-YOLO-T backbone and neck unchanged.
- Add video-specific modules after the neck and inside the detection head.
- Move the input organization and offline video evaluation path toward video
  detection while keeping the temporal head simple and ablation-friendly.

## Current Main Model

Current variant:

```text
Mamba-YOLO-T-VID-ScoreSmooth-v5
```

High-level flow:

```text
16-frame video window
  -> official Mamba-YOLO-T backbone
  -> official Mamba-YOLO-T neck / feature pyramid
  -> Detect_VID
      - YOLOv8-style bbox regression branch
      - raw classification branch
      - lightweight TemporalScoreSmoother
      - smoothed class logits
  -> offline VID detections / flicker / MOT-ID metrics
```

The official backbone and neck are not structurally modified. The current v5
experiment keeps the detection structure simple: current-frame boxes remain
unchanged, while nearby reference frames only smooth and boost class scores.

## Quick Start

On the training server:

```bash
cd /root/autodl-tmp/MYT
source .venv/bin/activate
git pull
git submodule update --init --recursive

python tools/model_variant.py train-command score_smooth_v5_2026-05-14 \
  --name score_smooth_v5
```

To inspect saved model variants:

```bash
python tools/model_variant.py list
python tools/model_variant.py show score_smooth_v5_2026-05-14
python tools/model_variant.py train-command score_smooth_v5_2026-05-14
```

## Important Files

### Main Entry Points

| File | Purpose |
| --- | --- |
| `mbyolo_train.py` | Unified entry for train/val/test/predict/export. It also wires VID window input, temporal-fusion arguments, and periodic extra video evaluation. |
| `setup.sh` | Server setup script for system packages, Python environment, dependencies, and VisDrone toolkit files. |
| `RUN_VISDRONE_VID.md` | VisDrone-VID preparation and training notes. |
| `RUN_VISDRONE_VID_YOLOV.md` | Notes for the YOLOV-style VID model path. |
| `CURRENT_MODEL_STRUCTURE.md` | Current model-structure marker for thesis writing and reproducibility. |
| `MODEL_HISTORY.md` | Index of saved model-stage records. |

### Model and Config Files

| Path | Purpose |
| --- | --- |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml` | VID model YAML. It keeps the official Mamba-YOLO-T backbone/neck and uses `Detect_VID`. |
| `ultralytics/cfg/datasets/VisDrone-VID.yaml` | VisDrone-VID dataset config. |
| `ultralytics/cfg/default.yaml` | Default Ultralytics arguments, extended with VID window and score-smoothing options. |
| `ultralytics/nn/modules/head.py` | Detection heads. `Detect_VID` is the current video head and owns the lightweight temporal score smoother. |
| `ultralytics/models/yolo/detect/train.py` | Passes VID clip layout and score-smoothing options into `Detect_VID` during training. |
| `ultralytics/models/yolo/detect/val.py` | Passes VID clip layout and score-smoothing options during validation. |
| `ultralytics/utils/loss.py` | YOLO detection losses plus the optional VID reference-frame auxiliary loss. |

### Model Variant Records

| Path | Purpose |
| --- | --- |
| `model_variants/README.md` | Explains the model-variant record directory. |
| `model_variants/score_smooth_v5_2026-05-14.yaml` | Current main model record: lightweight temporal score smoothing, key hyperparameters, notes, and training command. |
| `model_variants/temporal_adapter_p4p5_yolov_v4_2026-05-14.yaml` | Previous P4/P5 temporal adapter model record for rollback and ablation. |
| `model_variants/temporal_adapter_yolov_v3_2026-05-13.yaml` | Previous all-level temporal adapter record for rollback and ablation. |
| `model_variants/yolov_proposal_v2_2026-05-13.yaml` | Previous YOLOV proposal-only model record for rollback and ablation. |
| `tools/model_variant.py` | Small utility to list variants, show YAML records, and print stored training commands. |

When a new architecture stage becomes important, add a new YAML file under
`model_variants/` instead of overwriting old records.

### VisDrone Data Tools

| File | Purpose |
| --- | --- |
| `tools/download_visdrone_vid_zips.py` | Download VisDrone-VID zip files. |
| `tools/prepare_visdrone_vid_yolo.py` | Convert VisDrone-VID annotations into YOLO/VID training format. |
| `tools/verify_visdrone_vid_source.py` | Check whether the raw VisDrone-VID split is complete. |
| `tools/check_visdrone_vid_runtime.py` | Check dataset paths and runtime dependencies. |

### Export and Evaluation Tools

| File | Purpose |
| --- | --- |
| `tools/export_visdrone_vid_results.py` | Single-frame detection export for baseline Mamba-YOLO or YOLOv8. |
| `tools/export_visdrone_vid_clip_results.py` | Offline clip/window detection export for the VID model. |
| `tools/export_visdrone_vid_tracks.py` | Single-frame detection plus ByteTrack export. |
| `tools/export_visdrone_vid_clip_tracks.py` | Offline clip/window detection plus ByteTrack export. Current MOT/ID evaluation should use this path. |
| `tools/eval_visdrone_vid_cls_flicker.py` | Classification flicker metrics: `macro_flicker` and `micro_flicker`. |
| `tools/eval_visdrone_vid_mot.py` | Current MOT/ID metrics: IDF1, IDP, IDR, ID switches, and fragmentation. |
| `tools/eval_visdrone_vid_official.py` | Wrapper for official VisDrone AP/AR evaluation. Periodic extra-eval does not run official AP/AR by default. |
| `tools/run_visdrone_vid_official_eval.py` | Manual official VisDrone VID evaluation runner. |
| `tools/temporal_state.py` | Helper for resetting temporal state across video sequences. |

### Qualitative Comparison Tools

| File | Purpose |
| --- | --- |
| `tools/run_visdrone_comparison_examples.py` | One-command pipeline to export baseline/new predictions and select visual examples where the new model is better. |
| `tools/select_visdrone_comparison_examples.py` | Selects thesis-friendly qualitative examples using GT, baseline predictions, and new-model predictions. |
| `asserts/` | Figures used by the README or thesis, such as architecture diagrams and ODSSBlock illustrations. |

### Official Reference Code

| Path | Purpose |
| --- | --- |
| `third_party/Mamba-YOLO-official/` | Clean official Mamba-YOLO repository pinned as a Git submodule. Use it as the untouched reference implementation for baseline checks and code comparison. Do not edit experiment code directly inside this folder unless intentionally updating the reference. |

After cloning this repository on a new machine, initialize the official reference
code with:

```bash
git submodule update --init --recursive
```

## Current Main Hyperparameters

These options define `Mamba-YOLO-T-VID-ScoreSmooth-v5`:

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_fusion score_smooth
--score_smooth_sigma 0.03
--score_smooth_cls_gain 0.60
--score_smooth_conf_gain 0.70
--score_smooth_min_ref_score 0.001
--score_smooth_warmup_epochs 5
--score_smooth_alpha_target 1.0
```

If these options or the head structure change substantially, treat the result as
a new experiment variant and create a new `model_variants/*.yaml` record.

## Experiment Notes

Recommended comparisons:

- Official Mamba-YOLO-T baseline: single-frame Mamba-YOLO.
- Official YOLOv8 baseline: single-frame YOLOv8.
- Current new model: Mamba-YOLO-T backbone/neck plus lightweight temporal score
  smoothing inside `Detect_VID`.
- Ablations: disable `score_smooth_conf_gain`, disable `score_smooth_cls_gain`,
  vary `score_smooth_sigma`, compare against `temporal_fusion none`.

The previous best new model improved precision and identity stability but was
too complex for the observed gains. The v5 score-smoothing model is intended to
target video metrics directly with a simpler and more explainable temporal
module.

## Acknowledgement

This repository is based on:

- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOV](https://github.com/YuHengsss/YOLOV)
- [VMamba selective scan](https://github.com/MzeroMiko/VMamba)

## Citation

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
      title={Mamba YOLO: SSMs-Based YOLO For Object Detection},
      author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
      year={2024},
      eprint={2406.05835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.05835},
}
```
