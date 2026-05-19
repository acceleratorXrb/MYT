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
Mamba-YOLO-T-VID-VideoStable-v10
```

High-level flow:

```text
16-frame video window
  -> official Mamba-YOLO-T backbone
  -> official Mamba-YOLO-T neck / feature pyramid
  -> Detect_VID
      - TemporalResidualFeatureAdapter on P3/P4/P5 classification branch
      - YOLOv8-style bbox regression branch from original features
      - YOLOv8-style classification branch from temporal features
  -> track-id tube supervision during training
      - tube class recall loss
      - same-track confidence continuity loss
      - same-track class distribution consistency loss
  -> GT-free temporal export stabilization
      - detection class smoothing for flicker
      - track class smoothing, strict fragment relinking, one-frame gap filling
  -> offline VID detections / flicker / MOT-ID metrics
```

The official backbone and neck are not structurally modified. The current main
experiment uses one simple temporal residual feature adapter before the Detect
classification branch, while bbox regression keeps current-frame features. The
training objective also uses VisDrone `track_id` annotations to supervise
same-object temporal continuity inside each 16-frame window. Periodic extra
evaluation adds a conservative GT-free stabilizer to directly target `ID
Switches`, `Frag`, and `flicker`.

## Quick Start

On the training server:

```bash
cd /root/autodl-tmp/MYT
source .venv/bin/activate
git pull
git submodule update --init --recursive

python tools/model_variant.py train-command video_stable_v10_2026-05-18 \
  --name video_stable_v10
```

To inspect saved model variants:

```bash
python tools/model_variant.py list
python tools/model_variant.py show video_stable_v10_2026-05-18
python tools/model_variant.py train-command video_stable_v10_2026-05-18
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
| `CURRENT_NEW_MODEL.md` | Chinese thesis-oriented explanation of the current new model, including input flow, Detect_VID, TRFA, losses, and video metric evaluation. |
| `MODEL_HISTORY.md` | Index of saved model-stage records. |

### Model and Config Files

| Path | Purpose |
| --- | --- |
| `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml` | VID model YAML. It keeps the official Mamba-YOLO-T backbone/neck and uses `Detect_VID`. |
| `ultralytics/cfg/datasets/VisDrone-VID.yaml` | VisDrone-VID dataset config. |
| `ultralytics/cfg/default.yaml` | Default Ultralytics arguments, extended with VID window and temporal residual adapter options. |
| `ultralytics/nn/modules/head.py` | Detection heads. `Detect_VID` is the current video head and owns `TemporalResidualFeatureAdapter`. |
| `ultralytics/models/yolo/detect/train.py` | Passes VID clip layout and TRFA options into `Detect_VID` during training. |
| `ultralytics/models/yolo/detect/val.py` | Passes VID clip layout and TRFA options during validation. |
| `ultralytics/utils/loss.py` | YOLO detection losses plus the optional VID reference-frame auxiliary loss. |

### Model Variant Records

| Path | Purpose |
| --- | --- |
| `model_variants/README.md` | Explains the model-variant record directory. |
| `model_variants/video_stable_v10_2026-05-18.yaml` | Current main experiment record: classification-branch TRFA, track-id tube supervision, VID-stable ByteTrack, GT-free video export stabilization, key hyperparameters, notes, and training command. |
| `model_variants/video_stable_v9_2026-05-18.yaml` | Previous video-stable model record. |
| `model_variants/cls_stable_v8_2026-05-18.yaml` | Previous main model record: classification-branch TRFA plus track-id tube supervision. |
| `model_variants/track_tube_v7_2026-05-17.yaml` | Previous main model record: TRFA plus track-id tube supervision. |
| `model_variants/temporal_residual_v6_2026-05-16.yaml` | Previous temporal residual feature adapter model record kept for rollback and ablation. |
| `model_variants/score_smooth_v5_2026-05-14.yaml` | Previous score-smoothing model record kept for rollback and ablation. |
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
| `tools/visdrone_temporal_stabilize.py` | GT-free temporal smoothing/linking used by clip detection and track exports to reduce class flicker, ID switches, and short fragments. |
| `tools/eval_visdrone_vid_cls_flicker.py` | Classification flicker metrics: `macro_flicker` and `micro_flicker`. |
| `tools/eval_visdrone_vid_mot.py` | Current MOT/ID metrics: IDF1, IDP, IDR, ID switches, and fragmentation. |
| `tools/validate_visdrone_video_metrics.py` | Synthetic self-check for the local flicker and MOT/ID metric implementations. |
| `tools/eval_visdrone_vid_official.py` | Wrapper for official VisDrone AP/AR evaluation. Periodic extra-eval does not run official AP/AR by default. |
| `tools/run_visdrone_vid_official_eval.py` | Manual official VisDrone VID evaluation runner. |
| `tools/temporal_state.py` | Helper for resetting temporal state across video sequences. |
| `ultralytics/cfg/trackers/bytetrack_vidstable.yaml` | VID-stable ByteTrack config: longer lost-track buffer, lower second-stage threshold, and a small class-aware association penalty. |

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

These options define `Mamba-YOLO-T-VID-VideoStable-v10`:

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_fusion trfa
--trfa_levels all
--trfa_branch cls
--trfa_warmup_epochs 5
--trfa_alpha_target 1.0
--track_recall_loss 0.5
--track_consistency_loss 0.2
--track_cls_consistency_loss 0.1
--extra_eval_tracker ultralytics/cfg/trackers/bytetrack_vidstable.yaml
--extra_eval_track_conf 0.05
```

Periodic clip/window extra evaluation uses GT-free temporal stabilization by
default. It smooths high-overlap short detection tracklets for flicker, then
uses `bytetrack_vidstable.yaml`, smooths ByteTrack classes, absorbs bridge
tracklets inside short gaps, relinks strict short fragments, and fills one-frame
track gaps for MOT/ID. Add `--extra_eval_no_temporal_stabilize` only for the
ablation without this video-metric stabilizer.

If these options or the head structure change substantially, treat the result as
a new experiment variant and create a new `model_variants/*.yaml` record.

## Metric Validation

The periodic `flicker` and `MOT/ID` numbers are local auxiliary video metrics,
not official VisDrone AP/AR metrics. Before trusting them after code changes,
run the synthetic self-check:

```bash
python tools/validate_visdrone_video_metrics.py
```

This script builds a tiny VisDrone-style GT/prediction fixture with hand-checked
expected values and verifies `macro_flicker`, `micro_flicker`, `IDF1`, `IDP`,
`IDR`, `ID Switches`, `Frag`, and the GT-free temporal stabilizer. If this
script fails, do not compare model video metrics until the metric implementation
or stabilizer implementation is fixed.

When `mbyolo_train.py` is launched with `--extra_eval_period > 0`, this
self-check runs once before periodic video evaluation is registered. Use
`--skip_metric_self_check` only when deliberately debugging the metric scripts.

Periodic extra evaluation also writes detection export speed to
`extra_eval/epochXXX/speed.json` and copies it into `summary.json` under
`speed`. The reported `detection_export_fps` is measured after model loading and
includes preprocessing, model inference, NMS, and result txt export; it does not
include flicker/MOT metric computation.

## Experiment Notes

Recommended comparisons:

- Official Mamba-YOLO-T baseline: single-frame Mamba-YOLO.
- Official YOLOv8 baseline: single-frame YOLOv8.
- Current new model: Mamba-YOLO-T backbone/neck plus temporal residual feature
  adaptation on the `Detect_VID` classification branch and track-id tube
  supervision in the loss.
- Ablations: compare `trfa_levels all/p3p4/p4p5`, vary `trfa_alpha_target`,
  compare against `temporal_fusion none`.

The previous score-smoothing/proposal branches improved precision and identity
stability inconsistently. The current v10 path is intentionally simpler on the
model side: one local temporal residual adapter is added before the existing
Detect classification branch, and the extra-eval export applies conservative
GT-free temporal stabilization for the video metrics.

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
