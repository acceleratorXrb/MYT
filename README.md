# Mamba-YOLO-T-VID for VisDrone Video Detection

This repository adapts the official Mamba-YOLO codebase for VisDrone-VID video
object detection. The current research constraint is:

- Keep the official Mamba-YOLO-T backbone and neck unchanged.
- Add video-specific modules after the neck and inside the detection head.
- Move the input organization, temporal proposal refinement, and offline video
  evaluation path toward YOLOV.

## Current Main Model

Current variant:

```text
Mamba-YOLO-T-VID-TemporalAdapter-YOLOV-v3
```

High-level flow:

```text
16-frame video window
  -> official Mamba-YOLO-T backbone
  -> official Mamba-YOLO-T neck / feature pyramid
  -> TemporalFeatureAdapter
  -> Detect_VID
      - YOLOv8-style bbox regression branch
      - raw classification branch
      - YOLOV-style two-stage ProposalTemporalRefiner
      - refined class logits
  -> offline VID detections / flicker / MOT-ID metrics
```

The official backbone and neck are not structurally modified. The new temporal
feature adapter is added inside `Detect_VID` before the detection branches, so
previous baseline head weights can still mostly load into `model.21.*`.

## Quick Start

On the training server:

```bash
cd /root/autodl-tmp/MYT
source .venv/bin/activate
git pull

python tools/model_variant.py train-command temporal_adapter_yolov_v3_2026-05-13 \
  --name temporal_adapter_yolov_v3
```

To inspect saved model variants:

```bash
python tools/model_variant.py list
python tools/model_variant.py show temporal_adapter_yolov_v3_2026-05-13
python tools/model_variant.py train-command temporal_adapter_yolov_v3_2026-05-13
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
| `ultralytics/cfg/default.yaml` | Default Ultralytics arguments, extended with VID, temporal adapter, and proposal-refinement options. |
| `ultralytics/nn/modules/head.py` | Detection heads. `Detect_VID` is the current video head and owns the temporal adapter plus YOLOV-style proposal refiner. |
| `ultralytics/nn/modules/temporal_adapter.py` | Added feature-level temporal adapter between neck features and detect branches. |
| `ultralytics/nn/modules/yolov_fam.py` | FAM and YOLOV-style proposal temporal refinement modules. |
| `ultralytics/models/yolo/detect/train.py` | Passes VID clip layout and temporal options into `Detect_VID` during training. |
| `ultralytics/models/yolo/detect/val.py` | Passes VID clip layout and temporal options during validation. |
| `ultralytics/utils/loss.py` | YOLO detection losses plus VID auxiliary losses, including YOLOV-style refined-class auxiliary loss. |

### Model Variant Records

| Path | Purpose |
| --- | --- |
| `model_variants/README.md` | Explains the model-variant record directory. |
| `model_variants/temporal_adapter_yolov_v3_2026-05-13.yaml` | Current main model record: structure, key hyperparameters, notes, and training command. |
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

## Current Main Hyperparameters

These options define `Mamba-YOLO-T-VID-TemporalAdapter-YOLOV-v3`:

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_adapter affinity
--temporal_adapter_time_sigma 4.0
--temporal_fusion yolov
--yolov_cls_loss 0.30
--proposal_topk 700
--proposal_after_topk 220
--proposal_nms_radius 1
--proposal_spatial_sigma 0.12
--proposal_time_sigma 4.0
--proposal_loc_gain 0.5
--proposal_cls_sim_gain 0.55
--proposal_vote_gain 0.50
--proposal_recall_gain 1.25
--proposal_recall_radius 1
--fam_warmup_epochs 5
--fam_alpha_target 0.65
```

If these options or the head structure change substantially, treat the result as
a new experiment variant and create a new `model_variants/*.yaml` record.

## Experiment Notes

Recommended comparisons:

- Official Mamba-YOLO-T baseline: single-frame Mamba-YOLO.
- Official YOLOv8 baseline: single-frame YOLOv8.
- Current new model: Mamba-YOLO-T backbone/neck plus TemporalFeatureAdapter and
  YOLOV-style proposal head.
- Ablations: disable `temporal_adapter`, disable `proposal_vote_gain`, disable
  `proposal_recall_gain`, disable `proposal_after_topk/nms/time/loc`.

The previous best new model improved precision and identity stability. The v3
adapter is intended to improve recall by enhancing weak current-frame features
before detection.

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

