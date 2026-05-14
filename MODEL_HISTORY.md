# Model History

This file is the index of saved model-stage records. Each detailed record lives
under `model_variants/` as a YAML file and can be printed with:

```bash
python tools/model_variant.py list
python tools/model_variant.py show score_smooth_v5_2026-05-14
python tools/model_variant.py train-command score_smooth_v5_2026-05-14
```

## Current Main Variant

- ID: `score_smooth_v5_2026-05-14`
- File: `model_variants/score_smooth_v5_2026-05-14.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. A new
  lightweight TemporalScoreSmoother is added inside Detect_VID after raw class
  logits. It keeps current-frame boxes unchanged and smooths only local class
  scores across nearby reference frames.

## Previous Main Variant

- ID: `temporal_adapter_p4p5_yolov_v4_2026-05-14`
- File: `model_variants/temporal_adapter_p4p5_yolov_v4_2026-05-14.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. A
  TemporalFeatureAdapter is added before the YOLOV-style proposal head, but it
  is applied only to P4/P5 so P3 small-object features are preserved.

## Older Main Variant

- ID: `temporal_adapter_yolov_v3_2026-05-13`
- File: `model_variants/temporal_adapter_yolov_v3_2026-05-13.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. The temporal
  feature adapter is applied to all P3/P4/P5 levels.

## Earlier Main Variant

- ID: `yolov_proposal_v2_2026-05-13`
- File: `model_variants/yolov_proposal_v2_2026-05-13.yaml`
- Summary: Mamba-YOLO-T backbone and neck are kept fixed. The Detect_VID head
  uses YOLOV-style two-stage proposal temporal refinement over 16-frame windows.

## Saved Variants

| ID | Date | Backbone/Neck | Temporal Head | Notes |
| --- | --- | --- | --- | --- |
| `score_smooth_v5_2026-05-14` | 2026-05-14 | Official Mamba-YOLO-T | TemporalScoreSmoother | Current main variant. Simple score-level temporal smoothing and positive confidence boost; no feature adapter, no proposal refiner in the main path. |
| `temporal_adapter_p4p5_yolov_v4_2026-05-14` | 2026-05-14 | Official Mamba-YOLO-T | P4/P5 TemporalFeatureAdapter + YOLOV-style proposal refinement | Previous main variant. Bypasses P3 to protect small-object recall while keeping temporal feature aggregation on P4/P5. |
| `temporal_adapter_yolov_v3_2026-05-13` | 2026-05-13 | Official Mamba-YOLO-T | P3/P4/P5 TemporalFeatureAdapter + YOLOV-style proposal refinement | Previous main variant. Improved stability but reduced recall compared with baseline. |
| `yolov_proposal_v2_2026-05-13` | 2026-05-13 | Official Mamba-YOLO-T | YOLOV-style two-stage proposal refinement | Proposal-only variant after adding local NMS, after-topk attention, time bias, learned location bias, temporal voting, and recall boost. |
