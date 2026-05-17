# Model History

This file is the index of saved model-stage records. Each detailed record lives
under `model_variants/` as a YAML file and can be printed with:

```bash
python tools/model_variant.py list
python tools/model_variant.py show ttrm_v8_2026-05-17
python tools/model_variant.py train-command ttrm_v8_2026-05-17
```

## Current Main Variant

- ID: `ttrm_v8_2026-05-17`
- File: `model_variants/ttrm_v8_2026-05-17.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. A new
  TemporalResidualFeatureAdapter is added inside Detect_VID before the original
  bbox/class branches. VisDrone track_id annotations are also used to build
  same-object tubes inside each 16-frame window, adding class recall and
  confidence-continuity losses. During periodic video evaluation, raw detections
  are refined by the model-agnostic Temporal Tubelet Refinement Module (TTRM):
  tubelet association, class voting, score propagation, short-gap recovery, and
  box smoothing.

## Previous Main Variant

These older variants are kept as experiment records. Their removed modules are
not part of the current active code path; use the corresponding historical Git
commit if an exact rerun is needed.

- ID: `score_smooth_v5_2026-05-14`
- File: `model_variants/score_smooth_v5_2026-05-14.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. A
  lightweight TemporalScoreSmoother was added inside Detect_VID after raw class
  logits. It kept current-frame boxes unchanged and smoothed only local class
  scores across nearby reference frames.

- ID: `temporal_residual_v6_2026-05-16`
- File: `model_variants/temporal_residual_v6_2026-05-16.yaml`
- Summary: Official Mamba-YOLO-T backbone and neck are kept fixed. A
  TemporalResidualFeatureAdapter is added inside Detect_VID before the original
  bbox/class branches.

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
| `ttrm_v8_2026-05-17` | 2026-05-17 | Official Mamba-YOLO-T | P3/P4/P5 TRFA + track-id tube loss + TTRM video refinement | Current main variant. Adds model-agnostic tubelet refinement for video metrics: class voting, score propagation, gap recovery, and box smoothing. |
| `track_tube_v7_2026-05-17` | 2026-05-17 | Official Mamba-YOLO-T | P3/P4/P5 TRFA + track-id tube loss | Previous main variant. Uses raw VisDrone track_id annotations to add tube class recall and same-track confidence continuity supervision. |
| `temporal_residual_v6_2026-05-16` | 2026-05-16 | Official Mamba-YOLO-T | P3/P4/P5 TemporalResidualFeatureAdapter | Previous main variant. Removes old proposal/score-smoothing branches and adapts local temporal features before bbox/class Detect branches. |
| `score_smooth_v5_2026-05-14` | 2026-05-14 | Official Mamba-YOLO-T | TemporalScoreSmoother | Previous main variant. Simple score-level temporal smoothing and positive confidence boost; no feature adapter, no proposal refiner in the main path. |
| `temporal_adapter_p4p5_yolov_v4_2026-05-14` | 2026-05-14 | Official Mamba-YOLO-T | P4/P5 TemporalFeatureAdapter + YOLOV-style proposal refinement | Previous main variant. Bypasses P3 to protect small-object recall while keeping temporal feature aggregation on P4/P5. |
| `temporal_adapter_yolov_v3_2026-05-13` | 2026-05-13 | Official Mamba-YOLO-T | P3/P4/P5 TemporalFeatureAdapter + YOLOV-style proposal refinement | Previous main variant. Improved stability but reduced recall compared with baseline. |
| `yolov_proposal_v2_2026-05-13` | 2026-05-13 | Official Mamba-YOLO-T | YOLOV-style two-stage proposal refinement | Proposal-only variant after adding local NMS, after-topk attention, time bias, learned location bias, temporal voting, and recall boost. |
