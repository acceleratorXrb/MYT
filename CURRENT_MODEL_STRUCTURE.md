# Current Model Structure Marker

This file marks the model structure currently used as the main VID experiment
configuration in this repository.

Last updated: 2026-05-17

## Model Name

**Mamba-YOLO-T-VID with TRFA, Track-ID Tube Supervision, and TTRM**

Short name used in notes:

```text
Mamba-YOLO-T-VID-TTRM-v8
```

## Fixed Backbone and Neck

The backbone and neck are kept from the official Mamba-YOLO-T detection model:

```text
Input video window
  -> frame-wise Mamba-YOLO-T backbone
  -> Mamba-YOLO neck / feature pyramid
  -> P3, P4, P5 multi-scale features
```

The official Mamba-YOLO backbone and neck are not structurally modified. Frames
in a video window are flattened into a normal image batch, so each frame passes
through the official backbone and neck once.

## Current Detection Head Structure

The current main detection head is:

```text
P3/P4/P5 window features
  -> TemporalResidualFeatureAdapter per level
       1x1 reduce
       3D depthwise local temporal-spatial conv
       1x1 expand
       residual alpha gate
  -> original YOLOv8/Mamba-YOLO Detect branches
       cv2 -> bbox distribution
       cv3 -> class logits
  -> output bbox regression + class logits
```

This replaces the failed score-level smoothing path. Temporal information now
acts before the bbox/classification branches, so both localization and class
prediction can learn from nearby frames. The residual gate is initialized as an
identity path and can be warmed up during training.

## Current Training-Time Video Supervision

The current main training objective additionally uses VisDrone `track_id`
annotations. During 16-frame window training, the dataset reads raw
`VisDrone2019-VID-*/annotations/*.txt`, attaches `track_ids` to labels, and the
VID loss builds object tubes inside each window.

```text
GT labels with track_id
  -> group by (window, track_id)
  -> sample Detect_VID class logits at GT box centers on P3/P4/P5
  -> tube class recall loss
  -> same-track confidence continuity loss
```

This is the current primary video-metric optimization path. It directly targets
missed detections, class flicker, and track fragmentation instead of relying on
feature fusion alone.

## Current Video-Level Refinement

Periodic extra evaluation now uses a model-agnostic Temporal Tubelet Refinement
Module (TTRM) after raw detection export:

```text
raw per-frame detections
  -> greedy local association into tubelets
  -> tubelet class voting
  -> tubelet score propagation
  -> short-gap interpolation
  -> light box smoothing
  -> refined detections for flicker
  -> refined track txt files for MOT/ID
```

This module is designed specifically for video metrics. It reduces class flicker
by assigning a stable majority category inside each tubelet, reduces
fragmentation by filling short missing gaps, and improves identity continuity by
using stable tubelet ids for MOT/ID evaluation.

## Current Main Structural Hyperparameters

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_fusion trfa
--trfa_levels all
--ref_aux_loss 0.0
--trfa_warmup_epochs 5
--trfa_alpha_target 1.0
--track_recall_loss 0.5
--track_consistency_loss 0.2
--extra_eval_ttrm
--extra_eval_conf 0.01
--ttrm_seed_score 0.12
--ttrm_keep_score 0.08
--ttrm_track_keep_score 0.12
--ttrm_track_min_len 4
--ttrm_track_min_score 0.20
--ttrm_gap_fill 1
--ttrm_vote_gain 0.35
--ttrm_recall_gain 0.25
--ttrm_input_topk 60
--ttrm_max_per_frame 80
--debug_ttrm
```

## What These Options Mean

`vid_clip_mode=window` means the model uses a consecutive video window, and all
frames in the window are treated as key frames.

`temporal_fusion=trfa` selects the temporal residual feature adapter.

`trfa_levels=all` applies the adapter to P3, P4, and P5. This is the current main
setting because the goal is a real video-detection module, not a weak score-only
post-processing path.

`trfa_warmup_epochs` and `trfa_alpha_target` warm up the residual adapter gate.

`track_recall_loss` encourages each GT object in a track tube to keep a strong
class response at its center location.

`track_consistency_loss` pulls lower same-track class confidence toward the best
same-track confidence inside the window, reducing temporal dropouts.

## Supported Modes

```text
trfa  -> current temporal residual feature adapter head
none  -> single-frame Detect-like head for ablation
```

For thesis figures and main experiment descriptions, use the `trfa` structure
above unless a section explicitly describes an ablation.
