# Current Model Structure Marker

This file marks the model structure currently used as the main VID experiment
configuration in this repository.

Last updated: 2026-05-16

## Model Name

**Mamba-YOLO-T-VID with Temporal Residual Feature Adapter**

Short name used in notes:

```text
Mamba-YOLO-T-VID-TRFA-v6
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
```

## What These Options Mean

`vid_clip_mode=window` means the model uses a consecutive video window, and all
frames in the window are treated as key frames.

`temporal_fusion=trfa` selects the temporal residual feature adapter.

`trfa_levels=all` applies the adapter to P3, P4, and P5. This is the current main
setting because the goal is a real video-detection module, not a weak score-only
post-processing path.

`trfa_warmup_epochs` and `trfa_alpha_target` warm up the residual adapter gate.

## Supported Modes

```text
trfa  -> current temporal residual feature adapter head
none  -> single-frame Detect-like head for ablation
```

For thesis figures and main experiment descriptions, use the `trfa` structure
above unless a section explicitly describes an ablation.
