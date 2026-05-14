# Current Model Structure Marker

This file marks the model structure currently used as the main VID experiment
configuration in this repository.

Last updated: 2026-05-14

## Model Name

**Mamba-YOLO-T-VID with lightweight temporal score smoothing**

Short name used in notes:

```text
Mamba-YOLO-T-VID-ScoreSmooth-v5
```

## Fixed Backbone and Neck

The backbone and neck are kept from the official Mamba-YOLO-T detection model:

```text
Input video window
  -> frame-wise Mamba-YOLO-T backbone
  -> Mamba-YOLO neck / feature pyramid
  -> P3, P4, P5 multi-scale features
```

The official Mamba-YOLO backbone and neck do not perform explicit temporal
fusion and are not structurally modified. Frames in a video window are
flattened into a normal image batch, so each frame passes through the backbone
and neck once. The current temporal module is deliberately simple and is
inserted inside `Detect_VID` after raw class logits are produced.

## Current Detection Head Structure

The current main detection head is:

```text
Detect_VID Head
  - Reg branch: cv2 -> bbox distribution
  - Cls branch: cv3_pre -> class logits
  - TemporalScoreSmoother
      - local reference-frame class-probability support
      - class probability smoothing for nearby grid cells
      - low-current-confidence positive boost from adjacent frames
      - no bbox update; current-frame boxes stay primary
  - Output: original bbox regression + smoothed class logits
```

The frame-local bbox regression branch is kept unchanged. Temporal information
is used only at class-score level, which keeps the structure simple and aims
directly at video metrics such as flicker, ID switches, fragmentation, and
short confidence drops.

## Current Main Structural Hyperparameters

These hyperparameters define the current main structure:

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_adapter none
--temporal_fusion score_smooth
--ref_aux_loss 0.0
--score_smooth_sigma 0.03
--score_smooth_cls_gain 0.60
--score_smooth_conf_gain 0.70
--score_smooth_min_ref_score 0.03
--fam_warmup_epochs 5
--fam_alpha_target 1.0
```

## What These Options Mean

`vid_clip_mode=window` means the model uses a consecutive video window, and all
frames in the window are treated as key frames.

`temporal_adapter=none` disables the heavier feature-level adapter. This variant
keeps the official Mamba-YOLO-T backbone and neck unchanged and avoids extra
feature-fusion complexity.

`temporal_fusion=score_smooth` selects a lightweight score-level temporal module.
For each frame, nearby grid cells from adjacent reference frames provide class
probability support. The current-frame bbox branch is not modified.

`score_smooth_sigma` controls the local feature-cell radius used to borrow
reference-frame support. `score_smooth_cls_gain` controls how much class
probabilities are smoothed toward nearby temporal support.
`score_smooth_conf_gain` controls the positive boost for low-current-confidence
locations. `score_smooth_min_ref_score` filters weak reference support.

`fam_warmup_epochs` and `fam_alpha_target` warm up all temporal residual gates,
including the score smoothing gate.

## Other Supported Modes

The code also supports older `Detect_VID` modes, but they are not the current
main model structure:

```text
none          -> single-frame Detect-like head
fam           -> dense feature aggregation module
proposal      -> proposal-level class refinement used directly
yolov         -> YOLOV-style proposal auxiliary/refined head
fam_proposal  -> dense FAM followed by proposal refinement
score_smooth  -> current lightweight temporal score smoothing head
logits        -> direct average logits fusion
logits_gated  -> confidence-gated logits fusion
```

The feature adapter can be disabled with:

```bash
--temporal_adapter none
```

For thesis figures and main experiment descriptions, use the
`score_smooth` structure above unless a section explicitly describes
an ablation.

## Current Training Command Identity

The current model structure corresponds to commands that include:

```bash
--vid_clip_mode window \
--vid_window_size 16 \
--num_ref_frames 15 \
--temporal_adapter none \
--temporal_fusion score_smooth \
--score_smooth_sigma 0.03 \
--score_smooth_cls_gain 0.60 \
--score_smooth_conf_gain 0.70 \
--score_smooth_min_ref_score 0.03 \
--fam_warmup_epochs 5 \
--fam_alpha_target 1.0
```

If these options are changed, the model head structure or behavior should be
treated as a different experimental variant.
