# Current Model Structure Marker

This file marks the model structure currently used as the main VID experiment
configuration in this repository.

Last updated: 2026-05-13

## Model Name

**Mamba-YOLO-T-VID with Temporal Feature Adapter and YOLOV-style proposal temporal refinement**

Short name used in notes:

```text
Mamba-YOLO-T-VID-TemporalAdapter-YOLOV-v3
```

## Fixed Backbone and Neck

The backbone and neck are kept from the official Mamba-YOLO-T detection model:

```text
Input video window
  -> frame-wise Mamba-YOLO-T backbone
  -> Mamba-YOLO neck / feature pyramid
  -> TemporalFeatureAdapter
  -> P3, P4, P5 multi-scale features
```

The official Mamba-YOLO backbone and neck do not perform explicit temporal
fusion and are not structurally modified. Frames in a video window are
flattened into a normal image batch, so each frame passes through the backbone
and neck once. The new temporal module is inserted after the neck and inside
`Detect_VID`, before the detect branches run.

## Current Detection Head Structure

The current main detection head is:

```text
Detect_VID Head
  - TemporalFeatureAdapter
      - per-scale P3/P4/P5 temporal affinity over the 16-frame window
      - adjacent-frame reference mask
      - temporal-distance attention bias
      - depthwise local spatial context before feature aggregation
      - residual feature update with alpha warmup
  - Reg branch: cv2 -> bbox distribution
  - Cls branch: cv3_pre -> class logits
  - YOLOV-style ProposalTemporalRefiner
      - pre top-k proposal selection
      - local proposal NMS
      - second-stage proposal subset for temporal attention
      - proposal-center spatial matching plus learned location bias
      - temporal-distance attention bias
      - YOLOV-style [support, key] proposal classification refine
      - temporal class voting
      - temporal recall boost
  - Output: original bbox regression + refined class logits
```

The frame-local bbox regression branch is kept unchanged. Temporal information
is now used twice: first at feature level by `TemporalFeatureAdapter`, then at
proposal/class-logit level by `ProposalTemporalRefiner`.

## Current Main Structural Hyperparameters

These hyperparameters define the current main structure:

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_adapter affinity
--temporal_adapter_time_sigma 4.0
--temporal_fusion yolov
--ref_aux_loss 0.0
--yolov_cls_loss 0.30
--proposal_topk 700
--proposal_after_topk 220
--proposal_nms_radius 1
--proposal_spatial_sigma 0.12
--proposal_time_sigma 4.0
--proposal_loc_gain 0.5
--proposal_cls_sim_gain 0.55
--proposal_reg_sim_gain 0.0
--proposal_score_gain 0.0
--proposal_vote_gain 0.50
--proposal_recall_gain 1.25
--proposal_recall_radius 1
--fam_warmup_epochs 5
--fam_alpha_target 0.65
```

## What These Options Mean

`vid_clip_mode=window` means the model uses a consecutive video window, and all
frames in the window are treated as key frames.

`temporal_adapter=affinity` enables the feature-level temporal adapter. It
aggregates P3/P4/P5 features across the same 16-frame window before the detect
branches run. This is inspired by FGFA/SELSA-style video feature aggregation:
neighboring frames can enhance weak current-frame features before box/class
prediction.

`temporal_fusion=yolov` selects the YOLOV-style proposal temporal refinement
path. During training, the original detection output is supervised by the normal
YOLO losses, while the proposal-refined class logits receive an additional
auxiliary classification loss. During inference and extra evaluation, the
refined class logits are used as the final class prediction.

`proposal_topk` is the pre-selection proposal count per scale and per frame.
`proposal_after_topk` controls how many of those candidates enter the
cross-frame temporal attention. This follows YOLOV's spirit of selecting a
larger proposal set first, then aggregating a smaller high-quality proposal
list.

`proposal_nms_radius` applies local feature-cell suppression before the top-k
selection, reducing repeated low-value neighboring grid points in dense scenes.

`proposal_spatial_sigma` controls how local cross-frame proposal matching should
be. Smaller values enforce stronger spatial locality.

`proposal_time_sigma` biases proposal attention toward nearer frames in the
16-frame window, while still allowing useful support from farther frames.

`proposal_loc_gain` enables a small learned proposal-location attention bias,
similar in purpose to YOLOV's location embedding.

`proposal_vote_gain` enables temporal class voting. Nearby proposals in adjacent
frames can strengthen the current proposal's class logits.

`proposal_recall_gain` and `proposal_recall_radius` allow neighboring-frame
support to pull low-current-confidence locations into the proposal candidate
set, which is intended to improve recall.

`fam_warmup_epochs` and `fam_alpha_target` warm up all temporal residual gates,
including FAM/proposal refinement and the new feature adapter.

## Other Supported Modes

The code also supports older `Detect_VID` modes, but they are not the current
main model structure:

```text
none          -> single-frame Detect-like head
fam           -> dense feature aggregation module
proposal      -> proposal-level class refinement used directly
yolov         -> current main YOLOV-style proposal auxiliary/refined head
fam_proposal  -> dense FAM followed by proposal refinement
logits        -> direct average logits fusion
logits_gated  -> confidence-gated logits fusion
```

The feature adapter can be disabled with:

```bash
--temporal_adapter none
```

For thesis figures and main experiment descriptions, use the
`temporal_adapter + yolov` structure above unless a section explicitly describes
an ablation.

## Current Training Command Identity

The current model structure corresponds to commands that include:

```bash
--vid_clip_mode window \
--vid_window_size 16 \
--num_ref_frames 15 \
--temporal_adapter affinity \
--temporal_adapter_time_sigma 4.0 \
--temporal_fusion yolov \
--proposal_after_topk 220 \
--proposal_nms_radius 1 \
--proposal_time_sigma 4.0 \
--proposal_loc_gain 0.5 \
--proposal_vote_gain 0.50 \
--proposal_recall_gain 1.25 \
--proposal_recall_radius 1 \
--fam_warmup_epochs 5 \
--fam_alpha_target 0.65
```

If these options are changed, the model head structure or behavior should be
treated as a different experimental variant.

