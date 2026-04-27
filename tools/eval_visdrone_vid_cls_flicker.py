#!/usr/bin/env python3
"""Evaluate per-track classification flicker on VisDrone-VID predictions.

Headline metric: macro-averaged classification flicker rate, computed by
matching predictions to ground-truth tracks (IoU >= 0.5, same category not
required) and counting how often the predicted class for the same GT track
changes between consecutive frames.

GT format (per sequence, ground-truth):
    frame,track_id,left,top,width,height,score,category,truncation,occlusion

Prediction format (per sequence, exported by Ultralytics predict + save_txt
or by tools/export_visdrone_vid_results.py):
    frame_id,score,category,left,top,width,height
or VisDrone official:
    frame,track_id,left,top,width,height,score,category,truncation,occlusion
(both supported via auto-detection on field count and content).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, required=True,
                   help="Directory of GT txt files (one per sequence).")
    p.add_argument("--pred", type=Path, required=True,
                   help="Directory of prediction txt files (one per sequence).")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for GT<->pred matching.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional output JSON path.")
    return p.parse_args()


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-12)


def _read_gt(path: Path) -> dict:
    """Returns {frame_id: list of {track_id, category, box(xyxy)}}."""
    by_frame = defaultdict(list)
    if not path.exists():
        return by_frame
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) < 8:
            raise ValueError(f"{path}:{line_no} has {len(fields)} fields (need >=8 GT)")
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        l, t, w, h = (float(fields[i]) for i in range(2, 6))
        score = float(fields[6])
        cat = int(float(fields[7]))
        if w <= 0 or h <= 0 or cat <= 0 or score <= 0 or track_id < 0:
            continue
        by_frame[frame_id].append({
            "track_id": track_id,
            "category": cat,
            "box": np.array([l, t, l + w, t + h], dtype=np.float32),
        })
    return by_frame


def _read_pred(path: Path) -> dict:
    """Auto-detect VisDrone-official (10 fields) or simple (7 fields) format.

    Returns {frame_id: list of {category, box(xyxy), score}}.
    Predicted track_ids (if present) are ignored — we use GT tracks for flicker.
    """
    by_frame = defaultdict(list)
    if not path.exists():
        return by_frame
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) >= 10:
            # VisDrone official: frame,track,l,t,w,h,score,cat,trunc,occ
            frame_id = int(float(fields[0]))
            l, t, w, h = (float(fields[i]) for i in range(2, 6))
            score = float(fields[6])
            cat = int(float(fields[7]))
        elif len(fields) >= 7:
            # Simple: frame,score,cat,l,t,w,h
            frame_id = int(float(fields[0]))
            score = float(fields[1])
            cat = int(float(fields[2]))
            l, t, w, h = (float(fields[i]) for i in range(3, 7))
        else:
            raise ValueError(f"{path}:{line_no} has {len(fields)} fields (need >=7 pred)")
        if w <= 0 or h <= 0 or cat <= 0:
            continue
        by_frame[frame_id].append({
            "category": cat,
            "box": np.array([l, t, l + w, t + h], dtype=np.float32),
            "score": score,
        })
    return by_frame


def _best_match(gt_box, preds, iou_thr):
    """Return predicted category of the highest-IoU pred above threshold, else None."""
    best_iou, best_cat = 0.0, None
    for p in preds:
        v = _iou(gt_box, p["box"])
        if v > best_iou and v >= iou_thr:
            best_iou, best_cat = v, p["category"]
    return best_cat


def evaluate_flicker(gt_dir: Path, pred_dir: Path, iou_thr: float = 0.5) -> dict:
    gt_files = sorted(gt_dir.glob("*.txt"))
    per_seq_flicker = {}
    total_pairs = 0
    total_changes = 0
    total_tracks = 0
    total_dropped_frames = 0

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        gt_by_frame = _read_gt(gt_file)
        pr_by_frame = _read_pred(pred_file)
        # collect per-track sequences of predicted classes
        track_seqs: dict[int, list[int | None]] = defaultdict(list)
        for frame_id in sorted(gt_by_frame):
            for gt in gt_by_frame[frame_id]:
                cat = _best_match(gt["box"], pr_by_frame.get(frame_id, []), iou_thr)
                track_seqs[gt["track_id"]].append(cat)

        # compute flicker per track (drop None entries before counting transitions)
        seq_pairs = 0
        seq_changes = 0
        seq_tracks_evaluated = 0
        for tid, seq in track_seqs.items():
            kept = [c for c in seq if c is not None]
            total_dropped_frames += len(seq) - len(kept)
            if len(kept) < 2:
                continue
            seq_tracks_evaluated += 1
            for a, b in zip(kept[:-1], kept[1:]):
                seq_pairs += 1
                if a != b:
                    seq_changes += 1
        seq_flicker = (seq_changes / seq_pairs) if seq_pairs else 0.0
        per_seq_flicker[gt_file.stem] = {
            "flicker": seq_flicker,
            "pairs": seq_pairs,
            "changes": seq_changes,
            "tracks": seq_tracks_evaluated,
        }
        total_pairs += seq_pairs
        total_changes += seq_changes
        total_tracks += seq_tracks_evaluated

    macro_flicker = (
        np.mean([s["flicker"] for s in per_seq_flicker.values()])
        if per_seq_flicker else 0.0
    )
    micro_flicker = (total_changes / total_pairs) if total_pairs else 0.0
    return {
        "macro_flicker": float(macro_flicker),
        "micro_flicker": float(micro_flicker),
        "total_pairs": total_pairs,
        "total_changes": total_changes,
        "tracks_evaluated": total_tracks,
        "frames_dropped_no_match": total_dropped_frames,
        "iou_threshold": iou_thr,
        "per_seq": per_seq_flicker,
    }


def main():
    args = parse_args()
    metrics = evaluate_flicker(args.gt, args.pred, iou_thr=args.iou)
    summary = {k: v for k, v in metrics.items() if k != "per_seq"}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
