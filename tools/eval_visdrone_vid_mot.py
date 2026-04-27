#!/usr/bin/env python3
"""Evaluate VisDrone-VID tracking txt files with local ID metrics.

The expected input is one official-format txt per sequence:
frame,id,left,top,width,height,score,category,truncation,occlusion
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Directory containing VisDrone ground-truth txt files.")
    parser.add_argument("--pred", type=Path, required=True, help="Directory containing predicted tracking txt files.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold used to match detections to GT.")
    return parser.parse_args()


def _read_visdrone_txt(path, is_gt):
    by_frame = defaultdict(list)
    if not path.exists():
        return by_frame
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_no} has {len(fields)} fields, expected at least 8.")
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            left, top, width, height = (float(fields[i]) for i in range(2, 6))
            score = float(fields[6])
            category = int(float(fields[7]))
            if width <= 0 or height <= 0 or category <= 0:
                continue
            if is_gt and (score <= 0 or track_id < 0):
                continue
            if not is_gt and track_id < 0:
                continue
            by_frame[frame_id].append(
                {
                    "track_id": track_id,
                    "category": category,
                    "box": np.array([left, top, left + width, top + height], dtype=np.float32),
                }
            )
    return by_frame


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def _match_frame(gt_items, pred_items, iou_threshold):
    if not gt_items or not pred_items:
        return []
    costs = np.ones((len(gt_items), len(pred_items)), dtype=np.float32)
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if gt["category"] == pred["category"]:
                costs[gi, pi] = 1.0 - _iou(gt["box"], pred["box"])
    gt_idx, pred_idx = linear_sum_assignment(costs)
    matches = []
    for gi, pi in zip(gt_idx, pred_idx):
        iou = 1.0 - float(costs[gi, pi])
        if iou >= iou_threshold:
            matches.append((gi, pi))
    return matches


def evaluate_mot(gt_dir, pred_dir, iou_threshold=0.5):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.txt"))
    state = {}
    totals = {
        "GT": 0,
        "Pred": 0,
        "IDTP": 0,
        "IDFP": 0,
        "IDFN": 0,
        "ID Switches": 0,
        "Frag": 0,
    }

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        gt_by_frame = _read_visdrone_txt(gt_file, is_gt=True)
        pred_by_frame = _read_visdrone_txt(pred_file, is_gt=False)
        frames = sorted(set(gt_by_frame) | set(pred_by_frame))
        for frame_id in frames:
            gt_items = gt_by_frame.get(frame_id, [])
            pred_items = pred_by_frame.get(frame_id, [])
            totals["GT"] += len(gt_items)
            totals["Pred"] += len(pred_items)
            matches = _match_frame(gt_items, pred_items, iou_threshold)
            matched_gt = {gi for gi, _ in matches}
            matched_pred = {pi for _, pi in matches}

            for gi, pi in matches:
                gt = gt_items[gi]
                pred = pred_items[pi]
                key = (gt_file.stem, gt["track_id"])
                gt_state = state.setdefault(
                    key,
                    {"last_pred": None, "ever_matched": False, "was_matched": False, "switched": False},
                )
                pred_id = pred["track_id"]
                if gt_state["ever_matched"] and not gt_state["was_matched"]:
                    totals["Frag"] += 1
                if gt_state["last_pred"] is not None and gt_state["last_pred"] != pred_id:
                    totals["ID Switches"] += 1
                    gt_state["switched"] = True
                if not gt_state["switched"]:
                    totals["IDTP"] += 1
                gt_state["last_pred"] = pred_id
                gt_state["ever_matched"] = True
                gt_state["was_matched"] = True

            for gi, gt in enumerate(gt_items):
                if gi not in matched_gt:
                    key = (gt_file.stem, gt["track_id"])
                    gt_state = state.setdefault(
                        key,
                        {"last_pred": None, "ever_matched": False, "was_matched": False, "switched": False},
                    )
                    gt_state["was_matched"] = False

            totals["IDFN"] += len(gt_items) - len(matched_gt)
            totals["IDFP"] += len(pred_items) - len(matched_pred)

    switched_or_wrong = totals["GT"] - totals["IDTP"] - totals["IDFN"]
    totals["IDFN"] += max(0, switched_or_wrong)
    totals["IDFP"] += max(0, switched_or_wrong)
    denom_f1 = 2 * totals["IDTP"] + totals["IDFP"] + totals["IDFN"]
    totals["IDF1"] = (2 * totals["IDTP"] / denom_f1) if denom_f1 else 0.0
    totals["IDP"] = (totals["IDTP"] / (totals["IDTP"] + totals["IDFP"])) if totals["IDTP"] + totals["IDFP"] else 0.0
    totals["IDR"] = (totals["IDTP"] / (totals["IDTP"] + totals["IDFN"])) if totals["IDTP"] + totals["IDFN"] else 0.0
    return totals


def main():
    args = parse_args()
    metrics = evaluate_mot(args.gt, args.pred, iou_threshold=args.iou)
    payload = {"metrics": metrics, "gt": str(args.gt), "pred": str(args.pred), "iou": args.iou}
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
