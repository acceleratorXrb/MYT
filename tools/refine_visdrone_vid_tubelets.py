#!/usr/bin/env python3
"""Refine VisDrone-VID detections with tubelet-level temporal consistency.

The script consumes per-sequence VisDrone-format detection txt files and writes:

1. refined detections for flicker evaluation
2. refined tracks with stable tubelet ids for MOT/ID evaluation

It is intentionally model-agnostic, so the same video refinement path can be
used for YOLOv8, official Mamba-YOLO-T, and the VID model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    frame: int
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int
    source: str = "det"

    @property
    def box(self) -> np.ndarray:
        return np.array([self.left, self.top, self.left + self.width, self.top + self.height], dtype=np.float32)

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width * 0.5, self.top + self.height * 0.5

    @property
    def scale(self) -> float:
        return math.sqrt(max(self.width * self.height, 1.0))


@dataclass
class Tubelet:
    tid: int
    dets: list[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.dets[-1]

    @property
    def first_frame(self) -> int:
        return self.dets[0].frame

    @property
    def last_frame(self) -> int:
        return self.dets[-1].frame

    @property
    def max_score(self) -> float:
        return max((d.score for d in self.dets), default=0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True, help="Directory with raw detection txt files.")
    p.add_argument("--out_det", type=Path, required=True, help="Directory for refined detection txt files.")
    p.add_argument("--out_tracks", type=Path, required=True, help="Directory for refined track txt files.")
    p.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    p.add_argument("--seed_score", type=float, default=0.12, help="Minimum score for starting a new tubelet.")
    p.add_argument("--attach_score", type=float, default=0.01, help="Minimum score for attaching to an existing tubelet.")
    p.add_argument("--keep_score", type=float, default=0.08, help="Minimum refined score to output as a detection.")
    p.add_argument("--track_keep_score", type=float, default=0.12, help="Minimum refined score to output as a track.")
    p.add_argument("--track_min_len", type=int, default=4, help="Suppress tubelets shorter than this unless very confident.")
    p.add_argument("--track_min_score", type=float, default=0.20, help="Minimum max score for output tubelets.")
    p.add_argument("--track_min_avg_score", type=float, default=0.10, help="Minimum average score for output tubelets.")
    p.add_argument("--assoc_iou", type=float, default=0.22, help="IoU threshold for frame-to-frame association.")
    p.add_argument("--assoc_center_factor", type=float, default=2.5, help="Center-distance gate in object-scale units.")
    p.add_argument("--max_gap", type=int, default=2, help="Maximum frame gap for keeping a tubelet active.")
    p.add_argument("--gap_fill", type=int, default=1, help="Interpolate missing detections for gaps up to this length.")
    p.add_argument("--smooth", type=float, default=0.55, help="Box smoothing weight toward the temporal moving average.")
    p.add_argument("--vote_gain", type=float, default=0.35, help="Class-vote score gain for tubelet majority category.")
    p.add_argument("--recall_gain", type=float, default=0.25, help="Score propagation gain from tubelet max score.")
    p.add_argument("--input_topk", type=int, default=60, help="Maximum raw detections kept per frame before association.")
    p.add_argument("--max_per_frame", type=int, default=80, help="Maximum refined outputs per frame.")
    p.add_argument("--output_nms_iou", type=float, default=0.65, help="Class-wise NMS IoU for refined per-frame outputs.")
    p.add_argument("--debug", action="store_true", help="Print per-sequence TTRM sanity summaries.")
    p.add_argument("--debug_sequences", type=int, default=3, help="Number of sequences to print when --debug is enabled.")
    return p.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + ab - inter, 1e-12)


def norm_center_distance(a: Detection, b: Detection) -> float:
    ax, ay = a.center
    bx, by = b.center
    dist = math.hypot(ax - bx, ay - by)
    return dist / max((a.scale + b.scale) * 0.5, 1.0)


def read_detections(path: Path) -> dict[int, list[Detection]]:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    if not path.exists():
        return by_frame
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) >= 10:
            frame = int(float(fields[0]))
            left, top, width, height = (float(fields[i]) for i in range(2, 6))
            score = float(fields[6])
            category = int(float(fields[7]))
        elif len(fields) >= 7:
            frame = int(float(fields[0]))
            score = float(fields[1])
            category = int(float(fields[2]))
            left, top, width, height = (float(fields[i]) for i in range(3, 7))
        else:
            raise ValueError(f"{path}:{line_no} has {len(fields)} fields; expected >=7")
        if width <= 0 or height <= 0 or category <= 0:
            continue
        by_frame[frame].append(Detection(frame, left, top, width, height, score, category))
    return by_frame


def association_score(tube: Tubelet, det: Detection, args: argparse.Namespace) -> float:
    last = tube.last
    gap = det.frame - last.frame
    if gap <= 0 or gap > args.max_gap:
        return -1.0
    ov = iou(last.box, det.box)
    cd = norm_center_distance(last, det)
    same_cls = 1.0 if last.category == det.category else 0.0
    if ov < args.assoc_iou and cd > args.assoc_center_factor:
        return -1.0
    gap_penalty = 0.08 * max(gap - 1, 0)
    center_affinity = math.exp(-0.5 * cd * cd)
    return 0.70 * ov + 0.25 * center_affinity + 0.12 * same_cls + 0.08 * det.score - gap_penalty


def build_tubelets(by_frame: dict[int, list[Detection]], args: argparse.Namespace) -> tuple[list[Tubelet], dict]:
    tubelets: list[Tubelet] = []
    active: list[Tubelet] = []
    next_id = 1
    stats = {
        "frames_seen": len(by_frame),
        "raw_candidates_after_score": 0,
        "candidates_used_after_topk": 0,
        "candidates_dropped_by_topk": 0,
        "associations": 0,
        "new_tubelets": 0,
    }

    for frame in sorted(by_frame):
        detections = [d for d in by_frame[frame] if d.score >= args.attach_score]
        detections.sort(key=lambda d: d.score, reverse=True)
        stats["raw_candidates_after_score"] += len(detections)
        if args.input_topk > 0:
            stats["candidates_dropped_by_topk"] += max(0, len(detections) - args.input_topk)
            detections = detections[: args.input_topk]
        stats["candidates_used_after_topk"] += len(detections)
        active = [t for t in active if frame - t.last_frame <= args.max_gap]

        candidates: list[tuple[float, int, int]] = []
        for ti, tube in enumerate(active):
            for di, det in enumerate(detections):
                score = association_score(tube, det, args)
                if score >= 0.0:
                    candidates.append((score, ti, di))
        candidates.sort(reverse=True)

        used_tubes: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in candidates:
            if ti in used_tubes or di in used_dets:
                continue
            active[ti].dets.append(detections[di])
            used_tubes.add(ti)
            used_dets.add(di)
            stats["associations"] += 1

        for di, det in enumerate(detections):
            if di in used_dets or det.score < args.seed_score:
                continue
            tube = Tubelet(next_id, [det])
            next_id += 1
            tubelets.append(tube)
            active.append(tube)
            stats["new_tubelets"] += 1

    return tubelets, stats


def interpolate_gap(left: Detection, right: Detection) -> list[Detection]:
    gap = right.frame - left.frame
    if gap <= 1:
        return []
    out = []
    for step in range(1, gap):
        r = step / gap
        score = min(left.score, right.score) * 0.85
        category = left.category if left.score >= right.score else right.category
        out.append(
            Detection(
                frame=left.frame + step,
                left=left.left * (1 - r) + right.left * r,
                top=left.top * (1 - r) + right.top * r,
                width=left.width * (1 - r) + right.width * r,
                height=left.height * (1 - r) + right.height * r,
                score=score,
                category=category,
                source="interp",
            )
        )
    return out


def refine_tubelet(tube: Tubelet, args: argparse.Namespace) -> tuple[Tubelet, dict]:
    dets = sorted(tube.dets, key=lambda d: d.frame)
    filled: list[Detection] = []
    interpolated = 0
    for i, det in enumerate(dets):
        if i:
            prev = dets[i - 1]
            gap = det.frame - prev.frame - 1
            if 0 < gap <= args.gap_fill:
                gap_dets = interpolate_gap(prev, det)
                interpolated += len(gap_dets)
                filled.extend(gap_dets)
        filled.append(det)

    votes: Counter[int] = Counter()
    for det in filled:
        votes[det.category] += max(det.score, 1e-6)
    majority_cat, majority_weight = votes.most_common(1)[0]
    total_weight = max(sum(votes.values()), 1e-6)
    vote_ratio = majority_weight / total_weight
    max_score = max(d.score for d in filled)

    refined: list[Detection] = []
    smooth_box = None
    category_changed = 0
    score_increased = 0
    smoothed_boxes = 0
    for det in filled:
        box = det.box.astype(np.float32)
        if smooth_box is None:
            smooth_box = box
        else:
            smooth_box = args.smooth * smooth_box + (1.0 - args.smooth) * box
            smoothed_boxes += 1
        x1, y1, x2, y2 = smooth_box.tolist()
        category = majority_cat
        if det.category != majority_cat:
            category_changed += 1
        vote_boost = args.vote_gain * vote_ratio if det.category == majority_cat else args.vote_gain * 0.5 * vote_ratio
        propagated = det.score + 0.5 * args.recall_gain * max(0.0, max_score - det.score)
        score = propagated + (1.0 - propagated) * min(vote_boost * 0.35, 0.5)
        score = min(0.999999, max(score, det.score))
        if score > det.score + 1e-6:
            score_increased += 1
        refined.append(
            Detection(
                frame=det.frame,
                left=x1,
                top=y1,
                width=max(0.0, x2 - x1),
                height=max(0.0, y2 - y1),
                score=score,
                category=category,
                source=det.source,
            )
        )
    return Tubelet(tube.tid, refined), {
        "interpolated_detections": interpolated,
        "category_changed": category_changed,
        "score_increased": score_increased,
        "smoothed_boxes": smoothed_boxes,
        "majority_category": majority_cat,
        "vote_ratio": vote_ratio,
        "raw_len": len(dets),
        "refined_len": len(refined),
    }


def keep_tubelet(tube: Tubelet, args: argparse.Namespace) -> bool:
    if tube.max_score < args.track_min_score:
        return False
    avg_score = sum(d.score for d in tube.dets) / max(len(tube.dets), 1)
    if avg_score < args.track_min_avg_score:
        return False
    if len(tube.dets) >= args.track_min_len:
        return True
    return tube.max_score >= max(args.track_min_score * 2.0, 0.20)


def format_line(det: Detection, track_id: int) -> str:
    return (
        f"{int(det.frame)},{int(track_id)},{det.left:.2f},{det.top:.2f},{det.width:.2f},{det.height:.2f},"
        f"{float(det.score):.6f},{int(det.category)},-1,-1\n"
    )


def cap_per_frame(items: list[tuple[int, Detection]], max_per_frame: int) -> list[tuple[int, Detection]]:
    if max_per_frame <= 0:
        return items
    grouped: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for item in items:
        grouped[item[1].frame].append(item)
    kept = []
    for frame in sorted(grouped):
        frame_items = sorted(grouped[frame], key=lambda x: x[1].score, reverse=True)[:max_per_frame]
        kept.extend(frame_items)
    return kept


def nms_per_frame(items: list[tuple[int, Detection]], iou_thr: float) -> list[tuple[int, Detection]]:
    if iou_thr <= 0:
        return items
    grouped: dict[tuple[int, int], list[tuple[int, Detection]]] = defaultdict(list)
    for item in items:
        grouped[(item[1].frame, item[1].category)].append(item)
    kept = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda x: x[1].score, reverse=True)
        selected: list[tuple[int, Detection]] = []
        for item in candidates:
            if all(iou(item[1].box, chosen[1].box) < iou_thr for chosen in selected):
                selected.append(item)
        kept.extend(selected)
    return kept


def refine_sequence(path: Path, args: argparse.Namespace) -> dict:
    raw = read_detections(path)
    tubes, build_stats = build_tubelets(raw, args)
    refined_pairs = [refine_tubelet(t, args) for t in tubes]
    refine_stats = [s for _, s in refined_pairs]
    refined = [t for t, _ in refined_pairs]
    refined = [t for t in refined if keep_tubelet(t, args)]

    det_items: list[tuple[int, Detection]] = []
    track_items: list[tuple[int, Detection]] = []
    for tube in refined:
        for det in tube.dets:
            if det.score >= args.keep_score:
                det_items.append((-1, det))
            if det.score >= args.track_keep_score:
                track_items.append((tube.tid, det))
    det_items = nms_per_frame(det_items, args.output_nms_iou)
    track_items = nms_per_frame(track_items, args.output_nms_iou)
    det_items = cap_per_frame(det_items, args.max_per_frame)
    track_items = cap_per_frame(track_items, args.max_per_frame)
    det_items.sort(key=lambda x: (x[1].frame, -x[1].score))
    track_items.sort(key=lambda x: (x[1].frame, -x[1].score))

    args.out_det.mkdir(parents=True, exist_ok=True)
    args.out_tracks.mkdir(parents=True, exist_ok=True)
    (args.out_det / path.name).write_text("".join(format_line(det, tid) for tid, det in det_items), encoding="utf-8")
    (args.out_tracks / path.name).write_text(
        "".join(format_line(det, tid) for tid, det in track_items), encoding="utf-8"
    )
    raw_count = sum(len(v) for v in raw.values())
    raw_frames = len(raw)
    refined_frames = len({det.frame for _, det in det_items})
    tube_lengths = [len(t.dets) for t in refined]
    stats = {
        **build_stats,
        "raw_frames": raw_frames,
        "refined_frames": refined_frames,
        "interpolated_detections": sum(s["interpolated_detections"] for s in refine_stats),
        "category_changed": sum(s["category_changed"] for s in refine_stats),
        "score_increased": sum(s["score_increased"] for s in refine_stats),
        "smoothed_boxes": sum(s["smoothed_boxes"] for s in refine_stats),
        "avg_kept_tubelet_len": float(np.mean(tube_lengths)) if tube_lengths else 0.0,
        "max_kept_tubelet_len": max(tube_lengths) if tube_lengths else 0,
    }
    return {
        "sequence": path.stem,
        "raw_detections": raw_count,
        "tubelets": len(tubes),
        "kept_tubelets": len(refined),
        "refined_detections": len(det_items),
        "refined_tracks": len(track_items),
        **stats,
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.pred.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No prediction txt files found in {args.pred}")
    per_seq = [refine_sequence(path, args) for path in files]
    totals = {
        "raw_detections": sum(x["raw_detections"] for x in per_seq),
        "tubelets": sum(x["tubelets"] for x in per_seq),
        "kept_tubelets": sum(x["kept_tubelets"] for x in per_seq),
        "refined_detections": sum(x["refined_detections"] for x in per_seq),
        "refined_tracks": sum(x["refined_tracks"] for x in per_seq),
        "raw_frames": sum(x["raw_frames"] for x in per_seq),
        "refined_frames": sum(x["refined_frames"] for x in per_seq),
        "associations": sum(x["associations"] for x in per_seq),
        "interpolated_detections": sum(x["interpolated_detections"] for x in per_seq),
        "category_changed": sum(x["category_changed"] for x in per_seq),
        "score_increased": sum(x["score_increased"] for x in per_seq),
        "smoothed_boxes": sum(x["smoothed_boxes"] for x in per_seq),
        "candidates_dropped_by_topk": sum(x["candidates_dropped_by_topk"] for x in per_seq),
    }
    totals["refined_per_raw_ratio"] = totals["refined_detections"] / max(totals["raw_detections"], 1)
    totals["frame_coverage_ratio"] = totals["refined_frames"] / max(totals["raw_frames"], 1)
    warnings = []
    if totals["raw_detections"] > 0 and totals["refined_detections"] == 0:
        warnings.append("TTRM produced zero refined detections from non-empty raw detections.")
    if totals["tubelets"] > 0 and totals["kept_tubelets"] == 0:
        warnings.append("TTRM built tubelets but all were filtered out.")
    if totals["frame_coverage_ratio"] < 0.25:
        warnings.append("TTRM refined frame coverage is very low; thresholds may be too strict.")
    if totals["refined_tracks"] > totals["raw_frames"] * max(args.max_per_frame, 1):
        warnings.append("TTRM refined track count exceeds the configured per-frame cap; check output filtering.")
    if totals["raw_frames"] and totals["refined_tracks"] / totals["raw_frames"] > 120:
        warnings.append("TTRM is producing unusually many tracks per frame; thresholds may be too loose.")
    payload = {"method": "TTRM", "args": vars(args) | {"pred": str(args.pred), "out_det": str(args.out_det), "out_tracks": str(args.out_tracks), "summary": str(args.summary) if args.summary else None}, "totals": totals, "per_seq": per_seq}
    if warnings:
        payload["warnings"] = warnings
    print(json.dumps({"method": "TTRM", "totals": totals, "warnings": warnings}, indent=2))
    if args.debug:
        for item in per_seq[: max(0, args.debug_sequences)]:
            print(
                "[ttrm-debug] "
                f"seq={item['sequence']} raw={item['raw_detections']} refined={item['refined_detections']} "
                f"tubelets={item['tubelets']}/{item['kept_tubelets']} assoc={item['associations']} "
                f"interp={item['interpolated_detections']} cls_fix={item['category_changed']} "
                f"score_up={item['score_increased']} coverage={item['refined_frames']}/{item['raw_frames']}",
                flush=True,
            )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
