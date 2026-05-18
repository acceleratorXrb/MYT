"""Temporal stabilization helpers for VisDrone-VID exports.

The helpers in this file are deliberately GT-free. They only use neighboring
predictions from the same sequence, with strict spatial overlap checks, to
reduce short-term class flicker and obvious tracker fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass
class VidRecord:
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width * 0.5, self.top + self.height * 0.5


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def center_distance_ratio(a: VidRecord, b: VidRecord) -> float:
    ax, ay = a.center
    bx, by = b.center
    scale = max((a.width + b.width + a.height + b.height) * 0.25, 1e-6)
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / scale


def weighted_majority(records: Iterable[VidRecord]) -> tuple[int | None, float, int]:
    weights: dict[int, float] = {}
    count = 0
    for rec in records:
        weights[rec.category] = weights.get(rec.category, 0.0) + max(float(rec.score), 1e-6)
        count += 1
    if not weights:
        return None, 0.0, 0
    cat, weight = max(weights.items(), key=lambda kv: (kv[1], -kv[0]))
    total = sum(weights.values())
    return cat, weight / max(total, 1e-12), count


def _format_record(rec: VidRecord) -> str:
    return (
        f"{int(rec.frame)},{int(rec.track_id)},{rec.left:.2f},{rec.top:.2f},"
        f"{rec.width:.2f},{rec.height:.2f},{float(rec.score):.6f},"
        f"{int(rec.category)},-1,-1\n"
    )


def format_records(records: Iterable[VidRecord]) -> list[str]:
    ordered = sorted(records, key=lambda r: (r.frame, -r.score, r.track_id, r.left, r.top))
    return [_format_record(rec) for rec in ordered]


def build_detection_tracklets(
    records: list[VidRecord],
    iou_thr: float = 0.65,
    max_gap: int = 1,
) -> list[list[int]]:
    """Greedily group raw detections into short tracklets for class smoothing."""
    by_frame: dict[int, list[int]] = {}
    for idx, rec in enumerate(records):
        by_frame.setdefault(rec.frame, []).append(idx)

    active: list[dict] = []
    tracklets: list[list[int]] = []
    for frame in sorted(by_frame):
        frame_indices = sorted(by_frame[frame], key=lambda i: records[i].score, reverse=True)
        for active_item in active:
            active_item["matched"] = False
        for idx in frame_indices:
            rec = records[idx]
            best = None
            best_score = iou_thr
            for active_item in active:
                last = records[active_item["last"]]
                gap = rec.frame - last.frame
                if active_item["matched"] or gap <= 0 or gap > max_gap:
                    continue
                score = iou_xyxy(rec.xyxy, last.xyxy)
                if score > best_score:
                    best_score = score
                    best = active_item
            if best is None:
                tracklets.append([idx])
                active.append({"last": idx, "tracklet": len(tracklets) - 1, "matched": True})
            else:
                tracklets[best["tracklet"]].append(idx)
                best["last"] = idx
                best["matched"] = True
        active = [a for a in active if frame - records[a["last"]].frame < max_gap]
    return tracklets


def stabilize_detection_classes(
    records: list[VidRecord],
    iou_thr: float = 0.65,
    max_gap: int = 1,
    min_len: int = 3,
    vote_ratio: float = 0.58,
    max_score_to_change: float = 0.55,
) -> tuple[list[VidRecord], dict[str, int | float]]:
    """Smooth raw detection classes along high-IoU short tracklets."""
    records = [replace(r) for r in records]
    changes = 0
    tracklets = build_detection_tracklets(records, iou_thr=iou_thr, max_gap=max_gap)
    eligible = 0
    for indices in tracklets:
        if len(indices) < min_len:
            continue
        cat, ratio, _ = weighted_majority(records[i] for i in indices)
        if cat is None or ratio < vote_ratio:
            continue
        eligible += 1
        for idx in indices:
            rec = records[idx]
            if rec.category != cat and rec.score <= max_score_to_change:
                records[idx] = replace(rec, category=cat)
                changes += 1
    return records, {
        "det_tracklets": len(tracklets),
        "det_tracklets_smoothed": eligible,
        "det_class_changes": changes,
    }


def _group_by_track(records: list[VidRecord]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, rec in enumerate(records):
        groups.setdefault(rec.track_id, []).append(idx)
    for indices in groups.values():
        indices.sort(key=lambda i: records[i].frame)
    return groups


def smooth_track_classes(
    records: list[VidRecord],
    min_len: int = 3,
    vote_ratio: float = 0.55,
) -> tuple[list[VidRecord], int]:
    records = [replace(r) for r in records]
    changes = 0
    for indices in _group_by_track(records).values():
        if len(indices) < min_len:
            continue
        cat, ratio, _ = weighted_majority(records[i] for i in indices)
        if cat is None or ratio < vote_ratio:
            continue
        for idx in indices:
            rec = records[idx]
            if rec.category != cat:
                records[idx] = replace(rec, category=cat)
                changes += 1
    return records, changes


def link_short_track_fragments(
    records: list[VidRecord],
    max_gap: int = 3,
    iou_thr: float = 0.45,
    center_ratio_thr: float = 0.55,
) -> tuple[list[VidRecord], int]:
    """Relabel short separated track fragments when endpoints strongly overlap."""
    records = [replace(r) for r in records]
    groups = _group_by_track(records)
    summaries = []
    for tid, indices in groups.items():
        first = records[indices[0]]
        last = records[indices[-1]]
        cat, _, _ = weighted_majority(records[i] for i in indices)
        summaries.append({"tid": tid, "first": first, "last": last, "category": cat})
    summaries.sort(key=lambda x: (x["first"].frame, x["tid"]))

    parent = {s["tid"]: s["tid"] for s in summaries}

    def find(tid: int) -> int:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    links = 0
    for child in summaries:
        best_tid = None
        best_score = 0.0
        for prev in summaries:
            gap = child["first"].frame - prev["last"].frame
            if prev["tid"] == child["tid"] or gap <= 0 or gap > max_gap:
                continue
            if child["category"] != prev["category"]:
                continue
            overlap = iou_xyxy(prev["last"].xyxy, child["first"].xyxy)
            center_ratio = center_distance_ratio(prev["last"], child["first"])
            if overlap < iou_thr and center_ratio > center_ratio_thr:
                continue
            score = overlap + max(0.0, center_ratio_thr - center_ratio) * 0.1
            if score > best_score:
                best_score = score
                best_tid = prev["tid"]
        if best_tid is not None:
            root_child = find(child["tid"])
            root_parent = find(best_tid)
            if root_child != root_parent:
                parent[root_child] = root_parent
                links += 1

    if links:
        for idx, rec in enumerate(records):
            records[idx] = replace(rec, track_id=find(rec.track_id))
    return records, links


def fill_short_track_gaps(
    records: list[VidRecord],
    max_gap: int = 1,
    min_endpoint_score: float = 0.15,
    iou_thr: float = 0.35,
) -> tuple[list[VidRecord], int]:
    """Interpolate one or two missing frames inside stable tracks."""
    records = [replace(r) for r in records]
    existing = {(rec.frame, rec.track_id) for rec in records}
    additions: list[VidRecord] = []
    for indices in _group_by_track(records).values():
        seq = [records[i] for i in indices]
        for a, b in zip(seq[:-1], seq[1:]):
            gap = b.frame - a.frame - 1
            if gap <= 0 or gap > max_gap:
                continue
            if a.category != b.category or min(a.score, b.score) < min_endpoint_score:
                continue
            if iou_xyxy(a.xyxy, b.xyxy) < iou_thr:
                continue
            for step in range(1, gap + 1):
                frame = a.frame + step
                if (frame, a.track_id) in existing:
                    continue
                r = step / (gap + 1)
                additions.append(
                    VidRecord(
                        frame=frame,
                        track_id=a.track_id,
                        left=a.left * (1 - r) + b.left * r,
                        top=a.top * (1 - r) + b.top * r,
                        width=a.width * (1 - r) + b.width * r,
                        height=a.height * (1 - r) + b.height * r,
                        score=max(min(a.score, b.score) * 0.85, 1e-4),
                        category=a.category,
                    )
                )
                existing.add((frame, a.track_id))
    return records + additions, len(additions)


def stabilize_track_records(records: list[VidRecord]) -> tuple[list[VidRecord], dict[str, int | float]]:
    """Apply conservative class smoothing, fragment linking, and short-gap filling."""
    smoothed, class_changes = smooth_track_classes(records)
    linked, links = link_short_track_fragments(smoothed)
    filled, fills = fill_short_track_gaps(linked)
    return filled, {
        "track_class_changes": class_changes,
        "track_fragment_links": links,
        "track_gap_fills": fills,
    }
