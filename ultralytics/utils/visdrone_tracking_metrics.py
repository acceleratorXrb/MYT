# Ultralytics YOLO 🚀, AGPL-3.0 license
"""VisDrone-VID tracking benchmark utilities for IDF1, ID switches, and fragments."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ultralytics.utils import LOGGER


TRACKING_METRIC_KEYS = ["metrics/IDF1", "metrics/IDSwitches", "metrics/Frag"]

# VisDrone categories: 0 ignored regions, 1 pedestrian, 2 people, 3 bicycle, 4 car,
# 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor.
VISDRONE_CATEGORY_TO_YOLO = {category: category - 1 for category in range(1, 11)}
VISDRONE_SPLIT_NAMES = {"train": "train", "val": "val", "test": "test-dev", "test-dev": "test-dev"}


@dataclass(frozen=True)
class TrackingBox:
    """One ground-truth or predicted box with a sequence-local identity."""

    track_id: int
    cls: int
    xyxy: tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class TrackerInput:
    """Minimal object shaped like Ultralytics Results boxes for BYTETracker."""

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


def frame_index(path: Path, fallback: int | None = None) -> int:
    """Infer a 1-based frame index from a VisDrone frame filename."""
    if path.stem.isdigit():
        return int(path.stem)
    matches = re.findall(r"\d+", path.stem)
    if matches:
        return int(matches[-1])
    if fallback is None:
        raise ValueError(f"Cannot infer frame index from {path}")
    return fallback


def parse_visdrone_annotations(annotation_file: Path) -> dict[int, list[TrackingBox]]:
    """Parse one VisDrone-VID annotation txt file into frame-indexed tracking boxes."""
    frames = defaultdict(list)
    with Path(annotation_file).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 10:
                raise ValueError(f"{annotation_file}:{line_no} has {len(fields)} fields, expected at least 10.")

            frame = int(float(fields[0]))
            target_id = int(float(fields[1]))
            left, top, width, height = (float(fields[i]) for i in range(2, 6))
            score = float(fields[6])
            category = int(float(fields[7]))
            if score <= 0 or target_id <= 0 or category not in VISDRONE_CATEGORY_TO_YOLO or width <= 0 or height <= 0:
                continue
            frames[frame].append(
                TrackingBox(
                    track_id=target_id,
                    cls=VISDRONE_CATEGORY_TO_YOLO[category],
                    xyxy=(left, top, left + width, top + height),
                    score=score,
                )
            )
    return dict(frames)


def resolve_visdrone_annotation_dir(data: dict, split: str) -> Path | None:
    """Find the official VisDrone-VID annotation directory for a converted YOLO dataset."""
    root = data.get("path") if isinstance(data, dict) else None
    if root is None:
        return None
    root = Path(root)
    split_name = VISDRONE_SPLIT_NAMES.get(split, split)
    candidates = [
        root / "raw" / f"VisDrone2019-VID-{split_name}" / "annotations",
        root / f"VisDrone2019-VID-{split_name}" / "annotations",
        root / "annotations",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def tracking_available_for_data(data: dict, split: str = "val") -> bool:
    """Return True when VisDrone-VID identity annotations are available for the split."""
    return resolve_visdrone_annotation_dir(data, split) is not None


def load_visdrone_ground_truth(annotation_dir: Path) -> dict[str, dict[int, list[TrackingBox]]]:
    """Load all sequence annotation files from a VisDrone-VID annotations directory."""
    return {path.stem: parse_visdrone_annotations(path) for path in sorted(Path(annotation_dir).glob("*.txt"))}


def _iou_matrix(gt_boxes: list[TrackingBox], pred_boxes: list[TrackingBox]) -> np.ndarray:
    """Compute class-aware IoU matrix for one frame."""
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    gt = np.asarray([box.xyxy for box in gt_boxes], dtype=np.float32)
    pred = np.asarray([box.xyxy for box in pred_boxes], dtype=np.float32)
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    gt_area = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    pred_area = np.clip(pred[:, 2] - pred[:, 0], 0, None) * np.clip(pred[:, 3] - pred[:, 1], 0, None)
    iou = inter / (gt_area[:, None] + pred_area[None, :] - inter + 1e-7)
    class_match = np.asarray([[g.cls == p.cls for p in pred_boxes] for g in gt_boxes], dtype=bool)
    return iou * class_match


def _match_frame(
    gt_boxes: list[TrackingBox], pred_boxes: list[TrackingBox], iou_threshold: float
) -> list[tuple[int, int]]:
    """Match GT and predicted boxes in one frame by maximizing IoU."""
    iou = _iou_matrix(gt_boxes, pred_boxes)
    if not iou.size:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def compute_tracking_metrics(
    ground_truth: dict[str, dict[int, list[TrackingBox]]],
    predictions: dict[str, dict[int, list[TrackingBox]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute sequence-level IDF1, ID switches, and fragmentation over VisDrone-VID tracks."""
    total_gt = 0
    total_pred = 0
    pair_counts = defaultdict(int)
    gt_match_states = defaultdict(list)

    for sequence in sorted(set(ground_truth) | set(predictions)):
        gt_frames = ground_truth.get(sequence, {})
        pred_frames = predictions.get(sequence, {})
        for frame in sorted(set(gt_frames) | set(pred_frames)):
            gt_boxes = gt_frames.get(frame, [])
            pred_boxes = pred_frames.get(frame, [])
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
            matches = _match_frame(gt_boxes, pred_boxes, iou_threshold)
            matched_gt = {gt_index: pred_boxes[pred_index].track_id for gt_index, pred_index in matches}

            for gt_index, gt_box in enumerate(gt_boxes):
                gt_key = (sequence, gt_box.track_id)
                pred_id = matched_gt.get(gt_index)
                gt_match_states[gt_key].append((frame, pred_id))
                if pred_id is not None:
                    pair_counts[(gt_key, (sequence, pred_id))] += 1

    idtp = _global_identity_true_positives(pair_counts)
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    denominator = 2 * idtp + idfp + idfn
    idf1 = 0.0 if denominator == 0 else (2 * idtp) / denominator

    return {
        "metrics/IDF1": float(idf1),
        "metrics/IDSwitches": float(_count_id_switches(gt_match_states)),
        "metrics/Frag": float(_count_fragments(gt_match_states)),
    }


def _global_identity_true_positives(pair_counts: dict[tuple[tuple[str, int], tuple[str, int]], int]) -> int:
    """Find the best one-to-one GT-id to predicted-id assignment."""
    if not pair_counts:
        return 0
    gt_ids = sorted({key[0] for key in pair_counts})
    pred_ids = sorted({key[1] for key in pair_counts})
    gt_index = {track_id: i for i, track_id in enumerate(gt_ids)}
    pred_index = {track_id: i for i, track_id in enumerate(pred_ids)}
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for (gt_id, pred_id), count in pair_counts.items():
        counts[gt_index[gt_id], pred_index[pred_id]] = count
    rows, cols = linear_sum_assignment(-counts)
    return int(sum(counts[row, col] for row, col in zip(rows, cols) if counts[row, col] > 0))


def _count_id_switches(gt_match_states: dict[tuple[str, int], list[tuple[int, int | None]]]) -> int:
    """Count matched identity changes for each GT trajectory."""
    switches = 0
    for states in gt_match_states.values():
        previous_id = None
        for _, pred_id in sorted(states):
            if pred_id is None:
                continue
            if previous_id is not None and pred_id != previous_id:
                switches += 1
            previous_id = pred_id
    return switches


def _count_fragments(gt_match_states: dict[tuple[str, int], list[tuple[int, int | None]]]) -> int:
    """Count interruptions between matched segments for each GT trajectory."""
    fragments = 0
    for states in gt_match_states.values():
        matched_segments = 0
        in_segment = False
        for _, pred_id in sorted(states):
            if pred_id is None:
                in_segment = False
            elif not in_segment:
                matched_segments += 1
                in_segment = True
        fragments += max(0, matched_segments - 1)
    return fragments


def make_tracker_input(predictions: np.ndarray) -> TrackerInput:
    """Convert native xyxy/conf/class predictions to the minimal BYTETracker input object."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.size == 0:
        return TrackerInput(
            xywh=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.float32),
        )
    xyxy = predictions[:, :4]
    xywh = xyxy.copy()
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return TrackerInput(xywh=xywh, conf=predictions[:, 4], cls=predictions[:, 5])


class VisDroneTrackingBenchmark:
    """Collect validator predictions and compute VisDrone-VID tracking metrics at validation end."""

    def __init__(
        self,
        ground_truth: dict[str, dict[int, list[TrackingBox]]] | None = None,
        enabled: bool = False,
        reason: str = "",
    ):
        self.ground_truth = ground_truth or {}
        self.enabled = enabled
        self.reason = reason
        self.predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    @classmethod
    def from_data(cls, data: dict, split: str) -> "VisDroneTrackingBenchmark":
        """Build a benchmark for a validator data dictionary and split."""
        annotation_dir = resolve_visdrone_annotation_dir(data, split)
        if annotation_dir is None:
            return cls(enabled=False, reason="VisDrone-VID annotations were not found")
        return cls(ground_truth=load_visdrone_ground_truth(annotation_dir), enabled=True)

    def add_frame_predictions(self, image_file: str | Path, predictions: np.ndarray) -> None:
        """Collect one image's predictions in native xyxy/conf/class format."""
        if not self.enabled:
            return
        path = Path(image_file)
        self.predictions[path.parent.name][frame_index(path)] = np.asarray(predictions, dtype=np.float32)

    def compute(self) -> dict[str, float]:
        """Run BYTETracker per sequence and compute IDF1, ID switches, and fragments."""
        if not self.enabled:
            return {}
        pred_tracks = self._run_tracker()
        return compute_tracking_metrics(self.ground_truth, pred_tracks)

    def _run_tracker(self) -> dict[str, dict[int, list[TrackingBox]]]:
        """Convert collected detections to tracked boxes with sequence-local tracker IDs."""
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, ROOT, yaml_load

        tracker_cfg = IterableSimpleNamespace(**yaml_load(ROOT / "cfg" / "trackers" / "bytetrack.yaml"))
        tracked_predictions = defaultdict(dict)
        for sequence in sorted(set(self.ground_truth) | set(self.predictions)):
            tracker = BYTETracker(args=tracker_cfg, frame_rate=30)
            sequence_predictions = self.predictions.get(sequence, {})
            frames = sorted(set(self.ground_truth.get(sequence, {})) | set(sequence_predictions))
            for frame in frames:
                tracker_input = make_tracker_input(sequence_predictions.get(frame, np.zeros((0, 6), dtype=np.float32)))
                tracks = tracker.update(tracker_input, img=None)
                tracked_predictions[sequence][frame] = [
                    TrackingBox(
                        track_id=int(track[4]),
                        cls=int(track[6]),
                        xyxy=(float(track[0]), float(track[1]), float(track[2]), float(track[3])),
                        score=float(track[5]),
                    )
                    for track in tracks
                ]
        return dict(tracked_predictions)


def warn_tracking_unavailable_once(benchmark: VisDroneTrackingBenchmark) -> None:
    """Log why tracking metrics are absent without raising on non-VisDrone datasets."""
    if not benchmark.enabled and benchmark.reason:
        LOGGER.warning(f"WARNING ⚠️ skipping VisDrone tracking metrics: {benchmark.reason}")
