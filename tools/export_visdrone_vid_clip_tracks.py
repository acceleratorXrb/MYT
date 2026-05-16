#!/usr/bin/env python3
"""Export VisDrone-VID tracks from explicit key+ref clip detections.

This keeps MOT/ID evaluation on the same offline clip inference path used by
``export_visdrone_vid_clip_results.py`` for flicker evaluation. Each key frame
is inferred with its sampled references, then ByteTrack links those detections
within the sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_visdrone_vid_clip_results import (
    configure_temporal_options,
    frame_index,
    image_files,
    load_clip,
    resolve_sequence_root,
    sample_ref_positions,
    set_clip_layout,
)
from temporal_state import reset_video_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Official split root or sequence root.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-sequence tracking txt files.")
    parser.add_argument("--tracker", default="ultralytics/cfg/trackers/bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--num_ref_frames", type=int, default=4)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--ref_sample", default="adjacent", choices=["adjacent", "causal"])
    parser.add_argument("--all_keys", action="store_true", help="Infer non-overlapping windows and track every frame once.")
    parser.add_argument("--window_size", type=int, default=16, help="Frames per window when --all_keys is enabled.")
    parser.add_argument("--temporal_fusion", default=None, choices=["trfa", "none"])
    parser.add_argument("--trfa_levels", default=None, choices=["all", "p3", "p4", "p5", "p3p4", "p4p5", "none"])
    return parser.parse_args()


def xyxy_to_xywh_center(xyxy: np.ndarray) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = xyxy.astype(np.float32, copy=True)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    out[:, 0] = out[:, 0] + out[:, 2] / 2.0
    out[:, 1] = out[:, 1] + out[:, 3] / 2.0
    return out


def xyxy_to_xywh_left_top(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def build_tracker(tracker_yaml):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace, yaml_load

    cfg = IterableSimpleNamespace(**yaml_load(tracker_yaml))
    if cfg.tracker_type not in TRACKER_MAP:
        raise ValueError(f"Unsupported tracker_type={cfg.tracker_type!r}")
    return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)


def det_namespace(det: torch.Tensor) -> SimpleNamespace:
    if len(det):
        xyxy = det[:, :4].detach().cpu().numpy().astype(np.float32)
        conf = det[:, 4].detach().cpu().numpy().astype(np.float32)
        cls = det[:, 5].detach().cpu().numpy().astype(np.float32)
        xywh = xyxy_to_xywh_center(xyxy)
    else:
        xywh = np.zeros((0, 4), dtype=np.float32)
        conf = np.zeros((0,), dtype=np.float32)
        cls = np.zeros((0,), dtype=np.float32)
    return SimpleNamespace(xywh=xywh, conf=conf, cls=cls)


def format_track_line(frame_id, track_id, xyxy, score, cls, width, height):
    left, top, box_width, box_height = xyxy_to_xywh_left_top(xyxy, width, height)
    return (
        f"{int(frame_id)},{int(track_id)},{left:.2f},{top:.2f},{box_width:.2f},{box_height:.2f},"
        f"{float(score):.6f},{int(cls) + 1},-1,-1\n"
    )


def main():
    args = parse_args()
    from ultralytics import YOLO
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import select_device

    args.out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    yolo = YOLO(str(args.weights))
    yolo.model.to(device).eval()
    configure_temporal_options(yolo.model, args)
    stride = int(max(getattr(yolo.model, "stride", torch.tensor([32])).max().item(), 32))
    seq_root = resolve_sequence_root(args.source)
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found in {seq_root}")

    for seq_dir in seq_dirs:
        reset_video_state(yolo)
        tracker = build_tracker(args.tracker)
        frames = image_files(seq_dir)
        lines = []
        if args.all_keys:
            window_size = max(1, int(args.window_size or 1))
            for start in range(0, len(frames), window_size):
                window_frames = frames[start : start + window_size]
                tensor, img0s = load_clip(window_frames, args.imgsz, stride, return_all=True)
                tensor = tensor.to(device, non_blocking=True).float() / 255.0
                set_clip_layout(yolo.model, (1, len(window_frames)), all_keys=True, num_ref_frames=args.num_ref_frames)
                with torch.inference_mode():
                    pred = yolo.model(tensor)
                dets = ops.non_max_suppression(pred, args.conf, args.iou, max_det=args.max_det, in_place=False)
                for local_pos, det in enumerate(dets):
                    img0 = img0s[local_pos]
                    if len(det):
                        ops.scale_boxes(tensor.shape[2:], det[:, :4], img0.shape)
                    tracks = tracker.update(det_namespace(det))
                    height, width = img0.shape[:2]
                    index = frame_index(window_frames[local_pos], start + local_pos + 1)
                    if len(tracks):
                        order = np.argsort(tracks[:, 5])[::-1]
                        for row_index in order:
                            row = tracks[row_index]
                            lines.append(format_track_line(index, row[4], row[:4], row[5], row[6], width, height))
            mode_label = "window"
        else:
            for key_pos, frame_path in enumerate(frames):
                ref_pos = sample_ref_positions(
                    len(frames), key_pos, args.num_ref_frames, max(1, args.clip_stride), args.ref_sample
                )
                clip_paths = [frame_path] + [frames[p] for p in ref_pos]
                tensor, key_img0 = load_clip(clip_paths, args.imgsz, stride)
                tensor = tensor.to(device, non_blocking=True).float() / 255.0
                set_clip_layout(yolo.model, (1, len(clip_paths)), num_ref_frames=args.num_ref_frames)
                with torch.inference_mode():
                    pred = yolo.model(tensor)
                det = ops.non_max_suppression(pred, args.conf, args.iou, max_det=args.max_det, in_place=False)[0]
                if len(det):
                    ops.scale_boxes(tensor.shape[2:], det[:, :4], key_img0.shape)

                tracks = tracker.update(det_namespace(det))
                height, width = key_img0.shape[:2]
                index = frame_index(frame_path, key_pos + 1)
                if len(tracks):
                    order = np.argsort(tracks[:, 5])[::-1]
                    for row_index in order:
                        row = tracks[row_index]
                        lines.append(format_track_line(index, row[4], row[:4], row[5], row[6], width, height))
            mode_label = "clip"

        output = args.out / f"{seq_dir.name}.txt"
        output.write_text("".join(lines), encoding="utf-8")
        print(f"{seq_dir.name}: {len(frames)} frames, {len(lines)} {mode_label} tracks -> {output}")


if __name__ == "__main__":
    main()
