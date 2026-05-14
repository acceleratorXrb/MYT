#!/usr/bin/env python3
"""Export VisDrone-VID detections with explicit key+ref clip inference."""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from temporal_state import reset_video_state


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Official split root or sequence root.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-sequence txt files.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--num_ref_frames", type=int, default=4)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--ref_sample", default="adjacent", choices=["adjacent", "causal"])
    parser.add_argument("--all_keys", action="store_true", help="Infer non-overlapping windows and output every frame once.")
    parser.add_argument("--window_size", type=int, default=16, help="Frames per window when --all_keys is enabled.")
    parser.add_argument("--temporal_fusion", default=None, choices=["score_smooth", "none"])
    parser.add_argument("--score_smooth_sigma", type=float, default=None)
    parser.add_argument("--score_smooth_cls_gain", type=float, default=None)
    parser.add_argument("--score_smooth_conf_gain", type=float, default=None)
    parser.add_argument("--score_smooth_min_ref_score", type=float, default=None)
    return parser.parse_args()


def frame_index(path, fallback):
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else fallback


def image_files(seq_dir):
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def resolve_sequence_root(path):
    if path.name == "sequences" and path.is_dir():
        return path
    candidate = path / "sequences"
    if candidate.is_dir():
        return candidate
    return path


def sample_adjacent_positions(length, key_pos, num_refs, stride):
    left_count = num_refs // 2
    right_count = num_refs - left_count
    desired = list(range(-left_count, 0)) + list(range(1, right_count + 1))
    picked = []
    for offset in desired:
        pos = key_pos + offset * stride
        if 0 <= pos < length and pos != key_pos and pos not in picked:
            picked.append(pos)
    if len(picked) < num_refs:
        candidates = [p for p in range(length) if p != key_pos and p not in picked]
        candidates.sort(key=lambda p: (abs(p - key_pos), p))
        picked.extend(candidates[: num_refs - len(picked)])
    if len(picked) < num_refs:
        picked.extend([picked[-1] if picked else key_pos] * (num_refs - len(picked)))
    return picked[:num_refs]


def sample_causal_positions(length, key_pos, num_refs, stride):
    if key_pos <= 0:
        return [key_pos] * num_refs
    return [max(0, key_pos - offset * stride) for offset in range(num_refs, 0, -1)]


def sample_ref_positions(length, key_pos, num_refs, stride, mode):
    if num_refs <= 0:
        return []
    if mode == "causal":
        return sample_causal_positions(length, key_pos, num_refs, stride)
    return sample_adjacent_positions(length, key_pos, num_refs, stride)


def xyxy_to_xywh(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def set_clip_layout(model, layout, all_keys=False, num_ref_frames=None):
    from ultralytics.nn.modules import Detect_VID

    for module in model.modules():
        if isinstance(module, Detect_VID):
            module.clip_layout = layout
            module.clip_all_keys = bool(all_keys)
            if num_ref_frames is not None:
                module.num_ref_frames = int(num_ref_frames)


def configure_temporal_options(model, args):
    from ultralytics.nn.modules import Detect_VID

    for module in model.modules():
        if isinstance(module, Detect_VID):
            if args.temporal_fusion is not None:
                module.temporal_fusion = args.temporal_fusion
            if args.score_smooth_sigma is not None:
                module.score_smooth_sigma = float(args.score_smooth_sigma)
            if args.score_smooth_cls_gain is not None:
                module.score_smooth_cls_gain = float(args.score_smooth_cls_gain)
            if args.score_smooth_conf_gain is not None:
                module.score_smooth_conf_gain = float(args.score_smooth_conf_gain)
            if args.score_smooth_min_ref_score is not None:
                module.score_smooth_min_ref_score = float(args.score_smooth_min_ref_score)


def load_clip(frame_paths, imgsz, stride, return_all=False):
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)
    tensors = []
    key_img0 = None
    img0s = []
    for i, path in enumerate(frame_paths):
        img0 = cv2.imread(str(path))
        if img0 is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img0s.append(img0)
        if i == 0:
            key_img0 = img0
        img = letterbox(image=img0)
        img = img[..., ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        tensors.append(torch.from_numpy(np.ascontiguousarray(img)))
    if return_all:
        return torch.stack(tensors, 0), img0s
    return torch.stack(tensors, 0), key_img0


def export_window_detections(yolo, frames, args, stride, device):
    from ultralytics.utils import ops

    lines = []
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
            height, width = img0.shape[:2]
            frame_path = window_frames[local_pos]
            index = frame_index(frame_path, start + local_pos + 1)
            for row in det.detach().cpu().numpy():
                left, top, box_width, box_height = xyxy_to_xywh(row[:4], width, height)
                lines.append(
                    f"{index},-1,{left:.2f},{top:.2f},{box_width:.2f},{box_height:.2f},"
                    f"{float(row[4]):.6f},{int(row[5]) + 1},-1,-1\n"
                )
    return lines


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
        frames = image_files(seq_dir)
        if args.all_keys:
            lines = export_window_detections(yolo, frames, args, stride, device)
            mode_label = "window"
        else:
            lines = []
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
                height, width = key_img0.shape[:2]
                index = frame_index(frame_path, key_pos + 1)
                for row in det.detach().cpu().numpy():
                    left, top, box_width, box_height = xyxy_to_xywh(row[:4], width, height)
                    lines.append(
                        f"{index},-1,{left:.2f},{top:.2f},{box_width:.2f},{box_height:.2f},"
                        f"{float(row[4]):.6f},{int(row[5]) + 1},-1,-1\n"
                    )
            mode_label = "clip"

        output = args.out / f"{seq_dir.name}.txt"
        output.write_text("".join(lines), encoding="utf-8")
        print(f"{seq_dir.name}: {len(frames)} frames, {len(lines)} {mode_label} detections -> {output}")


if __name__ == "__main__":
    main()
