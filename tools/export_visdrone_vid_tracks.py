#!/usr/bin/env python3
"""Export YOLO tracking predictions to VisDrone-VID official txt format."""

import argparse
import re
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained checkpoint.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Official VisDrone split root or sequence root containing sequences/.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-sequence tracking txt files.")
    parser.add_argument("--tracker", default="ultralytics/cfg/trackers/bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
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


def xyxy_to_xywh(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def format_track_line(frame_id, track_id, xyxy, score, cls, width, height):
    left, top, box_width, box_height = xyxy_to_xywh(xyxy, width, height)
    return (
        f"{int(frame_id)},{int(track_id)},{left:.2f},{top:.2f},{box_width:.2f},{box_height:.2f},"
        f"{float(score):.6f},{int(cls) + 1},-1,-1\n"
    )


def reset_model_trackers(model):
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) if predictor is not None else []:
        tracker.reset()


def export_tracks(weights, source, out, tracker, imgsz=640, device="0", conf=0.1, iou=0.7):
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    seq_root = resolve_sequence_root(source)
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found in {seq_root}")

    for seq_dir in seq_dirs:
        reset_model_trackers(model)
        frames = image_files(seq_dir)
        lines = []
        for fallback, frame_path in enumerate(frames, start=1):
            results = model.track(
                source=str(frame_path),
                persist=True,
                tracker=str(tracker),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                save=False,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            height, width = result.orig_shape
            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or boxes.id is None:
                continue
            frame_id = frame_index(frame_path, fallback)
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            order = confs.argsort()[::-1]
            for row_index in order:
                lines.append(
                    format_track_line(
                        frame_id,
                        track_ids[row_index],
                        xyxy[row_index],
                        confs[row_index],
                        classes[row_index],
                        width,
                        height,
                    )
                )
        output = out / f"{seq_dir.name}.txt"
        output.write_text("".join(lines), encoding="utf-8")
        print(f"{seq_dir.name}: {len(frames)} frames, {len(lines)} tracks -> {output}")


def main():
    args = parse_args()
    export_tracks(args.weights, args.source, args.out, args.tracker, args.imgsz, args.device, args.conf, args.iou)


if __name__ == "__main__":
    main()
