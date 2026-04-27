#!/usr/bin/env python3
"""Export Mamba-YOLO predictions to the official VisDrone-VID result txt format."""

import argparse
import re
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained checkpoint, usually best.pt.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Official VisDrone sequence root or split root containing sequences/, e.g. VisDrone2019-VID-test-dev.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-sequence txt files.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence threshold for official AP evaluation.")
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


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def xyxy_to_xywh(xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def main():
    args = parse_args()
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    seq_root = resolve_sequence_root(args.source)

    model = YOLO(str(args.weights))
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found in {seq_root}")

    for seq_dir in seq_dirs:
        frames = image_files(seq_dir)
        lines = []
        for frame_batch in chunks(frames, args.batch):
            results = model.predict(
                source=[str(p) for p in frame_batch],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=len(frame_batch),
                save=False,
                verbose=False,
            )
            for fallback, (frame_path, result) in enumerate(zip(frame_batch, results), start=1):
                height, width = result.orig_shape
                index = frame_index(frame_path, fallback)
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                order = confs.argsort()[::-1]
                for row_index in order:
                    box = xyxy[row_index]
                    conf = confs[row_index]
                    cls = classes[row_index]
                    left, top, box_width, box_height = xyxy_to_xywh(box, width, height)
                    category = cls + 1
                    lines.append(
                        f"{index},-1,{left:.2f},{top:.2f},{box_width:.2f},{box_height:.2f},"
                        f"{float(conf):.6f},{category},-1,-1\n"
                    )

        output = args.out / f"{seq_dir.name}.txt"
        output.write_text("".join(lines), encoding="utf-8")
        print(f"{seq_dir.name}: {len(frames)} frames, {len(lines)} detections -> {output}")


if __name__ == "__main__":
    main()
