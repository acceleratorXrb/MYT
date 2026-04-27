#!/usr/bin/env python3
"""Convert VisDrone2019-VID sequences to Ultralytics YOLO detection layout."""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


SPLIT_CANDIDATES = {
    "train": ("VisDrone2019-VID-train", "trainset", "train"),
    "val": ("VisDrone2019-VID-val", "valset", "val"),
    "test-dev": ("VisDrone2019-VID-test-dev", "testset-dev", "test-dev"),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# VisDrone category IDs:
# 0 ignored regions, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
# 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
CATEGORY_TO_YOLO = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Root containing raw VisDrone2019-VID split folders.")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset root.")
    parser.add_argument("--yaml", type=Path, default=None, help="Optional output dataset yaml path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test-dev"),
        choices=tuple(SPLIT_CANDIDATES),
        help="Splits to convert.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels and image links/copies.")
    return parser.parse_args()


def write_dataset_yaml(path, dataset_root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "# Auto-generated VisDrone2019-VID dataset config",
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test-dev",
                "",
                "names:",
                "  0: pedestrian",
                "  1: people",
                "  2: bicycle",
                "  3: car",
                "  4: van",
                "  5: truck",
                "  6: tricycle",
                "  7: awning-tricycle",
                "  8: bus",
                "  9: motor",
                "",
            )
        ),
        encoding="utf-8",
    )


def has_yolo_layout(root):
    required = (
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    )
    return all(p.is_dir() for p in required)


def find_split_dir(src, split):
    for name in SPLIT_CANDIDATES[split]:
        candidate = src / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {split} split under {src}. Tried: {SPLIT_CANDIDATES[split]}")


def find_sequence_root(split_dir):
    seq_root = split_dir / "sequences"
    return seq_root if seq_root.is_dir() else split_dir


def image_size(path):
    with Image.open(path) as im:
        return im.size


def convert_box(width, height, left, top, box_width, box_height):
    return (
        (left + box_width / 2) / width,
        (top + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def link_or_copy(src, dst, copy, overwrite):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def read_annotations(annotation_file, frame_count):
    by_frame = defaultdict(list)
    if not annotation_file.exists():
        return by_frame

    with annotation_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 10:
                raise ValueError(f"{annotation_file}:{line_no} has {len(fields)} fields, expected at least 10.")

            frame_index = int(float(fields[0]))
            track_id = int(float(fields[1]))
            left, top, box_width, box_height = (float(fields[i]) for i in range(2, 6))
            score = float(fields[6])
            category = int(float(fields[7]))

            if frame_index < 1 or frame_index > frame_count:
                continue
            if score <= 0 or track_id < 0 or category not in CATEGORY_TO_YOLO or box_width <= 0 or box_height <= 0:
                continue
            by_frame[frame_index].append((track_id, CATEGORY_TO_YOLO[category], left, top, box_width, box_height))
    return by_frame


def convert_split(src, out, split, copy, overwrite):
    split_dir = find_split_dir(src, split)
    seq_root = find_sequence_root(split_dir)
    ann_root = split_dir / "annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {ann_root}")

    sequences = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not sequences:
        raise FileNotFoundError(f"No sequence directories found in {seq_root}")

    image_total = 0
    label_total = 0
    track_records = []
    for seq_dir in sequences:
        frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not frames:
            continue

        annotations = read_annotations(ann_root / f"{seq_dir.name}.txt", len(frames))
        track_lines = []
        for frame_index, frame_path in enumerate(frames, start=1):
            rel_frame = Path(seq_dir.name) / frame_path.name
            out_image = out / "images" / split / rel_frame
            out_label = out / "labels" / split / rel_frame.with_suffix(".txt")

            link_or_copy(frame_path, out_image, copy=copy, overwrite=overwrite)
            out_label.parent.mkdir(parents=True, exist_ok=True)

            width, height = image_size(frame_path)
            lines = []
            for track_id, cls, left, top, box_width, box_height in annotations.get(frame_index, []):
                x, y, w, h = convert_box(width, height, left, top, box_width, box_height)
                x, y, w, h = (min(max(v, 0.0), 1.0) for v in (x, y, w, h))
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                track_lines.append(f"{frame_index},{track_id},{cls},{x:.6f},{y:.6f},{w:.6f},{h:.6f}\n")
                track_records.append(
                    {
                        "sequence_id": seq_dir.name,
                        "frame_id": frame_index,
                        "image": str(Path("images") / split / rel_frame).replace("\\", "/"),
                        "track_id": track_id,
                        "cls": cls,
                        "bbox": [round(float(v), 6) for v in (x, y, w, h)],
                    }
                )

            if overwrite or not out_label.exists():
                out_label.write_text("".join(lines), encoding="utf-8")
            image_total += 1
            label_total += len(lines)

        out_track = out / "tracks" / split / f"{seq_dir.name}.txt"
        out_track.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not out_track.exists():
            out_track.write_text("".join(track_lines), encoding="utf-8")

    manifest = out / "tracks" / f"{split}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not manifest.exists():
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in track_records),
            encoding="utf-8",
        )
    return image_total, label_total


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        images, labels = convert_split(args.src, args.out, split, args.copy, args.overwrite)
        print(f"{split}: {images} images, {labels} boxes")
    if args.yaml:
        write_dataset_yaml(args.yaml, args.out.resolve())
        print(f"Dataset yaml written to: {args.yaml}")
    print(f"YOLO dataset written to: {args.out}")


if __name__ == "__main__":
    main()
