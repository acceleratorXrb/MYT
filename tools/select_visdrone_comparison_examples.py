#!/usr/bin/env python3
"""Select side-by-side examples where the new VID model improves over baseline.

Inputs are VisDrone-VID annotations and two prediction directories in the
repository's exported txt format. The script ranks frames by detection gains and
saves paper-friendly side-by-side visualizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


@dataclass
class Item:
    category: int
    box: tuple[float, float, float, float]
    score: float = 1.0
    track_id: int = -1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-root", type=Path, required=True, help="VisDrone-VID split root with sequences/ and annotations/.")
    p.add_argument("--baseline-pred", type=Path, required=True, help="Baseline prediction txt directory.")
    p.add_argument("--new-pred", type=Path, required=True, help="New model prediction txt directory.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for selected visual examples.")
    p.add_argument("--seq", action="append", default=None, help="Optional sequence name. Can be repeated.")
    p.add_argument("--top-k", type=int, default=20, help="Number of selected frames to save.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for GT/pred matching.")
    p.add_argument("--min-score", type=float, default=0.05, help="Ignore predictions below this score.")
    p.add_argument("--frame-stride", type=int, default=1, help="Score every Nth frame.")
    p.add_argument("--max-per-seq", type=int, default=6, help="Maximum saved frames per sequence.")
    p.add_argument("--draw-gt", action="store_true", help="Also draw GT boxes in blue on both panels.")
    return p.parse_args()


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def frame_index(path: Path, fallback: int) -> int:
    if path.stem.isdigit():
        return int(path.stem)
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else fallback


def read_gt(path: Path) -> dict[int, list[Item]]:
    by_frame: dict[int, list[Item]] = {}
    if not path.exists():
        return by_frame
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) < 8:
            raise ValueError(f"{path}:{line_no} has {len(fields)} fields, expected >=8")
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        left, top, width, height = (float(fields[i]) for i in range(2, 6))
        score = float(fields[6])
        category = int(float(fields[7]))
        if width <= 0 or height <= 0 or category <= 0 or score <= 0 or track_id < 0:
            continue
        by_frame.setdefault(frame, []).append(
            Item(category, (left, top, left + width, top + height), score=score, track_id=track_id)
        )
    return by_frame


def read_pred(path: Path, min_score: float) -> dict[int, list[Item]]:
    by_frame: dict[int, list[Item]] = {}
    if not path.exists():
        return by_frame
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
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
            raise ValueError(f"{path}:{line_no} has {len(fields)} fields, expected >=7")
        if width <= 0 or height <= 0 or category <= 0 or score < min_score:
            continue
        by_frame.setdefault(frame, []).append(
            Item(category, (left, top, left + width, top + height), score=score)
        )
    return by_frame


def greedy_match(gt_items: list[Item], pred_items: list[Item], iou_thr: float, same_category: bool = True):
    pairs = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if same_category and gt.category != pred.category:
                continue
            v = iou(gt.box, pred.box)
            if v >= iou_thr:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, v))
    return matches


def best_any_category(gt: Item, pred_items: list[Item], iou_thr: float):
    best = None
    for pi, pred in enumerate(pred_items):
        v = iou(gt.box, pred.box)
        if v >= iou_thr and (best is None or v > best[2]):
            best = (pi, pred, v)
    return best


def score_frame(gt_items, base_items, new_items, iou_thr):
    base_matches = greedy_match(gt_items, base_items, iou_thr, same_category=True)
    new_matches = greedy_match(gt_items, new_items, iou_thr, same_category=True)
    base_gt = {gi for gi, _, _ in base_matches}
    new_gt = {gi for gi, _, _ in new_matches}
    base_pred = {pi for _, pi, _ in base_matches}
    new_pred = {pi for _, pi, _ in new_matches}

    gained_gt = sorted(new_gt - base_gt)
    lost_gt = sorted(base_gt - new_gt)
    base_fp = len(base_items) - len(base_pred)
    new_fp = len(new_items) - len(new_pred)

    class_fix = []
    for gi, gt in enumerate(gt_items):
        if gi in new_gt:
            old_any = best_any_category(gt, base_items, iou_thr)
            if old_any is not None and old_any[1].category != gt.category:
                class_fix.append(gi)

    score = (
        5.0 * len(gained_gt)
        - 4.0 * len(lost_gt)
        + 2.0 * max(0, base_fp - new_fp)
        - 1.0 * max(0, new_fp - base_fp)
        + 4.0 * len(class_fix)
    )
    return {
        "score": score,
        "gt": len(gt_items),
        "base_tp": len(base_matches),
        "new_tp": len(new_matches),
        "base_fp": base_fp,
        "new_fp": new_fp,
        "gained_gt": gained_gt,
        "lost_gt": lost_gt,
        "class_fix": class_fix,
        "base_matches": base_matches,
        "new_matches": new_matches,
    }


def try_font(size, bold=False):
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for c in candidates:
        p = Path(c)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


FONT = try_font(16)
FONT_BOLD = try_font(18, bold=True)


def draw_label(draw, xy, text, fill):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=FONT)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=fill)
    draw.text((x, y), text, fill=(0, 0, 0), font=FONT)


def draw_items(img: Image.Image, items: list[Item], color, title: str, gt_items=None, highlight_gt=None, draw_gt=False):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 42], fill=(0, 0, 0))
    draw.text((12, 11), title, fill=(255, 255, 255), font=FONT_BOLD)
    if draw_gt and gt_items:
        for gi, gt in enumerate(gt_items):
            x1, y1, x2, y2 = gt.box
            width = 4 if highlight_gt and gi in highlight_gt else 2
            draw.rectangle([x1, y1, x2, y2], outline=(90, 170, 255), width=width)
            draw_label(draw, (x1, max(42, y1 - 18)), f"GT {NAMES.get(gt.category, gt.category)}", (90, 170, 255))
    for pred in items:
        x1, y1, x2, y2 = pred.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{NAMES.get(pred.category, pred.category)} {pred.score:.2f}"
        draw_label(draw, (x1, max(42, y1 - 18)), label, color)
    return img


def image_files(seq_dir: Path):
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main():
    args = parse_args()
    seq_root = args.official_root / "sequences"
    gt_root = args.official_root / "annotations"
    seq_dirs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if args.seq:
        wanted = set(args.seq)
        seq_dirs = [p for p in seq_dirs if p.name in wanted]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found under {seq_root}")

    candidates = []
    per_seq_count: dict[str, int] = {}
    stride = max(1, int(args.frame_stride))
    for seq_dir in seq_dirs:
        seq = seq_dir.name
        gt = read_gt(gt_root / f"{seq}.txt")
        base = read_pred(args.baseline_pred / f"{seq}.txt", args.min_score)
        new = read_pred(args.new_pred / f"{seq}.txt", args.min_score)
        frames = image_files(seq_dir)
        for fallback, frame_path in enumerate(frames, start=1):
            frame = frame_index(frame_path, fallback)
            if frame % stride != 0:
                continue
            gt_items = gt.get(frame, [])
            if not gt_items:
                continue
            info = score_frame(gt_items, base.get(frame, []), new.get(frame, []), args.iou)
            if info["score"] <= 0:
                continue
            info.update({"seq": seq, "frame": frame, "image": frame_path})
            candidates.append(info)

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["new_tp"] - x["base_tp"],
            x["base_fp"] - x["new_fp"],
            -x["frame"],
        ),
        reverse=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    selected = []
    for info in candidates:
        seq = info["seq"]
        if per_seq_count.get(seq, 0) >= args.max_per_seq:
            continue
        selected.append(info)
        per_seq_count[seq] = per_seq_count.get(seq, 0) + 1
        if len(selected) >= args.top_k:
            break

    rows = []
    for rank, info in enumerate(selected, start=1):
        seq, frame = info["seq"], info["frame"]
        img = Image.open(info["image"]).convert("RGB")
        gt_items = read_gt(gt_root / f"{seq}.txt").get(frame, [])
        base_items = read_pred(args.baseline_pred / f"{seq}.txt", args.min_score).get(frame, [])
        new_items = read_pred(args.new_pred / f"{seq}.txt", args.min_score).get(frame, [])
        highlight = set(info["gained_gt"]) | set(info["class_fix"])

        left = draw_items(
            img,
            base_items,
            (255, 210, 60),
            f"Baseline | TP {info['base_tp']} FP {info['base_fp']}",
            gt_items,
            highlight,
            args.draw_gt,
        )
        right = draw_items(
            img,
            new_items,
            (90, 220, 120),
            f"Ours | TP {info['new_tp']} FP {info['new_fp']} | gain {info['score']:.1f}",
            gt_items,
            highlight,
            args.draw_gt,
        )
        canvas = Image.new("RGB", (img.width * 2, img.height + 58), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (img.width, 0))
        draw = ImageDraw.Draw(canvas)
        reason = (
            f"{seq} frame {frame} | GT {info['gt']} | "
            f"TP +{info['new_tp'] - info['base_tp']} | "
            f"FP {info['base_fp']} -> {info['new_fp']} | "
            f"class_fix {len(info['class_fix'])}"
        )
        draw.text((14, img.height + 18), reason, fill=(30, 40, 48), font=FONT_BOLD)
        filename = f"{rank:02d}_{seq}_{frame:07d}_score_{info['score']:.1f}.jpg"
        canvas.save(args.out / filename, quality=95)

        row = {
            "rank": rank,
            "file": filename,
            "seq": seq,
            "frame": frame,
            "score": round(float(info["score"]), 3),
            "gt": info["gt"],
            "base_tp": info["base_tp"],
            "new_tp": info["new_tp"],
            "base_fp": info["base_fp"],
            "new_fp": info["new_fp"],
            "gained_gt": len(info["gained_gt"]),
            "lost_gt": len(info["lost_gt"]),
            "class_fix": len(info["class_fix"]),
        }
        rows.append(row)

    with (args.out / "selected_examples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "selected_examples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Selected {len(rows)} examples -> {args.out}")


if __name__ == "__main__":
    main()
