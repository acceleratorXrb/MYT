#!/usr/bin/env python3
"""Self-check local VisDrone-VID auxiliary video metrics.

This script builds a tiny synthetic VisDrone-style fixture with known outcomes
and verifies:

1. classification flicker counting on GT tracks;
2. local MOT/ID counting, including ID switches, fragmentation, FP, and FN.

It is intentionally independent of model checkpoints and real datasets. Run it
after changing metric scripts to make sure the local auxiliary metric semantics
have not drifted.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

from eval_visdrone_vid_cls_flicker import evaluate_flicker
from eval_visdrone_vid_mot import evaluate_mot
from visdrone_temporal_stabilize import VidRecord, stabilize_detection_classes, stabilize_track_records


SEQ_NAME = "toy_sequence.txt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-fixture",
        type=Path,
        default=None,
        help="Optional directory where the synthetic GT/pred files are kept for inspection.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full metric payloads.")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create GT, flicker predictions, and MOT predictions.

    GT has two tracks:
    - track 1, class 1, visible on frames 1-4;
    - track 2, class 2, visible on frames 1-2.

    Flicker predictions are simple 7-field detection files:
    frame,score,category,left,top,width,height

    MOT predictions are official 10-field tracking files:
    frame,id,left,top,width,height,score,category,truncation,occlusion
    """
    gt_dir = root / "gt"
    flicker_pred_dir = root / "pred_flicker"
    mot_pred_dir = root / "pred_mot"

    gt = """
1,1,0,0,10,10,1,1,0,0
1,2,100,100,10,10,1,2,0,0
2,1,0,0,10,10,1,1,0,0
2,2,100,100,10,10,1,2,0,0
3,1,0,0,10,10,1,1,0,0
4,1,0,0,10,10,1,1,0,0
"""
    write_text(gt_dir / SEQ_NAME, gt)

    # Expected flicker:
    # track 1 predicted classes: [1, 1, 2, None] -> 2 kept pairs, 1 change
    # track 2 predicted classes: [2, 3]          -> 1 kept pair, 1 change
    # total_pairs=3, total_changes=2, dropped_no_match=1
    flicker_pred = """
1,0.90,1,0,0,10,10
1,0.90,2,100,100,10,10
2,0.90,1,0,0,10,10
2,0.90,3,100,100,10,10
3,0.90,2,0,0,10,10
"""
    write_text(flicker_pred_dir / SEQ_NAME, flicker_pred)

    # Expected MOT:
    # f1: two correct matches and one extra FP.
    # f2: track 1 stays with pred id 10, track 2 changes 20 -> 21 (ID switch).
    # f3: track 1 is missed and there is one extra FP.
    # f4: track 1 reappears with id 11 after a miss (fragment + ID switch).
    mot_pred = """
1,10,0,0,10,10,0.90,1,-1,-1
1,20,100,100,10,10,0.90,2,-1,-1
1,99,200,200,10,10,0.30,1,-1,-1
2,10,0,0,10,10,0.90,1,-1,-1
2,21,100,100,10,10,0.90,2,-1,-1
3,30,200,200,10,10,0.30,1,-1,-1
4,11,0,0,10,10,0.90,1,-1,-1
"""
    write_text(mot_pred_dir / SEQ_NAME, mot_pred)

    return gt_dir, flicker_pred_dir, mot_pred_dir


def assert_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected:.12g}, got {actual:.12g}")


def validate_fixture(root: Path, verbose: bool = False) -> None:
    gt_dir, flicker_pred_dir, mot_pred_dir = build_fixture(root)

    flicker = evaluate_flicker(gt_dir, flicker_pred_dir, iou_thr=0.5)
    if verbose:
        print("[validate] flicker")
        print(json.dumps(flicker, indent=2))
    assert_equal("flicker.total_pairs", flicker["total_pairs"], 3)
    assert_equal("flicker.total_changes", flicker["total_changes"], 2)
    assert_equal("flicker.tracks_evaluated", flicker["tracks_evaluated"], 2)
    assert_equal("flicker.frames_dropped_no_match", flicker["frames_dropped_no_match"], 1)
    assert_close("flicker.micro_flicker", flicker["micro_flicker"], 2 / 3)
    assert_close("flicker.macro_flicker", flicker["macro_flicker"], 2 / 3)

    mot = evaluate_mot(gt_dir, mot_pred_dir, iou_threshold=0.5)
    if verbose:
        print("[validate] mot")
        print(json.dumps(mot, indent=2))
    assert_equal("mot.GT", mot["GT"], 6)
    assert_equal("mot.Pred", mot["Pred"], 7)
    assert_equal("mot.IDTP", mot["IDTP"], 3)
    assert_equal("mot.IDFP", mot["IDFP"], 4)
    assert_equal("mot.IDFN", mot["IDFN"], 3)
    assert_equal("mot.ID Switches", mot["ID Switches"], 2)
    assert_equal("mot.Frag", mot["Frag"], 1)
    assert_close("mot.IDF1", mot["IDF1"], 6 / 13)
    assert_close("mot.IDP", mot["IDP"], 3 / 7)
    assert_close("mot.IDR", mot["IDR"], 3 / 6)

    det_records = [
        VidRecord(1, -1, 0, 0, 10, 10, 0.40, 1),
        VidRecord(2, -1, 0, 0, 10, 10, 0.35, 2),
        VidRecord(3, -1, 0, 0, 10, 10, 0.45, 1),
    ]
    det_stable, det_stats = stabilize_detection_classes(det_records)
    assert_equal("stabilizer.det_class_changes", det_stats["det_class_changes"], 1)
    assert_equal("stabilizer.det_categories", [r.category for r in det_stable], [1, 1, 1])

    track_records = [
        VidRecord(1, 10, 0, 0, 10, 10, 0.80, 1),
        VidRecord(2, 10, 0, 0, 10, 10, 0.80, 1),
        VidRecord(4, 11, 0, 0, 10, 10, 0.80, 1),
    ]
    track_stable, track_stats = stabilize_track_records(track_records)
    assert_equal("stabilizer.track_fragment_links", track_stats["track_fragment_links"], 1)
    assert_equal("stabilizer.track_gap_fills", track_stats["track_gap_fills"], 1)
    assert_equal("stabilizer.track_frames", [r.frame for r in track_stable], [1, 2, 4, 3])
    assert_equal("stabilizer.track_ids", sorted({r.track_id for r in track_stable}), [10])


def main() -> None:
    args = parse_args()
    if args.keep_fixture:
        args.keep_fixture.mkdir(parents=True, exist_ok=True)
        validate_fixture(args.keep_fixture, verbose=args.verbose)
        print(f"[validate] OK. Fixture kept at {args.keep_fixture}")
        return

    with tempfile.TemporaryDirectory(prefix="visdrone_metric_validate_") as tmp:
        validate_fixture(Path(tmp), verbose=args.verbose)
    print("[validate] OK. Local flicker and MOT/ID metric checks passed.")


if __name__ == "__main__":
    main()
