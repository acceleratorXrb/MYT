#!/usr/bin/env python3
"""Export VisDrone-VID detections and run the official evaluation toolkit."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained checkpoint, usually best.pt.")
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="Official VisDrone-VID split root containing annotations/ and sequences/.",
    )
    parser.add_argument(
        "--toolkit",
        type=Path,
        default=Path("third_party/VisDrone2018-VID-toolkit"),
        help="Path to the official VisDrone2018-VID-toolkit checkout.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory for exported per-sequence txt files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for official metrics outputs. Defaults to <results>/official_eval.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence threshold for official AP evaluation.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--engine",
        choices=("auto", "matlab", "octave"),
        default="auto",
        help="MATLAB-compatible engine used to execute the official toolkit.",
    )
    parser.add_argument("--keep-runner", action="store_true", help="Keep the generated MATLAB runner file.")
    return parser.parse_args()


def run(cmd):
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    export_script = root / "export_visdrone_vid_results.py"
    eval_script = root / "eval_visdrone_vid_official.py"

    run(
        [
            sys.executable,
            str(export_script),
            "--weights",
            str(args.weights),
            "--source",
            str(args.official_root),
            "--out",
            str(args.results),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            str(args.device),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
        ]
    )

    cmd = [
        sys.executable,
        str(eval_script),
        "--toolkit",
        str(args.toolkit),
        "--official-root",
        str(args.official_root),
        "--results",
        str(args.results),
        "--engine",
        str(args.engine),
    ]
    if args.out:
        cmd.extend(["--out", str(args.out)])
    if args.keep_runner:
        cmd.append("--keep-runner")
    run(cmd)


if __name__ == "__main__":
    main()
