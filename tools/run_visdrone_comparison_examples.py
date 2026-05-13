#!/usr/bin/env python3
"""One-command pipeline for VisDrone baseline-vs-new qualitative examples.

The pipeline runs:
1. baseline detection export;
2. new VID-window detection export with the current YOLOV-style head options;
3. automatic selection of frames where the new model is visually better.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-weights", default="output_dir/visdrone_vid/baseline/weights/best.pt")
    p.add_argument("--new-weights", default="output_dir/visdrone_vid/temporal_adapter_p4p5_yolov_v4/weights/best.pt")
    p.add_argument("--official-root", default="datasets/VisDrone-VID/raw/VisDrone2019-VID-val")
    p.add_argument("--out", default="output_dir/compare_vis")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.10, help="Confidence threshold for exported predictions.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for exported predictions.")
    p.add_argument("--match-iou", type=float, default=0.5, help="GT/pred IoU threshold for selecting examples.")
    p.add_argument("--min-score", type=float, default=0.10, help="Ignore predictions below this score while selecting examples.")
    p.add_argument("--top-k", type=int, default=30, help="Number of selected side-by-side examples.")
    p.add_argument("--frame-stride", type=int, default=1, help="Score every Nth frame while selecting examples.")
    p.add_argument("--max-per-seq", type=int, default=6, help="Maximum selected examples per sequence.")
    p.add_argument("--seq", action="append", default=None, help="Optional sequence name for selector. Can be repeated.")
    p.add_argument("--skip-export", action="store_true", help="Reuse existing baseline_txt/new_txt and only select examples.")
    p.add_argument("--no-draw-gt", action="store_true", help="Do not draw GT boxes on selected visualizations.")

    # Current main model structure defaults. Keep these in sync with CURRENT_MODEL_STRUCTURE.md.
    p.add_argument("--num-ref-frames", type=int, default=15)
    p.add_argument("--clip-stride", type=int, default=1)
    p.add_argument("--ref-sample", default="adjacent", choices=["adjacent", "causal"])
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--temporal-fusion", default="yolov", choices=["fam", "proposal", "yolov", "fam_proposal", "logits", "logits_gated", "none"])
    p.add_argument("--temporal-adapter", default="affinity", choices=["none", "affinity"])
    p.add_argument("--temporal-adapter-time-sigma", type=float, default=4.0)
    p.add_argument("--temporal-adapter-levels", default="p4p5", choices=["all", "p3", "p4", "p5", "p3p4", "p4p5", "none"])
    p.add_argument("--proposal-topk", type=int, default=700)
    p.add_argument("--proposal-spatial-sigma", type=float, default=0.12)
    p.add_argument("--proposal-cls-sim-gain", type=float, default=0.55)
    p.add_argument("--proposal-reg-sim-gain", type=float, default=0.0)
    p.add_argument("--proposal-score-gain", type=float, default=0.0)
    p.add_argument("--proposal-vote-gain", type=float, default=0.50)
    p.add_argument("--proposal-recall-gain", type=float, default=1.25)
    p.add_argument("--proposal-recall-radius", type=int, default=1)
    p.add_argument("--proposal-after-topk", type=int, default=220)
    p.add_argument("--proposal-nms-radius", type=int, default=1)
    p.add_argument("--proposal-time-sigma", type=float, default=4.0)
    p.add_argument("--proposal-loc-gain", type=float, default=0.5)
    return p.parse_args()


def q(cmd: list[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd: list[object], env: dict[str, str]):
    print(f"\n[run] {q(cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    baseline_txt = out / "baseline_txt"
    new_txt = out / "new_txt"
    selected = out / "selected_better_examples"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(env.get("OMP_NUM_THREADS") or "1")

    py = sys.executable
    if not args.skip_export:
        run(
            [
                py,
                repo / "tools" / "export_visdrone_vid_results.py",
                "--weights",
                args.baseline_weights,
                "--source",
                args.official_root,
                "--out",
                baseline_txt,
                "--imgsz",
                args.imgsz,
                "--device",
                args.device,
                "--conf",
                args.conf,
                "--iou",
                args.iou,
            ],
            env,
        )
        run(
            [
                py,
                repo / "tools" / "export_visdrone_vid_clip_results.py",
                "--weights",
                args.new_weights,
                "--source",
                args.official_root,
                "--out",
                new_txt,
                "--imgsz",
                args.imgsz,
                "--device",
                args.device,
                "--conf",
                args.conf,
                "--iou",
                args.iou,
                "--num_ref_frames",
                args.num_ref_frames,
                "--clip_stride",
                args.clip_stride,
                "--ref_sample",
                args.ref_sample,
                "--all_keys",
                "--window_size",
                args.window_size,
                "--temporal_fusion",
                args.temporal_fusion,
                "--temporal_adapter",
                args.temporal_adapter,
                "--temporal_adapter_time_sigma",
                args.temporal_adapter_time_sigma,
                "--temporal_adapter_levels",
                args.temporal_adapter_levels,
                "--proposal_topk",
                args.proposal_topk,
                "--proposal_spatial_sigma",
                args.proposal_spatial_sigma,
                "--proposal_cls_sim_gain",
                args.proposal_cls_sim_gain,
                "--proposal_reg_sim_gain",
                args.proposal_reg_sim_gain,
                "--proposal_score_gain",
                args.proposal_score_gain,
                "--proposal_vote_gain",
                args.proposal_vote_gain,
                "--proposal_recall_gain",
                args.proposal_recall_gain,
                "--proposal_recall_radius",
                args.proposal_recall_radius,
                "--proposal_after_topk",
                args.proposal_after_topk,
                "--proposal_nms_radius",
                args.proposal_nms_radius,
                "--proposal_time_sigma",
                args.proposal_time_sigma,
                "--proposal_loc_gain",
                args.proposal_loc_gain,
            ],
            env,
        )

    select_cmd: list[object] = [
        py,
        repo / "tools" / "select_visdrone_comparison_examples.py",
        "--official-root",
        args.official_root,
        "--baseline-pred",
        baseline_txt,
        "--new-pred",
        new_txt,
        "--out",
        selected,
        "--top-k",
        args.top_k,
        "--iou",
        args.match_iou,
        "--min-score",
        args.min_score,
        "--frame-stride",
        args.frame_stride,
        "--max-per-seq",
        args.max_per_seq,
    ]
    if not args.no_draw_gt:
        select_cmd.append("--draw-gt")
    if args.seq:
        for seq in args.seq:
            select_cmd.extend(["--seq", seq])
    run(select_cmd, env)

    print("\n[done] Qualitative comparison examples saved to:", selected, flush=True)
    print("[done] Summary CSV:", selected / "selected_examples.csv", flush=True)


if __name__ == "__main__":
    main()
