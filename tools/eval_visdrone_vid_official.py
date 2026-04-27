#!/usr/bin/env python3
"""Run the official VisDrone-VID MATLAB toolkit and collect its metrics.

This wrapper intentionally does not reimplement VisDrone metrics. It delegates
AP/AR computation to the official VisDrone2018-VID-toolkit so local numbers stay
aligned with the official evaluation code.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


METRIC_PATTERNS = {
    "AP_50_95_maxDets_500": r"Average Precision .*IoU=0\.50:0\.95.*maxDets=500.*=\s*([0-9.]+)%",
    "AP_50_maxDets_500": r"Average Precision .*IoU=0\.50 \| maxDets=500.*=\s*([0-9.]+)%",
    "AP_75_maxDets_500": r"Average Precision .*IoU=0\.75 \| maxDets=500.*=\s*([0-9.]+)%",
    "AR_50_95_maxDets_1": r"Average Recall .*IoU=0\.50:0\.95.*maxDets=\s*1.*=\s*([0-9.]+)%",
    "AR_50_95_maxDets_10": r"Average Recall .*IoU=0\.50:0\.95.*maxDets=\s*10.*=\s*([0-9.]+)%",
    "AR_50_95_maxDets_100": r"Average Recall .*IoU=0\.50:0\.95.*maxDets=100.*=\s*([0-9.]+)%",
    "AR_50_95_maxDets_500": r"Average Recall .*IoU=0\.50:0\.95.*maxDets=500.*=\s*([0-9.]+)%",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toolkit",
        type=Path,
        default=Path("third_party/VisDrone2018-VID-toolkit"),
        help="Path to the official VisDrone2018-VID-toolkit checkout.",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="Official VisDrone-VID split root containing annotations/ and sequences/.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing one official-format result txt per sequence.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for official_metrics.json/txt. Defaults to <results>/official_eval.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "matlab", "octave"),
        default="auto",
        help="MATLAB-compatible engine used to execute the official toolkit.",
    )
    parser.add_argument("--keep-runner", action="store_true", help="Keep the generated MATLAB runner file.")
    return parser.parse_args()


def resolve(path):
    return path.expanduser().resolve()


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path, description):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def check_result_files(official_root, results):
    expected = sorted(path.name for path in (official_root / "annotations").glob("*.txt"))
    found = {path.name for path in results.glob("*.txt")}
    missing = [name for name in expected if name not in found]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} result txt files in {results}. "
            f"Examples: {preview}{suffix}"
        )


def choose_engine(engine):
    if engine == "auto":
        for candidate in ("matlab", "octave"):
            exe = shutil.which(candidate)
            if exe:
                return candidate, exe
        raise RuntimeError("Neither matlab nor octave was found in PATH.")
    exe = shutil.which(engine)
    if not exe:
        raise RuntimeError(f"{engine} was not found in PATH.")
    return engine, exe


def matlab_string(path):
    return str(path).replace("\\", "\\\\").replace("'", "''")


def build_runner(toolkit, official_root, results):
    return f"""
clear all; close all; warning off all;
addpath('{matlab_string(toolkit)}');
isSeqDisplay = false;
datasetPath = '{matlab_string(official_root)}';
resPath = '{matlab_string(results)}';
gtPath = fullfile(datasetPath, 'annotations');
seqPath = fullfile(datasetPath, 'sequences');
nameSeqs = findSeqList(gtPath);
numSeqs = length(nameSeqs);
[allgt, alldet] = saveAnnoRes(gtPath, resPath, seqPath, numSeqs, nameSeqs);
displaySeq(seqPath, numSeqs, nameSeqs, allgt, alldet, isSeqDisplay);
[AP, AR, AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500] = calcAccuracy(numSeqs, allgt, alldet);
fprintf('Average Precision (AP) @[ IoU=0.50:0.95 | maxDets=500 ] = %.2f%%.\\n', AP_all);
fprintf('Average Precision (AP) @[ IoU=0.50 | maxDets=500 ] = %.2f%%.\\n', AP_50);
fprintf('Average Precision (AP) @[ IoU=0.75 | maxDets=500 ] = %.2f%%.\\n', AP_75);
fprintf('Average Recall (AR) @[ IoU=0.50:0.95 | maxDets= 1 ] = %.2f%%.\\n', AR_1);
fprintf('Average Recall (AR) @[ IoU=0.50:0.95 | maxDets= 10 ] = %.2f%%.\\n', AR_10);
fprintf('Average Recall (AR) @[ IoU=0.50:0.95 | maxDets=100 ] = %.2f%%.\\n', AR_100);
fprintf('Average Recall (AR) @[ IoU=0.50:0.95 | maxDets=500 ] = %.2f%%.\\n', AR_500);
""".strip()


def run_runner(engine, executable, runner):
    if engine == "matlab":
        cmd = [executable, "-batch", f"run('{matlab_string(runner)}')"]
    else:
        cmd = [executable, "--no-gui", "--quiet", "--eval", f"run('{matlab_string(runner)}')"]

    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"Official VisDrone-VID evaluation failed with exit code {completed.returncode}.\n{output}")
    return output


def parse_metrics(output):
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    missing = sorted(set(METRIC_PATTERNS) - set(metrics))
    if missing:
        raise RuntimeError(f"Could not parse official metrics {missing} from toolkit output.\n{output}")
    return metrics


def main():
    args = parse_args()
    toolkit = resolve(args.toolkit)
    official_root = resolve(args.official_root)
    results = resolve(args.results)
    out_dir = resolve(args.out) if args.out else results / "official_eval"

    require_dir(toolkit, "official VisDrone-VID toolkit directory")
    for filename in ("findSeqList.m", "saveAnnoRes.m", "displaySeq.m", "calcAccuracy.m"):
        require_file(toolkit / filename, f"official toolkit file {filename}")
    require_dir(official_root / "annotations", "official annotations directory")
    require_dir(official_root / "sequences", "official sequences directory")
    require_dir(results, "official-format result directory")
    check_result_files(official_root, results)

    engine, executable = choose_engine(args.engine)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="visdrone_vid_eval_") as tmp:
        runner = Path(tmp) / "run_official_visdrone_vid_eval.m"
        runner.write_text(build_runner(toolkit, official_root, results), encoding="utf-8")
        output = run_runner(engine, executable, runner)
        if args.keep_runner:
            shutil.copy2(runner, out_dir / runner.name)

    metrics = parse_metrics(output)
    payload = {
        "metrics_percent": metrics,
        "engine": engine,
        "toolkit": str(toolkit),
        "official_root": str(official_root),
        "results": str(results),
    }
    (out_dir / "official_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "official_metrics.txt").write_text(output, encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved official metrics to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
