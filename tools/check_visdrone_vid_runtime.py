#!/usr/bin/env python3
"""Check local readiness for the Mamba-YOLO-T VisDrone-VID pipeline."""

from pathlib import Path


def count_files(path):
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "datasets" / "VisDrone-VID"
    print(f"project: {root}")
    print(f"dataset: {data}")
    for split in ("train", "val", "test"):
        images = count_files(data / "images" / split)
        labels = count_files(data / "labels" / split)
        print(f"{split}: images={images}, labels={labels}")

    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Mamba-YOLO selective_scan requires a visible CUDA GPU.")

    from ultralytics import YOLO

    model = YOLO(str(root / "ultralytics" / "cfg" / "models" / "mamba-yolo" / "Mamba-YOLO-T-VID.yaml"))
    print(f"model: {model.model.__class__.__name__}")
    print("runtime check passed")


if __name__ == "__main__":
    main()
