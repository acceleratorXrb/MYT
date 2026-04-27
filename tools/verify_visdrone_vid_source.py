#!/usr/bin/env python3
"""Verify project-local VisDrone2019-VID raw archives and converted YOLO splits."""

from pathlib import Path


EXPECTED_COUNTS = {
    "train": 24201,
    "val": 2846,
    "test": 6635,
}

EXPECTED_ZIPS = (
    "VisDrone2019-VID-train.zip",
    "VisDrone2019-VID-val.zip",
    "VisDrone2019-VID-test-dev.zip",
)


def count_files(path):
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "datasets" / "VisDrone-VID"
    raw_dirs = (data / "raw_zips", data)

    missing = []
    for name in EXPECTED_ZIPS:
        candidates = [d / name for d in raw_dirs]
        found = next((p for p in candidates if p.exists() and p.stat().st_size > 10000), None)
        if found:
            print(f"zip: {name} -> {found}")
        else:
            print(f"zip: {name} -> MISSING")
            missing.append(name)

    bad_counts = []
    for split, expected in EXPECTED_COUNTS.items():
        image_count = count_files(data / "images" / split)
        label_count = count_files(data / "labels" / split)
        print(f"{split}: images={image_count}, labels={label_count}, expected_images={expected}")
        if image_count != expected or label_count != expected:
            bad_counts.append(split)

    if missing or bad_counts:
        raise SystemExit(
            "VisDrone-VID source verification failed. "
            f"missing_zips={missing}, bad_counts={bad_counts}. "
            "Run tools/download_visdrone_vid_zips.py on a networked machine, then rebuild/verify."
        )

    print("VisDrone-VID source verification passed.")


if __name__ == "__main__":
    main()
