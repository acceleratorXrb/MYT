#!/usr/bin/env python3
"""Download VisDrone2019-VID official split archives into the project."""

import argparse
from pathlib import Path

import requests


HF_MIRROR = "https://hf-mirror.com/datasets/AndriiDemk/visDrone_copy/resolve/main"
GDRIVE_BASE = "https://drive.usercontent.google.com/download"

FILES = {
    "VisDrone2019-VID-train.zip": {
        "hf": f"{HF_MIRROR}/VisDrone2019-VID-train.zip",
        "gd_id": "1NSNapZQHar22OYzQYuXCugA3QlMndzvw",
    },
    "VisDrone2019-VID-val.zip": {
        "hf": f"{HF_MIRROR}/VisDrone2019-VID-val.zip",
        "gd_id": "1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B",
    },
    "VisDrone2019-VID-test-dev.zip": {
        "hf": f"{HF_MIRROR}/VisDrone2019-VID-test-dev.zip",
        "gd_id": "1-BEq--FcjshTF1UwUabby_LHhYj41os5",
    },
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "datasets" / "VisDrone-VID" / "raw_zips")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def download(url, dst, timeout):
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, timeout)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{dst.name}: {done / total * 100:5.1f}%", end="", flush=True)
    print()
    tmp.replace(dst)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, info in FILES.items():
        dst = args.out / name
        if dst.exists() and dst.stat().st_size > 10000:
            print(f"{name}: exists")
            continue
        urls = [info["hf"], f"{GDRIVE_BASE}?id={info['gd_id']}&export=download&confirm=t"]
        for url in urls:
            try:
                print(f"Downloading {name} from {url}")
                download(url, dst, args.timeout)
                break
            except Exception as exc:
                print(f"{name}: failed from {url}: {exc}")
        if not dst.exists():
            raise SystemExit(f"Failed to download {name}")


if __name__ == "__main__":
    main()
