"""VIDClipDataset sampling tests.

Verifies:
- Reference frames are sampled from the same sequence as the key frame
- Reference sampling honours the local temporal window
- The collate_fn flattens (B, T, 3, H, W) -> (B*T, 3, H, W) and emits clip_layout

These tests instantiate the dataset against a small synthetic on-disk
fixture rather than mocking the parent class, so they exercise the real
sequence-discovery and sampling code paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.fixture
def fake_visdrone_root(tmp_path: Path):
    """Create a minimal VisDrone-VID-shaped directory with two sequences."""
    root = tmp_path / "VisDrone-VID"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    sequences = {"uav0000001_seq": 12, "uav0000002_seq": 8}
    for seq, n in sequences.items():
        (img_dir / seq).mkdir(parents=True)
        (lbl_dir / seq).mkdir(parents=True)
        for f in range(1, n + 1):
            stem = f"{f:07d}"
            # tiny RGB image (32x32) saved as jpg
            from PIL import Image
            Image.fromarray(np.full((32, 32, 3), f, dtype=np.uint8)).save(
                img_dir / seq / f"{stem}.jpg")
            # one synthetic label per frame
            (lbl_dir / seq / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root, sequences


def _build_dataset(root: Path, num_ref_frames: int = 4, ref_sample: str = "uniform_local"):
    from types import SimpleNamespace
    from ultralytics.data.vid_dataset import VIDClipDataset

    hyp = SimpleNamespace(
        # image transforms
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
        mask_ratio=4, overlap_mask=True,
    )
    return VIDClipDataset(
        img_path=str(root / "images" / "train"),
        imgsz=32,
        batch_size=2,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=8,
        pad=0.0,
        prefix="test: ",
        task="detect",
        classes=None,
        data={"task": "vid", "names": {0: "x"}, "channels": 3},
        fraction=1.0,
        num_ref_frames=num_ref_frames,
        clip_stride=1,
        ref_sample=ref_sample,
    )


def test_seq_index_groups_by_parent_dir(fake_visdrone_root):
    root, seqs = fake_visdrone_root
    ds = _build_dataset(root)
    assert set(ds.seqs.keys()) == set(seqs.keys())
    for name, n in seqs.items():
        assert len(ds.seqs[name]) == n


def test_ref_sampling_stays_in_same_sequence(fake_visdrone_root):
    root, seqs = fake_visdrone_root
    ds = _build_dataset(root, num_ref_frames=4, ref_sample="uniform_local")
    for idx in range(len(ds)):
        seq_name, _ = ds.idx2seqpos[idx]
        ref_idxs = ds._sample_refs(idx)
        for r in ref_idxs:
            assert ds.idx2seqpos[r][0] == seq_name, (
                f"ref leaked across sequences: key={seq_name}, ref={ds.idx2seqpos[r][0]}")


def test_ref_sampling_local_window_bounds(fake_visdrone_root):
    root, seqs = fake_visdrone_root
    N, S = 3, 1
    ds = _build_dataset(root, num_ref_frames=N, ref_sample="uniform_local")
    ds._vid_stride = S
    for idx in range(len(ds)):
        seq_name, key_pos = ds.idx2seqpos[idx]
        seq_idxs = ds.seqs[seq_name]
        for r in ds._sample_refs(idx):
            r_pos = seq_idxs.index(r)
            assert abs(r_pos - key_pos) <= N * S


def test_collate_emits_clip_layout(fake_visdrone_root):
    root, _ = fake_visdrone_root
    ds = _build_dataset(root, num_ref_frames=2)
    sample0 = ds[0]
    sample1 = ds[1]
    batch = ds.collate_fn([sample0, sample1])
    assert "clip_layout" in batch
    B, T = batch["clip_layout"].view(-1).tolist()
    assert (B, T) == (2, 3)
    # img is flattened (B*T, 3, H, W)
    assert batch["img"].shape == (B * T, 3, 32, 32)


def test_zero_refs_makes_t_equal_one(fake_visdrone_root):
    root, _ = fake_visdrone_root
    ds = _build_dataset(root, num_ref_frames=0)
    sample0 = ds[0]
    batch = ds.collate_fn([sample0])
    B, T = batch["clip_layout"].view(-1).tolist()
    assert (B, T) == (1, 1)
    assert batch["img"].shape == (1, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
