# Ultralytics YOLO, AGPL-3.0 license
"""Video clip dataset for YOLOV-style temporal feature aggregation training.

Each `__getitem__` returns a "clip" of (1 key frame + N reference frames) drawn
from the same video sequence. Ground-truth labels are kept only for the key
frame; references are unlabeled context. The collate function flattens the
clip layout to `(B*T, 3, H, W)` so the unmodified Mamba-YOLO backbone can run
without changes, and stashes `clip_layout=(B, T)` in the batch dict.

Random augmentations that would break geometric correspondence between key and
references (mosaic, mixup, affine, flips, copy_paste) are disabled in clip mode.
"""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .dataset import YOLODataset


class VIDClipDataset(YOLODataset):
    """YOLO video clip dataset for VID training.

    Args:
        num_ref_frames: number of reference frames per clip (refs unlabeled).
        clip_stride: temporal stride between sampled refs (in `uniform_local`).
        ref_sample: 'uniform_local' (within +/-W of key) or 'uniform_global'
            (anywhere in the same sequence).
        seq_key: 'parent' (group by parent dir name) or 'stem_prefix' (group
            by leading non-digit prefix in basename).
    """

    def __init__(
        self,
        *args,
        num_ref_frames: int = 4,
        clip_stride: int = 1,
        ref_sample: str = "uniform_local",
        seq_key: str = "parent",
        **kwargs,
    ):
        self._vid_num_ref = max(0, int(num_ref_frames))
        self._vid_stride = max(1, int(clip_stride))
        if ref_sample not in {"uniform_local", "uniform_global"}:
            raise ValueError(f"ref_sample must be uniform_local|uniform_global, got {ref_sample}")
        self._vid_ref_sample = ref_sample
        self._vid_seq_key = seq_key
        super().__init__(*args, **kwargs)
        self._build_seq_index()

    # ----- transforms: disable spatial random aug (preserves clip correspondence) -----
    def build_transforms(self, hyp=None):
        if hyp is not None:
            for k in (
                "mosaic", "mixup", "copy_paste",
                "degrees", "translate", "scale", "shear", "perspective",
                "fliplr", "flipud",
            ):
                if hasattr(hyp, k):
                    setattr(hyp, k, 0.0)
        return super().build_transforms(hyp)

    # ----- sequence indexing -----
    def _seq_name(self, path: str) -> str:
        if self._vid_seq_key == "parent":
            return Path(path).parent.name
        # 'stem_prefix': use the non-digit prefix of the basename
        stem = Path(path).stem
        out = []
        for ch in stem:
            if ch.isdigit():
                break
            out.append(ch)
        return "".join(out) or stem

    def _build_seq_index(self) -> None:
        seqs: dict[str, list[int]] = defaultdict(list)
        for idx, p in enumerate(self.im_files):
            seqs[self._seq_name(p)].append(idx)
        for name, idxs in seqs.items():
            idxs.sort(key=lambda i: Path(self.im_files[i]).stem)
        self.seqs = dict(seqs)
        self.idx2seqpos: dict[int, tuple[str, int]] = {}
        for name, idxs in self.seqs.items():
            for pos, idx in enumerate(idxs):
                self.idx2seqpos[idx] = (name, pos)

    # ----- reference sampling -----
    def _sample_refs(self, idx: int) -> list[int]:
        if self._vid_num_ref == 0:
            return []
        if idx not in self.idx2seqpos:
            return [idx] * self._vid_num_ref
        seq_name, key_pos = self.idx2seqpos[idx]
        seq_idxs = self.seqs[seq_name]
        L = len(seq_idxs)
        N = self._vid_num_ref

        if self._vid_ref_sample == "uniform_global":
            choices_pos = [i for i in range(L) if i != key_pos]
        else:
            W = N * self._vid_stride
            lo, hi = max(0, key_pos - W), min(L, key_pos + W + 1)
            choices_pos = [i for i in range(lo, hi) if i != key_pos]

        if not choices_pos:
            return [idx] * N
        if len(choices_pos) >= N:
            picked = random.sample(choices_pos, N)
        else:
            picked = random.choices(choices_pos, k=N)
        return [seq_idxs[p] for p in picked]

    # ----- per-sample assembly -----
    def __getitem__(self, idx: int):
        # Key frame: full transforms (deterministic since random aug is disabled)
        key_sample = super().__getitem__(idx)

        if self._vid_num_ref == 0:
            # Promote key-only to a 1-frame clip
            key_sample["img"] = key_sample["img"].unsqueeze(0)  # (1, 3, H, W)
            key_sample["clip_T"] = 1
            return key_sample

        ref_idxs = self._sample_refs(idx)
        ref_imgs = []
        for r_idx in ref_idxs:
            r_sample = super().__getitem__(r_idx)
            ref_imgs.append(r_sample["img"])  # (3, H, W) tensor

        # Stack: key first, then refs -> (T, 3, H, W)
        clip = torch.stack([key_sample["img"]] + ref_imgs, dim=0)
        key_sample["img"] = clip
        key_sample["clip_T"] = clip.shape[0]
        return key_sample

    # ----- collate: flatten (B, T, 3, H, W) -> (B*T, 3, H, W) and emit clip_layout -----
    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        T_per_sample = [b.get("clip_T", 1) for b in batch]
        T = T_per_sample[0]
        if any(t != T for t in T_per_sample):
            raise RuntimeError(f"Inconsistent clip lengths in batch: {T_per_sample}")
        B = len(batch)

        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                # each value is (T, 3, H, W); stack to (B, T, 3, H, W); flatten to (B*T, 3, H, W)
                stacked = torch.stack(value, 0)
                new_batch[k] = stacked.view(B * T, *stacked.shape[2:])
            elif k == "clip_T":
                continue  # consumed
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                new_batch[k] = torch.cat(value, 0)
            else:
                new_batch[k] = list(value)

        # batch_idx: per-sample tensor of length n_objects[b]; offset by sample index
        # NOTE: indexing is into KEY-frame samples (length B), NOT the flattened B*T —
        # the loss takes preds of shape (B, no, H, W) which is what Detect_VID emits.
        bi = list(new_batch.get("batch_idx", []))
        for i in range(len(bi)):
            bi[i] = bi[i] + i
        new_batch["batch_idx"] = torch.cat(bi, 0) if bi else torch.zeros(0)

        # Stash clip layout for the trainer / Detect_VID head
        new_batch["clip_layout"] = torch.tensor([B, T], dtype=torch.long)
        return new_batch
