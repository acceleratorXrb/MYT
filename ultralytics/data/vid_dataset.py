# Ultralytics YOLO, AGPL-3.0 license
"""Video clip dataset for YOLOV-style temporal feature aggregation training.

Each `__getitem__` returns a "clip" of (1 key frame + N reference frames) drawn
from the same video sequence. Ground-truth labels are kept for the key frame
and, optionally, for reference frames so Detect_VID can use auxiliary ref loss.
The collate function flattens the
clip layout to `(B*T, 3, H, W)` so the unmodified Mamba-YOLO backbone can run
without changes, and stashes `clip_layout=(B, T)` in the batch dict.

Clip-breaking multi-image augmentations (mosaic, mixup, copy_paste) are disabled
in clip mode. Single-image random augmentations such as affine/perspective,
flip, HSV, and BGR are applied with synchronized random seeds across the key
and reference frames so the clip keeps a shared transform.
"""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
        debug_clip_aug: bool = False,
        **kwargs,
    ):
        self._vid_num_ref = max(0, int(num_ref_frames))
        self._vid_stride = max(1, int(clip_stride))
        self._vid_mosaic = 0.0
        self._vid_mixup = 0.0
        self._debug_clip_aug = bool(debug_clip_aug)
        self._debug_clip_aug_printed = 0
        if ref_sample not in {"uniform_local", "uniform_global"}:
            raise ValueError(f"ref_sample must be uniform_local|uniform_global, got {ref_sample}")
        self._vid_ref_sample = ref_sample
        self._vid_seq_key = seq_key
        super().__init__(*args, **kwargs)
        self._build_seq_index()

    # ----- transforms: disable multi-image aug; sync single-image aug in __getitem__ -----
    def build_transforms(self, hyp=None):
        if hyp is not None:
            self._vid_mosaic = float(getattr(hyp, "mosaic", 0.0) or 0.0)
            self._vid_mixup = float(getattr(hyp, "mixup", 0.0) or 0.0)
            for k in ("mosaic", "mixup", "copy_paste"):
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

    def _get_sync_aug_sample(self, idx: int, seed: int):
        """Load one frame while replaying the same random augment decisions for a clip."""
        if not self.augment:
            return super().__getitem__(idx)

        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            return super().__getitem__(idx)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _build_clip(self, idx: int):
        # Use one seed for the whole clip so single-image random transforms
        # (affine/perspective, flips, HSV, BGR) are shared by key and refs.
        aug_seed = random.randint(0, 2**32 - 1)
        key_sample = self._get_sync_aug_sample(idx, aug_seed)

        if self._vid_num_ref == 0:
            # Promote key-only to a 1-frame clip
            key_sample["img"] = key_sample["img"].unsqueeze(0)  # (1, 3, H, W)
            key_sample["clip_T"] = 1
            key_sample["ref_cls"] = torch.zeros((0, 1), dtype=key_sample["cls"].dtype)
            key_sample["ref_bboxes"] = torch.zeros((0, 4), dtype=key_sample["bboxes"].dtype)
            key_sample["ref_batch_idx"] = torch.zeros((0,), dtype=key_sample["batch_idx"].dtype)
            return key_sample

        ref_idxs = self._sample_refs(idx)
        ref_imgs = []
        ref_cls = []
        ref_bboxes = []
        ref_batch_idx = []
        for ref_pos, r_idx in enumerate(ref_idxs):
            r_sample = self._get_sync_aug_sample(r_idx, aug_seed)
            ref_imgs.append(r_sample["img"])  # (3, H, W) tensor
            if r_sample.get("cls") is not None and len(r_sample["cls"]):
                ref_cls.append(r_sample["cls"])
                ref_bboxes.append(r_sample["bboxes"])
                ref_batch_idx.append(torch.full((len(r_sample["cls"]),), ref_pos, dtype=r_sample["batch_idx"].dtype))

        # Stack: key first, then refs -> (T, 3, H, W)
        clip = torch.stack([key_sample["img"]] + ref_imgs, dim=0)
        key_sample["img"] = clip
        key_sample["clip_T"] = clip.shape[0]
        key_sample["ref_cls"] = (
            torch.cat(ref_cls, 0) if ref_cls else torch.zeros((0, 1), dtype=key_sample["cls"].dtype)
        )
        key_sample["ref_bboxes"] = (
            torch.cat(ref_bboxes, 0) if ref_bboxes else torch.zeros((0, 4), dtype=key_sample["bboxes"].dtype)
        )
        key_sample["ref_batch_idx"] = (
            torch.cat(ref_batch_idx, 0) if ref_batch_idx else torch.zeros((0,), dtype=key_sample["batch_idx"].dtype)
        )
        return key_sample

    @staticmethod
    def _shift_scale_boxes(bboxes: torch.Tensor, x0: int, y0: int, tile_w: int, tile_h: int, out_w: int, out_h: int):
        if bboxes.numel() == 0:
            return bboxes
        out = bboxes.clone()
        out[:, 0] = (bboxes[:, 0] * tile_w + x0) / out_w
        out[:, 1] = (bboxes[:, 1] * tile_h + y0) / out_h
        out[:, 2] = bboxes[:, 2] * tile_w / out_w
        out[:, 3] = bboxes[:, 3] * tile_h / out_h
        return out.clamp_(0.0, 1.0)

    @staticmethod
    def _merge_label_tensors(samples: list[dict], boxes: list[torch.Tensor] | None = None):
        cls = [s["cls"] for s in samples if s.get("cls") is not None and len(s["cls"])]
        merged_cls = torch.cat(cls, 0) if cls else torch.zeros((0, 1), dtype=samples[0]["cls"].dtype)
        if boxes is None:
            boxes = [s["bboxes"] for s in samples if s.get("bboxes") is not None and len(s["bboxes"])]
        else:
            boxes = [b for b in boxes if b is not None and len(b)]
        merged_boxes = torch.cat(boxes, 0) if boxes else torch.zeros((0, 4), dtype=samples[0]["bboxes"].dtype)
        return merged_cls, merged_boxes

    @staticmethod
    def _merge_ref_label_tensors(samples: list[dict], boxes: list[torch.Tensor] | None = None):
        cls = [s["ref_cls"] for s in samples if s.get("ref_cls") is not None and len(s["ref_cls"])]
        merged_cls = torch.cat(cls, 0) if cls else torch.zeros((0, 1), dtype=samples[0]["cls"].dtype)
        if boxes is None:
            boxes = [s["ref_bboxes"] for s in samples if s.get("ref_bboxes") is not None and len(s["ref_bboxes"])]
        else:
            boxes = [b for b in boxes if b is not None and len(b)]
        merged_boxes = torch.cat(boxes, 0) if boxes else torch.zeros((0, 4), dtype=samples[0]["bboxes"].dtype)
        batch_idx = [s["ref_batch_idx"] for s in samples if s.get("ref_batch_idx") is not None and len(s["ref_batch_idx"])]
        merged_batch_idx = torch.cat(batch_idx, 0) if batch_idx else torch.zeros((0,), dtype=samples[0]["batch_idx"].dtype)
        return merged_cls, merged_boxes, merged_batch_idx

    def _apply_clip_mosaic(self, sample: dict):
        """Apply synchronized 2x2 mosaic to whole clips after per-frame transforms."""
        indexes = [random.randint(0, len(self) - 1) for _ in range(3)]
        samples = [sample] + [self._build_clip(i) for i in indexes]
        clip = samples[0]["img"]
        T, C, H, W = clip.shape
        tile_h, tile_w = H // 2, W // 2
        out = torch.full_like(clip, 114)
        boxes = []
        ref_boxes = []
        placements = ((0, 0), (tile_w, 0), (0, tile_h), (tile_w, tile_h))

        for s, (x0, y0) in zip(samples, placements):
            tile = F.interpolate(s["img"].float(), size=(tile_h, tile_w), mode="bilinear", align_corners=False)
            tile = tile.round().clamp_(0, 255).to(dtype=clip.dtype)
            out[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
            boxes.append(self._shift_scale_boxes(s["bboxes"], x0, y0, tile_w, tile_h, W, H))
            ref_boxes.append(self._shift_scale_boxes(s["ref_bboxes"], x0, y0, tile_w, tile_h, W, H))

        sample["img"] = out
        sample["cls"], sample["bboxes"] = self._merge_label_tensors(samples, boxes)
        sample["ref_cls"], sample["ref_bboxes"], sample["ref_batch_idx"] = self._merge_ref_label_tensors(samples, ref_boxes)
        if "batch_idx" in sample:
            sample["batch_idx"] = torch.zeros((len(sample["cls"]),), dtype=sample["batch_idx"].dtype)
        return sample

    def _apply_clip_mixup(self, sample: dict):
        """Apply MixUp to whole clips and concatenate key-frame labels."""
        other = self._build_clip(random.randint(0, len(self) - 1))
        r = float(np.random.beta(32.0, 32.0))
        mixed = sample["img"].float() * r + other["img"].float() * (1.0 - r)
        sample["img"] = mixed.round().clamp_(0, 255).to(dtype=sample["img"].dtype)
        sample["cls"], sample["bboxes"] = self._merge_label_tensors([sample, other])
        sample["ref_cls"], sample["ref_bboxes"], sample["ref_batch_idx"] = self._merge_ref_label_tensors([sample, other])
        if "batch_idx" in sample:
            sample["batch_idx"] = torch.zeros((len(sample["cls"]),), dtype=sample["batch_idx"].dtype)
        return sample

    def _debug_print_clip_aug(self, idx: int, sample: dict, mosaic_applied: bool, mixup_applied: bool):
        if not self._debug_clip_aug or self._debug_clip_aug_printed >= 5:
            return
        img = sample["img"]
        bboxes = sample.get("bboxes")
        labels = int(len(sample.get("cls", [])))
        if bboxes is not None and bboxes.numel():
            box_min = float(bboxes.min().item())
            box_max = float(bboxes.max().item())
        else:
            box_min = box_max = None
        print(
            "[debug-clip-aug] "
            f"idx={idx} augment={self.augment} "
            f"num_ref_frames={self._vid_num_ref} clip_stride={self._vid_stride} ref_sample={self._vid_ref_sample} "
            f"mosaic_p={self._vid_mosaic} mosaic_applied={mosaic_applied} "
            f"mixup_p={self._vid_mixup} mixup_applied={mixup_applied} "
            f"img_shape={tuple(img.shape)} labels={labels} bbox_min={box_min} bbox_max={box_max}",
            flush=True,
        )
        self._debug_clip_aug_printed += 1

    # ----- per-sample assembly -----
    def __getitem__(self, idx: int):
        sample = self._build_clip(idx)
        mosaic_applied = False
        mixup_applied = False
        if self.augment and self._vid_mosaic > 0.0 and random.random() < self._vid_mosaic:
            sample = self._apply_clip_mosaic(sample)
            mosaic_applied = True
        if self.augment and self._vid_mixup > 0.0 and random.random() < self._vid_mixup:
            sample = self._apply_clip_mixup(sample)
            mixup_applied = True
        self._debug_print_clip_aug(idx, sample, mosaic_applied, mixup_applied)
        return sample

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
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "ref_bboxes", "ref_cls"}:
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

        # ref_batch_idx indexes auxiliary predictions over flattened ref frames:
        # image id = sample_index * (T - 1) + ref_position.
        rbi = list(new_batch.get("ref_batch_idx", []))
        if T > 1:
            for i in range(len(rbi)):
                rbi[i] = rbi[i] + i * (T - 1)
        new_batch["ref_batch_idx"] = torch.cat(rbi, 0) if rbi else torch.zeros(0)

        # Stash clip layout for the trainer / Detect_VID head
        new_batch["clip_layout"] = torch.tensor([B, T], dtype=torch.long)
        return new_batch
