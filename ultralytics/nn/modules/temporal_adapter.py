# Ultralytics YOLO, AGPL-3.0 license
"""Temporal feature adapters for VID models.

These modules are inserted after the Mamba-YOLO neck and before Detect_VID.
The official Mamba-YOLO backbone/neck stay unchanged; the adapter only refines
the P3/P4/P5 feature maps when a VID clip layout is available.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalScaleAdapter(nn.Module):
    """Per-scale temporal feature aggregation over a video window.

    The block is inspired by FGFA/SELSA/YOLOV-style video detection: each frame
    keeps its own feature map, but each spatial location can borrow feature
    evidence from nearby frames in the same window. Motion in VisDrone adjacent
    frames is usually local, so the value branch includes a small depthwise
    spatial context before temporal attention.
    """

    def __init__(self, channels: int, attn_ratio: float = 0.5, spatial_kernel: int = 3):
        super().__init__()
        self.channels = channels
        hidden = max(16, int(channels * attn_ratio))
        hidden = min(hidden, channels)
        pad = spatial_kernel // 2

        self.q = nn.Conv2d(channels, hidden, 1, bias=False)
        self.k = nn.Conv2d(channels, hidden, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_context = nn.Sequential(
            nn.Conv2d(channels, channels, spatial_kernel, padding=pad, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        layout: tuple[int, int] | None,
        frame_mask: torch.Tensor | None,
        time_sigma: float,
    ) -> torch.Tensor:
        if layout is None:
            return x
        B, T = layout
        if T <= 1 or x.shape[0] != B * T:
            return x

        N, C, H, W = x.shape
        q = self.q(x).view(B, T, -1, H, W)
        k = self.k(x).view(B, T, -1, H, W)
        v = (self.v(x) + self.spatial_context(x)).view(B, T, C, H, W)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        attn = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=3) / math.sqrt(max(q.shape[2], 1))

        if frame_mask is not None:
            mask = frame_mask.to(device=x.device, dtype=torch.bool)
            attn = attn.masked_fill(~mask.view(1, T, T, 1, 1), float("-inf"))

        if time_sigma > 0:
            idx = torch.arange(T, device=x.device, dtype=x.dtype)
            dist2 = (idx.view(T, 1) - idx.view(1, T)).pow(2)
            attn = attn - dist2.view(1, T, T, 1, 1) / (2.0 * max(time_sigma**2, 1e-6))

        row_valid = torch.isfinite(attn).any(dim=2, keepdim=True)
        attn = torch.where(row_valid, attn, torch.zeros_like(attn))
        weights = attn.softmax(dim=2)
        agg = (weights.unsqueeze(3) * v.unsqueeze(1)).sum(dim=2).reshape(N, C, H, W)
        agg = self.out(agg)

        gate = torch.sigmoid(self.gate(torch.cat((x, agg), dim=1)))
        return x + self.alpha.to(dtype=x.dtype, device=x.device) * gate * (agg - x)


class TemporalFeatureAdapter(nn.Module):
    """Multi-scale VID feature adapter placed between neck and Detect_VID."""

    def __init__(
        self,
        ch: list[int],
        num_ref_frames: int = 15,
        time_sigma: float = 4.0,
        attn_ratio: float = 0.5,
        spatial_kernel: int = 3,
        enabled: bool = True,
    ):
        super().__init__()
        self.ch = list(ch)
        self.num_ref_frames = num_ref_frames
        self.time_sigma = time_sigma
        self.enabled = enabled
        self.debug_temporal_adapter = False
        self._debug_printed = 0
        self.clip_layout: tuple[int, int] | None = None
        self.blocks = nn.ModuleList(TemporalScaleAdapter(c, attn_ratio, spatial_kernel) for c in ch)

    def _window_ref_indices(self, key_pos: int, T: int, num_refs: int) -> list[int]:
        if T <= 1 or num_refs <= 0:
            return []
        left_count = num_refs // 2
        right_count = num_refs - left_count
        desired = list(range(key_pos - left_count, key_pos)) + list(range(key_pos + 1, key_pos + right_count + 1))
        picked = [p for p in desired if 0 <= p < T and p != key_pos]
        if len(picked) < num_refs:
            candidates = [p for p in range(T) if p != key_pos and p not in picked]
            candidates.sort(key=lambda p: (abs(p - key_pos), p))
            picked.extend(candidates[: num_refs - len(picked)])
        return picked[:num_refs]

    def _frame_mask(self, T: int, device: torch.device) -> torch.Tensor:
        refs = min(int(getattr(self, "num_ref_frames", T - 1) or 0), max(T - 1, 0))
        mask = torch.eye(T, dtype=torch.bool, device=device)
        for key_pos in range(T):
            ref_idx = self._window_ref_indices(key_pos, T, refs)
            if ref_idx:
                mask[key_pos, torch.tensor(ref_idx, device=device, dtype=torch.long)] = True
        return mask

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.enabled or self.clip_layout is None:
            return xs
        if not isinstance(xs, (list, tuple)):
            return xs
        B, T = self.clip_layout
        if T <= 1:
            return list(xs)

        mask = self._frame_mask(T, xs[0].device)
        out = []
        for i, x in enumerate(xs):
            if i < len(self.blocks):
                out.append(self.blocks[i](x, (B, T), mask, float(getattr(self, "time_sigma", 0.0) or 0.0)))
            else:
                out.append(x)

        if bool(getattr(self, "debug_temporal_adapter", False)) and self._debug_printed < 4:
            shapes = [tuple(x.shape) for x in xs]
            refs = [(i, self._window_ref_indices(i, T, min(self.num_ref_frames, max(T - 1, 0)))) for i in range(min(T, 5))]
            alpha = [float(b.alpha.detach().cpu()) for b in self.blocks[: len(xs)]]
            print(
                "[debug-temporal-adapter] "
                f"B={B} T={T} enabled={self.enabled} num_ref_frames={self.num_ref_frames} "
                f"time_sigma={self.time_sigma} shapes={shapes} ref_debug={refs} alpha={alpha}",
                flush=True,
            )
            self._debug_printed += 1
        return out

    def extra_repr(self) -> str:
        return (
            f"ch={self.ch}, enabled={self.enabled}, num_ref_frames={self.num_ref_frames}, "
            f"time_sigma={self.time_sigma}"
        )


