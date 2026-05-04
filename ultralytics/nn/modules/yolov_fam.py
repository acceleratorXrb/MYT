# Ultralytics YOLO, AGPL-3.0 license
"""YOLOV-style temporal Feature Aggregation Module (FAM).

Aggregates classification features across reference frames into the key frame
via cosine-similarity attention over top-K objectness-filtered tokens. Reduces
per-track classification flicker on video streams.

Reference: YOLOV (AAAI'23), https://github.com/YuHengsss/YOLOV
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAggregationModule(nn.Module):
    """Cross-frame classification feature aggregation.

    Operates on the second-stage cls feature `(B, C, H, W)` taken from the
    Detect head's `cv3_pre` (the two Conv layers before the final 1x1 cls
    projection). Refs are batched as `(B, R, C, H, W)`.
    """

    def __init__(
        self,
        cls_channels: int,
        num_ref_frames: int = 4,
        topk: int = 750,
        conf_thr: float = 0.001,
        use_qkv: bool = True,
    ):
        super().__init__()
        self.cls_channels = cls_channels
        self.num_ref_frames = num_ref_frames
        self.topk = topk
        self.conf_thr = conf_thr
        self.use_qkv = use_qkv

        # Residual gate. Init to 0 so first forward = identity (cold-start safe).
        self.alpha = nn.Parameter(torch.tensor(0.0))

        if use_qkv:
            self.q = nn.Linear(cls_channels, cls_channels, bias=False)
            self.k = nn.Linear(cls_channels, cls_channels, bias=False)
            self.v = nn.Linear(cls_channels, cls_channels, bias=False)
        self.gate = nn.Conv2d(cls_channels * 2, cls_channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _project(self, proj: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
        return proj(x) if proj is not None else x

    def forward(
        self,
        key_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate ref features into the key frame.

        Args:
            key_feat:   (B, C, H, W)         pre-projection cls feature of key
            ref_feat:   (B, R, C, H, W)      pre-projection cls features of refs
            ref_logits: (B, R, nc, H, W)     final cls logits of refs (objectness proxy)

        Returns:
            aggregated key feature, shape (B, C, H, W)
        """
        if ref_feat.numel() == 0 or ref_feat.shape[1] == 0:
            return key_feat

        B, C, H, W = key_feat.shape
        R = ref_feat.shape[1]
        if ref_feat.shape[-2:] != (H, W):
            ref_feat = F.interpolate(
                ref_feat.reshape(B * R, C, *ref_feat.shape[-2:]), size=(H, W), mode="bilinear", align_corners=False
            ).view(B, R, C, H, W)
            ref_logits = F.interpolate(
                ref_logits.reshape(B * R, ref_logits.shape[2], *ref_logits.shape[-2:]),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).view(B, R, ref_logits.shape[2], H, W)
        HW = H * W

        # ---- objectness proxy: spatial-max over class probabilities ----
        ref_score = ref_logits.sigmoid().amax(dim=2)             # (B, R, H, W)
        ref_score_flat = ref_score.reshape(B, R, HW)             # (B, R, HW)

        # ---- balanced top-K selection per ref frame ----
        per_ref_k = max(1, min(HW, self.topk // max(R, 1)))
        topk_vals, topk_spatial_idx = ref_score_flat.topk(per_ref_k, dim=2)  # (B, R, K)
        ref_offsets = torch.arange(R, device=ref_score.device).view(1, R, 1) * HW
        topk_idx = (topk_spatial_idx + ref_offsets).reshape(B, R * per_ref_k)
        topk_vals = topk_vals.reshape(B, R * per_ref_k)

        # mask out tokens below conf_thr (keep the slot but zero its weight later)
        valid_mask = topk_vals > self.conf_thr                   # (B, K)
        valid_any = valid_mask.any(dim=1)                        # (B,)
        if not valid_any.any():
            return key_feat

        # ---- gather ref tokens ----
        ref_tok_full = ref_feat.permute(0, 1, 3, 4, 2).reshape(B, R * HW, C)  # (B, R*HW, C)
        selected_k = topk_idx.shape[1]
        gather_idx = topk_idx.unsqueeze(-1).expand(B, selected_k, C)
        ref_tok = ref_tok_full.gather(1, gather_idx)             # (B, K, C)

        # ---- key tokens ----
        key_tok = key_feat.permute(0, 2, 3, 1).reshape(B, HW, C) # (B, HW, C)

        q = self._project(getattr(self, "q", None), key_tok)
        k = self._project(getattr(self, "k", None), ref_tok)
        v = self._project(getattr(self, "v", None), ref_tok)

        # ---- cosine affinity ----
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(k, dim=-1)
        aff = torch.bmm(q_n, k_n.transpose(1, 2))                # (B, HW, K)

        # mask invalid ref tokens by setting their logit to -inf. Rows with no
        # valid refs are temporarily filled with zeros and restored to key below.
        if not valid_mask.all():
            aff = aff.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
            row_all_invalid = ~valid_any.view(B, 1)              # (B, 1)
            aff = torch.where(
                row_all_invalid.unsqueeze(1).expand_as(aff),
                torch.zeros_like(aff),
                aff,
            )

        w = aff.softmax(dim=-1)
        agg = torch.bmm(w, v)                                    # (B, HW, C)

        agg_2d = agg.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        gate = torch.sigmoid(self.gate(torch.cat((key_feat, agg_2d), dim=1)))
        out = key_feat + self.alpha * gate * (agg_2d - key_feat)
        if not valid_any.all():
            out = torch.where(valid_any.view(B, 1, 1, 1), out, key_feat)
        return out

    def extra_repr(self) -> str:
        return (
            f"C={self.cls_channels}, R={self.num_ref_frames}, "
            f"topk={self.topk}, conf_thr={self.conf_thr}, qkv={self.use_qkv}"
        )


def set_alpha_warmup(model: nn.Module, target: float) -> None:
    """Set FAM alpha residual gate to a fixed value across all FAMs in a model.

    Use this from the trainer to warm up alpha from 0 -> target over the first
    few epochs.
    """
    for m in model.modules():
        if isinstance(m, FeatureAggregationModule):
            with torch.no_grad():
                m.alpha.fill_(float(target))
