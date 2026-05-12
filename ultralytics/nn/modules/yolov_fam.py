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
        spatial_sigma: float = 0.2,
    ):
        super().__init__()
        self.cls_channels = cls_channels
        self.num_ref_frames = num_ref_frames
        self.topk = topk
        self.conf_thr = conf_thr
        self.use_qkv = use_qkv
        self.spatial_sigma = spatial_sigma

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
        ref_y = (topk_spatial_idx // W).to(dtype=key_feat.dtype) / max(H - 1, 1)
        ref_x = (topk_spatial_idx % W).to(dtype=key_feat.dtype) / max(W - 1, 1)
        ref_yx = torch.stack((ref_y, ref_x), dim=-1).reshape(B, R * per_ref_k, 2)

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

        # Adjacent-frame VID motion is usually local. A soft spatial prior keeps
        # small targets from borrowing features from far-away lookalikes while
        # still allowing attention to move within a neighborhood.
        spatial_sigma = float(getattr(self, "spatial_sigma", 0.2) or 0.0)
        if spatial_sigma > 0:
            ys, xs = torch.meshgrid(
                torch.linspace(0, 1, H, device=key_feat.device, dtype=key_feat.dtype),
                torch.linspace(0, 1, W, device=key_feat.device, dtype=key_feat.dtype),
                indexing="ij",
            )
            key_yx = torch.stack((ys.reshape(-1), xs.reshape(-1)), dim=-1)  # (HW, 2)
            dist2 = (key_yx.view(1, HW, 1, 2) - ref_yx.view(B, 1, -1, 2)).pow(2).sum(dim=-1)
            sigma2 = max(spatial_sigma ** 2, 1e-6)
            aff = aff - dist2 / (2.0 * sigma2)

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
            f"topk={self.topk}, conf_thr={self.conf_thr}, "
            f"spatial_sigma={getattr(self, 'spatial_sigma', 0.2)}, qkv={self.use_qkv}"
        )


class ProposalTemporalRefiner(nn.Module):
    """YOLOV-style sparse proposal refinement for VID classification logits.

    Dense FAM updates every grid cell. This module is intentionally sparser:
    it selects top-scoring key/ref locations, lets key proposals attend to
    proposal tokens from adjacent frames, then scatters a small class-logit
    correction back to those key locations. Boxes stay on the current frame.
    """

    def __init__(
        self,
        cls_channels: int,
        nc: int,
        topk: int = 150,
        conf_thr: float = 0.001,
        spatial_sigma: float = 0.05,
        cls_sim_gain: float = 0.5,
        reg_sim_gain: float = 0.25,
        score_gain: float = 0.25,
        vote_gain: float = 0.0,
        recall_gain: float = 0.0,
        recall_radius: int = 1,
        after_topk: int = 0,
        nms_radius: int = 0,
        time_sigma: float = 0.0,
        loc_gain: float = 0.0,
    ):
        super().__init__()
        self.cls_channels = cls_channels
        self.nc = nc
        self.topk = topk
        self.conf_thr = conf_thr
        self.spatial_sigma = spatial_sigma
        self.cls_sim_gain = cls_sim_gain
        self.reg_sim_gain = reg_sim_gain
        self.score_gain = score_gain
        self.vote_gain = vote_gain
        self.recall_gain = recall_gain
        self.recall_radius = recall_radius
        self.after_topk = after_topk
        self.nms_radius = nms_radius
        self.time_sigma = time_sigma
        self.loc_gain = loc_gain

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.q = nn.Linear(cls_channels, cls_channels, bias=False)
        self.k = nn.Linear(cls_channels, cls_channels, bias=False)
        self.v = nn.Linear(cls_channels, cls_channels, bias=False)
        self.gate = nn.Linear(cls_channels * 2, 1)
        self.norm = nn.LayerNorm(cls_channels)
        self.out = nn.Linear(cls_channels, nc)
        # YOLOV aggregates proposal features and classifies the fused
        # [support, key] token. This branch keeps that idea while only adding a
        # positive class-logit correction to preserve current-frame boxes.
        self.concat_norm = nn.LayerNorm(cls_channels * 2)
        self.concat_out = nn.Linear(cls_channels * 2, nc)
        # Lightweight learned geometry bias, analogous to YOLOV's loc embedding.
        self.loc_mlp = nn.Sequential(
            nn.Linear(4, max(16, cls_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, cls_channels // 4), 1),
        )

        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_uniform_(self.concat_out.weight, gain=0.01)
        nn.init.zeros_(self.concat_out.bias)
        nn.init.zeros_(self.loc_mlp[-1].weight)
        nn.init.zeros_(self.loc_mlp[-1].bias)

    @staticmethod
    def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather flattened spatial tokens from `(B, C, H, W)` using `(B, K)` indices."""
        B, C, _, _ = x.shape
        flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        return flat.gather(1, idx.unsqueeze(-1).expand(B, idx.shape[1], C))

    @staticmethod
    def _gather_logits(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather flattened spatial logits from `(B, C, H, W)` using `(B, K)` indices."""
        B, C, _, _ = x.shape
        flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        return flat.gather(1, idx.unsqueeze(-1).expand(B, idx.shape[1], C))

    def _topk_positions(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        score_map = logits.sigmoid().amax(dim=1)  # (B, H, W)
        if int(getattr(self, "nms_radius", 0) or 0) > 0:
            score_map = self._local_nms_scores(score_map.unsqueeze(1)).squeeze(1)
        score = score_map.flatten(1)  # (B, HW)
        k = max(1, min(int(k), score.shape[1]))
        return score.topk(k, dim=1)

    def _candidate_scores(
        self,
        score: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend current-frame scores with local temporal support before top-k.

        This lets weak current-frame locations enter the proposal set when
        neighboring frames repeatedly support the same spatial neighborhood.
        """
        recall_gain = float(getattr(self, "recall_gain", 0.0) or 0.0)
        B, T, H, W = score.shape
        if recall_gain <= 0.0 or T <= 1:
            return score

        if frame_mask is None:
            frame_mask = ~torch.eye(T, device=score.device, dtype=torch.bool)
        else:
            frame_mask = frame_mask.to(device=score.device, dtype=torch.bool)

        support = []
        fallback = score.max(dim=1).values
        for q in range(T):
            refs = torch.where(frame_mask[q])[0]
            support_q = score[:, refs].max(dim=1).values if refs.numel() else fallback
            radius = max(0, int(getattr(self, "recall_radius", 1) or 0))
            if radius > 0:
                k = 2 * radius + 1
                support_q = F.max_pool2d(support_q.unsqueeze(1), kernel_size=k, stride=1, padding=radius).squeeze(1)
            support.append(support_q)
        support_score = torch.stack(support, dim=1)
        return torch.maximum(score, recall_gain * support_score)

    def _local_nms_scores(self, score: torch.Tensor) -> torch.Tensor:
        """Keep only local score maxima before proposal top-k selection."""
        radius = int(getattr(self, "nms_radius", 0) or 0)
        if radius <= 0:
            return score
        B, T, H, W = score.shape
        k = 2 * radius + 1
        pooled = F.max_pool2d(score.reshape(B * T, 1, H, W), kernel_size=k, stride=1, padding=radius)
        keep = score.reshape(B * T, 1, H, W) >= (pooled - 1e-12)
        filtered = score.reshape(B * T, 1, H, W).masked_fill(~keep, -1.0)
        return filtered.view(B, T, H, W)

    def _proposal_k(self, HW: int) -> tuple[int, int]:
        """Return pre-selection K and attention K for YOLOV-style two-stage proposals."""
        pre_k = max(1, min(int(getattr(self, "topk", 150) or 1), HW))
        after_k = int(getattr(self, "after_topk", 0) or 0)
        if after_k <= 0:
            return pre_k, pre_k
        return pre_k, max(1, min(after_k, pre_k))

    def _relative_loc_bias(self, key_yx: torch.Tensor, ref_yx: torch.Tensor) -> torch.Tensor:
        """Learned local geometry bias from normalized proposal centers."""
        gain = float(getattr(self, "loc_gain", 0.0) or 0.0)
        if gain <= 0.0:
            return key_yx.new_zeros((*key_yx.shape[:-1], ref_yx.shape[-2]))
        rel = ref_yx.unsqueeze(-3) - key_yx.unsqueeze(-2)
        geom = torch.cat((rel, rel.abs()), dim=-1)
        return gain * self.loc_mlp(geom).squeeze(-1)

    @staticmethod
    def _gather_window_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather `(B, T, K, C)` tokens from `(B, T, C, H, W)` by spatial indices `(B, T, K)`."""
        B, T, C, _, _ = x.shape
        flat = x.flatten(3).permute(0, 1, 3, 2)  # (B, T, HW, C)
        return flat.gather(2, idx.unsqueeze(-1).expand(B, T, idx.shape[2], C))

    @staticmethod
    def _gather_window_logits(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather `(B, T, K, C)` logits/probs from `(B, T, C, H, W)` by `(B, T, K)` indices."""
        B, T, C, _, _ = x.shape
        flat = x.flatten(3).permute(0, 1, 3, 2)  # (B, T, HW, C)
        return flat.gather(2, idx.unsqueeze(-1).expand(B, T, idx.shape[2], C))

    @staticmethod
    def _grid_yx(idx: torch.Tensor, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        y = (idx // W).to(dtype=dtype) / max(H - 1, 1)
        x = (idx % W).to(dtype=dtype) / max(W - 1, 1)
        return torch.stack((y, x), dim=-1)

    @staticmethod
    def _proposal_centers_window(reg: torch.Tensor | None, idx: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Approximate decoded proposal centers in normalized feature coordinates.

        YOLOv8 predicts distance distributions from each grid point. For temporal
        attention, using the decoded center is closer to proposal-level fusion
        than comparing only raw grid indices.
        """
        dtype = reg.dtype if reg is not None else torch.float32
        if reg is None:
            return ProposalTemporalRefiner._grid_yx(idx, H, W, dtype)
        B, T, Cr, _, _ = reg.shape
        reg_max = max(1, Cr // 4)
        proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
        dist = (reg.view(B, T, 4, reg_max, H, W).softmax(3) * proj.view(1, 1, 1, reg_max, 1, 1)).sum(3)
        dist_tok = ProposalTemporalRefiner._gather_window_logits(dist, idx)  # (B, T, K, 4)
        grid_yx = ProposalTemporalRefiner._grid_yx(idx, H, W, reg.dtype)
        grid_y, grid_x = grid_yx[..., 0], grid_yx[..., 1]
        l, t, r, b = dist_tok.unbind(-1)
        center_x = (grid_x * max(W - 1, 1) + (r - l) * 0.5) / max(W - 1, 1)
        center_y = (grid_y * max(H - 1, 1) + (b - t) * 0.5) / max(H - 1, 1)
        return torch.stack((center_y.clamp(0, 1), center_x.clamp(0, 1)), dim=-1)

    @staticmethod
    def _proposal_centers(reg: torch.Tensor | None, idx: torch.Tensor, H: int, W: int) -> torch.Tensor:
        dtype = reg.dtype if reg is not None else torch.float32
        if reg is None:
            return ProposalTemporalRefiner._grid_yx(idx, H, W, dtype)
        B, Cr, _, _ = reg.shape
        reg_max = max(1, Cr // 4)
        proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
        dist = (reg.view(B, 4, reg_max, H, W).softmax(2) * proj.view(1, 1, reg_max, 1, 1)).sum(2)
        dist_tok = ProposalTemporalRefiner._gather_logits(dist, idx)  # (B, K, 4)
        grid_yx = ProposalTemporalRefiner._grid_yx(idx, H, W, reg.dtype)
        grid_y, grid_x = grid_yx[..., 0], grid_yx[..., 1]
        l, t, r, b = dist_tok.unbind(-1)
        center_x = (grid_x * max(W - 1, 1) + (r - l) * 0.5) / max(W - 1, 1)
        center_y = (grid_y * max(H - 1, 1) + (b - t) * 0.5) / max(H - 1, 1)
        return torch.stack((center_y.clamp(0, 1), center_x.clamp(0, 1)), dim=-1)

    def forward_window(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        reg: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine all key frames in a video window in one proposal attention pass.

        Args:
            feat:       (B, T, C, H, W)
            logits:     (B, T, nc, H, W)
            reg:        optional (B, T, Cr, H, W)
            frame_mask: optional bool (T, T), where mask[q, r] allows q to
                attend to proposals from frame r. Diagonal is usually False.

        Returns:
            Refined logits with shape (B, T, nc, H, W).
        """
        B, T, C, H, W = feat.shape
        if T <= 1:
            return logits

        HW = H * W
        pre_topk, topk = self._proposal_k(HW)
        score_map = logits.sigmoid().amax(dim=2)  # (B, T, H, W)
        if frame_mask is None:
            frame_mask = ~torch.eye(T, device=feat.device, dtype=torch.bool)
        else:
            frame_mask = frame_mask.to(device=feat.device, dtype=torch.bool)
        candidate_score = self._local_nms_scores(self._candidate_scores(score_map, frame_mask)).flatten(2)
        vals, idx = candidate_score.topk(pre_topk, dim=2)       # (B, T, pre-K)
        if topk < pre_topk:
            vals = vals[:, :, :topk]
            idx = idx[:, :, :topk]
        current_vals = score_map.flatten(2).gather(2, idx)  # (B, T, K)
        valid_mask = vals.reshape(B, T * topk) > float(getattr(self, "conf_thr", 0.001) or 0.0)
        valid_any = valid_mask.any(dim=1)
        if not valid_any.any():
            return logits
        safe_mask = valid_mask.clone()
        safe_mask[~valid_any, 0] = True

        feat_tok = self._gather_window_tokens(feat, idx)             # (B, T, K, C)
        prob_tok = self._gather_window_logits(logits.sigmoid(), idx) # (B, T, K, nc)
        q_tok = feat_tok.reshape(B, T * topk, C)
        r_tok = q_tok
        q = F.normalize(self.q(q_tok), dim=-1)
        k = F.normalize(self.k(r_tok), dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) / max(C ** 0.5, 1.0)  # (B, T*K, T*K)

        q_prob = prob_tok.reshape(B, T * topk, self.nc)
        r_prob = q_prob
        cls_gain = float(getattr(self, "cls_sim_gain", 0.5) or 0.0)
        if cls_gain > 0:
            cls_sim = torch.bmm(F.normalize(q_prob, dim=-1), F.normalize(r_prob, dim=-1).transpose(1, 2))
            attn = attn + cls_gain * cls_sim

        reg_gain = float(getattr(self, "reg_sim_gain", 0.25) or 0.0)
        if reg_gain > 0 and reg is not None:
            Cr = reg.shape[2]
            reg_tok = self._gather_window_tokens(reg, idx).reshape(B, T * topk, Cr)
            reg_sim = torch.bmm(F.normalize(reg_tok, dim=-1), F.normalize(reg_tok, dim=-1).transpose(1, 2))
            attn = attn + reg_gain * reg_sim

        spatial_sigma = float(getattr(self, "spatial_sigma", 0.05) or 0.0)
        centers = None
        if spatial_sigma > 0:
            centers = self._proposal_centers_window(reg, idx, H, W).reshape(B, T * topk, 2)
            dist2 = (centers.unsqueeze(2) - centers.unsqueeze(1)).pow(2).sum(dim=-1)
            attn = attn - dist2 / (2.0 * max(spatial_sigma ** 2, 1e-6))
        loc_gain = float(getattr(self, "loc_gain", 0.0) or 0.0)
        if loc_gain > 0:
            if centers is None:
                centers = self._proposal_centers_window(reg, idx, H, W).reshape(B, T * topk, 2)
            attn = attn + self._relative_loc_bias(centers, centers)

        time_sigma = float(getattr(self, "time_sigma", 0.0) or 0.0)
        if time_sigma > 0:
            frame_ids = torch.arange(T, device=feat.device, dtype=feat.dtype).repeat_interleave(topk)
            time_dist2 = (frame_ids.view(1, -1, 1) - frame_ids.view(1, 1, -1)).pow(2)
            attn = attn - time_dist2 / (2.0 * max(time_sigma ** 2, 1e-6))

        score_gain = float(getattr(self, "score_gain", 0.25) or 0.0)
        if score_gain > 0:
            attn = attn + score_gain * torch.log(vals.reshape(B, T * topk).clamp_min(1e-6)).unsqueeze(1)

        token_mask = frame_mask.repeat_interleave(topk, dim=0).repeat_interleave(topk, dim=1)  # (T*K, T*K)
        safe_mask = safe_mask & token_mask.any(dim=0).view(1, -1)
        attn = attn.masked_fill(~token_mask.view(1, T * topk, T * topk), float("-inf"))
        attn = attn.masked_fill(~safe_mask.unsqueeze(1), float("-inf"))
        row_has_valid = (token_mask.view(1, T * topk, T * topk) & safe_mask.unsqueeze(1)).any(dim=2)
        attn = torch.where(row_has_valid.unsqueeze(2), attn, torch.zeros_like(attn))

        weights = attn.softmax(dim=-1)
        support_prob = torch.bmm(weights, r_prob)
        agg = torch.bmm(weights, self.v(r_tok))
        gate = torch.sigmoid(self.gate(torch.cat((q_tok, agg), dim=-1)))
        fused = q_tok + gate * (agg - q_tok)

        concat_feat = torch.cat((agg, q_tok), dim=-1)
        learned_scale = torch.sigmoid(self.out(self.norm(fused)) + self.concat_out(self.concat_norm(concat_feat)))
        delta = (
            learned_scale
            * torch.relu(support_prob - q_prob)
            * self.alpha.to(dtype=logits.dtype, device=logits.device)
        )
        vote_gain = float(getattr(self, "vote_gain", 0.0) or 0.0)
        if vote_gain > 0:
            delta = delta + vote_gain * torch.relu(support_prob - q_prob) * self.alpha.to(
                dtype=logits.dtype, device=logits.device
            )
        center_uncertain = 1.0 - current_vals.reshape(B, T * topk, 1).clamp(0.0, 1.0)
        delta = delta * center_uncertain
        delta = torch.where(row_has_valid.unsqueeze(-1), delta, torch.zeros_like(delta))
        delta = torch.where(valid_any.view(B, 1, 1), delta, torch.zeros_like(delta))

        out_flat = logits.flatten(3).clone()
        scatter_idx = idx.unsqueeze(2).expand(B, T, self.nc, topk)
        scatter_src = delta.view(B, T, topk, self.nc).permute(0, 1, 3, 2)
        out_flat.scatter_add_(3, scatter_idx, scatter_src)
        return out_flat.view(B, T, self.nc, H, W)

    def forward(
        self,
        key_feat: torch.Tensor,
        key_logits: torch.Tensor,
        ref_feat: torch.Tensor,
        ref_logits: torch.Tensor,
        key_reg: torch.Tensor | None = None,
        ref_reg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ref_feat.numel() == 0 or ref_feat.shape[1] == 0:
            return key_logits

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
            if ref_reg is not None:
                ref_reg = F.interpolate(
                    ref_reg.reshape(B * R, ref_reg.shape[2], *ref_reg.shape[-2:]),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).view(B, R, ref_reg.shape[2], H, W)

        HW = H * W
        pre_topk, topk = self._proposal_k(HW)
        key_vals, key_idx = self._topk_positions(key_logits, pre_topk)
        if topk < pre_topk:
            key_vals = key_vals[:, :topk]
            key_idx = key_idx[:, :topk]
        per_ref_k = max(1, min(H * W, topk // max(R, 1)))
        ref_score_map = ref_logits.sigmoid().amax(dim=2)
        if int(getattr(self, "nms_radius", 0) or 0) > 0:
            ref_score_map = self._local_nms_scores(ref_score_map)
        ref_score = ref_score_map.reshape(B, R, H * W)
        ref_vals, ref_spatial_idx = ref_score.topk(per_ref_k, dim=2)  # (B, R, K)
        ref_offsets = torch.arange(R, device=ref_logits.device).view(1, R, 1) * (H * W)
        ref_idx = (ref_spatial_idx + ref_offsets).reshape(B, R * per_ref_k)
        ref_vals = ref_vals.reshape(B, R * per_ref_k)

        valid_mask = ref_vals > float(getattr(self, "conf_thr", 0.001) or 0.0)
        valid_any = valid_mask.any(dim=1)
        if not valid_any.any():
            return key_logits
        safe_mask = valid_mask.clone()
        safe_mask[~valid_any, 0] = True

        key_tok = self._gather_tokens(key_feat, key_idx)  # (B, Kq, C)
        ref_tok_full = ref_feat.permute(0, 1, 3, 4, 2).reshape(B, R * H * W, C)
        ref_tok = ref_tok_full.gather(1, ref_idx.unsqueeze(-1).expand(B, ref_idx.shape[1], C))

        q = F.normalize(self.q(key_tok), dim=-1)
        k = F.normalize(self.k(ref_tok), dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) / max(C ** 0.5, 1.0)

        key_prob = self._gather_logits(key_logits.sigmoid(), key_idx)
        ref_prob_full = ref_logits.sigmoid().permute(0, 1, 3, 4, 2).reshape(B, R * H * W, self.nc)
        ref_prob = ref_prob_full.gather(1, ref_idx.unsqueeze(-1).expand(B, ref_idx.shape[1], self.nc))
        cls_gain = float(getattr(self, "cls_sim_gain", 0.5) or 0.0)
        if cls_gain > 0:
            cls_sim = torch.bmm(F.normalize(key_prob, dim=-1), F.normalize(ref_prob, dim=-1).transpose(1, 2))
            attn = attn + cls_gain * cls_sim

        reg_gain = float(getattr(self, "reg_sim_gain", 0.25) or 0.0)
        if reg_gain > 0 and key_reg is not None and ref_reg is not None:
            key_reg_tok = self._gather_tokens(key_reg, key_idx)
            Cr = ref_reg.shape[2]
            ref_reg_full = ref_reg.permute(0, 1, 3, 4, 2).reshape(B, R * H * W, Cr)
            ref_reg_tok = ref_reg_full.gather(1, ref_idx.unsqueeze(-1).expand(B, ref_idx.shape[1], Cr))
            reg_sim = torch.bmm(
                F.normalize(key_reg_tok, dim=-1), F.normalize(ref_reg_tok, dim=-1).transpose(1, 2)
            )
            attn = attn + reg_gain * reg_sim

        spatial_sigma = float(getattr(self, "spatial_sigma", 0.05) or 0.0)
        key_yx = None
        ref_yx = None
        if spatial_sigma > 0:
            key_yx = self._proposal_centers(key_reg, key_idx, H, W)
            ref_spatial = ref_spatial_idx.reshape(B, R * per_ref_k)
            ref_yx = self._proposal_centers(
                ref_reg.reshape(B * R, ref_reg.shape[2], H, W) if ref_reg is not None else None,
                ref_spatial.reshape(B * R, per_ref_k),
                H,
                W,
            ).reshape(B, R * per_ref_k, 2)
            dist2 = (key_yx.unsqueeze(2) - ref_yx.unsqueeze(1)).pow(2).sum(dim=-1)
            attn = attn - dist2 / (2.0 * max(spatial_sigma ** 2, 1e-6))
        loc_gain = float(getattr(self, "loc_gain", 0.0) or 0.0)
        if loc_gain > 0:
            if key_yx is None:
                key_yx = self._proposal_centers(key_reg, key_idx, H, W)
            if ref_yx is None:
                ref_spatial = ref_spatial_idx.reshape(B, R * per_ref_k)
                ref_yx = self._proposal_centers(
                    ref_reg.reshape(B * R, ref_reg.shape[2], H, W) if ref_reg is not None else None,
                    ref_spatial.reshape(B * R, per_ref_k),
                    H,
                    W,
                ).reshape(B, R * per_ref_k, 2)
            attn = attn + self._relative_loc_bias(key_yx, ref_yx)

        score_gain = float(getattr(self, "score_gain", 0.25) or 0.0)
        if score_gain > 0:
            attn = attn + score_gain * torch.log(ref_vals.clamp_min(1e-6)).unsqueeze(1)

        attn = attn.masked_fill(~safe_mask.unsqueeze(1), float("-inf"))
        weights = attn.softmax(dim=-1)
        support_prob = torch.bmm(weights, ref_prob)
        agg = torch.bmm(weights, self.v(ref_tok))
        gate = torch.sigmoid(self.gate(torch.cat((key_tok, agg), dim=-1)))
        fused = key_tok + gate * (agg - key_tok)

        concat_feat = torch.cat((agg, key_tok), dim=-1)
        learned_scale = torch.sigmoid(self.out(self.norm(fused)) + self.concat_out(self.concat_norm(concat_feat)))
        delta = (
            learned_scale
            * torch.relu(support_prob - key_prob)
            * self.alpha.to(dtype=key_logits.dtype, device=key_logits.device)
        )
        vote_gain = float(getattr(self, "vote_gain", 0.0) or 0.0)
        if vote_gain > 0:
            delta = delta + vote_gain * torch.relu(support_prob - key_prob) * self.alpha.to(
                dtype=key_logits.dtype, device=key_logits.device
            )
        # Strong key predictions need less help; uncertain proposals get more temporal correction.
        center_uncertain = 1.0 - key_vals.unsqueeze(-1).clamp(0.0, 1.0)
        delta = delta * center_uncertain
        delta = torch.where(valid_any.view(B, 1, 1), delta, torch.zeros_like(delta))

        out_flat = key_logits.flatten(2).clone()
        scatter_idx = key_idx.unsqueeze(1).expand(B, self.nc, key_idx.shape[1])
        out_flat.scatter_add_(2, scatter_idx, delta.transpose(1, 2))
        return out_flat.view(B, self.nc, H, W)

    def extra_repr(self) -> str:
        return (
            f"C={self.cls_channels}, nc={self.nc}, topk={self.topk}, conf_thr={self.conf_thr}, "
            f"after_topk={self.after_topk}, nms_radius={self.nms_radius}, "
            f"spatial_sigma={self.spatial_sigma}, cls_sim={self.cls_sim_gain}, "
            f"reg_sim={self.reg_sim_gain}, score={self.score_gain}, vote={self.vote_gain}, "
            f"recall={self.recall_gain}, recall_radius={self.recall_radius}, "
            f"time_sigma={self.time_sigma}, loc_gain={self.loc_gain}"
        )


def set_alpha_warmup(model: nn.Module, target: float) -> None:
    """Set FAM alpha residual gate to a fixed value across all FAMs in a model.

    Use this from the trainer to warm up alpha from 0 -> target over the first
    few epochs.
    """
    for m in model.modules():
        if isinstance(m, (FeatureAggregationModule, ProposalTemporalRefiner)):
            with torch.no_grad():
                m.alpha.fill_(float(target))
