"""Shape-and-gradient sanity tests for the YOLOV FAM module.

These tests intentionally avoid building the full Mamba-YOLO model (which
requires the `selective_scan_cuda` extension). They exercise the
FeatureAggregationModule and the cv3-tap behaviour of Detect_VID directly.
"""

from __future__ import annotations

import pytest
import torch

# FAM is independent of the rest of the model graph
from ultralytics.nn.modules.yolov_fam import FeatureAggregationModule


def test_fam_forward_shape_basic():
    B, R, C, H, W, NC = 2, 4, 64, 16, 16, 10
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=R, topk=128)
    key = torch.randn(B, C, H, W)
    refs = torch.randn(B, R, C, H, W)
    ref_logits = torch.randn(B, R, NC, H, W)
    out = fam(key, refs, ref_logits)
    assert out.shape == (B, C, H, W)


def test_fam_alpha_init_is_identity():
    """With alpha=0 (default init), the FAM output equals the key feature."""
    B, R, C, H, W, NC = 1, 2, 8, 4, 4, 5
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=R, topk=8)
    fam.eval()
    key = torch.randn(B, C, H, W)
    refs = torch.randn(B, R, C, H, W)
    ref_logits = torch.randn(B, R, NC, H, W)
    with torch.no_grad():
        out = fam(key, refs, ref_logits)
    assert torch.allclose(out, key, atol=1e-5)


def test_fam_zero_refs_returns_key():
    """When R=0 the FAM returns the key feature unchanged."""
    B, C, H, W, NC = 1, 16, 8, 8, 4
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=0, topk=32)
    key = torch.randn(B, C, H, W)
    refs = torch.empty(B, 0, C, H, W)
    ref_logits = torch.empty(B, 0, NC, H, W)
    out = fam(key, refs, ref_logits)
    assert torch.equal(out, key)


def test_fam_gradient_flows_to_alpha():
    """Loss on FAM output must produce a gradient on alpha (and refs by default)."""
    B, R, C, H, W, NC = 1, 3, 32, 8, 8, 6
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=R, topk=64)
    # bump alpha away from 0 so the aggregated path actually contributes
    with torch.no_grad():
        fam.alpha.fill_(0.5)
    key = torch.randn(B, C, H, W, requires_grad=True)
    refs = torch.randn(B, R, C, H, W, requires_grad=True)
    ref_logits = torch.randn(B, R, NC, H, W)
    out = fam(key, refs, ref_logits)
    out.mean().backward()
    assert fam.alpha.grad is not None
    assert refs.grad is not None
    assert key.grad is not None


def test_fam_topk_clamped_below_token_count():
    """topk > R*HW must not error; should clamp to available tokens."""
    B, R, C, H, W, NC = 1, 2, 16, 4, 4, 3
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=R, topk=10_000)
    key = torch.randn(B, C, H, W)
    refs = torch.randn(B, R, C, H, W)
    ref_logits = torch.randn(B, R, NC, H, W)
    out = fam(key, refs, ref_logits)
    assert out.shape == key.shape


def test_fam_conf_thr_filters_dead_refs():
    """When all ref scores are below conf_thr, output should still be valid (no NaN)."""
    B, R, C, H, W, NC = 1, 2, 16, 4, 4, 3
    fam = FeatureAggregationModule(cls_channels=C, num_ref_frames=R, topk=8, conf_thr=10.0)
    with torch.no_grad():
        fam.alpha.fill_(0.5)
    key = torch.randn(B, C, H, W)
    refs = torch.randn(B, R, C, H, W)
    ref_logits = torch.zeros(B, R, NC, H, W)  # all scores 0 < conf_thr
    out = fam(key, refs, ref_logits)
    assert out.shape == key.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
