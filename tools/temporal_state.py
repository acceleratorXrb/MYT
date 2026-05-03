#!/usr/bin/env python3
"""Helpers for resetting temporal state in VID inference/export scripts."""

from __future__ import annotations


def _module_roots(model):
    roots = [getattr(model, "model", None)]
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        roots.append(getattr(predictor, "model", None))
    for root in roots:
        if root is not None:
            yield root


def reset_detect_vid_state(model) -> int:
    """Reset Detect_VID clip/streaming state on YOLO and predictor models."""
    from ultralytics.nn.modules import Detect_VID

    seen = set()
    count = 0
    for root in _module_roots(model):
        modules = root.modules() if hasattr(root, "modules") else []
        for module in modules:
            ident = id(module)
            if ident in seen:
                continue
            seen.add(ident)
            if isinstance(module, Detect_VID):
                module.clip_layout = None
                module.reset_buffer()
                count += 1
    return count


def reset_model_trackers(model) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) if predictor is not None else []:
        tracker.reset()


def reset_video_state(model, trackers: bool = False) -> int:
    """Reset model temporal buffers, and optionally tracker state, at sequence boundaries."""
    if trackers:
        reset_model_trackers(model)
    return reset_detect_vid_state(model)
