# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
from .yolov_fam import FeatureAggregationModule

__all__ = "Detect", "Detect_VID", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class Detect_VID(Detect):
    """Video Detect head with cross-frame classification feature aggregation.

    Inherits Detect, taps the cls branch (`cv3[i]`) between its two Conv blocks
    and the final 1x1 projection, runs a per-scale FeatureAggregationModule on
    the pre-projection features, then projects to class logits. Reg branch
    (`cv2`) is unchanged. The head emits **key-frame only** outputs so the
    downstream loss/decoder sees a regular YOLOv8-shaped tensor.

    Two forward modes:
      - Training / clip-mode val: receives `(B*T, C, H, W)` features and reads
        `self.clip_layout = (B, T)` to slice and aggregate.
      - Streaming inference: when `self.clip_layout` is None and `self.training`
        is False, maintains a rolling deque of past `cv3_pre` features per
        detection level (causal). Call `reset_buffer()` on source change.
    """

    def __init__(self, nc=80, topk=750, conf_thr=0.001, num_ref_frames=4, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # = cls feature channels per Detect.__init__
        self.fams = nn.ModuleList(
            FeatureAggregationModule(
                cls_channels=c3,
                num_ref_frames=num_ref_frames,
                topk=topk,
                conf_thr=conf_thr,
            )
            for _ in ch
        )
        self.num_ref_frames = num_ref_frames
        self.temporal_fusion = "fam"  # fam | logits | logits_gated | none
        self.fam_conf_boost = 0.0
        self.temporal_cls_consistency = 0.0
        self.debug_vid_head = False
        self._debug_vid_head_printed = 0
        self.clip_layout: tuple[int, int] | None = None  # (B, T); set by trainer
        self.clip_all_keys = False
        self.aux_outputs: list[torch.Tensor] | None = None  # ref-frame predictions for auxiliary loss
        self.temporal_consistency_losses: list[torch.Tensor] | None = None
        # streaming inference buffer (per detection level)
        self._stream_buffers: list[list[torch.Tensor]] | None = None

    @torch.no_grad()
    def reset_buffer(self) -> None:
        """Clear the streaming inference buffer; call on video source switch."""
        self._stream_buffers = [[] for _ in range(self.nl)] if self.nl else None

    def _cv3_pre(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """Run the first two Conv blocks of cv3[i] (pre-projection cls feature)."""
        s = self.cv3[i]
        return s[1](s[0](x))

    def _cv3_cls(self, i: int, pre: torch.Tensor) -> torch.Tensor:
        """Run the final 1x1 projection of cv3[i]."""
        return self.cv3[i][2](pre)

    def _fusion_mode(self) -> str:
        mode = str(getattr(self, "temporal_fusion", "fam") or "fam").lower()
        if mode not in {"fam", "logits", "logits_gated", "none"}:
            raise ValueError(f"temporal_fusion must be fam|logits|logits_gated|none, got {mode!r}")
        return mode

    def _logits_fuse(
        self, i: int, key_logits: torch.Tensor, ref_logits: torch.Tensor, gated: bool = False
    ) -> torch.Tensor:
        """Direct temporal cls-logit fusion for a simple FAM ablation."""
        if ref_logits.numel() == 0 or ref_logits.shape[1] == 0:
            return key_logits
        alpha = self.fams[i].alpha.to(device=key_logits.device, dtype=key_logits.dtype)
        if not gated:
            return key_logits + alpha * ref_logits.mean(dim=1)

        ref_conf = ref_logits.sigmoid().amax(dim=2, keepdim=True)  # (B, R, 1, H, W)
        ref_weight = ref_conf / ref_conf.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ref_mean = (ref_logits * ref_weight).sum(dim=1)
        center_gate = 1.0 - key_logits.sigmoid().amax(dim=1, keepdim=True)
        return key_logits + alpha * center_gate * ref_mean

    def _ref_conf_boost(self, i: int, key_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """Positive-only ref confidence compensation for FAM mode.

        This keeps the center-frame logits as the base prediction and only raises
        classes that adjacent refs support more strongly. It is deliberately
        asymmetric, so refs cannot suppress a confident center-frame class.
        """
        gain = float(getattr(self, "fam_conf_boost", 0.0) or 0.0)
        if gain <= 0.0 or ref_logits.numel() == 0 or ref_logits.shape[1] == 0:
            return key_logits
        alpha = self.fams[i].alpha.to(device=key_logits.device, dtype=key_logits.dtype)
        ref_prob = ref_logits.sigmoid()
        ref_conf = ref_prob.amax(dim=2, keepdim=True)
        ref_weight = ref_conf / ref_conf.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ref_prob_mean = (ref_prob * ref_weight).sum(dim=1)
        key_prob = key_logits.sigmoid()
        center_uncertain = 1.0 - key_prob.amax(dim=1, keepdim=True)
        boost = torch.relu(ref_prob_mean - key_prob) * center_uncertain
        return key_logits + gain * alpha * boost

    def _clip_cls_consistency_loss(self, key_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """Small optional clip-level class-probability consistency regularizer."""
        gain = float(getattr(self, "temporal_cls_consistency", 0.0) or 0.0)
        if gain <= 0.0 or ref_logits.numel() == 0 or ref_logits.shape[1] == 0:
            return key_logits.new_zeros(())
        key_prob = key_logits.sigmoid()
        ref_prob = ref_logits.sigmoid().mean(dim=1).detach()
        with torch.no_grad():
            key_conf = key_prob.amax(dim=1, keepdim=True)
            ref_conf = ref_prob.amax(dim=1, keepdim=True)
            weight = torch.maximum(key_conf, ref_conf).clamp(0.0, 1.0)
        loss = F.smooth_l1_loss(key_prob, ref_prob, reduction="none").mean(dim=1, keepdim=True)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

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
        if len(picked) < num_refs and picked:
            picked.extend([picked[-1]] * (num_refs - len(picked)))
        return picked[:num_refs]

    def _aggregate_one_key(
        self,
        i: int,
        key_pre: torch.Tensor,
        key_logits: torch.Tensor,
        ref_pre: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        mode = self._fusion_mode()
        if mode == "fam":
            agg_pre = self.fams[i](key_pre, ref_pre, ref_logits)
            cls_out = self._cv3_cls(i, agg_pre)
            cls_out = self._ref_conf_boost(i, cls_out, ref_logits)
        elif mode == "logits":
            cls_out = self._logits_fuse(i, key_logits, ref_logits)
        elif mode == "logits_gated":
            cls_out = self._logits_fuse(i, key_logits, ref_logits, gated=True)
        else:
            cls_out = key_logits
        if self.training and self.temporal_consistency_losses is not None:
            self.temporal_consistency_losses.append(self._clip_cls_consistency_loss(cls_out, ref_logits))
        return cls_out

    def _aggregate_clip(
        self, i: int, x_i: torch.Tensor, B: int, T: int, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Clip-mode: input `(B*T, C, H, W)`, output key-frame `(B, no, H, W)`."""
        reg_full = self.cv2[i](x_i)                                 # (B*T, 4*reg_max, H, W)
        pre_full = self._cv3_pre(i, x_i)                            # (B*T, c3, H, W)

        if T <= 1:
            cls_out = self._cv3_cls(i, pre_full)
            out = torch.cat((reg_full, cls_out), 1)
            return (out, None) if return_aux else out

        # reshape to (B, T, ...)
        Cp, H, W = pre_full.shape[1:]
        pre_clip = pre_full.view(B, T, Cp, H, W)
        cls_full = self._cv3_cls(i, pre_full)                       # (B*T, nc, H, W)
        cls_clip = cls_full.view(B, T, self.nc, H, W)

        if self.clip_all_keys:
            num_refs = min(int(getattr(self, "num_ref_frames", T - 1)), max(T - 1, 0))
            ref_debug = []
            cls_outs = []
            for key_pos in range(T):
                ref_idx = self._window_ref_indices(key_pos, T, num_refs)
                if key_pos < 5:
                    ref_debug.append((key_pos, ref_idx))
                if ref_idx:
                    cls_out = self._aggregate_one_key(
                        i,
                        pre_clip[:, key_pos],
                        cls_clip[:, key_pos],
                        pre_clip[:, ref_idx],
                        cls_clip[:, ref_idx],
                    )
                else:
                    cls_out = cls_clip[:, key_pos]
                cls_outs.append(cls_out)
            cls_all = torch.stack(cls_outs, dim=1).reshape(B * T, self.nc, H, W)
            out = torch.cat((reg_full, cls_all), 1)
            if bool(getattr(self, "debug_vid_head", False)) and self._debug_vid_head_printed < 6:
                print(
                    "[debug-vid-head] "
                    f"mode=window level={i} B={B} T={T} clip_all_keys={self.clip_all_keys} "
                    f"num_ref_frames={self.num_ref_frames} ref_debug={ref_debug} "
                    f"pre_shape={tuple(pre_full.shape)} reg_shape={tuple(reg_full.shape)} "
                    f"out_shape={tuple(out.shape)} aux=None",
                    flush=True,
                )
                self._debug_vid_head_printed += 1
            return (out, None) if return_aux else out

        key_pre = pre_clip[:, 0]                                    # (B, c3, H, W)
        ref_pre = pre_clip[:, 1:]                                   # (B, T-1, c3, H, W)
        ref_logits = cls_clip[:, 1:]                                # (B, T-1, nc, H, W)

        cls_out = self._aggregate_one_key(i, key_pre, cls_clip[:, 0], ref_pre, ref_logits)

        Cr = reg_full.shape[1]
        reg_clip = reg_full.view(B, T, Cr, H, W)
        reg_key = reg_clip[:, 0]                                    # (B, 4*reg_max, H, W)
        key_out = torch.cat((reg_key, cls_out), 1)
        if bool(getattr(self, "debug_vid_head", False)) and self._debug_vid_head_printed < 6:
            print(
                "[debug-vid-head] "
                f"mode=center level={i} B={B} T={T} clip_all_keys={self.clip_all_keys} "
                f"num_ref_frames={self.num_ref_frames} ref_count={ref_pre.shape[1]} "
                f"pre_shape={tuple(pre_full.shape)} reg_key_shape={tuple(reg_key.shape)} "
                f"out_shape={tuple(key_out.shape)}",
                flush=True,
            )
            self._debug_vid_head_printed += 1
        if not return_aux:
            return key_out

        # Auxiliary ref detection uses the same unaggregated branches, so refs
        # remain supervised as ordinary single-frame detections.
        ref_reg = reg_clip[:, 1:].reshape(B * (T - 1), Cr, H, W)
        ref_cls = cls_clip[:, 1:].reshape(B * (T - 1), self.nc, H, W)
        ref_out = torch.cat((ref_reg, ref_cls), 1)
        return key_out, ref_out

    def _aggregate_stream(self, i: int, x_i: torch.Tensor) -> torch.Tensor:
        """Streaming-mode: input `(B, C, H, W)` (single frame). Use rolling buffer."""
        if self._stream_buffers is None:
            self.reset_buffer()
        reg_full = self.cv2[i](x_i)
        pre_full = self._cv3_pre(i, x_i)

        buf = self._stream_buffers[i]
        if not buf:
            cls_out = self._cv3_cls(i, pre_full)
            buf.append(pre_full.detach())
            return torch.cat((reg_full, cls_out), 1)

        # Ref features = entire buffer. Predictor letterboxing can change
        # feature-map shapes between frames, so resize buffered refs to the
        # current key feature shape before stacking.
        Hk, Wk = pre_full.shape[-2:]
        ref_items = [
            z if z.shape[-2:] == (Hk, Wk) else F.interpolate(z, size=(Hk, Wk), mode="bilinear", align_corners=False)
            for z in buf
        ]
        ref_pre = torch.stack(ref_items, dim=1)                     # (B, R, C, H, W)
        B, R, Cp, H, W = ref_pre.shape
        ref_logits = self._cv3_cls(i, ref_pre.view(B * R, Cp, H, W)).view(B, R, self.nc, H, W)

        mode = self._fusion_mode()
        if mode == "fam":
            agg_pre = self.fams[i](pre_full, ref_pre, ref_logits)
            cls_out = self._cv3_cls(i, agg_pre)
            cls_out = self._ref_conf_boost(i, cls_out, ref_logits)
        elif mode == "logits":
            cls_out = self._logits_fuse(i, self._cv3_cls(i, pre_full), ref_logits)
        elif mode == "logits_gated":
            cls_out = self._logits_fuse(i, self._cv3_cls(i, pre_full), ref_logits, gated=True)
        else:
            cls_out = self._cv3_cls(i, pre_full)

        # push current pre into buffer (capped)
        buf.append(pre_full.detach())
        if len(buf) > self.num_ref_frames:
            buf.pop(0)
        return torch.cat((reg_full, cls_out), 1)

    def forward(self, x):
        """Detect_VID forward: clip-mode if clip_layout set, else streaming."""
        self.aux_outputs = None
        self.temporal_consistency_losses = [] if self.training else None
        if self.clip_layout is not None:
            B, T = self.clip_layout
        else:
            B, T = x[0].shape[0], 1

        aux_outputs = []
        for i in range(self.nl):
            if self.clip_layout is not None or self.training:
                out = self._aggregate_clip(i, x[i], B, T, return_aux=self.training)
                if self.training:
                    x[i], aux_i = out
                    if aux_i is not None:
                        aux_outputs.append(aux_i)
                else:
                    x[i] = out
            else:
                x[i] = self._aggregate_stream(i, x[i])
        if self.training and len(aux_outputs) == self.nl:
            self.aux_outputs = aux_outputs

        if self.training:
            return x

        # Inference path — same as Detect but on key-frame-only x[i]
        shape = x[0].shape  # (B, no, H, W)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (z.transpose(0, 1) for z in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
