import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = os.path.abspath('.') + "/"


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def bool_or_str(value):
    try:
        return str2bool(value)
    except argparse.ArgumentTypeError:
        return value


def parse_freeze(value):
    if value is None:
        return None
    value = str(value)
    if "," in value:
        return [int(x) for x in value.split(",") if x.strip()]
    return int(value)


def run_extra_eval_step(name, cmd, strict):
    cmd = [str(x) for x in cmd]
    print(f"[extra-eval] {name}: {' '.join(map(str, cmd))}", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    ok = completed.returncode == 0
    if not ok:
        message = f"[extra-eval] {name} failed with exit code {completed.returncode}"
        if strict:
            raise RuntimeError(message)
        print(message, flush=True)
    return {"name": name, "ok": ok, "returncode": completed.returncode, "cmd": cmd}


def build_extra_eval_callback(opt):
    period = max(int(opt.extra_eval_period), 0)
    official_root = Path(resolve_path(opt.extra_eval_official_root))
    tracker = Path(resolve_path(opt.extra_eval_tracker))
    strict = bool(opt.extra_eval_strict)

    def on_model_save(trainer):
        if period <= 0:
            return
        epoch = int(trainer.epoch) + 1
        if epoch % period != 0:
            return

        weights = Path(trainer.last)
        if not weights.exists():
            message = f"[extra-eval] skipping epoch {epoch}: checkpoint not found at {weights}"
            if strict:
                raise FileNotFoundError(message)
            print(message, flush=True)
            return

        if not (official_root / "annotations").is_dir() or not (official_root / "sequences").is_dir():
            message = (
                f"[extra-eval] skipping epoch {epoch}: official root must contain annotations/ and sequences/: "
                f"{official_root}"
            )
            if strict:
                raise FileNotFoundError(message)
            print(message, flush=True)
            return

        out_root = Path(trainer.save_dir) / "extra_eval" / f"epoch{epoch:03d}"
        detections_dir = out_root / "detections"
        tracks_dir = out_root / "tracks"
        flicker_json = out_root / "flicker.json"
        mot_json = out_root / "mot.json"
        summary_json = out_root / "summary.json"
        out_root.mkdir(parents=True, exist_ok=True)

        steps = []
        py = sys.executable
        scripts = Path(ROOT) / "tools"

        try:
            use_clip_export = opt.extra_eval_clip_inference or opt.extra_eval_window_inference
            export_script = (
                scripts / "export_visdrone_vid_clip_results.py"
                if use_clip_export
                else scripts / "export_visdrone_vid_results.py"
            )
            export_cmd = [
                py,
                export_script,
                "--weights",
                weights,
                "--source",
                official_root,
                "--out",
                detections_dir,
                "--imgsz",
                opt.imgsz,
                "--device",
                opt.device,
                "--conf",
                opt.extra_eval_conf,
                "--iou",
                opt.extra_eval_iou,
            ]
            if use_clip_export:
                export_cmd.extend(
                    [
                        "--num_ref_frames",
                        opt.num_ref_frames,
                        "--clip_stride",
                        opt.clip_stride,
                        "--ref_sample",
                        opt.ref_sample if opt.ref_sample in {"adjacent", "causal"} else "adjacent",
                    ]
                )
                if opt.extra_eval_window_inference:
                    export_cmd.extend(["--all_keys", "--window_size", opt.vid_window_size or opt.num_ref_frames + 1])
                export_cmd.extend(
                    [
                        "--temporal_fusion",
                        opt.temporal_fusion,
                        "--score_smooth_sigma",
                        opt.score_smooth_sigma,
                        "--score_smooth_cls_gain",
                        opt.score_smooth_cls_gain,
                        "--score_smooth_conf_gain",
                        opt.score_smooth_conf_gain,
                        "--score_smooth_min_ref_score",
                        opt.score_smooth_min_ref_score,
                    ]
                )
            else:
                export_cmd.extend(["--batch", opt.extra_eval_batch])
            steps.append(run_extra_eval_step("export_detections", export_cmd, strict))

            steps.append(
                run_extra_eval_step(
                    "flicker",
                    [
                        py,
                        scripts / "eval_visdrone_vid_cls_flicker.py",
                        "--gt",
                        official_root / "annotations",
                        "--pred",
                        detections_dir,
                        "--out",
                        flicker_json,
                    ],
                    strict,
                )
            )
            track_export_script = (
                scripts / "export_visdrone_vid_clip_tracks.py"
                if use_clip_export
                else scripts / "export_visdrone_vid_tracks.py"
            )
            track_cmd = [
                py,
                track_export_script,
                "--weights",
                weights,
                "--source",
                official_root,
                "--out",
                tracks_dir,
                "--tracker",
                tracker,
                "--imgsz",
                opt.imgsz,
                "--device",
                opt.device,
                "--conf",
                opt.extra_eval_track_conf,
                "--iou",
                opt.extra_eval_iou,
            ]
            if use_clip_export:
                track_cmd.extend(
                    [
                        "--num_ref_frames",
                        opt.num_ref_frames,
                        "--clip_stride",
                        opt.clip_stride,
                        "--ref_sample",
                        opt.ref_sample if opt.ref_sample in {"adjacent", "causal"} else "adjacent",
                    ]
                )
                if opt.extra_eval_window_inference:
                    track_cmd.extend(["--all_keys", "--window_size", opt.vid_window_size or opt.num_ref_frames + 1])
                track_cmd.extend(
                    [
                        "--temporal_fusion",
                        opt.temporal_fusion,
                        "--score_smooth_sigma",
                        opt.score_smooth_sigma,
                        "--score_smooth_cls_gain",
                        opt.score_smooth_cls_gain,
                        "--score_smooth_conf_gain",
                        opt.score_smooth_conf_gain,
                        "--score_smooth_min_ref_score",
                        opt.score_smooth_min_ref_score,
                    ]
                )
            steps.append(run_extra_eval_step("export_tracks", track_cmd, strict))
            steps.append(
                run_extra_eval_step(
                    "mot",
                    [
                        py,
                        scripts / "eval_visdrone_vid_mot.py",
                        "--gt",
                        official_root / "annotations",
                        "--pred",
                        tracks_dir,
                        "--out",
                        mot_json,
                    ],
                    strict,
                )
            )
        finally:
            payload = {
                "epoch": epoch,
                "weights": str(weights),
                "official_root": str(official_root),
                "out_root": str(out_root),
                "steps": steps,
            }
            summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"[extra-eval] epoch {epoch} summary saved to {summary_json}", flush=True)

    return on_model_save


def build_score_smooth_warmup_callback(opt):
    warmup_epochs = float(opt.score_smooth_warmup_epochs or 0.0)
    alpha_target = float(opt.score_smooth_alpha_target)

    def on_train_epoch_start(trainer):
        if warmup_epochs <= 0.0:
            return
        epoch = int(trainer.epoch)
        if epoch >= warmup_epochs:
            return
        denom = max(warmup_epochs - 1.0, 1.0)
        ratio = min(max(epoch / denom, 0.0), 1.0)
        alpha = alpha_target * ratio
        for module in trainer.model.modules():
            setter = getattr(module, "set_score_smooth_alpha", None)
            if callable(setter):
                setter(alpha)
        if epoch == 0 or epoch == int(warmup_epochs) - 1:
            print(f"[score-smooth-warmup] epoch={epoch + 1} alpha={alpha:.6g} target={alpha_target}", flush=True)

    return on_train_epoch_start


def set_detect_vid_temporal_fusion(
    model,
    mode,
    score_smooth_sigma=None,
    score_smooth_cls_gain=None,
    score_smooth_conf_gain=None,
    score_smooth_min_ref_score=None,
):
    """Set Detect_VID score-smoothing options on a YOLO wrapper or raw model."""
    try:
        from ultralytics.nn.modules import Detect_VID
    except Exception:
        return 0
    roots = [getattr(model, "model", model)]
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        roots.append(getattr(predictor, "model", None))
    count = 0
    for root in roots:
        if root is None or not hasattr(root, "modules"):
            continue
        for module in root.modules():
            if isinstance(module, Detect_VID):
                module.temporal_fusion = mode
                if score_smooth_sigma is not None:
                    module.score_smooth_sigma = float(score_smooth_sigma)
                if score_smooth_cls_gain is not None:
                    module.score_smooth_cls_gain = float(score_smooth_cls_gain)
                if score_smooth_conf_gain is not None:
                    module.score_smooth_conf_gain = float(score_smooth_conf_gain)
                if score_smooth_min_ref_score is not None:
                    module.score_smooth_min_ref_score = float(score_smooth_min_ref_score)
                count += 1
    return count

def resolve_dataset_yaml(path, project, data_task="auto"):
    """Materialize project-local dataset roots as absolute paths for Ultralytics."""
    data_path = Path(resolve_path(path))
    if data_path.suffix.lower() not in {".yaml", ".yml"} or not data_path.exists():
        return str(data_path)

    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    changed = False
    dataset_root = data.get("path")
    if isinstance(dataset_root, str) and not os.path.isabs(dataset_root):
        project_dataset_root = Path(resolve_path(dataset_root))
        if project_dataset_root.exists():
            data["path"] = str(project_dataset_root.resolve())
            changed = True

    if data_task == "detect":
        if "task" in data:
            data.pop("task", None)
            changed = True
    elif data_task == "vid":
        if data.get("task") != "vid":
            data["task"] = "vid"
            changed = True
    elif data_task != "auto":
        raise ValueError(f"Unsupported data_task: {data_task}")

    if not changed:
        return str(data_path)

    resolved_dir = Path(resolve_path(project))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    suffix = data_task if data_task != "auto" else "resolved"
    resolved_yaml = resolved_dir / f"{data_path.stem}.{suffix}.yaml"
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return str(resolved_yaml)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/VisDrone-VID.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml', help='model yaml path')
    parser.add_argument('--weights', type=str, default='', help='checkpoint path for val/test/predict/export or fine-tuning')
    parser.add_argument('--init_weights', type=str, default='', help='partial checkpoint initialization for training a YAML model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, predict or export')
    parser.add_argument('--data_task', default='auto', choices=['auto', 'detect', 'vid'], help='override dataset task; use detect for single-frame official Mamba-YOLO baseline')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--val_period', type=int, default=1, help='validate every N epochs during training')
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--lr0', type=float, default=None, help='initial learning rate override')
    parser.add_argument('--lrf', type=float, default=None, help='final learning rate factor override')
    parser.add_argument('--momentum', type=float, default=None, help='optimizer momentum/beta1 override')
    parser.add_argument('--weight_decay', type=float, default=None, help='optimizer weight decay override')
    parser.add_argument('--warmup_epochs', type=float, default=None, help='warmup epochs override')
    parser.add_argument('--warmup_momentum', type=float, default=None, help='warmup initial momentum override')
    parser.add_argument('--warmup_bias_lr', type=float, default=None, help='initial bias learning rate during warmup')
    parser.add_argument('--cos_lr', action='store_true', help='use cosine learning rate schedule')
    parser.add_argument('--patience', type=int, default=None, help='early stopping patience')
    parser.add_argument('--save_period', type=int, default=None, help='save checkpoint every N epochs')
    parser.add_argument('--cache', nargs='?', const=True, default=None, help='cache images: ram, disk, true/false')
    parser.add_argument('--pretrained', nargs='?', const=True, default=None, type=bool_or_str, help='use pretrained weights or a pretrained checkpoint path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', nargs='?', const=True, default=None, type=str2bool, help='enable deterministic training')
    parser.add_argument('--single_cls', action='store_true', help='train all classes as one class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--close_mosaic', type=int, default=None, help='disable mosaic for final N epochs')
    parser.add_argument('--fraction', type=float, default=None, help='fraction of training data to use')
    parser.add_argument('--freeze', type=parse_freeze, default=None, help='freeze first N layers or comma-separated layer indices')
    parser.add_argument('--multi_scale', action='store_true', help='enable multi-scale training')
    parser.add_argument('--box', type=float, default=None, help='box loss gain')
    parser.add_argument('--cls', type=float, default=None, help='classification loss gain')
    parser.add_argument('--dfl', type=float, default=None, help='DFL loss gain')
    parser.add_argument('--label_smoothing', type=float, default=None, help='label smoothing fraction')
    parser.add_argument('--nbs', type=int, default=None, help='nominal batch size')
    parser.add_argument('--hsv_h', type=float, default=None, help='HSV hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=None, help='HSV saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=None, help='HSV value augmentation')
    parser.add_argument('--degrees', type=float, default=None, help='rotation augmentation degrees')
    parser.add_argument('--translate', type=float, default=None, help='translation augmentation fraction')
    parser.add_argument('--scale', type=float, default=None, help='scale augmentation gain')
    parser.add_argument('--shear', type=float, default=None, help='shear augmentation degrees')
    parser.add_argument('--perspective', type=float, default=None, help='perspective augmentation fraction')
    parser.add_argument('--flipud', type=float, default=None, help='vertical flip probability')
    parser.add_argument('--fliplr', type=float, default=None, help='horizontal flip probability')
    parser.add_argument('--bgr', type=float, default=None, help='BGR channel swap probability')
    parser.add_argument('--mosaic', type=float, default=None, help='mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=None, help='mixup augmentation probability')
    parser.add_argument('--copy_paste', type=float, default=None, help='copy-paste augmentation probability')
    parser.add_argument('--conf', type=float, default=None, help='validation/prediction confidence threshold')
    parser.add_argument('--iou', type=float, default=None, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=None, help='maximum detections per image')
    parser.add_argument('--plots', nargs='?', const=True, default=None, type=str2bool, help='save training/validation plots')
    parser.add_argument('--save_json', action='store_true', help='save COCO-style validation JSON when supported')
    parser.add_argument('--save_hybrid', action='store_true', help='save hybrid validation labels')
    parser.add_argument('--project', default='output_dir/visdrone_vid', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_t', help='save to project/name')
    parser.add_argument('--source', type=str, default='', help='source path for predict')
    parser.add_argument('--format', type=str, default='onnx', help='export format')
    parser.add_argument('--save_txt', action='store_true', help='save prediction txt labels')
    parser.add_argument('--save_conf', action='store_true', help='save prediction confidences')
    parser.add_argument('--resume', action='store_true', help='resume training from the last checkpoint')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # VID clip-mode (consumed by VIDClipDataset when data yaml has task: vid)
    parser.add_argument('--num_ref_frames', type=int, default=4, help='reference frames per clip for VIDClipDataset (0 disables temporal smoothing)')
    parser.add_argument('--clip_stride', type=int, default=1, help='temporal stride between sampled refs')
    parser.add_argument('--ref_sample', default='adjacent', choices=['adjacent', 'causal', 'uniform_local', 'uniform_global'], help='ref-frame sampling strategy')
    parser.add_argument('--vid_clip_mode', default='center', choices=['center', 'window'], help='VID training clip layout: center repeats refs; window uses each frame once inside a temporal window')
    parser.add_argument('--vid_window_size', type=int, default=None, help='frames per window when --vid_clip_mode window; defaults to num_ref_frames+1')
    parser.add_argument('--ref_aux_loss', type=float, default=0.0, help='auxiliary detection loss weight for reference frames')
    parser.add_argument('--temporal_fusion', default='score_smooth', choices=['score_smooth', 'none'], help='Detect_VID temporal fusion mode')
    parser.add_argument('--score_smooth_sigma', type=float, default=0.03, help='normalized local radius for score_smooth temporal support')
    parser.add_argument('--score_smooth_cls_gain', type=float, default=0.6, help='class probability temporal smoothing gain for score_smooth')
    parser.add_argument('--score_smooth_conf_gain', type=float, default=0.7, help='low-current-confidence temporal boost gain for score_smooth')
    parser.add_argument('--score_smooth_min_ref_score', type=float, default=0.001, help='minimum reference score used by score_smooth')
    parser.add_argument('--score_smooth_warmup_epochs', type=float, default=0.0, help='linearly warm score smoothing alpha for this many epochs; 0 disables')
    parser.add_argument('--score_smooth_alpha_target', type=float, default=1.0, help='target score smoothing alpha value at the end of warmup')
    parser.add_argument('--debug_clip_shape', action='store_true', help='print the first training batch image shape and clip layout')
    parser.add_argument('--debug_clip_aug', action='store_true', help='print first few VID clip augmentation decisions')
    parser.add_argument('--debug_clip_refs', action='store_true', help='print first few VID key/ref frame paths and positions')
    parser.add_argument('--debug_vid_head', action='store_true', help='print first few Detect_VID temporal head shapes and ref indices')
    # Optional heavier video metrics, run after checkpoint save every N epochs.
    parser.add_argument('--extra_eval_period', type=int, default=1, help='run flicker/MOT video eval every N epochs; 0 disables')
    parser.add_argument('--extra_eval_official_root', default='datasets/VisDrone-VID/raw/VisDrone2019-VID-val', help='official VisDrone-VID split root with annotations/ and sequences/')
    parser.add_argument('--extra_eval_toolkit', default='third_party/VisDrone2018-VID-toolkit', help='deprecated; official AP/AR is no longer run by periodic extra eval')
    parser.add_argument('--extra_eval_tracker', default='ultralytics/cfg/trackers/bytetrack.yaml', help='tracker yaml for MOT export')
    parser.add_argument('--extra_eval_batch', type=int, default=16, help='batch size for detection export during extra eval')
    parser.add_argument('--extra_eval_conf', type=float, default=0.001, help='confidence threshold for detection export during extra eval')
    parser.add_argument('--extra_eval_track_conf', type=float, default=0.1, help='confidence threshold for tracking export')
    parser.add_argument('--extra_eval_iou', type=float, default=0.7, help='NMS IoU threshold for extra eval exports')
    parser.add_argument('--extra_eval_clip_inference', action='store_true', help='export detections with explicit key+ref clip inference instead of streaming')
    parser.add_argument('--extra_eval_window_inference', action='store_true', help='export detections/tracks with non-overlapping VID windows; every frame is inferred once')
    parser.add_argument('--extra_eval_strict', action='store_true', help='fail training if an extra-eval step fails')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    from ultralytics import YOLO

    task = opt.task.lower()
    data = resolve_dataset_yaml(opt.data, opt.project, opt.data_task)
    args = {
        "data": data,
        "epochs": opt.epochs,
        "val_period": opt.val_period,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": resolve_path(opt.project),
        "name": opt.name,
        # VID clip-mode params (ignored by non-VID datasets)
        "num_ref_frames": opt.num_ref_frames,
        "clip_stride": opt.clip_stride,
        "ref_sample": opt.ref_sample,
        "vid_clip_mode": opt.vid_clip_mode,
        "vid_window_size": opt.vid_window_size or 0,
        "ref_aux_loss": opt.ref_aux_loss,
        "temporal_fusion": opt.temporal_fusion,
        "score_smooth_sigma": opt.score_smooth_sigma,
        "score_smooth_cls_gain": opt.score_smooth_cls_gain,
        "score_smooth_conf_gain": opt.score_smooth_conf_gain,
        "score_smooth_min_ref_score": opt.score_smooth_min_ref_score,
        "score_smooth_warmup_epochs": opt.score_smooth_warmup_epochs,
        "score_smooth_alpha_target": opt.score_smooth_alpha_target,
        "debug_clip_shape": opt.debug_clip_shape,
        "debug_clip_aug": opt.debug_clip_aug,
        "debug_clip_refs": opt.debug_clip_refs,
        "debug_vid_head": opt.debug_vid_head,
    }
    passthrough = (
        "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "patience", "save_period", "cache", "pretrained", "seed", "deterministic", "close_mosaic",
        "fraction", "freeze", "box", "cls", "dfl", "label_smoothing", "nbs", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic",
        "mixup", "copy_paste", "conf", "iou", "max_det", "plots",
    )
    for k in passthrough:
        v = getattr(opt, k)
        if v is not None:
            args[k] = v
    if opt.cos_lr:
        args["cos_lr"] = True
    for k in ("single_cls", "rect", "multi_scale", "save_json", "save_hybrid"):
        if getattr(opt, k):
            args[k] = True
    model_path = resolve_path(opt.weights) if opt.weights else resolve_path(opt.config)
    model = YOLO(model_path)
    set_detect_vid_temporal_fusion(model, opt.temporal_fusion, opt.score_smooth_sigma, opt.score_smooth_cls_gain, opt.score_smooth_conf_gain, opt.score_smooth_min_ref_score)
    if task == "train":
        if opt.init_weights:
            init_weights = resolve_path(opt.init_weights)
            print(f"[init-weights] loading partial weights from {init_weights}", flush=True)
            model.load(init_weights)
            set_detect_vid_temporal_fusion(model, opt.temporal_fusion, opt.score_smooth_sigma, opt.score_smooth_cls_gain, opt.score_smooth_conf_gain, opt.score_smooth_min_ref_score)
        if opt.score_smooth_warmup_epochs > 0:
            model.add_callback("on_train_epoch_start", build_score_smooth_warmup_callback(opt))
        if opt.extra_eval_period > 0:
            model.add_callback("on_model_save", build_extra_eval_callback(opt))
        args["resume"] = opt.resume
        model.train(**args)
    elif task == "val":
        model.val(**args)
    elif task == "test":
        model.val(split="test", **args)
    elif task == "predict":
        source = resolve_path(opt.source) if opt.source else None
        model.predict(
            source=source,
            imgsz=opt.imgsz,
            batch=opt.batch_size,
            device=opt.device,
            half=opt.half,
            dnn=opt.dnn,
            project=resolve_path(opt.project),
            name=opt.name,
            save_txt=opt.save_txt,
            save_conf=opt.save_conf,
        )
    elif task == "export":
        model.export(format=opt.format, imgsz=opt.imgsz, device=opt.device, half=opt.half)
    else:
        raise ValueError(f"Unsupported task: {opt.task}. Expected one of: train, val, test, predict, export.")
