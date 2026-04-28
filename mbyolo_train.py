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
            steps.append(
                run_extra_eval_step(
                    "export_detections",
                    [
                        py,
                        scripts / "export_visdrone_vid_results.py",
                        "--weights",
                        weights,
                        "--source",
                        official_root,
                        "--out",
                        detections_dir,
                        "--imgsz",
                        opt.imgsz,
                        "--batch",
                        opt.extra_eval_batch,
                        "--device",
                        opt.device,
                        "--conf",
                        opt.extra_eval_conf,
                        "--iou",
                        opt.extra_eval_iou,
                    ],
                    strict,
                )
            )

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
            steps.append(
                run_extra_eval_step(
                    "export_tracks",
                    [
                        py,
                        scripts / "export_visdrone_vid_tracks.py",
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
                    ],
                    strict,
                )
            )
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


def resolve_dataset_yaml(path, project):
    """Materialize project-local dataset roots as absolute paths for Ultralytics."""
    data_path = Path(resolve_path(path))
    if data_path.suffix.lower() not in {".yaml", ".yml"} or not data_path.exists():
        return str(data_path)

    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = data.get("path")
    if not isinstance(dataset_root, str) or os.path.isabs(dataset_root):
        return str(data_path)

    project_dataset_root = Path(resolve_path(dataset_root))
    if not project_dataset_root.exists():
        return str(data_path)

    data["path"] = str(project_dataset_root.resolve())
    resolved_dir = Path(resolve_path(project))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = resolved_dir / f"{data_path.stem}.resolved.yaml"
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return str(resolved_yaml)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/VisDrone-VID.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml', help='model yaml path')
    parser.add_argument('--weights', type=str, default='', help='checkpoint path for val/test/predict/export or fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, predict or export')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--val_period', type=int, default=1, help='validate every N epochs during training')
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
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
    parser.add_argument('--num_ref_frames', type=int, default=4, help='reference frames per clip for VIDClipDataset (0 disables FAM)')
    parser.add_argument('--clip_stride', type=int, default=1, help='temporal stride between sampled refs')
    parser.add_argument('--ref_sample', default='uniform_local', choices=['uniform_local', 'uniform_global'], help='ref-frame sampling strategy')
    # Optional heavier video metrics, run after checkpoint save every N epochs.
    parser.add_argument('--extra_eval_period', type=int, default=0, help='run flicker/MOT video eval every N epochs; 0 disables')
    parser.add_argument('--extra_eval_official_root', default='datasets/VisDrone-VID/raw/VisDrone2019-VID-val', help='official VisDrone-VID split root with annotations/ and sequences/')
    parser.add_argument('--extra_eval_toolkit', default='third_party/VisDrone2018-VID-toolkit', help='deprecated; official AP/AR is no longer run by periodic extra eval')
    parser.add_argument('--extra_eval_tracker', default='ultralytics/cfg/trackers/bytetrack.yaml', help='tracker yaml for MOT export')
    parser.add_argument('--extra_eval_batch', type=int, default=16, help='batch size for detection export during extra eval')
    parser.add_argument('--extra_eval_conf', type=float, default=0.001, help='confidence threshold for detection export during extra eval')
    parser.add_argument('--extra_eval_track_conf', type=float, default=0.1, help='confidence threshold for tracking export')
    parser.add_argument('--extra_eval_iou', type=float, default=0.7, help='NMS IoU threshold for extra eval exports')
    parser.add_argument('--extra_eval_strict', action='store_true', help='fail training if an extra-eval step fails')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    from ultralytics import YOLO

    task = opt.task.lower()
    data = resolve_dataset_yaml(opt.data, opt.project)
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
    }
    model_path = resolve_path(opt.weights) if opt.weights else resolve_path(opt.config)
    model = YOLO(model_path)
    if task == "train":
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
