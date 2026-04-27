import argparse
import os
from pathlib import Path

import yaml

ROOT = os.path.abspath('.') + "/"


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


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
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model yaml path')
    parser.add_argument('--weights', type=str, default='', help='checkpoint path for val/test/predict/export or fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, predict or export')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default='output_dir/visdrone_vid', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_t', help='save to project/name')
    parser.add_argument('--source', type=str, default='', help='source path for predict')
    parser.add_argument('--format', type=str, default='onnx', help='export format')
    parser.add_argument(
        '--tracker',
        default='ultralytics/cfg/trackers/mambayolo_visdrone_track.yaml',
        help='tracker yaml path',
    )
    parser.add_argument('--official_root', type=str, default='', help='official VisDrone split root for track_export')
    parser.add_argument('--results', type=str, default='', help='tracking results directory for mot_eval')
    parser.add_argument('--out', type=str, default='', help='output path/directory for tracking tasks')
    parser.add_argument('--conf', type=float, default=0.1, help='confidence threshold for tracking/export')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS or MOT matching')
    parser.add_argument('--save_txt', action='store_true', help='save prediction txt labels')
    parser.add_argument('--save_conf', action='store_true', help='save prediction confidences')
    parser.add_argument('--resume', action='store_true', help='resume training from the last checkpoint')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    task = opt.task.lower()
    if task == "mot_eval":
        from tools.eval_visdrone_vid_mot import evaluate_mot

        gt_root = opt.official_root or opt.source
        if not gt_root:
            raise ValueError("--official_root or --source is required for task=mot_eval")
        results = opt.results or opt.out
        if not results:
            raise ValueError("--results or --out is required for task=mot_eval")
        gt_root = Path(resolve_path(gt_root))
        gt_dir = gt_root / "annotations" if (gt_root / "annotations").is_dir() else gt_root
        metrics = evaluate_mot(gt_dir, Path(resolve_path(results)), iou_threshold=opt.iou)
        out = (
            Path(resolve_path(opt.out))
            if opt.out
            else Path(resolve_path(opt.project)) / f"{opt.name}_mot_metrics.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps({"metrics": metrics}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"metrics": metrics, "out": str(out)}, indent=2))
        raise SystemExit(0)
    if task == "track_export":
        from tools.export_visdrone_vid_tracks import export_tracks

        if not opt.weights:
            raise ValueError("--weights is required for task=track_export")
        source = opt.official_root or opt.source
        if not source:
            raise ValueError("--official_root or --source is required for task=track_export")
        out = opt.out or os.path.join(opt.project, f"{opt.name}_tracks")
        export_tracks(
            weights=Path(resolve_path(opt.weights)),
            source=Path(resolve_path(source)),
            out=Path(resolve_path(out)),
            tracker=Path(resolve_path(opt.tracker)),
            imgsz=opt.imgsz,
            device=opt.device,
            conf=opt.conf,
            iou=opt.iou,
        )
        raise SystemExit(0)

    from ultralytics import YOLO

    data = resolve_dataset_yaml(opt.data, opt.project)
    args = {
        "data": data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": resolve_path(opt.project),
        "name": opt.name,
    }
    model_path = resolve_path(opt.weights) if opt.weights else resolve_path(opt.config)
    model = YOLO(model_path)
    if task in {"train", "train_track"}:
        args["resume"] = opt.resume
        if task == "train_track":
            args["pretrained"] = False
            args["tracker"] = resolve_path(opt.tracker)
            args.update(
                {
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "copy_paste": 0.0,
                    "degrees": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "fliplr": 0.0,
                    "flipud": 0.0,
                }
            )
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
        raise ValueError(
            f"Unsupported task: {opt.task}. Expected one of: train, train_track, val, test, predict, export, "
            "track_export, mot_eval."
        )
