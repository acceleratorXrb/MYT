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
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml', help='model yaml path')
    parser.add_argument('--weights', type=str, default='', help='checkpoint path for val/test/predict/export or fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, predict or export')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--val_period', type=int, default=10, help='validate every N epochs during training')
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
