from pathlib import Path

import yaml

from tools.prepare_visdrone_vid_yolo import write_dataset_yaml


def test_generated_visdrone_vid_yaml_enables_vid_task(tmp_path: Path):
    yaml_path = tmp_path / "VisDrone-VID.local.yaml"
    write_dataset_yaml(yaml_path, tmp_path / "VisDrone-VID")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert data["task"] == "vid"
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test-dev"
