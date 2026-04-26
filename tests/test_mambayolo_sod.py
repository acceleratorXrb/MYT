from pathlib import Path

import torch
import yaml

from ultralytics.utils import ROOT


def test_mambayolo_sod_yaml_is_detect_task():
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    cfg = yaml_model_load(ROOT / "cfg" / "models" / "mamba-yolo" / "Mamba-YOLO-T-SOD.yaml")

    assert guess_model_task(cfg) == "detect"


def test_mambayolo_sod_yaml_adds_p2_detection_branch():
    yaml_path = ROOT / "cfg" / "models" / "mamba-yolo" / "Mamba-YOLO-T-SOD.yaml"

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    layers = data["backbone"] + data["head"]
    detect_layer = layers[-1]

    assert detect_layer[2] == "LightP2Detect"
    assert len(detect_layer[0]) == 4
    assert any(layer[2] == "SmallObjectStateGate" for layer in layers)


def test_small_object_state_gate_projects_concat_features_to_p2_channels():
    from ultralytics.nn.modules import SmallObjectStateGate

    module = SmallObjectStateGate(96, 32)
    x = torch.randn(2, 96, 20, 20)

    y = module(x)

    assert y.shape == (2, 32, 20, 20)
