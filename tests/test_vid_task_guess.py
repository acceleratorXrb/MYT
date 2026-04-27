from pathlib import Path

from ultralytics.nn.tasks import guess_model_task, yaml_model_load


def test_detect_vid_yaml_guesses_detect_task():
    cfg = yaml_model_load(Path("ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml"))

    assert guess_model_task(cfg) == "detect"
