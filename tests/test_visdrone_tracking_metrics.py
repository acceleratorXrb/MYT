import math

import numpy as np


def test_parse_visdrone_annotations_keeps_track_ids_and_filters_ignored_categories(tmp_path):
    from ultralytics.utils.visdrone_tracking_metrics import parse_visdrone_annotations

    annotation = tmp_path / "uav000001.txt"
    annotation.write_text(
        "\n".join(
            [
                "1,7,10,20,30,40,1,1,0,0",
                "1,8,10,20,30,40,1,0,0,0",
                "2,7,11,21,30,40,0,1,0,0",
                "2,7,12,22,30,40,1,4,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frames = parse_visdrone_annotations(annotation)

    assert [item.track_id for item in frames[1]] == [7]
    assert frames[1][0].cls == 0
    assert frames[1][0].xyxy == (10.0, 20.0, 40.0, 60.0)
    assert [item.cls for item in frames[2]] == [3]


def test_tracking_metrics_are_perfect_when_identity_is_stable():
    from ultralytics.utils.visdrone_tracking_metrics import TrackingBox, compute_tracking_metrics

    ground_truth = {
        "seq": {
            1: [TrackingBox(track_id=1, cls=0, xyxy=(0.0, 0.0, 10.0, 10.0))],
            2: [TrackingBox(track_id=1, cls=0, xyxy=(1.0, 0.0, 11.0, 10.0))],
            3: [TrackingBox(track_id=1, cls=0, xyxy=(2.0, 0.0, 12.0, 10.0))],
        }
    }
    predictions = {
        "seq": {
            1: [TrackingBox(track_id=10, cls=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.9)],
            2: [TrackingBox(track_id=10, cls=0, xyxy=(1.0, 0.0, 11.0, 10.0), score=0.9)],
            3: [TrackingBox(track_id=10, cls=0, xyxy=(2.0, 0.0, 12.0, 10.0), score=0.9)],
        }
    }

    metrics = compute_tracking_metrics(ground_truth, predictions, iou_threshold=0.5)

    assert metrics["metrics/IDF1"] == 1.0
    assert metrics["metrics/IDSwitches"] == 0
    assert metrics["metrics/Frag"] == 0


def test_tracking_metrics_count_identity_switches_and_fragments():
    from ultralytics.utils.visdrone_tracking_metrics import TrackingBox, compute_tracking_metrics

    ground_truth = {
        "seq": {
            1: [TrackingBox(track_id=1, cls=0, xyxy=(0.0, 0.0, 10.0, 10.0))],
            2: [TrackingBox(track_id=1, cls=0, xyxy=(1.0, 0.0, 11.0, 10.0))],
            3: [TrackingBox(track_id=1, cls=0, xyxy=(2.0, 0.0, 12.0, 10.0))],
            4: [TrackingBox(track_id=1, cls=0, xyxy=(3.0, 0.0, 13.0, 10.0))],
        }
    }
    predictions = {
        "seq": {
            1: [TrackingBox(track_id=10, cls=0, xyxy=(0.0, 0.0, 10.0, 10.0), score=0.9)],
            3: [TrackingBox(track_id=11, cls=0, xyxy=(2.0, 0.0, 12.0, 10.0), score=0.9)],
            4: [TrackingBox(track_id=11, cls=0, xyxy=(3.0, 0.0, 13.0, 10.0), score=0.9)],
        }
    }

    metrics = compute_tracking_metrics(ground_truth, predictions, iou_threshold=0.5)

    assert math.isclose(metrics["metrics/IDF1"], 4 / 7)
    assert metrics["metrics/IDSwitches"] == 1
    assert metrics["metrics/Frag"] == 1


def test_make_tracker_inputs_returns_empty_arrays_for_frames_without_predictions():
    from ultralytics.utils.visdrone_tracking_metrics import TrackerInput, make_tracker_input

    empty = make_tracker_input(np.zeros((0, 6), dtype=np.float32))

    assert isinstance(empty, TrackerInput)
    assert empty.xywh.shape == (0, 4)
    assert empty.conf.shape == (0,)
    assert empty.cls.shape == (0,)


def test_detection_validator_metric_keys_include_tracking_keys_when_configured():
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.visdrone_tracking_metrics import TRACKING_METRIC_KEYS

    validator = DetectionValidator(args={"model": "yolov8n.pt", "data": "coco8.yaml"})
    validator.tracking_metric_keys = TRACKING_METRIC_KEYS

    assert validator.metric_keys[-3:] == TRACKING_METRIC_KEYS
