from pathlib import Path

from PIL import Image


def _write_image(path: Path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(127, 127, 127)).save(path)


def test_prepare_visdrone_preserves_track_ids(tmp_path):
    from tools.prepare_visdrone_vid_yolo import convert_split

    raw = tmp_path / "raw"
    seq = raw / "VisDrone2019-VID-train" / "sequences" / "uav000001"
    ann = raw / "VisDrone2019-VID-train" / "annotations"
    _write_image(seq / "0000001.jpg")
    _write_image(seq / "0000002.jpg")
    ann.mkdir(parents=True)
    ann.joinpath("uav000001.txt").write_text(
        "\n".join(
            [
                "1,42,10,20,30,20,1,4,0,0",
                "1,99,1,1,10,10,0,4,0,0",
                "2,42,12,20,30,20,1,4,0,0",
                "2,7,40,30,10,10,1,11,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "yolo"
    images, labels = convert_split(raw, out, "train", copy=True, overwrite=True)

    assert images == 2
    assert labels == 2
    yolo_lines = (out / "labels" / "train" / "uav000001" / "0000001.txt").read_text().splitlines()
    assert len(yolo_lines) == 1
    assert len(yolo_lines[0].split()) == 5

    track_file = out / "tracks" / "train" / "uav000001.txt"
    assert track_file.exists()
    track_lines = track_file.read_text(encoding="utf-8").splitlines()
    assert track_lines == [
        "1,42,3,0.250000,0.375000,0.300000,0.250000",
        "2,42,3,0.270000,0.375000,0.300000,0.250000",
    ]

    manifest = out / "tracks" / "train.jsonl"
    assert '"sequence_id": "uav000001"' in manifest.read_text(encoding="utf-8")
    assert '"track_id": 42' in manifest.read_text(encoding="utf-8")


def test_mot_metrics_count_id_switches_and_fragments(tmp_path):
    from tools.eval_visdrone_vid_mot import evaluate_mot

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    gt_dir.joinpath("uav000001.txt").write_text(
        "\n".join(
            [
                "1,1,10,10,10,10,1,4,0,0",
                "2,1,10,10,10,10,1,4,0,0",
                "3,1,10,10,10,10,1,4,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pred_dir.joinpath("uav000001.txt").write_text(
        "\n".join(
            [
                "1,10,10,10,10,10,0.9,4,-1,-1",
                "2,11,10,10,10,10,0.9,4,-1,-1",
                "3,10,10,10,10,10,0.9,4,-1,-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_mot(gt_dir, pred_dir, iou_threshold=0.5)

    assert metrics["ID Switches"] == 2
    assert metrics["Frag"] == 0
    assert 0.0 < metrics["IDF1"] < 1.0
    assert metrics["IDTP"] == 1


def test_mot_metrics_count_fragments_after_gap(tmp_path):
    from tools.eval_visdrone_vid_mot import evaluate_mot

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    gt_dir.joinpath("uav000001.txt").write_text(
        "1,1,10,10,10,10,1,4,0,0\n2,1,10,10,10,10,1,4,0,0\n3,1,10,10,10,10,1,4,0,0\n",
        encoding="utf-8",
    )
    pred_dir.joinpath("uav000001.txt").write_text(
        "1,10,10,10,10,10,0.9,4,-1,-1\n3,10,10,10,10,10,0.9,4,-1,-1\n",
        encoding="utf-8",
    )

    metrics = evaluate_mot(gt_dir, pred_dir, iou_threshold=0.5)

    assert metrics["ID Switches"] == 0
    assert metrics["Frag"] == 1
    assert metrics["IDFN"] == 1


def test_track_export_formats_non_negative_track_ids():
    from tools.export_visdrone_vid_tracks import format_track_line

    line = format_track_line(frame_id=7, track_id=123, xyxy=(10, 20, 30, 45), score=0.75, cls=3, width=100, height=80)

    assert line == "7,123,10.00,20.00,20.00,25.00,0.750000,4,-1,-1\n"


def test_dataset_attaches_track_ids_from_track_sidecar(tmp_path):
    import numpy as np

    from ultralytics.data.dataset import _attach_track_ids

    image = tmp_path / "images" / "train" / "uav000001" / "0000001.jpg"
    image.parent.mkdir(parents=True)
    track_file = tmp_path / "tracks" / "train" / "uav000001.txt"
    track_file.parent.mkdir(parents=True)
    track_file.write_text("1,42,3,0.250000,0.375000,0.300000,0.250000\n", encoding="utf-8")
    labels = [
        {
            "im_file": str(image),
            "cls": np.array([[3]], dtype=np.float32),
            "bboxes": np.array([[0.25, 0.375, 0.3, 0.25]], dtype=np.float32),
        }
    ]

    attached = _attach_track_ids(labels)

    assert attached[0]["track_ids"].tolist() == [42]


def test_botsort_class_aware_distance_blocks_cross_class_matches():
    from types import SimpleNamespace

    import numpy as np

    from ultralytics.trackers.bot_sort import BOTSORT, BOTrack

    args = SimpleNamespace(
        track_buffer=30,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        gmc_method="none",
        match_thresh=0.8,
        class_aware=True,
    )
    tracker = BOTSORT(args=args, frame_rate=30)
    track = BOTrack(np.array([5, 5, 10, 10, 0], dtype=np.float32), 0.9, 3)
    same_class = BOTrack(np.array([5, 5, 10, 10, 0], dtype=np.float32), 0.9, 3)
    other_class = BOTrack(np.array([5, 5, 10, 10, 1], dtype=np.float32), 0.9, 4)

    dists = tracker.get_dists([track], [same_class, other_class])

    assert dists[0, 0] < 1.0
    assert dists[0, 1] == 1.0
