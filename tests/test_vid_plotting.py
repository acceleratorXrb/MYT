import torch

from ultralytics.models.yolo.detect.train import _plot_images_for_training_batch


def test_plot_images_for_training_batch_uses_key_frames_for_clip_batch():
    batch = {
        "img": torch.arange(2 * 3 * 1 * 2 * 2).reshape(2 * 3, 1, 2, 2),
        "clip_layout": torch.tensor([2, 3]),
    }

    images = _plot_images_for_training_batch(batch)

    assert images.shape == (2, 1, 2, 2)
    assert torch.equal(images[0], batch["img"][0])
    assert torch.equal(images[1], batch["img"][3])


def test_plot_images_for_training_batch_leaves_regular_batch_unchanged():
    batch = {"img": torch.zeros(2, 3, 4, 4)}

    assert _plot_images_for_training_batch(batch) is batch["img"]
