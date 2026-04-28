from types import SimpleNamespace

import torch

from ultralytics.engine.validator import BaseValidator


class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = [batch]

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


class _DummyModel:
    def eval(self):
        return self

    def half(self):
        return self

    def float(self):
        return self

    def __call__(self, img, augment=False):
        return img

    def loss(self, batch, preds):
        return None, torch.zeros(1)


class _ModelAwareValidator(BaseValidator):
    def __init__(self, *args, expected_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_model = expected_model

    def preprocess(self, batch):
        assert self.model is self.expected_model
        return batch

    def get_desc(self):
        return "test"

    def get_stats(self):
        return {"fitness": 0.0}


def test_training_validation_binds_active_model_before_preprocess(tmp_path):
    model = _DummyModel()
    batch = {"img": torch.zeros(1, 3, 4, 4)}
    validator = _ModelAwareValidator(
        dataloader=_OneBatchLoader(batch),
        save_dir=tmp_path,
        args={"save_txt": False, "conf": 0.001, "imgsz": 4},
        expected_model=model,
    )
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        data={},
        ema=SimpleNamespace(ema=None),
        model=model,
        loss_items=torch.zeros(1),
        stopper=SimpleNamespace(possible_stop=False),
        epoch=0,
        epochs=1,
        label_loss_items=lambda loss, prefix: {},
    )

    validator(trainer)
