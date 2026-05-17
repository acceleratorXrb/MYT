# Model Variants

This directory stores experiment-stage records. A variant record captures the
model structure, key hyperparameters, known metrics, and the exact training
command used for that stage.

Use:

```bash
python tools/model_variant.py list
python tools/model_variant.py show track_tube_v7_2026-05-17
python tools/model_variant.py train-command track_tube_v7_2026-05-17
```

When a new architecture stage becomes important, add a new YAML file here
instead of overwriting an older one. This makes thesis ablations and historical
model rollback much easier.
