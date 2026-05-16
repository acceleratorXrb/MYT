# Model Variants

This directory stores experiment-stage records. A variant record captures the
model structure, key hyperparameters, known metrics, and the exact training
command used for that stage.

Use:

```bash
python tools/model_variant.py list
python tools/model_variant.py show temporal_residual_v6_2026-05-16
python tools/model_variant.py train-command temporal_residual_v6_2026-05-16
```

When a new architecture stage becomes important, add a new YAML file here
instead of overwriting an older one. This makes thesis ablations and historical
model rollback much easier.
