# Model Variants

This directory stores experiment-stage records. A variant record captures the
model structure, key hyperparameters, known metrics, and the exact training
command used for that stage.

Use:

```bash
python tools/model_variant.py list
python tools/model_variant.py show yolov_proposal_v2_2026-05-13
python tools/model_variant.py train-command yolov_proposal_v2_2026-05-13
```

When a new architecture stage becomes important, add a new YAML file here
instead of overwriting an older one. This makes thesis ablations and historical
model rollback much easier.

