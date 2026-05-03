# YOLOV-Mamba-YOLO-T 在 VisDrone-VID 上的复现指南

本分支 (`codex/yolov-mamba-yolo-t-vid`) 在原版 Mamba-YOLO-T 基础上加入 YOLOV (AAAI'23) 风格的跨帧特征聚合 (FAM)，目标是降低帧间分类抖动，提升视频流目标识别的时序一致性。

---

## 1. 环境准备

```bash
# 与原版 Mamba-YOLO 相同
conda create -n mambayolo -y python=3.11
conda activate mambayolo
pip install torch===2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install seaborn thop timm einops scipy
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

## 2. 数据准备

VisDrone-VID 2019 视频版数据集（注意：不是 DET）。

```bash
# 推荐：setup.sh 自动下载 zip 到 datasets/VisDrone-VID/，并解压到 datasets/VisDrone-VID/raw/
bash setup.sh

# 转换到 YOLO 布局，输出到 datasets/VisDrone-VID/
python tools/prepare_visdrone_vid_yolo.py \
    --src datasets/VisDrone-VID/raw \
    --out datasets/VisDrone-VID \
    --splits train val test-dev
```

预期布局：

```
datasets/VisDrone-VID/
├── images/{train,val,test-dev}/<seq>/<frame>.jpg
└── labels/{train,val,test-dev}/<seq>/<frame>.txt
```

`ultralytics/cfg/datasets/VisDrone-VID.yaml` 中 `task: vid` 字段会触发 `VIDClipDataset` 自动按 `<seq>` 父目录分组、按帧号排序。

## 3. 训练

### 3.1 基线（无 FAM）

```bash
python mbyolo_train.py \
    --task train \
    --data ultralytics/cfg/datasets/VisDrone-VID.yaml \
    --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
    --epochs 100 \
    --val_period 1 \
    --batch_size 8 \
    --imgsz 640 \
    --amp \
    --num_ref_frames 0 \
    --project output_dir/visdrone_vid \
    --name baseline_t
```

> 注意：`Mamba-YOLO-T.yaml` 默认 `nc: 80`，VisDrone-VID 是 10 类。Ultralytics 在数据 yaml 含 `names` 时会自动覆盖 nc，无需手改。

### 3.2 YOLOV-Mamba-YOLO-T

```bash
python mbyolo_train.py \
    --task train \
    --data ultralytics/cfg/datasets/VisDrone-VID.yaml \
    --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml \
    --epochs 100 \
    --val_period 1 \
    --batch_size 2 \
    --imgsz 640 \
    --amp \
    --num_ref_frames 4 \
    --clip_stride 1 \
    --ref_sample adjacent \
    --project output_dir/visdrone_vid \
    --name yolov_t_R4
```

`--val_period 1` 表示每个 epoch 结束后都跑一次 Ultralytics 常规验证；
当前代码中它已经是默认值。这个常规验证走 converted YOLO val split，且验证时为单帧输入，
会产生 `precision(B)`、`recall(B)`、`mAP50(B)`、`mAP50-95(B)` 以及验证 loss。

如果希望训练过程中每个 epoch 自动额外跑一次视频级指标，可以在训练命令后追加：

```bash
    --extra_eval_period 1 \
    --extra_eval_official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val
```

额外评测会保存到 `output_dir/visdrone_vid/yolov_t_R4/extra_eval/epochXXX/`，包括
cls flicker、ByteTrack 轨迹导出和 MOT identity 指标；训练过程中不再自动运行 CPU-only
的 VisDrone 官方 AP/AR。

显存提示：`B=2, T=5 (1 key + 4 ref), imgsz=640` 在 24 GB 卡上约 14 GB。如果 OOM：

- 优先把 `batch_size` 从 2 调到 1，配合 Ultralytics `accumulate` 维持等效 batch
- 退而求其次降低 `--num_ref_frames` 到 2

### 3.3 关键超参

| 参数 | 含义 | 默认 | 推荐扫描范围 |
|---|---|---|---|
| `--num_ref_frames` | 每个 clip 的 ref 帧数 | 4 | {0, 4, 8, 16} |
| `--clip_stride` | ref 帧采样时间步长 | 1 | {1, 2, 4} |
| `--ref_sample` | ref 帧采样策略 | adjacent | {adjacent, uniform_local, uniform_global} |

## 4. 评估

### 4.1 VisDrone 官方 mAP

```bash
# 1) 导出预测到 VisDrone 格式
python tools/export_visdrone_vid_results.py \
    --weights output_dir/visdrone_vid/yolov_t_R4/weights/best.pt \
    --source datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
    --out output_dir/visdrone_vid/yolov_t_R4/predictions

# 2) 跑官方 evaluator
python tools/eval_visdrone_vid_official.py \
    --toolkit third_party/VisDrone2018-VID-toolkit \
    --official-root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
    --results output_dir/visdrone_vid/yolov_t_R4/predictions \
    --out output_dir/visdrone_vid/yolov_t_R4/official_eval
```

### 4.2 帧间分类一致性 (cls flicker)

基于 GT 轨迹的 macro flicker rate。

```bash
python tools/eval_visdrone_vid_cls_flicker.py \
    --gt datasets/VisDrone-VID/raw/VisDrone2019-VID-val/annotations \
    --pred output_dir/visdrone_vid/yolov_t_R4/predictions \
    --iou 0.5 \
    --out output_dir/visdrone_vid/yolov_t_R4/flicker.json
```

输出关键字段：`macro_flicker`（按 sequence 宏平均的连续帧 cls 变化率，越低越好）。

### 4.3 MOT identity 指标 (IDF1, IDP, IDR, IDS)

```bash
# 1) 跑 ByteTrack 关联得到预测轨迹
python tools/export_visdrone_vid_tracks.py \
    --weights output_dir/visdrone_vid/yolov_t_R4/weights/best.pt \
    --source datasets/VisDrone-VID/raw/VisDrone2019-VID-val/sequences \
    --out output_dir/visdrone_vid/yolov_t_R4/tracks \
    --tracker ultralytics/cfg/trackers/bytetrack.yaml

# 2) MOT eval
python tools/eval_visdrone_vid_mot.py \
    --gt datasets/VisDrone-VID/raw/VisDrone2019-VID-val/annotations \
    --pred output_dir/visdrone_vid/yolov_t_R4/tracks \
    --out output_dir/visdrone_vid/yolov_t_R4/mot.json
```

## 5. 验收门槛

对比基线 vs YOLOV-Mamba-YOLO-T，预期：

| 指标 | 期望变化 | 备注 |
|---|---|---|
| VisDrone mAP / mAP@0.5 | ≥ baseline | FAM 不应损害检测精度 |
| macro_flicker | 相对降低 ≥ 20% | 项目核心目标 |
| IDF1 / IDS | baseline ±1 | FAM 不直接帮助轨迹关联；当前脚本不计算 MOTA |
| FPS | ~ baseline -8% | FAM 聚合开销 |

## 6. 消融建议

```bash
# Sweep num_ref_frames
for N in 0 4 8 16; do
    python mbyolo_train.py \
        --task train --data ... --config ...-VID.yaml \
        --num_ref_frames $N \
        --name yolov_t_R${N}
done
```

`N=0` 退化为 FAM 恒等（α 由训练决定，但起点为 0）；用作 sanity check 验证训练流程在 clip 维度退化时与基线行为一致。

`adjacent` vs `uniform_local` vs `uniform_global` 对照：

```bash
python mbyolo_train.py ... --ref_sample adjacent --name yolov_t_adjacent
python mbyolo_train.py ... --ref_sample uniform_local --name yolov_t_local
python mbyolo_train.py ... --ref_sample uniform_global --name yolov_t_global
```

## 7. 实现要点速查

- FAM 模块: `ultralytics/nn/modules/yolov_fam.py`
- Detect_VID 头: `ultralytics/nn/modules/head.py` (cv3 拆为 pre + cls，按 P3/P4/P5 各挂一个 FAM)
- 视频 clip dataset: `ultralytics/data/vid_dataset.py` (key + N refs；mosaic/mixup/copy_paste 关闭，affine/flip/HSV 等单帧增广在 clip 内同步随机参数)
- 训练路由: `ultralytics/models/yolo/detect/train.py` 在 `preprocess_batch` 中读取 `clip_layout` 写到 head
- 损失路由: `ultralytics/nn/tasks.py:DetectionModel.init_criterion` 在 head 是 Detect_VID 时返回 `v8VIDDetectionLoss`
- α 冷启动：`FeatureAggregationModule` 初始化 `alpha=0`（首次前向 = 恒等）；如需 warmup 可调用 `set_alpha_warmup(model, target)` 在 epoch callback 里渐增

## 8. 已知限制

- 增广策略：clip 内为保持 key-ref 几何对应，仍关闭 mosaic/mixup/copy_paste；affine/perspective/flip/HSV/BGR 已恢复为 clip 内同步随机参数。相比单帧基线，clip-mosaic/clip-mixup 仍未实现
- Ref 帧标签：当前已支持 ref 辅助检测 loss，可通过 `--ref_aux_loss` 控制权重
- 时序长度：默认 `num_ref_frames=4` 是显存/精度的折中，论文复现建议在显存允许下扫到 8 或 16
