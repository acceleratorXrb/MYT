# Mamba-YOLO-T-VID for VisDrone Video Detection

本仓库基于官方 Mamba-YOLO 和 Ultralytics YOLOv8，面向 VisDrone-VID
无人机视频目标检测任务做视频化改造。当前主线要求是：

- 保持官方 Mamba-YOLO-T 的 backbone 和 neck，不破坏其中的 ODSS/VSS/XSS 结构。
- 将视频输入、检测头、时序 proposal 融合、离线视频评估流程逐步向 YOLOV 靠齐。
- 同时评估单帧检测指标、分类 flicker、一致性、MOT/ID 等视频指标。

当前主模型记录在 `CURRENT_MODEL_STRUCTURE.md`，历史模型版本记录在
`model_variants/`。

## 当前主模型

当前主模型为：

```text
Mamba-YOLO-T-VID-YOLOV-Proposal-v2
```

整体结构：

```text
16-frame video window
  -> official Mamba-YOLO-T backbone
  -> official Mamba-YOLO-T neck / feature pyramid
  -> P3/P4/P5 features
  -> Detect_VID
      - YOLOv8-style bbox regression branch
      - raw classification branch
      - YOLOV-style two-stage ProposalTemporalRefiner
      - refined class logits
  -> offline VID detections / flicker / MOT-ID metrics
```

最新记录的模型版本：

```bash
python tools/model_variant.py list
python tools/model_variant.py show yolov_proposal_v2_2026-05-13
python tools/model_variant.py train-command yolov_proposal_v2_2026-05-13
```

## 快速训练

服务器上推荐从项目目录运行：

```bash
cd /root/autodl-tmp/MYT
source .venv/bin/activate
git pull

python tools/model_variant.py train-command yolov_proposal_v2_2026-05-13 \
  --name yolov_proposal_v2
```

如果需要直接执行完整命令，可以从上面的 `train-command` 输出复制运行。

## 重要文件说明

### 项目入口

| 文件 | 作用 |
| --- | --- |
| `mbyolo_train.py` | 训练、验证、预测、导出的统一入口。新增了 VID window 输入、时序融合参数、额外视频指标评估回调。 |
| `setup.sh` | 服务器环境准备脚本，安装系统依赖、Python 虚拟环境依赖、VisDrone 官方工具等。 |
| `RUN_VISDRONE_VID.md` | VisDrone-VID 数据准备、训练、额外评估的运行说明。 |
| `RUN_VISDRONE_VID_YOLOV.md` | 当前 YOLOV-style 视频模型相关运行说明。 |
| `CURRENT_MODEL_STRUCTURE.md` | 当前主实验模型结构标记文件，用于论文描述和实验复现。 |
| `MODEL_HISTORY.md` | 历史模型版本索引，记录每个阶段模型对应的 YAML 文件。 |

### 模型与配置

| 路径 | 作用 |
| --- | --- |
| `ultralytics/cfg/models/mamba-yolo/` | Mamba-YOLO 模型 YAML。当前 VID 主模型使用 `Mamba-YOLO-T-VID.yaml`。 |
| `ultralytics/cfg/datasets/VisDrone-VID.yaml` | VisDrone-VID 数据集配置。 |
| `ultralytics/cfg/default.yaml` | Ultralytics 默认训练参数，已加入 VID 和 proposal temporal fusion 参数。 |
| `ultralytics/nn/modules/head.py` | 检测头定义。`Detect_VID` 在这里实现，是当前视频模型的核心入口。 |
| `ultralytics/nn/modules/yolov_fam.py` | 视频时序融合模块，包括 FAM 和 YOLOV-style `ProposalTemporalRefiner`。 |
| `ultralytics/models/yolo/detect/train.py` | 训练时把 VID clip layout、时序融合参数传入 `Detect_VID`。 |
| `ultralytics/utils/loss.py` | YOLO 检测损失和 VID 辅助损失，包括 YOLOV-style refined cls auxiliary loss。 |

### 模型版本记录

| 路径 | 作用 |
| --- | --- |
| `model_variants/README.md` | 模型版本记录目录说明。 |
| `model_variants/yolov_proposal_v2_2026-05-13.yaml` | 当前主模型的完整记录：结构、关键超参、指标、训练命令。 |
| `tools/model_variant.py` | 模型版本管理小工具，可列出版本、查看 YAML、打印训练命令。 |

后续每次出现重要结构改动时，应该新增一个 `model_variants/*.yaml`，不要覆盖旧版本。

### VisDrone 数据工具

| 文件 | 作用 |
| --- | --- |
| `tools/download_visdrone_vid_zips.py` | 下载 VisDrone-VID 数据压缩包。 |
| `tools/prepare_visdrone_vid_yolo.py` | 将 VisDrone-VID 转成 YOLO/VID 训练所需目录与标签格式。 |
| `tools/verify_visdrone_vid_source.py` | 检查原始 VisDrone-VID 数据目录是否完整。 |
| `tools/check_visdrone_vid_runtime.py` | 检查训练和评估所需数据、路径、依赖是否可用。 |

### 检测结果导出

| 文件 | 作用 |
| --- | --- |
| `tools/export_visdrone_vid_results.py` | 单帧检测导出，适合 baseline Mamba-YOLO 或 YOLOv8。 |
| `tools/export_visdrone_vid_clip_results.py` | 离线 clip/window 检测导出，适合当前 VID 新模型。 |
| `tools/export_visdrone_vid_tracks.py` | 单帧检测 + ByteTrack 的轨迹导出。 |
| `tools/export_visdrone_vid_clip_tracks.py` | 离线 clip/window 检测 + ByteTrack 的轨迹导出，当前 MOT/ID 推荐使用这个。 |
| `tools/temporal_state.py` | 视频序列切换时重置模型时序状态的辅助工具。 |

### 指标评估

| 文件 | 作用 |
| --- | --- |
| `tools/eval_visdrone_vid_cls_flicker.py` | 计算分类抖动指标，包括 `macro_flicker` 和 `micro_flicker`。 |
| `tools/eval_visdrone_vid_mot.py` | 计算当前自实现 MOT/ID 指标，包括 IDF1、IDP、IDR、ID Switches、Frag。 |
| `tools/eval_visdrone_vid_official.py` | VisDrone 官方 AP/AR 工具封装，当前周期评估默认不跑官方 AP/AR。 |
| `tools/run_visdrone_vid_official_eval.py` | 手动运行官方 VisDrone VID 评估流程。 |

### 可视化对比工具

| 文件 | 作用 |
| --- | --- |
| `tools/run_visdrone_comparison_examples.py` | 一键导出 baseline 和新模型结果，并筛选新模型更优的可视化图片。 |
| `tools/select_visdrone_comparison_examples.py` | 根据 GT、baseline、新模型预测结果自动挑选论文展示样例。 |
| `asserts/` | 存放论文或 README 用图片，包括模型结构图、检测头流程图、ODSSBlock 示意图等。 |

## 当前主模型关键参数

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--temporal_fusion yolov
--yolov_cls_loss 0.30
--proposal_topk 700
--proposal_after_topk 220
--proposal_nms_radius 1
--proposal_spatial_sigma 0.12
--proposal_time_sigma 4.0
--proposal_loc_gain 0.5
--proposal_cls_sim_gain 0.55
--proposal_vote_gain 0.50
--proposal_recall_gain 1.25
--proposal_recall_radius 1
```

这些参数共同定义当前 `Mamba-YOLO-T-VID-YOLOV-Proposal-v2`。如果这些参数明显变化，应视为新的实验变体，并新增一个 `model_variants/*.yaml` 记录。

## 环境与依赖

推荐在服务器上执行：

```bash
bash setup.sh
source .venv/bin/activate
```

如果出现 `No module named cv2`，通常是当前 shell 没进入 `.venv`，或者虚拟环境依赖没有安装完整。

## 论文实验建议

推荐至少保留以下对比：

- 官方 Mamba-YOLO-T baseline：单帧模型。
- 官方 YOLOv8 baseline：单帧 YOLOv8。
- 当前新模型：Mamba-YOLO-T backbone/neck + YOLOV-style video head。
- Ablation：关闭 `proposal_vote_gain`、关闭 `proposal_recall_gain`、关闭 `proposal_after_topk/nms/time/loc` 等。

当前新模型的主要优势是更高 IDP、更少 ID Switches、更少 Frag；主要短板仍是 IDR 和召回。

## Acknowledgement

This repository is based on:

- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOV](https://github.com/YuHengsss/YOLOV)
- [VMamba selective scan](https://github.com/MzeroMiko/VMamba)

## Citation

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
      title={Mamba YOLO: SSMs-Based YOLO For Object Detection},
      author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
      year={2024},
      eprint={2406.05835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.05835},
}
```

