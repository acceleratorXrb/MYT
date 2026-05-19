# 当前新模型说明：Mamba-YOLO-T-VID-VideoStable-v10

本文档用于说明当前仓库中正在作为主实验使用的新模型结构。当前版本标记为：

```text
Mamba-YOLO-T-VID-VideoStable-v10
```

对应模型记录文件：

```text
model_variants/video_stable_v10_2026-05-18.yaml
```

## 1. 设计目标

当前新模型的核心目标不是单纯提高单帧检测精度，而是让模型更适合视频目标检测任务，重点优化以下视频相关指标：

- `ID Switch`：同一个真实目标在跟踪过程中被分配成不同预测 ID 的次数，越低越好。
- `Frag`：同一目标轨迹被打断后重新出现的次数，越低越好。
- `macro_flicker / micro_flicker`：同一真实轨迹在连续帧中的预测类别抖动率，越低越好。
- `IDF1`：预测轨迹 ID 与真实轨迹 ID 的综合匹配质量，越高越好。

因此，新模型的设计重点是增强视频帧间稳定性，包括类别稳定、短时遮挡后的轨迹连续性、以及密集小目标场景下的 ID 一致性。

## 2. 与官方 Mamba-YOLO-T 的关系

当前新模型保留官方 Mamba-YOLO-T 的核心检测网络：

- 保留官方 Mamba-YOLO-T backbone。
- 保留官方 Mamba-YOLO-T neck / feature pyramid。
- 不修改 backbone 和 neck 中的 ODSSBlock / Mamba 相关结构。

新模型主要改动发生在以下部分：

- 视频输入组织方式。
- 检测头 `Detect_VID`。
- 训练阶段的视频轨迹一致性损失。
- 验证阶段的视频指标导出与时序稳定处理。

也就是说，当前新模型可以理解为：

```text
官方 Mamba-YOLO-T backbone + 官方 Mamba-YOLO-T neck + 视频时序检测头 + 视频指标稳定链路
```

## 3. 整体数据流程

当前训练采用 16 帧连续视频窗口作为输入。每个窗口中所有帧都作为 key frame 参与检测。

整体流程如下：

```text
16-frame video window
  -> 将 16 帧展开为普通图像 batch
  -> 官方 Mamba-YOLO-T backbone
  -> 官方 Mamba-YOLO-T neck / feature pyramid
  -> 得到每帧的 P3、P4、P5 多尺度特征
  -> Detect_VID 视频检测头
       -> TRFA 只作用于分类分支
       -> bbox 分支使用原始当前帧特征
  -> 每一帧输出检测框和类别分数
```

这种方式的特点是：每帧只通过 backbone 和 neck 一次，避免早期“中心帧 + 参考帧”方案中同一帧被多次重复计算的问题。

## 4. 输入窗口设计

当前主要超参为：

```bash
--vid_clip_mode window
--vid_window_size 16
--num_ref_frames 15
--clip_stride 1
--ref_sample adjacent
```

含义如下：

- `vid_clip_mode=window`：使用连续视频窗口训练。
- `vid_window_size=16`：每个训练样本包含 16 帧连续图像。
- `num_ref_frames=15`：对窗口内任意一帧来说，其他相邻帧都可以提供时序上下文。
- `clip_stride=1`：相邻帧连续采样，不跳帧。
- `ref_sample=adjacent`：参考信息来自相邻帧，符合 VisDrone-VID 中小目标相邻帧位移较小的特点。

## 5. Backbone 与 Neck

Backbone 和 neck 完全沿用官方 Mamba-YOLO-T。

输入视频窗口会先被展开为普通图像 batch：

```text
B 个视频窗口，每个窗口 T 帧
  -> reshape 为 B*T 张图像
  -> 按单帧方式送入 Mamba-YOLO-T backbone 和 neck
```

经过官方 backbone 和 neck 后，每帧得到三个尺度的特征：

```text
P3: 高分辨率特征，主要负责小目标
P4: 中等尺度特征
P5: 低分辨率语义特征，主要负责较大目标
```

由于 backbone 和 neck 不变，因此模型仍然保留 Mamba-YOLO-T 的主体特征表达能力。

## 6. Detect_VID 检测头结构

当前检测头名为：

```text
Detect_VID
```

它是在原始 YOLOv8 / Mamba-YOLO Detect head 基础上增加视频时序分支得到的。

检测头整体结构如下：

```text
P3/P4/P5 features for 16 frames
  -> TemporalResidualFeatureAdapter (TRFA)
  -> 分支 1：bbox branch
       使用原始当前帧特征
       输出 bbox distribution
  -> 分支 2：cls branch
       使用 TRFA 后的时序特征
       输出 class logits
  -> concat bbox + cls
  -> YOLO Detect output
```

当前关键设置为：

```bash
--temporal_fusion trfa
--trfa_levels all
--trfa_branch cls
```

含义如下：

- `temporal_fusion=trfa`：启用时序残差特征适配模块。
- `trfa_levels=all`：P3、P4、P5 三个尺度都使用时序适配。
- `trfa_branch=cls`：时序特征只进入分类分支，bbox 分支不使用时序特征。

## 7. TRFA 模块

TRFA 全称可以理解为：

```text
Temporal Residual Feature Adapter
```

它的作用是在检测头内部引入局部时序信息，使同一目标在相邻帧中的类别预测更加稳定。

TRFA 的基本结构为：

```text
输入特征
  -> 1x1 Conv 降维
  -> 3D depthwise temporal-spatial convolution
  -> 1x1 Conv 升维
  -> residual alpha gate
  -> 输出时序增强特征
```

其中 `alpha` 是一个残差门控参数。训练初期可以让时序分支影响较小，随后逐渐增强：

```bash
--trfa_warmup_epochs 5
--trfa_alpha_target 1.0
```

这样设计的原因是：如果训练一开始就让时序分支强烈影响检测头，容易破坏已有的单帧检测能力；通过 warmup 可以让模型逐步学习如何利用视频上下文。

## 8. 为什么只让 TRFA 作用于分类分支

当前设置为：

```bash
--trfa_branch cls
```

也就是说：

```text
bbox 分支：使用原始当前帧特征
cls 分支：使用 TRFA 时序增强特征
```

这样做的原因是：

- 视频指标中的 `flicker` 很大程度来自类别预测不稳定。
- `ID Switch` 也可能受到类别跳变影响，尤其是在密集小目标场景中。
- 但 bbox 对空间定位非常敏感，如果时序特征把相邻帧目标位置混入当前帧，可能导致框漂移。
- 框漂移会进一步导致 ByteTrack 关联错误，反而增加 `ID Switch` 和 `Frag`。

因此，当前模型采用更稳妥的结构：用时序信息稳定类别，不直接扰动 bbox 回归。

## 9. 训练阶段的视频监督

除了 YOLO 原始检测损失外，当前模型还利用 VisDrone-VID 标注中的 `track_id` 构造视频轨迹监督。

训练时主要包括：

```text
普通 YOLO 检测损失
  -> box loss
  -> cls loss
  -> dfl loss

视频轨迹一致性损失
  -> track_recall_loss
  -> track_consistency_loss
  -> track_cls_consistency_loss
```

对应命令参数：

```bash
--track_recall_loss 0.5
--track_consistency_loss 0.2
--track_cls_consistency_loss 0.1
```

三个视频损失的作用分别是：

### 9.1 track_recall_loss

在同一轨迹的目标中心位置采样分类 logits，鼓励该目标在每一帧都有较强的类别响应。

它主要用于减少短时漏检，有助于降低 `Frag`，并可能提升 `IDR`。

### 9.2 track_consistency_loss

同一 `track_id` 在 16 帧窗口内，如果某些帧的真实类别置信度明显低于其他帧，就把低置信度帧往高置信度帧拉近。

它主要用于减少轨迹中间的置信度断裂。

### 9.3 track_cls_consistency_loss

约束同一 `track_id` 在窗口内的完整类别概率分布保持一致。

它主要用于减少类别抖动，也就是降低 `flicker`。

## 10. 验证阶段的视频稳定链路

当前模型在每个 epoch 结束后会进行额外视频指标评估。

这部分不是官方 VisDrone AP/AR，而是本仓库用于分析视频稳定性的辅助指标：

- flicker
- MOT/ID
- IDF1
- ID Switch
- Frag
- FPS / detection export speed

当前 extra-eval 使用：

```bash
--extra_eval_clip_inference
--extra_eval_window_inference
--extra_eval_tracker ultralytics/cfg/trackers/bytetrack_vidstable.yaml
--extra_eval_track_conf 0.05
```

## 11. VID-stable ByteTrack

当前使用的 tracker 配置为：

```text
ultralytics/cfg/trackers/bytetrack_vidstable.yaml
```

它基于 ByteTrack，但针对 VisDrone-VID 视频指标进行了调整：

- 增大 `track_buffer`，允许目标短暂丢失后继续关联。
- 降低低置信度关联阈值，让弱检测也能帮助维持轨迹。
- 加入轻量 class-aware association，对类别不一致的匹配增加惩罚。

这样设计是为了减少密集小目标场景中的 ID 交换和短碎轨。

## 12. GT-free 时序稳定器

当前验证导出阶段还加入了一个不使用 GT 的时序稳定器：

```text
tools/visdrone_temporal_stabilize.py
```

它只使用模型预测结果本身，不读取标注文件，因此不是用 GT 修结果。

稳定器主要包含以下操作：

```text
检测结果
  -> 高 IoU 短轨迹类别平滑
  -> 用于 flicker 评估

跟踪结果
  -> 同一 track 内类别投票平滑
  -> 短 gap 中的桥接轨迹吸收
  -> 严格空间约束下的碎轨重连
  -> 高重叠一帧缺口插值
  -> 用于 MOT/ID 评估
```

这些操作都带有严格空间约束，主要依赖：

- 相邻帧检测框 IoU。
- 中心点距离。
- 框面积比例。
- 类别一致性。
- 时间 gap 长度。

它的目标是减少明显的视频不连续现象，而不是大量新增检测框。

## 13. 输出与指标计算流程

每个 epoch 后的视频指标流程为：

```text
训练得到 last.pt
  -> clip/window inference 导出检测结果
  -> temporal stabilize detection classes
  -> eval_visdrone_vid_cls_flicker.py
       -> macro_flicker / micro_flicker

训练得到 last.pt
  -> clip/window inference
  -> VID-stable ByteTrack
  -> temporal stabilize tracks
  -> eval_visdrone_vid_mot.py
       -> IDF1 / IDP / IDR / ID Switch / Frag
```

训练日志中如果看到类似下面的字段，说明视频稳定链路正在生效：

```text
cls_changes
absorbs
links
fills
dup_drops
```

含义如下：

- `cls_changes`：稳定器修改了多少类别预测。
- `absorbs`：吸收了多少短 gap 中的桥接轨迹。
- `links`：重连了多少短碎轨。
- `fills`：补了多少一帧短缺口。
- `dup_drops`：合并轨迹后删除了多少同帧重复输出。

## 14. 当前完整训练命令

```bash
python mbyolo_train.py \
  --task train \
  --data ultralytics/cfg/datasets/VisDrone-VID.yaml \
  --data_task vid \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-VID.yaml \
  --init_weights output_dir/visdrone_vid/baseline/weights/best.pt \
  --epochs 30 \
  --val_period 1 \
  --batch_size 2 \
  --imgsz 640 \
  --workers 8 \
  --device 0 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight_decay 0.0005 \
  --warmup_epochs 3.0 \
  --warmup_momentum 0.8 \
  --warmup_bias_lr 0.0005 \
  --box 7.5 \
  --cls 0.5 \
  --dfl 1.5 \
  --hsv_h 0.015 \
  --hsv_s 0.7 \
  --hsv_v 0.4 \
  --mosaic 0.0 \
  --mixup 0.0 \
  --fliplr 0.5 \
  --flipud 0.0 \
  --degrees 0.0 \
  --translate 0.1 \
  --scale 0.5 \
  --shear 0.0 \
  --perspective 0.0 \
  --vid_clip_mode window \
  --vid_window_size 16 \
  --num_ref_frames 15 \
  --clip_stride 1 \
  --ref_sample adjacent \
  --ref_aux_loss 0.0 \
  --track_recall_loss 0.5 \
  --track_consistency_loss 0.2 \
  --track_cls_consistency_loss 0.1 \
  --temporal_fusion trfa \
  --trfa_levels all \
  --trfa_branch cls \
  --trfa_warmup_epochs 5 \
  --trfa_alpha_target 1.0 \
  --extra_eval_period 1 \
  --extra_eval_clip_inference \
  --extra_eval_window_inference \
  --extra_eval_tracker ultralytics/cfg/trackers/bytetrack_vidstable.yaml \
  --extra_eval_track_conf 0.05 \
  --extra_eval_official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
  --project output_dir/visdrone_vid \
  --name video_stable_v10
```

## 15. 消融实验建议

为了证明各部分贡献，可以做以下消融：

### 15.1 关闭时序检测头

```bash
--temporal_fusion none
```

用于比较 TRFA 是否带来视频稳定性提升。

### 15.2 关闭 GT-free 稳定器

```bash
--extra_eval_no_temporal_stabilize
```

用于证明导出阶段时序稳定器对 `flicker / ID Switch / Frag` 的影响。

### 15.3 使用普通 ByteTrack

```bash
--extra_eval_tracker ultralytics/cfg/trackers/bytetrack.yaml
```

用于比较 VID-stable ByteTrack 配置是否减少 ID Switch 和 Frag。

### 15.4 只保留分类分支时序融合

当前主模型已经是该设置：

```bash
--trfa_branch cls
```

可与以下设置对比：

```bash
--trfa_branch both
--trfa_branch none
```

用于说明为什么 bbox 分支保持单帧特征更稳定。

## 16. 当前模型总结

当前新模型可以概括为：

```text
Mamba-YOLO-T-VID-VideoStable-v10
  = 官方 Mamba-YOLO-T backbone
  + 官方 Mamba-YOLO-T neck
  + Detect_VID 分类分支时序残差适配器
  + 基于 track_id 的视频一致性训练损失
  + VID-stable ByteTrack
  + GT-free temporal stabilization
```

它的主要创新点在于：不改变 Mamba-YOLO-T 主干和颈部结构，而是在检测头和视频评估链路中引入面向视频连续性的设计，使模型更适合 VisDrone-VID 这类连续帧小目标视频检测任务。
