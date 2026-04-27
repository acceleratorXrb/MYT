# 在 VisDrone2019-VID 上运行 Mamba-YOLO

本仓库的主体是图像检测框架。针对 VisDrone2019-VID，需要先把视频序列展开成 Ultralytics YOLO 检测数据集，同时保留序列子目录结构。

## 1. 准备数据

从 VisDrone 官方数据集页面下载 VisDrone2019-VID 的 train、val 和 test-dev 三个 split，然后放到同一个根目录下，例如：

```text
/mnt/datasets/VisDrone2019-VID-raw/
  VisDrone2019-VID-train/
    sequences/
    annotations/
  VisDrone2019-VID-val/
    sequences/
    annotations/
  VisDrone2019-VID-test-dev/
    sequences/
    annotations/
```

转换成 YOLO 数据布局：

```bash
python tools/prepare_visdrone_vid_yolo.py \
  --src /mnt/datasets/VisDrone2019-VID-raw \
  --out /mnt/datasets/VisDrone2019-VID-YOLO \
  --yaml output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --overwrite
```

脚本会生成如下目录结构：

```text
/mnt/datasets/VisDrone2019-VID-YOLO/
  images/train/<sequence>/<frame>.jpg
  labels/train/<sequence>/<frame>.txt
  images/val/<sequence>/<frame>.jpg
  labels/val/<sequence>/<frame>.txt
  images/test-dev/<sequence>/<frame>.jpg
  labels/test-dev/<sequence>/<frame>.txt
```

默认情况下，图像会以符号链接方式写入，而不是复制文件。如果训练环境不支持符号链接，请添加 `--copy`。

如果转换后的数据集根目录不是 `/mnt/datasets/VisDrone2019-VID-YOLO`，训练时请使用生成的 `output_dir/visdrone_vid/VisDrone-VID.local.yaml`。

## 2. 安装环境

推荐环境为 Python 3.11、PyTorch 2.3.0、CUDA 12.x：

```bash
pip install torch==2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

`selective_scan` 会构建 CUDA 扩展，因此正式训练需要支持 CUDA 的机器。

同样的安装流程也封装在：

```bash
bash scripts/setup_mambayolo_cuda121.sh
```

在新机器上克隆本仓库后，可以使用一键入口：

```bash
bash setup.sh
```

该脚本会执行以下操作：

- 在可用时通过 `apt-get` 安装系统依赖
- 克隆 `third_party/VisDrone2018-VID-toolkit`
- 创建 `.venv` 并安装 Python/CUDA 依赖
- 在缺少官方压缩包时下载 `VisDrone2019-VID-{train,val,test-dev}.zip`
- 当 `datasets/VisDrone-VID/*.zip` 存在时，从本地压缩包恢复官方原始 split
- 在需要时构建 YOLO 格式数据集
- 执行运行时检查
- 默认停在“环境和数据已准备好、可开始训练”的状态

Python 依赖安装默认使用国内镜像：

- `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- `PYTORCH_WHEEL_BASE=https://mirrors.aliyun.com/pytorch-wheels/cu121`

如果你的机器需要其他镜像，可以自行覆盖这些环境变量。

新服务器上常用的覆盖参数示例：

```bash
RAW_ROOT=/path/to/VisDrone2019-VID-raw DEVICE=0 BATCH=4 WORKERS=4 EPOCHS=100 \
bash setup.sh
```

默认行为只恢复环境和数据，不会立即启动训练：

```bash
bash setup.sh
```

如果想禁止自动下载数据集，只使用本地压缩包：

```bash
DOWNLOAD_DATA=0 START_TRAIN=0 bash setup.sh
```

如果希望 setup 脚本最后直接启动正式训练：

```bash
START_TRAIN=1 bash setup.sh
```

## 3. 训练

建议先使用较小 batch 和 T 模型开始。

训练前，先检查项目本地数据集、Python 环境、CUDA 可见性和模型构建是否正常：

```bash
.venv/bin/python tools/check_visdrone_vid_runtime.py
```

如果要确认转换后的数据确实来自原始 VisDrone2019-VID split 压缩包：

```bash
.venv/bin/python tools/verify_visdrone_vid_source.py
```

如果缺少压缩包，可以把官方 split 压缩包下载到项目中：

```bash
.venv/bin/python tools/download_visdrone_vid_zips.py
```

启动 Mamba-YOLO-T 训练：

```bash
python mbyolo_train.py \
  --task train \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --epochs 100 \
  --workers 8 \
  --device 0 \
  --amp \
  --project output_dir/visdrone_vid \
  --name mambayolo_t
```

## 4. 验证

```bash
python mbyolo_train.py \
  --task val \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --workers 8 \
  --device 0 \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_val
```

## 5. Test-dev 与官方结果文件

使用训练好的 T 模型权重在 test-dev 上验证：

```bash
python mbyolo_train.py \
  --task test \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --workers 8 \
  --device 0 \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_testdev
```

最保守的本地评测方式是使用官方 split 根目录，并用一个命令完成导出和官方评测：

```bash
python tools/run_visdrone_vid_official_eval.py \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --official-root datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --toolkit third_party/VisDrone2018-VID-toolkit \
  --results output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --out output_dir/visdrone_vid/mambayolo_t_official_eval \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

如果需要分两步调试，先导出官方 VisDrone-VID toolkit 所需的逐序列 txt 文件：

```bash
python tools/export_visdrone_vid_results.py \
  --weights output_dir/visdrone_vid/mambayolo_t/weights/best.pt \
  --source datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --out output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

然后在这些导出结果上运行官方 VisDrone-VID MATLAB toolkit：

```bash
python tools/eval_visdrone_vid_official.py \
  --toolkit third_party/VisDrone2018-VID-toolkit \
  --official-root datasets/VisDrone-VID/raw/VisDrone2019-VID-test-dev \
  --results output_dir/visdrone_vid/mambayolo_t_visdrone_vid_results \
  --out output_dir/visdrone_vid/mambayolo_t_official_eval
```

官方评测封装脚本是保守实现：它不会用 Python 重新实现 VisDrone AP/AR，而是调用官方 toolkit。运行前需要官方 toolkit 文件
`findSeqList.m`、`saveAnnoRes.m`、`displaySeq.m`、`calcAccuracy.m`，并且 `PATH` 中需要有 MATLAB 兼容运行时（`matlab` 或 `octave`）。

`--official-root` 必须指向原始 VisDrone-VID split 目录，其中应包含 `annotations/` 和 `sequences/`；不要指向转换后的 YOLO 数据集。封装脚本还会检查每个官方序列是否都有对应结果 TXT，然后才会启动 toolkit。

## 6. 跟踪 ID 指标

跟踪流程不使用外部 ReID 权重。它使用 `tools/prepare_visdrone_vid_yolo.py` 保存在 `tracks/<split>/` 下的 VisDrone 序列 ID 标注，并结合 Mamba-YOLO 检测输出。

使用跟踪配置入口训练：

```bash
python mbyolo_train.py \
  --task train_track \
  --data output_dir/visdrone_vid/VisDrone-VID.local.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-Track.yaml \
  --imgsz 640 \
  --batch_size 16 \
  --epochs 100 \
  --workers 8 \
  --device 0 \
  --amp \
  --project output_dir/visdrone_vid \
  --name mambayolo_t_track
```

导出逐序列跟踪结果文件，输出中的预测轨迹 ID 为非负整数：

```bash
python mbyolo_train.py \
  --task track_export \
  --weights output_dir/visdrone_vid/mambayolo_t_track/weights/best.pt \
  --official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
  --out output_dir/visdrone_vid/mambayolo_t_track_val_tracks \
  --tracker ultralytics/cfg/trackers/mambayolo_visdrone_track.yaml \
  --imgsz 640 \
  --device 0
```

在 val split 上评估本地 ID 指标：

```bash
python mbyolo_train.py \
  --task mot_eval \
  --official_root datasets/VisDrone-VID/raw/VisDrone2019-VID-val \
  --results output_dir/visdrone_vid/mambayolo_t_track_val_tracks \
  --out output_dir/visdrone_vid/mambayolo_t_track_val_mot.json
```

本地 MOT 评估器会输出 `IDF1`、`IDP`、`IDR`、`ID Switches` 和 `Frag`。在该流程中 test-dev 没有公开 GT，因此 test-dev 只用于通过 `track_export` 生成提交文件；可量化的 ID 指标以 val split 为准。

完整 T 模型流程封装在：

```bash
bash scripts/run_visdrone_vid_t_full.sh
```

默认数据集路径为项目本地目录：`datasets/VisDrone-VID`。
