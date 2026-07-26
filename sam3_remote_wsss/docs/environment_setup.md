# 环境配置全流程

这份流程面向 Linux + NVIDIA GPU，默认兼容实验室的 2 张 2080Ti，也可以通过修改开头的路径变量在另一台设备上使用。

## 1. 前置文件

准备好以下内容：

```text
仓库根目录/
  sam3-main/sam3-main/
  sam3_remote_wsss/

Potsdam/
  4_Ortho_RGBIR/
  5_Labels_all/

checkpoints/
  sam3.pt
```

`sam3.pt` 需要从获批的 Hugging Face `facebook/sam3` 仓库下载。不要把 Hugging Face token 写进脚本、配置或 Git 仓库。

## 2. 设置路径变量

克隆整个 GitHub 仓库后，先根据机器修改这五行：

```bash
export SAM3_WSSS_ROOT=/path/to/Sheung_project1_sam3_remote_wsss
export SAM3_WSSS_PROJECT="$SAM3_WSSS_ROOT/sam3_remote_wsss"
export SAM3_WSSS_UPSTREAM="$SAM3_WSSS_ROOT/sam3-main/sam3-main"
export SAM3_WSSS_DATA=/path/to/Postdam
export SAM3_WSSS_CKPT=/path/to/checkpoints/sam3.pt
```

实验室服务器的现有路径可以这样设置：

```bash
export SAM3_WSSS_PROJECT=/home/undergr/1/sam3_remote_wsss
export SAM3_WSSS_UPSTREAM=/home/undergr/1/sam3-main/sam3-main
export SAM3_WSSS_DATA=/home/undergr/remote_dataset/Postdam
export SAM3_WSSS_CKPT=/home/undergr/1/checkpoints/sam3.pt
```

检查文件是否存在：

```bash
test -f "$SAM3_WSSS_PROJECT/pyproject.toml" && echo "project ok"
test -f "$SAM3_WSSS_UPSTREAM/pyproject.toml" && echo "sam3 source ok"
test -d "$SAM3_WSSS_DATA/4_Ortho_RGBIR" && echo "images ok"
test -d "$SAM3_WSSS_DATA/5_Labels_all" && echo "labels ok"
test -f "$SAM3_WSSS_CKPT" && echo "checkpoint ok"
```

## 3. 创建 Conda 环境

下面是已在本项目上跑通的 Python 3.10 路线：

```bash
conda create -n sam3_wsss python=3.10 -y
conda activate sam3_wsss

python -m pip install --upgrade pip wheel "setuptools<81"
```

2080Ti 可以使用 PyTorch CUDA 11.8 wheel。若目标服务器已经有可用的 PyTorch，并且下一节检测正常，可以跳过安装 PyTorch 的命令。

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

安装 SAM3、兼容依赖和本工程：

```bash
python -m pip install "numpy==1.26.4"
python -m pip install -e "$SAM3_WSSS_UPSTREAM"

python -m pip install \
  einops \
  pycocotools \
  psutil \
  pandas \
  python-rapidjson \
  decord

python -m pip install "opencv-python==4.11.0.86" --no-deps
python -m pip install -e "$SAM3_WSSS_PROJECT"
python -m pip check
```

这里固定 `numpy==1.26.4`，并用 `--no-deps` 安装 OpenCV，是为了避免 pip 把 NumPy 自动升级到 2.x。包名是 `python-rapidjson`，不是 `rapidjson`。

## 4. 检查 CUDA 和导入

```bash
python - <<'PY'
import cv2
import numpy
import torch
import sam3
import sam3_remote_wsss

print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"gpu{i}:", torch.cuda.get_device_name(i))
print("sam3 import ok")
print("sam3_remote_wsss import ok")
PY
```

预期 NumPy 为 `1.26.4`，OpenCV 为 `4.11.0`，并且能看到两张 2080Ti。`Flash Attention is disabled` 对 2080Ti 是正常提示，不是报错。

## 5. 2080Ti 的 BF16 兼容检查

2080Ti 属于 Turing 架构，不支持原生 BF16。当前仓库携带的 SAM3 已经在 `sam3/perflib/fused.py` 中加入 FP32 fallback：

```bash
grep -n "_supports_bf16_cuda" "$SAM3_WSSS_UPSTREAM/sam3/perflib/fused.py"
```

该命令应至少输出一行。如果使用的是另外下载的原始 SAM3，而这里没有输出，真实推理可能再次报：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

此时应使用本仓库中的 SAM3 版本，或把本仓库同一文件中的 FP32 fallback 合并到外部 SAM3。

## 6. 生成本机配置

下面命令从 2080Ti 配置复制一份本机配置，并使用刚才的环境变量填写路径：

```bash
cd "$SAM3_WSSS_PROJECT"
export SAM3_WSSS_CONFIG="$SAM3_WSSS_PROJECT/configs/potsdam_local.json"

python - <<'PY'
import json
import os
from pathlib import Path

project = Path(os.environ["SAM3_WSSS_PROJECT"])
source = project / "configs" / "potsdam_sam3_2080ti.json"
target = Path(os.environ["SAM3_WSSS_CONFIG"])
cfg = json.loads(source.read_text(encoding="utf-8"))
cfg["dataset_root"] = os.environ["SAM3_WSSS_DATA"]
cfg["sam3_repo"] = os.environ["SAM3_WSSS_UPSTREAM"]
cfg["checkpoint_path"] = os.environ["SAM3_WSSS_CKPT"]
target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("wrote:", target)
PY
```

验证配置：

```bash
python - <<'PY'
import os
from sam3_remote_wsss.config import load_config

cfg = load_config(os.environ["SAM3_WSSS_CONFIG"])
print("dataset_root:", cfg.dataset_root, cfg.dataset_root.exists())
print("sam3_repo:", cfg.sam3_repo, cfg.sam3_repo.exists())
print("checkpoint:", cfg.checkpoint_path, cfg.checkpoint_path.exists())
print("image_dir:", (cfg.dataset_root / cfg.image_dir).exists())
print("label_dir:", (cfg.dataset_root / cfg.label_dir).exists())
PY
```

五个结果都应为 `True`。

## 7. 首次 smoke test

先生成 1 张图的 image-level label：

```bash
cd "$SAM3_WSSS_PROJECT"

python -m sam3_remote_wsss.build_image_level_labels \
  --config "$SAM3_WSSS_CONFIG" \
  --output data/potsdam_image_level_labels_smoke.csv \
  --limit 1
```

再用单张 GPU 生成伪标签：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config "$SAM3_WSSS_CONFIG" \
  --labels-csv data/potsdam_image_level_labels_smoke.csv \
  --output-dir runs/smoke_sam3 \
  --limit 1
```

检查输出：

```bash
find runs/smoke_sam3 -maxdepth 2 -type f | sort
```

应看到 `pseudo_labels/*.png`、`overlays/*.jpg` 和 `metadata/*.json`。实验室 2080Ti 的一次已知结果约为 310 秒/图，但耗时会随 tile size、prompt 数量和 GPU 状态变化。

以上只用于验证环境。正式方法实验继续按照 `runbook.md` 生成显式 patch 数据集，不再让每个小块继承 `6000 x 6000` 父图的全部类别。

## 8. 换设备时只需修改什么

- 修改 `SAM3_WSSS_PROJECT`、`SAM3_WSSS_UPSTREAM`、`SAM3_WSSS_DATA` 和 `SAM3_WSSS_CKPT`。
- 根据显卡驱动选择合适的 PyTorch CUDA wheel。
- Ampere 或更新显卡可使用 BF16/Flash Attention；2080Ti 保持当前 FP32 fallback 和 student FP16 AMP。
- 不复制 `runs/` 也能重新运行；需要保留实验结果时应单独备份该目录。
- 权重和数据集不应提交到 GitHub。
