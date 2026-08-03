# 运行手册

本页用于日常实验。第一次配置新服务器时先完成 `environment_setup.md`。

## 1. 每次登录服务器

```bash
conda activate sam3_wsss

export SAM3_WSSS_PROJECT=/home/undergr/1/sam3_remote_wsss
export SAM3_WSSS_SOURCE_CONFIG="$SAM3_WSSS_PROJECT/configs/potsdam_sam3_smoke.json"
export SAM3_WSSS_CONFIG="$SAM3_WSSS_SOURCE_CONFIG"

cd "$SAM3_WSSS_PROJECT"
```

当前 smoke config 已指向：

```text
Potsdam: /home/undergr/remote_dataset/Postdam
SAM3:    /home/undergr/1/sam3-main/sam3-main
权重:    /home/undergr/1/checkpoints/sam3.pt
```

## 2. 先确认 GPU 与路径

```bash
nvidia-smi

python - <<'PY'
import os
import torch
from sam3_remote_wsss.config import load_config

cfg = load_config(os.environ["SAM3_WSSS_CONFIG"])
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("dataset:", cfg.dataset_root.exists())
print("sam3:", cfg.sam3_repo.exists())
print("checkpoint:", cfg.checkpoint_path.exists())
PY
```

## 3. 生成显式 patch 数据集

先用一张原始 `6000 x 6000` 图生成 patch smoke 数据集：

```bash
export SAM3_WSSS_PATCH_ROOT=/home/undergr/remote_dataset/Postdam_patches_512

python -m sam3_remote_wsss.prepare_potsdam_patches \
  --config "$SAM3_WSSS_CONFIG" \
  --output-root "$SAM3_WSSS_PATCH_ROOT" \
  --patch-size 512 \
  --patch-overlap 128 \
  --min-class-pixels 16 \
  --class-min-pixels car=4 \
  --limit 1 \
  --skip-existing
```

输出结构：

```text
Postdam_patches_512/
  4_Ortho_RGBIR/             # 模型输入
  5_Labels_all/              # 只用于弱标签派生和离线评估
  image_level_labels.csv     # 每个 patch 一行弱标签
  patches.csv                # 父图、坐标、像素计数和比例
  potsdam_patches_config.json
  patch_summary.json
```

检查 patch 数量和标签：

```bash
wc -l "$SAM3_WSSS_PATCH_ROOT/image_level_labels.csv"
head "$SAM3_WSSS_PATCH_ROOT/image_level_labels.csv"
cat "$SAM3_WSSS_PATCH_ROOT/patch_summary.json"
```

后续命令改用自动生成的配置和 CSV：

```bash
export SAM3_WSSS_CONFIG="$SAM3_WSSS_PATCH_ROOT/potsdam_patches_config.json"
export SAM3_WSSS_PATCH_LABELS="$SAM3_WSSS_PATCH_ROOT/image_level_labels.csv"
```

注意：GT patch 不会被伪标签生成器或 student 训练读取，只在生成 image-level 标签和计算最终指标时使用。

## 4. 五个 patch 伪标签实验

确认 patch-level CSV 已生成：

```bash
wc -l "$SAM3_WSSS_PATCH_LABELS"
```

CSV 通常包含多于 5 个 patch；下面通过 `--limit 5` 只取前 5 个进行 smoke。先 dry run 检查任务，不加载 SAM3：

```bash
python -m sam3_remote_wsss.generate_pseudo_labels \
  --config "$SAM3_WSSS_CONFIG" \
  --labels-csv "$SAM3_WSSS_PATCH_LABELS" \
  --output-dir runs/smoke_sam3_5patches \
  --limit 5 \
  --dry-run
```

真实生成：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config "$SAM3_WSSS_CONFIG" \
  --labels-csv "$SAM3_WSSS_PATCH_LABELS" \
  --output-dir runs/smoke_sam3_5patches \
  --limit 5 \
  --skip-existing
```

评估：

```bash
python -m sam3_remote_wsss.evaluate_pseudo_labels \
  --config "$SAM3_WSSS_CONFIG" \
  --pseudo-label-dir runs/smoke_sam3_5patches/pseudo_labels \
  --output runs/smoke_sam3_5patches/pseudo_metrics.json
```

检查以下三类输出：

```bash
ls runs/smoke_sam3_5patches/pseudo_labels
ls runs/smoke_sam3_5patches/overlays
ls runs/smoke_sam3_5patches/metadata
```

重点同时看 `foreground_miou`、每类 IoU 和 overlay，不能只看一个总均值。

修复后 `background` IoU 应成为有效指标。旧的 `foreground_mIoU=0.2846` 使用了错误的 background ignore 映射，只能作为历史参考。

## 5. 五个 patch student 实验

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.train_student \
  --config "$SAM3_WSSS_CONFIG" \
  --pseudo-label-dir runs/smoke_sam3_5patches/pseudo_labels \
  --output-dir runs/student_5patches \
  --epochs 5 \
  --batch-size 2 \
  --crop-size 384 \
  --samples-per-image 16 \
  --head segformer \
  --segformer-embed-dim 128 \
  --num-workers 2 \
  --amp
```

训练输出：

```text
runs/student_5patches/train_log.jsonl
runs/student_5patches/checkpoints/last.pt
```

这个实验只验证训练趋势，不用于汇报最终精度。

## 6. 两张 2080Ti 生成伪标签

先去掉 `--limit 1`，为所有原始大图补齐 patch 数据集。`--skip-existing` 会复用 smoke 阶段已经写好的 TIFF：

```bash
export SAM3_WSSS_SOURCE_CONFIG="$SAM3_WSSS_PROJECT/configs/potsdam_sam3_2080ti.json"
export SAM3_WSSS_PATCH_ROOT=/home/undergr/remote_dataset/Postdam_patches_512

python -m sam3_remote_wsss.prepare_potsdam_patches \
  --config "$SAM3_WSSS_SOURCE_CONFIG" \
  --output-root "$SAM3_WSSS_PATCH_ROOT" \
  --patch-size 512 \
  --patch-overlap 128 \
  --min-class-pixels 16 \
  --class-min-pixels car=4 \
  --skip-existing

export SAM3_WSSS_CONFIG="$SAM3_WSSS_PATCH_ROOT/potsdam_patches_config.json"
export SAM3_WSSS_PATCH_LABELS="$SAM3_WSSS_PATCH_ROOT/image_level_labels.csv"
```

两个进程分别使用一张 GPU 和一个数据 shard：

```bash
bash scripts/run_two_2080ti.sh \
  "$SAM3_WSSS_CONFIG" \
  "$SAM3_WSSS_PATCH_LABELS" \
  runs/potsdam_sam3_2080ti
```

监控：

```bash
tail -f runs/potsdam_sam3_2080ti/logs/gpu0.log
tail -f runs/potsdam_sam3_2080ti/logs/gpu1.log
nvidia-smi
find runs/potsdam_sam3_2080ti/pseudo_labels -name '*.png' | wc -l
```

脚本带有 `--skip-existing`，中断后可运行同一条命令续跑。

## 7. 两卡训练 student

当前训练脚本使用简单的 `torch.nn.DataParallel`：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.train_student \
  --config "$SAM3_WSSS_CONFIG" \
  --pseudo-label-dir runs/potsdam_sam3_2080ti/pseudo_labels \
  --val-labels-csv "$SAM3_WSSS_PATCH_ROOT/image_level_labels_val.csv" \
  --output-dir runs/student_segformer_resnet50 \
  --epochs 20 \
  --batch-size 8 \
  --crop-size 512 \
  --output-stride 16 \
  --head segformer \
  --segformer-embed-dim 256 \
  --samples-per-image 1 \
  --cat-max-ratio 0.75 \
  --min-component-area 16 \
  --ignore-boundary-width 1 \
  --num-workers 4 \
  --data-parallel \
  --amp
```

若 OOM，依次尝试：

1. 把 `--batch-size 8` 改为 4 或 2。
2. 把 `--crop-size 512` 改为 384。
3. 把 `--segformer-embed-dim 256` 改为 128。
4. 把 `--output-stride 16` 改为 32。

student 的 `--amp` 使用 FP16；不要在 2080Ti 上改成 BF16。

## 8. 常见错误速查

`pip install -e .` 提示不是 Python project：当前目录不对。应进入包含 `pyproject.toml` 的 SAM3 目录或 `sam3_remote_wsss` 目录。

`ModuleNotFoundError: sam3`：还没有执行 `python -m pip install -e "$SAM3_WSSS_UPSTREAM"`。

`ModuleNotFoundError: einops/pycocotools/psutil`：按 `environment_setup.md` 补齐兼容依赖。

NumPy 冲突：重新固定以下组合：

```bash
python -m pip install "numpy==1.26.4" --force-reinstall
python -m pip install "opencv-python==4.11.0.86" --force-reinstall --no-deps
```

`BFloat16 and Float`：确认使用仓库内带 FP32 fallback 的 `sam3/perflib/fused.py`。

进度长时间停在 `0/1`：先看 `nvidia-smi`。如果 GPU 显存和利用率有变化，通常仍在运行；本项目已出现过单图约 310 秒的正常情况。

## 9. 当前方法上的实验纪律

- 生成 N 张伪标签前，image-level CSV 必须有 N 行图像记录。
- 先跑 5 张并评估，再扩大规模。
- 保存配置、Git commit、日志、metrics JSON 和 overlay。
- 调一个因素时尽量固定其他参数，方便做可靠消融。
- 不再把原始整幅图级标签的全量结果作为最终实验；正式实验使用 patch-level CSV。
- 训练/验证划分必须按 `parent_image_id` 划分，不能把同一原图的重叠 patch 随机分到两边。
- 论文中明确说明 patch 标签由 GT 仅派生为存在/不存在标记；这比原始父图标签更局部，是数据协议的一部分。
