# 实验记录

本页只记录已经实际运行过的结果。后续每次实验建议追加配置、命令、指标和结论，不要只记录一个 mIoU 数字。

## 实验 A：真实 SAM3 单图 smoke

目的：验证 SAM3 权重、Potsdam 读取、prompt、切片和伪标签输出能完整运行。

```text
GPU: 1 x 2080Ti
config: configs/potsdam_sam3_smoke.json
tile_size: 512
tile_overlap: 128
max_prompts_per_class: 1
images: 1
elapsed: 约 5 分 09 秒（310 秒/图）
result: 成功生成 pseudo label、overlay 和 metadata
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config configs/potsdam_sam3_smoke.json \
  --labels-csv data/potsdam_image_level_labels.csv \
  --output-dir runs/smoke_sam3 \
  --limit 1
```

## 实验 B：student 单图 smoke

目的：验证 SAM3 伪标签能够被 SegFormer-style student dataset 和训练循环读取。

```text
epochs: 1
batch_size: 2
crop_size: 384
samples_per_image: 16
head: segformer
segformer_embed_dim: 128
AMP: enabled
final loss: 6.9738
checkpoint: runs/student_smoke_from_sam3/checkpoints/last.pt
result: 训练闭环跑通；该 loss 不代表模型已经收敛
```

## 实验 C：错误的 5 图评估

初次尝试得到：

```text
foreground_mIoU = 0.0630367
```

但检查后发现 CSV 只有 2 行：表头和 `top_potsdam_2_10` 一张图。后面四张图没有自己的 positive class 标签，因此这不是有效的 5 图实验，不能用于论文结论。

用于发现问题的命令：

```bash
wc -l data/potsdam_image_level_labels.csv
cat data/potsdam_image_level_labels.csv
```

## 实验 D：修正 image-level labels 后的 5 图评估

先重新为 5 张图生成 image-level CSV，再生成伪标签和评估。

结果：

```json
{
  "class_iou": {
    "background": 0.0,
    "impervious_surface": 0.06100306772738242,
    "building": 0.5040318517649861,
    "low_vegetation": 0.013216615332180015,
    "tree": 0.10532811295505962,
    "car": 0.7391752694788177
  },
  "miou": 0.237125819543071,
  "foreground_miou": 0.28455098345168517
}
```

结论：

- SAM3-only 原型不是完全失效，building 和 car 已表现出较强可用性。
- impervious surface、low vegetation 和 tree 是下一轮 prompt、阈值和区域约束的重点。
- 5 张样本只能用于工程验证和方向筛选，不能作为最终研究结果。
- background IoU 为 0 需要结合当前 clutter/background 映射和 ignore 策略单独检查。

后续代码检查确认，该实验运行时 Potsdam 红色 clutter/background 没有映射为 `0`，而是被当作 `255 ignore`。这会忽略 clutter 区域上的前景 false positive，使前景指标可能偏乐观。当前代码已修复映射，但本实验数值尚未重算，应标记为“工程可行性历史结果”，不能作为最终基线。

## 修复记录：background 与显式 patch 数据集

已完成但尚未在服务器重跑：

```text
Potsdam red clutter/background -> class ID 0
unknown GT colors -> ignore ID 255
6000 x 6000 parent tile -> explicit 512 x 512 patches
each patch -> independent image-level class row
patch GT -> only for tag simulation and offline evaluation
```

下一项实验必须报告修复后的 background IoU、各前景 IoU 和 foreground mIoU，并与本页实验 D 分开记录。

## 后续实验记录模板

```text
实验名称:
日期:
Git commit:
设备:
数据范围/划分:
配置文件:
与基线相比的唯一改动:
运行命令:
运行时间:
class IoU:
mIoU:
foreground mIoU:
student validation mIoU:
可视化观察:
结论:
下一步:
```
