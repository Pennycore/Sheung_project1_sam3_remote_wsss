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

## 实验 E：256 patch CAM smoke 与人工检查（进行中）

目的：验证多标签 CAM 训练、CAM 导出和可视化流程，并在融合 CAM/SAM3 前人工检查类别定位质量。

```text
日期: 2026-08-02
状态: CAM 训练完成，CAM 导出和人工检查待进行
Git commit: 43366c6 add CAM and SAM3 hybrid pseudo-label pipeline
设备: 1 x NVIDIA 2080Ti（CUDA_VISIBLE_DEVICES=0）
数据范围: 256 个 512 x 512 重叠 patch
父图范围: 仅 top_potsdam_2_10，一张父图
监督: patch-level image tags
CAM backbone: ResNet-50
output stride: 16
训练轮数: 1
训练 batch size: 4
预训练编码器: ImageNet ResNet-50，成功从 Torch 缓存加载
AMP: enabled
用途限制: 仅用于工程 smoke test 和方法初筛，不能作为正式训练/验证结果
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.train_cam \
  --config /home/undergr/remote_dataset/Postdam_patches_512/potsdam_patches_config_prompt4.json \
  --labels-csv /home/undergr/remote_dataset/Postdam_patches_512/image_level_labels.csv \
  --output-dir runs/cam_smoke_256 \
  --epochs 1 \
  --batch-size 4 \
  --image-size 512 \
  --backbone resnet50 \
  --output-stride 16 \
  --pretrained-backbone \
  --num-workers 2 \
  --amp
```

训练结果：

```json
{
  "epoch": 1,
  "loss": 0.4548998698592186,
  "micro_f1": 0.828328611898017,
  "per_class_f1": {
    "impervious_surface": 0.9082125603864735,
    "building": 0.832807570977918,
    "low_vegetation": 0.9718875502008032,
    "tree": 0.6037735849056604,
    "car": 0.6481481481481481
  },
  "final_lr": 2.368307135172497e-06
}
```

```text
checkpoint: runs/cam_smoke_256/checkpoints/last.pt
checkpoint size: 270 MB
训练运行时间: 未记录
峰值显存: 未记录
```

说明：F1 来自训练过程中的同一批 256 个 patch，不是独立验证集指标。它只能说明多标签分类训练链路有效，不能证明 CAM 空间定位质量。tree 和 car 的训练 F1 相对较低，后续可视化时应优先检查这两类。

2026-08-02 人工检查了首批 CAM overlay。观察结果：

- CAM 没有退化为整图均匀高响应，说明网络至少形成了局部判别热点。
- 响应大多是少量离散圆形热点，没有覆盖完整语义区域，区域召回明显不足。
- `top_potsdam_2_10_x0000_y0768` 的 low vegetation CAM 在车辆、硬质地面和屋顶附近出现明显响应，草地区域覆盖较弱，存在错误定位。
- 同一 patch 的 tree CAM 只覆盖少量树冠或植被热点，同时也在屋顶、庭院和草地上响应，tree/low vegetation 区分不足。
- 后续几组同一 patch 的不同类别 CAM 外观非常相似，热点位置高度重合，说明分类器可能利用类别共现和场景上下文完成分类，而没有学到充分的类别特异定位。
- 当前生成器会对每个图像级正类的 CAM 分别归一化到 `[0, 1]`，因此即使原始响应很弱，也会至少产生一个亮点。可视化亮度不能直接解释为跨类别的绝对置信度。

阶段结论：分类训练链路成功，但 1 epoch、单父图 CAM 的空间质量不足，不应直接生成并融合全部 256 个 patch 作为方法结果。可以先对同一 5 个 patch 执行一次 CAM/SAM3 融合以验证代码闭环；正式融合前应扩大到多个父图、增加训练轮数，并加入分类置信度门控或保存原始 CAM/logit 供诊断。

5 patch CAM/SAM3 融合闭环已完成。融合参数为：

```text
background_threshold = 0.2
foreground_threshold = 0.7
cam_support_threshold = 0.3
SAM3 input = runs/sam3_prompt4_256patches_ignore255/pseudo_labels
output = runs/cam_sam_fused_5patches_smoke
```

严格评估结果：

```json
{
  "class_iou": {
    "background": 0.0007755697916766714,
    "impervious_surface": 0.5965778580585575,
    "building": 0.6328582579599233,
    "low_vegetation": 0.40630096110475,
    "tree": 0.4861810655872818,
    "car": 0.8732378310597713
  },
  "miou": 0.4993219239269935,
  "foreground_miou": 0.5990311947540568
}
```

已标注区域结果：

```json
{
  "labeled_miou": 0.5385921410633229,
  "labeled_foreground_miou": 0.6461549415010068,
  "labeled_coverage": 0.9062026977539063,
  "unlabeled_prediction_pixels": 122942
}
```

与此前同一 5 patch 的 Prompt4 硬背景基线相比：

```text
strict mIoU:            0.5005 -> 0.4993（-0.0012）
strict foreground mIoU: 0.5999 -> 0.5990（-0.0009）
car IoU:                0.8818 -> 0.8732（-0.0085）
```

本批 5 patch 的真实 background 只有 `1,868 / 1,310,720` 像素。融合结果的 confusion matrix 显示，约 41.5 万像素被预测为背景，其中只有 323 个是真背景，背景 precision 约为 `0.00078`。因此 `90.62%` 的高覆盖率主要来自把 CAM 低响应区填成背景，而不是可靠的新伪标签。

结论：代码闭环成功，但当前“CAM 低响应直接作为背景”的实现失败，不能扩展到全部 256 patch。必须先在相同 5 patch 上计算 SAM3 Ignore255 的公平基线，并检查融合 summary 中 background/CAM/SAM/conflict 像素构成。方法上需要降低或门控背景填充、保留原始分类置信度，并在更多父图上训练更完整的 CAM。

同一 5 patch 的 SAM3 Ignore255 公平基线随后完成：

```text
strict mIoU              = 0.4997018
strict foreground mIoU   = 0.5996421
labeled mIoU             = 0.9063542
labeled foreground mIoU  = 0.9063542
labeled coverage         = 0.5865555
labeled pixels           = 768,810
ignored pixels           = 541,910
```

融合 summary 的像素构成：

```text
background_pixels        = 414,923
sam_foreground_pixels    = 768,670
cam_foreground_pixels    = 4,185
conflict_pixels          = 140
ignored_pixels           = 122,942
total_pixels             = 1,310,720
```

公平对照结论：

```text
strict foreground mIoU:  0.5996421 -> 0.5990312  (-0.0006109)
labeled foreground mIoU: 0.9063542 -> 0.6461549  (-0.2601993)
labeled coverage:        0.5865555 -> 0.9062027  (+0.3196472)
```

CAM 只新增 4,185 个前景像素，约占总像素的 0.32%；背景规则新增 414,923 个像素，约占总像素的 31.66%。因此覆盖率提升几乎全部来自背景填充，而不是 CAM 前景补全。SAM3 原有 768,810 个已标注像素中只因冲突删除了 140 个，说明当前冲突门控也几乎不起作用。

这组结果进一步说明：SAM3-only 是“高精度、低覆盖”的有效伪标签源；当前 CAM 是“分类可用、定位不足”，不能直接承担背景排除。下一步不能用当前融合标签训练 student。应先在包含更多 background 的 256 patch 诊断集上扫描更保守的背景阈值，并修改 CAM 生成器保存原始分类置信度，避免每个正类归一化后被强制产生高响应。

随后使用同一个 1 epoch checkpoint、`scale=1.0`、单张 2080Ti 补充生成其余 CAM，计时结果：

```text
real = 41.474 s
user = 44.834 s
sys  = 2.087 s
CAM 总数 = 256（后续阈值扫描评估了全部 256 个 patch）
```

固定 `foreground_threshold=0.7`、`cam_support_threshold=0.3`，对全部 256 patch 扫描背景阈值：

| background threshold | mIoU | foreground mIoU | labeled foreground mIoU | coverage | background IoU | background pixels | CAM foreground pixels | ignored pixels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.3620 | 0.4248 | 0.6113 | 0.5250 | 0.0482 | 507,581 | 329,899 | 31,874,516 |
| 0.01 | 0.3743 | 0.4248 | 0.6008 | 0.5527 | 0.1215 | 2,363,747 | 329,899 | 30,018,350 |
| 0.02 | 0.3802 | 0.4248 | 0.5848 | 0.5867 | 0.1571 | 4,648,350 | 329,899 | 27,733,747 |
| 0.05 | 0.3857 | 0.4248 | 0.5447 | 0.6768 | 0.1898 | 10,691,106 | 329,899 | 21,690,991 |
| 0.10 | **0.3858** | 0.4248 | 0.5046 | 0.7803 | **0.1907** | 17,638,691 | 329,899 | 14,743,406 |
| 0.20 | 0.3845 | 0.4248 | 0.4659 | 0.8927 | 0.1827 | 25,178,792 | 329,899 | 7,203,305 |

SAM3 Ignore255 的 256 patch 基线为 strict foreground mIoU `0.4255`、labeled foreground mIoU `0.6213`、coverage `0.5133`。所有背景阈值的 strict foreground mIoU 都约为 `0.4248`，说明当前 CAM 没有改善前景。`0.10` 的总 mIoU 最高，但 labeled foreground mIoU 已降到 `0.5046`；这一增益只来自背景 IoU，不能直接按总 mIoU 选择阈值用于训练。

阈值越高，coverage 越高、已标注区域前景精度越低。当前较保守候选是 `0.00` 或 `0.01`，但最终选择还需要计算背景伪标签 precision/recall。由于 CAM 前景像素在所有阈值下固定为 329,899，背景扫描不会解决 CAM 前景补全失效的问题。

背景种子 precision/recall：

| threshold | background precision | background recall | background F1 | predicted background | correct background |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | **0.8875** | 0.0485 | 0.0920 | 507,581 | 450,472 |
| 0.01 | 0.5338 | 0.1359 | 0.2166 | 2,363,747 | 1,261,765 |
| 0.02 | 0.4070 | 0.2038 | 0.2716 | 4,648,350 | 1,892,063 |
| 0.05 | 0.2981 | 0.3433 | 0.3191 | 10,691,106 | 3,187,323 |
| 0.10 | 0.2444 | 0.4643 | 0.3203 | 17,638,691 | 4,311,567 |
| 0.20 | 0.2114 | 0.5732 | 0.3089 | 25,178,792 | 5,322,799 |

用于 WSSS 伪标签训练时应优先保证 seed precision，而不是背景 F1 或总 mIoU。因此选择 `background_threshold=0.00` 作为当前候选：它能提供 450,472 个正确背景种子，precision 为 88.75%，但只召回 4.85% 的真实背景。`0.01` 的 precision 已降至 53.38%，不再适合作为可靠伪标签。

下一步固定背景阈值为 `0.00`，扫描 CAM 前景阈值。目标是判断 329,899 个 CAM 新增前景是否有价值，以及提高前景阈值后能否恢复 SAM3-only 的已标注区域精度。

固定 `background_threshold=0.00`、`cam_support_threshold=0.3` 后的前景阈值扫描：

| foreground threshold | mIoU | foreground mIoU | labeled foreground mIoU | coverage | CAM foreground pixels | conflicts | ignored pixels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 0.3620 | 0.4248 | 0.6113 | 0.5250 | 329,899 | 52,001 | 31,874,516 |
| 0.80 | 0.3628 | **0.4258** | 0.6166 | 0.5228 | 150,937 | 20,127 | 32,021,604 |
| 0.90 | **0.3629** | **0.4258** | 0.6195 | 0.5215 | 48,812 | 5,122 | 32,108,724 |
| 0.95 | 0.3627 | 0.4256 | 0.6203 | 0.5211 | 17,391 | 1,567 | 32,136,590 |
| 0.99 | 0.3626 | 0.4255 | 0.6207 | 0.5209 | 2,362 | 169 | 32,150,221 |
| 1.00 | 0.3626 | 0.4255 | **0.6208** | 0.5209 | 531 | 40 | 32,151,923 |

前景阈值越高，CAM 新增前景和 CAM/SAM3 冲突越少，labeled foreground mIoU 越接近 SAM3-only 基线 `0.6213`。`0.80/0.90` 的 strict foreground mIoU 仅有约 `0.0003` 的微小提升，不足以证明 CAM 前景补全有效。当前 CAM 的主要可用贡献是 `background_threshold=0.00` 的高精度背景种子。

据此新增正式 `background_only` 融合模式：完整保留 SAM3 前景，不使用 CAM 补前景，也不让 CAM 删除 SAM3 掩码；仅在所有正类 CAM 恰为零的位置生成背景，其余区域保持 `255`。full hybrid 模式保留为消融实验。

### 256 patch background-only 最终评估

服务器更新到 commit `ad1340e` 后，使用以下策略重新融合全部 256 patch：

```text
SAM3 foreground: 完整保留
CAM foreground completion: disabled
CAM/SAM3 conflict rejection: disabled
background threshold: 0.00
uncovered remainder: 255
output: runs/cam_sam_background_only_256
```

结果：

```json
{
  "class_iou": {
    "background": 0.048216974243363894,
    "impervious_surface": 0.4123100328041728,
    "building": 0.5775959893277807,
    "low_vegetation": 0.22887519200950893,
    "tree": 0.13897292995996338,
    "car": 0.769554920774433
  },
  "miou": 0.3625876731865371,
  "foreground_miou": 0.42546181297517177,
  "labeled_miou": 0.5386070276455444,
  "labeled_foreground_miou": 0.6208152782160715,
  "labeled_coverage": 0.5208916962146759,
  "evaluated_images": 256
}
```

与 SAM3 Ignore255 256 patch 基线对比：

| 指标 | SAM3 Ignore255 | CAM background-only | 变化 |
| --- | ---: | ---: | ---: |
| background IoU | 0.0000 | 0.0482 | +0.0482 |
| foreground mIoU | 0.4254618 | 0.4254618 | 0.0000 |
| mIoU | 0.3545515 | 0.3625877 | +0.0080362 |
| labeled foreground mIoU | 0.6213079 | 0.6208153 | -0.0004926 |
| labeled mIoU | 0.5177566 | 0.5386070 | +0.0208504 |
| labeled coverage | 0.5133281 | 0.5208917 | +0.0075635 |

背景种子统计：

```text
predicted background = 507,581
correct background   = 450,472
false background     = 57,109
background precision = 0.8875
background recall    = 0.0485
```

误判背景主要来自 low vegetation 的 56,264 个像素，另有 building 573、tree 272；impervious surface 和 car 没有被该规则误判为背景。

结论：background-only 是当前一父图诊断数据上最可信的融合策略。它严格保持 SAM3 的前景结果，以 88.75% precision 增加少量背景种子，使总 mIoU 提升 0.80 个百分点、labeled mIoU 提升 2.09 个百分点，代价是 labeled foreground mIoU 下降 0.05 个百分点。该结果证明 CAM 当前适合做保守背景排除，不适合补前景。由于全部 patch 仍来自单一父图，该策略仍需在按父图划分的完整 Potsdam 数据上验证。

### Background-only SegFormer student smoke

```text
日期: 2026-08-03
Git commit: ad1340e
设备: 2 x NVIDIA 2080Ti，torch.nn.DataParallel
伪标签: runs/cam_sam_background_only_256/pseudo_labels
encoder 初始化: runs/cam_smoke_256/checkpoints/last.pt
student: ResNet-50 + SegFormer-style head
epochs: 1
batch size: 4
crop size: 512
samples per image: 1
AMP: enabled
output: runs/student_background_only_smoke
```

同一输出目录被独立运行了两次，`train_log.jsonl` 采用追加写入：

```text
run 1: epoch=1 loss=4.69349205866456
run 2: epoch=1 loss=4.630455622449517
final lr: 2.368307135172497e-06
latest checkpoint: runs/student_background_only_smoke/checkpoints/last.pt
checkpoint size: 284 MB
```

当前 `last.pt` 对应第二次运行。两条日志不是一次训练的 epoch 1/2，第二次也没有从第一次 resume。后续实验应使用不同输出目录或在重新运行前归档旧日志，避免误读。

结论：background-only 伪标签、CAM encoder 初始化、SegFormer student、双 2080Ti DataParallel、AMP、反向传播和 checkpoint 保存全部跑通。loss 仅用于工程 smoke，不代表收敛或分割精度。单父图调试到此停止，下一阶段转向完整 Potsdam 父图清点、父图级划分和 student 验证闭环。

## 数据阶段 F：完整 Potsdam 清点与父图划分

服务器数据清点：

```text
日期: 2026-08-03
dataset_root: /home/undergr/remote_dataset/Postdam
RGBIR parent tiles: 38
pixel-label parent tiles: 38
RGBIR/label pairing: complete
dataset size: 9.0 GB
parent resolution: 6000 x 6000
```

采用固定父图划分：

```text
train:   17 parents -> 4,352 patches
val:      6 parents -> 1,536 patches
test:    14 parents -> 3,584 patches
exclude:  1 parent  -> top_potsdam_7_10
total generated: 37 parents -> 9,472 patches
patch size: 512 x 512
patch overlap: 128
```

test 使用常见的 14 张固定父图：`2_13`、`2_14`、`3_13`、`3_14`、`4_13`、`4_14`、`4_15`、`5_13`、`5_14`、`5_15`、`6_13`、`6_14`、`6_15`、`7_13`。train+val 使用剩余常用 23 张训练父图；validation 参考已发表的 17/7/14 划分，并移除经文献报告有标注问题的 `7_10`，最终为 17/6/14。

工程已新增：

```text
configs/potsdam_parent_split_17_6_14.json
configs/potsdam_server_prompt4.json
prepare_potsdam_patches.py --parent-split
```

生成器会验证每个可用父图恰好出现一次，拒绝跨 split 重复、未知 ID 和未分配 ID。输出 `image_level_labels_train.csv`、`image_level_labels_val.csv`、`image_level_labels_test.csv`，并在 `patches.csv` 中为每个 patch 写入 `parent_image_id` 和 `split`。这从数据结构上阻止重叠 patch 跨集合泄漏。

划分参考：[ISPRS Potsdam 官方数据说明](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)、[公开的 17/7/14 父图划分](https://doi.org/10.3390/rs15071836)、[23/14 且排除 7_10 的实验协议](https://www.mdpi.com/2072-4292/17/17/3054)。

尚待补录：

```text
训练运行时间与峰值显存:
异常或 warning:
完整 patch 生成运行时间、磁盘占用与 patch_summary.json:
```

### CAM 正式验证闭环代码

```text
日期: 2026-08-03
状态: 代码与单元测试完成，服务器正式训练尚未运行
训练集: image_level_labels_train.csv，17 个父图
验证集: image_level_labels_val.csv，6 个父图
checkpoint 选择: validation macro-F1 最大，val loss 用于同分决胜
输出: checkpoints/last.pt 与 checkpoints/best.pt
测试: 14/14 passed
```

`train_cam` 现在记录 train/validation loss、micro-F1、macro-F1 和每类 F1，
并从 patch ID 反推父图 ID，发现 train/validation 共享父图时直接终止。
验证集关闭随机增强。正式生成 CAM 时必须使用 `best.pt`，不能根据训练 F1 选择
`last.pt`。

一次误启动暴露了输出覆盖风险：原实现会在启动时清空同目录的
`train_log.jsonl`。现已改为发现 `train_log.jsonl`、`best.pt` 或 `last.pt` 时拒绝
启动；只有显式传入 `--overwrite-output` 才会替换。控制台日志应使用 `tee -a`。

### 正式 CAM 训练结果

```text
日期: 2026-08-03
数据: Potsdam_patches_512_full
train: 4,352 patches / 17 parent tiles
validation: 1,536 patches / 6 parent tiles
模型: ResNet-50 CAM, output stride 16, ImageNet initialization
设备: 2 x NVIDIA 2080Ti, DataParallel, AMP
计划 epochs: 20
best epoch: 19
best validation loss: 0.19989451327705865
best validation micro-F1: 0.9506204102304381
best validation macro-F1: 0.9471402317853196
validation per-class F1:
  impervious_surface: 0.9370199692780338
  building: 0.9495412844036697
  low_vegetation: 0.9765863590091619
  tree: 0.9507246376811594
  car: 0.9218289085545722
checkpoint: runs/cam_resnet50_full/checkpoints/best.pt
```

正式训练完成后，同一命令被误启动，旧 `train_log.jsonl` 在新训练第一个 epoch
完成前即被清空，因此完整 20 轮曲线不可恢复。误启动及时停止，原正式训练的
`best.pt` 仍保留 epoch 19 及完整 validation 指标。该 checkpoint 应复制到独立的
recovered 目录后用于 CAM 生成。上述分类指标证明图像级分类器训练有效，但不能
单独证明 CAM 空间定位质量；下一阶段仍需检查 CAM 可视化与伪标签 IoU。

### `top_potsdam_4_12` 标签调色板恢复

```text
日期: 2026-08-03
问题: 4_12 的 256 个 patch 在评估中全部 no_valid_gt
源标签: 6000 x 6000 x 3, uint8, 无 TIFF 压缩，但颜色不是离散 Potsdam 调色板
恢复策略: 每像素映射到最近的六类标准 RGB 颜色
距离阈值: 80
最大观测距离: 70.44146506142529
超过阈值像素: 0
最近与第二近颜色的最小抽样距离差: 154.21
```

恢复后恰好包含六种颜色，3,600 万像素全部完成映射：

```text
background:          1,020,265
impervious_surface: 11,976,195
building:           12,069,654
low_vegetation:      7,267,176
tree:                2,752,612
car:                   914,098
```

该恢复具有清晰的最近颜色间隔，不是任意阈值猜测。旧正式 CAM 曾把这 256 个
patch 当作全负样本，必须在修复 patch 标签与 image-level CSV 后重新训练；旧的
4_12 SAM3 伪标签与 CAM 文件也不能进入最终融合。

### 修复后的正式 SAM3 Prompt4 训练集伪标签

```text
日期: 2026-08-03
数据: 4,352 train patches / 17 parent tiles
输入 PNG: 4,352
实际评估: 4,352
跳过: 0
mIoU: 0.3823913517215775
foreground mIoU: 0.458869622065893
labeled mIoU: 0.5296900681612295
labeled foreground mIoU: 0.6356280817934754
labeled coverage: 0.6139396288815666
```

```text
strict class IoU:
background:          0.0
impervious_surface:  0.5475153584698246
building:            0.5865863696478153
low_vegetation:      0.26788723254099234
tree:                0.13875136664466262
car:                 0.7536077830261705
```

该结果是当前第一个覆盖全部 17 个训练父图且无静默跳过的正式 SAM3-only
Prompt4 指标。SAM3 仍不直接生成背景，因此 background IoU 为 0；car、building
和 impervious surface 较强，tree 与 low vegetation 是主要改进目标。下一步必须
用修复后的 image-level CSV 重新训练 CAM，再评估 exact-zero background-only
融合。

### 修复标签后的正式 CAM 重训

```text
日期: 2026-08-03
数据: 4,352 train patches / 17 parents
验证: 1,536 val patches / 6 parents
模型: ResNet-50 CAM, output stride 16, ImageNet initialization
设备: 2 x NVIDIA 2080Ti, DataParallel, AMP
best epoch: 20
train loss: 0.019860664512902183
train macro-F1: 0.9894704174511879
validation loss: 0.24388618000095144
validation micro-F1: 0.9547477121988078
validation macro-F1: 0.9509633993685762
```

```text
validation per-class F1:
impervious_surface: 0.9418960244648318
building:           0.9590536851683349
low_vegetation:     0.9790257104194858
tree:               0.9545782263878875
car:                0.9202633504023409
```

修复后 validation macro-F1 相比错误标签模型的 0.9471402 提升约 0.00382。
正式 CAM checkpoint 改为
`runs/cam_resnet50_full_repaired/checkpoints/best.pt`；旧 checkpoint 和旧 CAM
不得进入最终融合。

### 修复后的正式 CAM exact-zero / SAM3 融合

```text
日期: 2026-08-03
数据: 4,352 train patches / 17 parent tiles
SAM3 前景: runs/sam3_prompt4_full_train/pseudo_labels
CAM: runs/cam_resnet50_full_repaired/cams_train
融合输出: runs/cam_sam_background_only_full_train_corrected_v2
规则: SAM3 前景保持不变；所有正类 CAM 恰为 0 的未标注像素设为背景；其余为 255
输入/评估/跳过: 4,352 / 4,352 / 0
mIoU: 0.40847412193741706
foreground mIoU: 0.458869622065893
labeled mIoU: 0.5822235522913067
labeled foreground mIoU: 0.6348273459840041
labeled coverage: 0.6225840843775693
background IoU: 0.15649662129503694
background pixels: 9,862,033
SAM3 foreground pixels: 700,413,448
ignored pixels: 430,575,207
CAM foreground pixels: 0
conflict pixels: 0
```

相对修复后的 SAM3-only 基线，foreground mIoU 完全不变；总 mIoU 提升
`0.0260827702`，labeled mIoU 提升 `0.0525334841`，coverage 提升
`0.0086444555`。labeled foreground mIoU 仅下降 `0.0008007358`，同时背景
IoU 从 0 提升到 `0.15650`。这验证了当前核心假设：SAM3 负责高置信前景，
CAM 只负责极保守背景种子，不能补充前景。

第一次全量融合目录复用了旧结果并配合 `--skip-existing`，因此曾得到错误的旧
CAM 指标。融合程序现已记录输入指纹并拒绝静默复用改变过的输入；正式 student
只能使用上述 `corrected_v2` 输出。

### Student 独立验证闭环

Student 训练入口现支持 `--val-labels-csv`、`--val-limit` 和独立验证 batch
size。验证集读取像素 GT 计算 class IoU、mIoU、foreground mIoU 与 pixel
accuracy；训练集仍只读取融合伪标签。程序按父图检查 train/val 隔离，并默认以
validation mIoU 保存 `best.pt`。正式 512 patch 每轮使用
`--samples-per-image 1`，不再将每个现成 patch 重复采样 16 次。

### 正式 SegFormer Student 训练

```text
日期: 2026-08-03
训练: 4,352 train patches / CAM exact-zero + SAM3 Prompt4 伪标签
验证: 1,536 val patches / 6 parent tiles / pixel GT only for evaluation
模型: ResNet-50 encoder + SegFormer-style head
初始化: 修复后的 CAM best checkpoint
设备: 2 x NVIDIA 2080Ti, DataParallel, FP16 AMP
轮数: 20
best epoch: 12
best train loss: 0.47851847912020545
validation loss: 0.9688712566470107
validation mIoU: 0.4875610471782705
validation foreground mIoU: 0.5493236629082672
validation pixel accuracy: 0.6713276483981132
```

```text
validation class IoU:
background:          0.17874796852828687
impervious_surface:  0.47217184709328575
building:            0.7663985480255381
low_vegetation:      0.487684301912386
tree:                0.4019231410478634
car:                 0.6184404764622627
```

验证 mIoU 从 epoch 1 的 `0.3627` 上升到 epoch 12 的 `0.4876`，此后训练
loss 继续下降，但验证 mIoU 回落到 epoch 20 的 `0.4520`，说明模型开始拟合
伪标签噪声。最终模型必须使用 epoch 12 的 `best.pt`，不能使用 `last.pt`。
building 表现最佳，background 和 tree 仍是主要瓶颈。下一步只用冻结的
`best.pt` 在 test 父图上推理并拼接，test 结果不再用于调参。

### 锁定模型后的正式 Test 父图拼接评估

```text
日期: 2026-08-03
checkpoint: Student epoch 12 best.pt
test 数据: 3,584 patches / 14 parent tiles
patch valid pixels: 939,524,096
stitched unique parent pixels: 504,000,000
跳过: 0
```

```text
overlapping patch metrics:
mIoU:             0.5091463330771208
foreground mIoU:  0.5820734577720676
pixel accuracy:   0.6979997807315418

stitched parent metrics (final):
mIoU:             0.5259467813363186
foreground mIoU:  0.5983423572821176
pixel accuracy:   0.7205370952380953
```

```text
stitched test class IoU:
background:          0.16396890160732308
impervious_surface:  0.5966029324529248
building:            0.772483803399172
low_vegetation:      0.4655518490151158
tree:                0.45640781136046266
car:                 0.7006653901829129
```

中心权重拼接相对重叠 patch 统计将 mIoU 提升 `0.0168004483`、foreground
mIoU 提升 `0.0162688995`、pixel accuracy 提升 `0.0225373145`，且六类 IoU
全部提升。14 个父图 mIoU 范围为 `0.4521` 到 `0.5510`；最低为
`top_potsdam_4_15`，最高为 `top_potsdam_5_15`。完整像素数恰好等于
`14 x 6000 x 6000`，确认父图覆盖完整且每个原始像素只计一次。

这是锁定 epoch 12 后的一次性正式 test 结果，不用于继续选择 checkpoint、
阈值或训练超参数。当前完整方法闭环的最终主指标为 stitched test mIoU
`52.59%` 和 foreground mIoU `59.83%`。

### 同架构全监督上界：正式训练与验证

```text
日期: 2026-08-04
状态: 20 epochs 正式训练和锁定 test stitched 评估均已完成
目的: 量化当前 WSSS Student 相对同一数据划分和网络结构全监督上界的差距
训练: 4,352 train patches / pixel GT
验证: 1,536 val patches / 6 parent tiles
测试: 3,584 patches / 14 parent tiles，仅在验证选定 best.pt 后执行一次
模型: ResNet-50 encoder + SegFormer-style head
初始化: ImageNet，不加载 CAM checkpoint
损失: 标准 cross-entropy
设备: 2 x NVIDIA 2080Ti / DataParallel / FP16 AMP
best epoch: 20
best train loss: 0.2837014668621123
validation loss: 0.3703024876303971
validation mIoU: 0.7164237514416052
validation foreground mIoU: 0.7832898501955425
validation pixel accuracy: 0.8640236925342666
```

```text
validation class IoU:
background:          0.38209325767191815
impervious_surface:  0.7871020966380811
building:            0.9065860397917864
low_vegetation:      0.7450842764822087
tree:                0.6789743943266183
car:                 0.7987024437390186
```

该实验使用与 WSSS Student 相同的父图划分、增强、20 epochs、验证选模和
stitched test 评估。训练入口以 `--train-labels-csv` 和
`--pseudo-label-dir` 区分全监督与弱监督来源，二者不能同时出现。检查点额外
记录 `training_supervision` 和 `training_loss`，防止实验身份混淆。

相对 WSSS Student 的最佳验证结果，全监督上界的 mIoU 提高
`0.2288627043`，foreground mIoU 提高 `0.2339661873`，pixel accuracy
提高 `0.1926960441`。全监督验证 mIoU 在 epoch 20 达到最高值，训练期间未
出现 WSSS Student 在 epoch 12 后那样明显的伪标签噪声过拟合。为保持训练
预算一致，本次上界仍固定为 20 epochs，不继续用验证集延长训练。

```text
stitched test metrics (14 parents / 504,000,000 pixels):
mIoU:             0.7318
foreground mIoU:  0.8113
pixel accuracy:   0.8821

stitched test class IoU:
background:          0.3342
impervious_surface:  0.8331
building:            0.9118
low_vegetation:      0.7265
tree:                0.7536
car:                 0.8313
```

当前 WSSS Student 的 stitched test mIoU 保留率为 `71.87%`，foreground
mIoU 保留率为 `73.75%`。WSSS 相对全监督的 class IoU 缺口从大到小为
tree `0.2972`、low vegetation `0.2610`、impervious surface `0.2365`、
background `0.1703`、building `0.1393`、car `0.1306`。后续方法改进优先
面向 tree、low vegetation 与 impervious surface，building/car 暂不作为
主要瓶颈。

### 消融：去掉 CAM Encoder 初始化

```text
日期: 2026-08-04
与主方法相比的唯一改动: Student 使用 ImageNet ResNet-50 初始化，不加载 CAM checkpoint
伪标签: cam_sam_background_only_full_train_corrected_v2（保持不变）
损失/训练预算/划分: 与主方法保持不变
best epoch: 11
train loss: 0.6141521892619922
validation loss: 1.0520522979398568
validation mIoU: 0.4625677122535825
validation foreground mIoU: 0.5257087413776282
validation pixel accuracy: 0.6295606943165989
```

```text
validation class IoU:
background:          0.1468625666333539
impervious_surface:  0.43907346526679997
building:            0.7065317692314996
low_vegetation:      0.43947912271807404
tree:                0.4051002338046895
car:                 0.6383591158670779
```

加载 CAM encoder 的主方法相对该消融将验证 mIoU 提高 `0.0249933349`、
foreground mIoU 提高 `0.0236149215`、pixel accuracy 提高 `0.0417669541`。
class IoU 变化为 background `+0.0319`、impervious surface `+0.0331`、
building `+0.0599`、low vegetation `+0.0482`、tree `-0.0032`、car
`-0.0199`。因此 CAM 初始化总体有效，但收益集中在前三类大面积地物，并非
所有类别一致改善。

### 消融：去掉 CAM Exact-Zero 背景种子

```text
日期: 2026-08-04
与主方法相比的唯一改动: 使用 SAM3 Prompt4-only 伪标签，不加入背景种子
Student 初始化: CAM encoder（保持不变）
损失/训练预算/划分: 与主方法保持不变
best epoch: 12
train loss: 0.2894199748441358
validation loss: 1.283834427439918
validation mIoU: 0.4539533188714562
validation foreground mIoU: 0.5447439826457474
validation pixel accuracy: 0.6763497795853379
```

```text
validation class IoU:
background:          0.0
impervious_surface:  0.4662236303190551
building:            0.7896597917577922
low_vegetation:      0.5059507916902245
tree:                0.3353462038254278
car:                 0.626539495636237
```

加入 exact-zero 背景种子的主方法相对该消融将验证 mIoU 提高
`0.0336077283`，foreground mIoU 提高 `0.0045796803`，pixel accuracy
降低 `0.0050221312`。class IoU 变化为 background `+0.1787`、impervious
surface `+0.0059`、building `-0.0233`、low vegetation `-0.0183`、tree
`+0.0666`、car `-0.0081`。该策略的主要价值是使第六类 background 可学习，
并明显减少 tree 混淆；它对标准五前景类的总体提升有限，且存在类别间取舍。

### 消融：SAM3-Only + ImageNet 初始化

```text
日期: 2026-08-04
伪标签: SAM3 Prompt4-only，不加入背景种子
Student 初始化: ImageNet ResNet-50，不加载 CAM checkpoint
损失/训练预算/划分: 与主方法保持不变
best epoch: 4
train loss: 0.39543757542474745
validation loss: 1.0101072518154979
validation mIoU: 0.46812475300810963
validation foreground mIoU: 0.5617497036097315
validation pixel accuracy: 0.7068264757049412
```

```text
validation class IoU:
background:          0.0
impervious_surface:  0.5297253187771079
building:            0.777956602004426
low_vegetation:      0.5434054826563928
tree:                0.3576415501948277
car:                 0.6000195644159032
```

### Background × CAM 初始化 2×2 验证矩阵

| Background seeds | CAM init | mIoU | foreground mIoU | pixel accuracy |
| --- | --- | ---: | ---: | ---: |
| no | no | 0.4681 | 0.5617 | 0.7068 |
| no | yes | 0.4540 | 0.5447 | 0.6763 |
| yes | no | 0.4626 | 0.5257 | 0.6296 |
| yes | yes | **0.4876** | 0.5493 | 0.6713 |

无背景种子时，CAM 初始化对 mIoU 的作用为 `-0.0141714341`；有背景种子时
为 `+0.0249933349`，二者的 mIoU 交互项为 `+0.0391647691`。foreground
mIoU 交互项为 `+0.0406206425`。这表明 CAM encoder 与 CAM exact-zero
背景监督存在明显耦合：单独加载 CAM encoder 并不稳定，只有在训练标签也
包含与其来源一致的背景证据时才产生正收益。

同时，主方法相对真正 SAM3-only 基线的总 mIoU 提高 `0.0194362942`，但
foreground mIoU 降低 `0.0124260407`。因此当前方法适合六类（包含
background/clutter）目标，却尚未证明对 Potsdam 标准五前景类更优。上述
差异目前来自单随机种子，后续需要重复种子确认交互是否稳定。

### Background × CAM 初始化 2×2 锁定 Test 矩阵

| Background seeds | CAM init | mIoU | foreground mIoU | pixel accuracy | background IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| no | no | 0.5068 | **0.6082** | **0.7435** | 0.0000 |
| no | yes | 0.4999 | 0.5999 | 0.7284 | 0.0000 |
| yes | no | 0.5048 | 0.5716 | 0.6865 | **0.1706** |
| yes | yes | **0.5259** | 0.5983 | 0.7205 | 0.1640 |

test 结论与 validation 一致。主方法六类 mIoU 最佳；SAM3-only + ImageNet
的五前景 mIoU 和 pixel accuracy 最佳。主方法相对后者提高 `0.0191` 六类
mIoU，但降低 `0.0099` foreground mIoU 和 `0.0230` pixel accuracy。CAM
初始化在有背景时贡献约 `+0.0211` test mIoU，在无背景时贡献约
`-0.0069`，test 交互约为 `+0.0280`。

按类比较，主方法相对 SAM3-only + ImageNet 提高 background `0.1640`、
tree `0.0583` 和 car `0.0098`，但降低 impervious surface `0.0541`、
building `0.0208` 和 low vegetation `0.0422`。下一步在固定主方法伪标签和
CAM 初始化的条件下，仅扫描 ToCo background loss 权重，尝试保留背景/tree
收益并恢复其余前景类。

### Background Loss Weight 验证扫描（完成）

| background weight | best epoch | mIoU | foreground mIoU | pixel accuracy | background IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 12 | 0.4876 | 0.5493 | 0.6713 | 0.1787 |
| 0.50 | 8 | 0.4886 | 0.5469 | 0.6601 | **0.1976** |
| 0.25 | 16 | **0.4901** | **0.5506** | **0.6788** | 0.1872 |
| 0.10 | 16 | 0.4852 | 0.5485 | 0.6826 | 0.1687 |

`0.25` 是阶段一验证最优权重，相对 `1.0` 将 mIoU 提高约 `0.0025`、
foreground mIoU 提高约 `0.0013`、pixel accuracy 提高约 `0.0075`。
但 class IoU 变化并不均匀：impervious surface、building、car 和 background
分别约提高 `0.0087/0.0285/0.0253/0.0085`，tree 约降低 `0.0556`，low
vegetation 基本不变。`0.10` 的 mIoU 回落到 `0.4852`，且 tree IoU 继续
下降到 `0.3221`，因此停止继续细分权重，并按预先固定的 validation mIoU
规则冻结 `0.25`。由于总体增益较小且来自单 seed，尚不能视为稳定改进；
下一步仅对冻结的 `0.25` 做一次 test，之后若要宣称提升必须补多 seed。

```text
冻结 weight=0.25 stitched test:
mIoU:             0.5216  (vs 1.00: -0.0043244574)
foreground mIoU:  0.5996  (vs 1.00: +0.0012509892)
pixel accuracy:   0.7272  (vs 1.00: +0.0067072361)
background IoU:   0.1318  (vs 1.00: -0.0322016907)
best epoch: 16
```

相对 `1.0`，weight `0.25` 的 class IoU 变化为 impervious surface
`+0.0054`、building `+0.0348`、low vegetation `-0.0082`、tree
`-0.0437`、car `+0.0180`。它轻微提高标准五前景均值和 pixel accuracy，
但六类 mIoU 未泛化，且 background/tree 明显下降。由于预先选择指标为六类
validation mIoU，正式主方法回退并固定默认 `1.0`；不再根据该 test 结果继续
搜索权重。加权损失保留为负结果和后续独立研究选项。

### 主方法 Validation 多随机种子

| seed | best epoch | mIoU | foreground mIoU | pixel accuracy | background IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 12 | 0.4876 | 0.5493 | 0.6713 | 0.1787 |
| 43 | 17 | 0.4791 | 0.5431 | 0.6479 | 0.1590 |
| 44 | 18 | 0.4835 | 0.5463 | 0.6624 | 0.1695 |
| mean | - | 0.4834 | 0.5462 | 0.6606 | 0.1691 |
| sample std | - | 0.0042 | 0.0031 | 0.0118 | 0.0099 |

主方法 validation mIoU/foreground mIoU 的样本标准差分别约为
`0.42/0.31` 个百分点，整体训练结论较稳定。pixel accuracy 和 background
IoU 波动更大，进一步表明背景学习是当前主要不稳定来源。seed 42 高于均值，
后续方法比较必须给 SAM3-only 基线补相同种子，并用均值/标准差而非单 seed
结论。

### Main vs SAM3-Only Validation 多种子对照

| method | mIoU mean ± std | foreground mIoU mean ± std | pixel accuracy mean ± std | background IoU mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Main | 0.4834 ± 0.0042 | 0.5462 ± 0.0031 | 0.6606 ± 0.0118 | 0.1691 ± 0.0099 |
| SAM3-only + ImageNet | 0.4702 ± 0.0022 | **0.5643 ± 0.0026** | **0.7012 ± 0.0126** | 0.0000 ± 0.0000 |
| paired Main - SAM3 | **+0.0131 ± 0.0064** | -0.0180 ± 0.0057 | -0.0407 ± 0.0194 | **+0.1691 ± 0.0099** |

SAM3-only seed 42/43/44 的最佳 epoch 为 4/4/8，validation mIoU 为
`0.4681/0.4725/0.4701`。三个配对 seed 上，Main 的六类 mIoU 都更高，
foreground mIoU 和 pixel accuracy 都更低。因此性能取舍不是 seed 42 的
偶然波动：当前背景策略稳定提供 background/clutter 能力并提高六类均值，
但稳定牺牲标准五前景均值。下一步只对已由 validation 锁定的六个 checkpoint
各做一次 test 推理，不能再根据 test 调参。

### Main vs SAM3-Only Stitched Test 多种子对照

| method | mIoU mean ± std | foreground mIoU mean ± std | pixel accuracy mean ± std | background IoU mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Main | **0.5244 ± 0.0014** | 0.5964 ± 0.0017 | 0.7137 ± 0.0061 | **0.1642 ± 0.0044** |
| SAM3-only + ImageNet | 0.5081 ± 0.0035 | **0.6097 ± 0.0042** | **0.7396 ± 0.0106** | 0.0000 ± 0.0000 |
| paired Main - SAM3 | **+0.0163 ± 0.0038** | -0.0133 ± 0.0053 | -0.0260 ± 0.0114 | **+0.1642 ± 0.0044** |

Main 的 test class IoU 均值为 background `0.1642`、impervious surface
`0.5835`、building `0.7710`、low vegetation `0.4419`、tree `0.4679`、
car `0.7179`。SAM3-only 对应为 `0/0.6245/0.7812/0.5097/0.4348/0.6983`。
主方法稳定提升 background、tree 和 car，却稳定损失 impervious surface、
building 和 low vegetation。基于该重复结果，下一版不再调整全局背景权重，
而是新增 background-vs-foreground 与条件前景语义解耦损失；所有新选择仍只
使用 validation。

### Background Loss Weight 验证扫描（完成）

| background weight | best epoch | mIoU | foreground mIoU | pixel accuracy | background IoU | tree IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 12 | 0.4876 | 0.5493 | 0.6713 | 0.1787 | 0.4019 |
| 0.50 | 8 | 0.4886 | 0.5469 | 0.6601 | **0.1976** | **0.4117** |
| 0.25 | 16 | **0.4901** | **0.5506** | 0.6788 | 0.1872 | 0.3463 |
| 0.10 | 16 | 0.4852 | 0.5485 | **0.6826** | 0.1687 | 0.3221 |

验证扫描到此停止。按预先声明的 validation mIoU 选择 `0.25`，不继续搜索
更细权重。`0.10` 的 mIoU 已回落，说明降低背景权重并非单调改善。`0.25`
相对 `1.0` 的提升很小且 tree 下降明显，因此只将其冻结为 test 候选；在
test 和多随机种子验证前，不宣称它优于默认权重。

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
