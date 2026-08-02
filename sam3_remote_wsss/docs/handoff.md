# SAM3 Remote WSSS 项目交接文档

更新日期：2026-08-02

## 1. 项目目标

本项目研究如何将 SAM3 用于 Potsdam 遥感图像的 image-level 弱监督语义分割（WSSS）。训练阶段只向模型提供每张图像或 patch 的类别存在标签，不向 SAM3、CAM 分类器或 student 分割网络输入像素级 GT。

当前主路线为：

```text
Potsdam 原始大图
-> 512 x 512 重叠 patch 和 patch-level image tags
-> RemoteCLIP/B2C 风格类别文本模板
-> SAM3 生成类别相关的高精度前景掩码
-> CAM 补充前景覆盖并通过排除法确定背景
-> 冲突区域设为 255 ignore
-> 融合语义伪标签
-> SegFormer-style student
```

SAM3-only 仍保留为基线。PromptBG 也保留为负向消融，但不再作为主路线。

## 2. 方法定位

SAM3 在当前方案中负责生成具有类别语义和较好边界的前景掩码。它不是训练中的普通无类别实例提议器，因为输入给 SAM3 的是由图像级正类标签构造的类别文本 prompt。

CAM 分类器负责两个 SAM3 不擅长的问题：

1. 补充 SAM3 没有覆盖的前景区域。
2. 当所有正类 CAM 响应都很低时，用排除法生成背景种子。

Potsdam 的 background/clutter 是异质残余类别，不是一个外观统一的具体目标。因此，直接用 `background`、`clutter` 等抽象文本让 SAM3 分割背景的效果很差。

当前 CAM/SAM3 融合规则：

```text
SAM3 前景 + CAM 没有强烈反对       -> 保留 SAM3 类别
SAM3 未覆盖 + CAM >= 前景阈值      -> CAM 补充前景
SAM3 未覆盖 + 所有 CAM <= 背景阈值 -> 背景 0
SAM3 与 CAM 强冲突                 -> ignore 255
中间置信区域                        -> ignore 255
```

默认阈值：

```text
background_threshold = 0.2
foreground_threshold = 0.7
cam_support_threshold = 0.3
ignore_index = 255
```

## 3. 数据集和监督协议

类别定义：

| ID | 类别 |
| ---: | --- |
| 0 | background/clutter |
| 1 | impervious_surface |
| 2 | building |
| 3 | low_vegetation |
| 4 | tree |
| 5 | car |
| 255 | ignore/unlabeled |

Potsdam 官方大图约为 `6000 x 6000`。当前使用 `512 x 512` patch，重叠为 `128`。

为了模拟 patch-level image supervision，`prepare_potsdam_patches.py` 会使用 GT 统计每个 patch 中是否存在各前景类别，并生成 `image_level_labels.csv`。GT 同时被复制到隔离目录，供离线评估使用。SAM3、CAM 和 student 的训练代码不会读取像素 GT。

论文中必须明确说明：这些 image-level tags 是从像素 GT 派生的 patch-level tags。它们比原始父图级标签更局部，不能表述为未经处理的原始图像级标注。

### 当前最重要的数据限制

服务器上的 `/home/undergr/remote_dataset/Postdam_patches_512` 当前只有一个父图 `top_potsdam_2_10`，共 256 个 patch。这些数据只能用于工程 smoke test 和方法初筛，不能作为正式训练/验证结果，也不能与论文的完整 Potsdam 划分直接比较。

正式实验必须按父图划分 train/val/test。不能随机拆分重叠 patch，否则相邻区域会造成空间泄漏。

## 4. 已完成实现

### SAM3 伪标签

- Potsdam RGBIR 读取和 RGB 波段选择。
- RemoteCLIP/B2C 风格 prompt 模板。
- Prompt4，每个正类最多使用 4 个候选 prompt。
- SAM3 文本推理、掩码阈值过滤和类别融合。
- 伪标签 PNG、overlay JPG、metadata JSON 和 summary JSON。
- 未覆盖区域可设为 `255`。
- Potsdam 红色 clutter/background 已修复为类别 `0`，未知颜色映射为 `255`。
- 2080Ti/Turing 环境使用 FP32 fallback，避免 BF16/FP32 dtype 冲突。

### Patch 数据集

- 显式生成 512 patch、重叠坐标和 patch metadata。
- 每个 patch 单独生成 image-level 标签。
- 支持 `car=4` 的独立最小像素阈值。
- GT 与训练输入隔离，仅用于标签派生和评估。

### CAM/SAM3 混合模块

- ResNet 多标签 CAM 分类器。
- BCEWithLogitsLoss 和类别正样本权重。
- 90 度旋转、水平/垂直翻转、颜色扰动和模糊增强。
- 不使用随机裁剪，避免裁掉小车后仍保留 car 图像标签。
- 多尺度 CAM 和水平翻转 TTA。
- 每类 CAM 独立归一化并屏蔽图像级负类。
- CAM/SAM3 保守融合和逐图统计。
- CAM 编码器权重可初始化 student 的 ResNet 编码器。

### Student

- ResNet 多尺度编码器。
- SegFormer-style MLP decoder。
- ToCo LargeFOV/ASPP head 作为消融选项。
- 随机尺度裁剪、类别比例约束、翻转、90 度旋转、颜色增强和模糊。
- 小连通域清理和边界 ignore。
- 支持单卡和 `torch.nn.DataParallel` 双卡训练。

### 测试状态

最新本地代码已经通过：

```text
10/10 unit tests passed
Python compile check passed
train_cam CLI passed
generate_cams CLI passed
fuse_cam_sam CLI passed
CAM checkpoint -> student encoder loading smoke passed
```

CAM 代码已经完成，并已在服务器上完成 256 patch、1 epoch 的训练 smoke；尚未产生真实 CAM/SAM3 融合指标。

## 5. 已运行实验和结果

### 5.1 原始大图 SAM3 单图 smoke

```text
GPU: 1 x NVIDIA 2080Ti
images: 1
耗时: 约 5 分 09 秒
结果: 成功生成 pseudo label、overlay 和 metadata
```

这一步较慢是因为输入为原始大图并在内部切片。显式 512 patch 后速度明显提高。

### 5.2 Student smoke

```text
epochs: 1
loss: 6.9738
checkpoint: runs/student_smoke_from_sam3/checkpoints/last.pt
```

这里只证明数据读取、前向、反向和权重保存正常，不代表模型收敛。

### 5.3 历史 5 图标签错误实验

曾得到 `foreground_mIoU = 0.0630`，之后发现 CSV 只有表头和一张图的标签，其余四张没有各自的 image-level tags。该结果无效，不得用于结论。

### 5.4 修正标签后的 5 图历史结果

```text
impervious_surface IoU = 0.0610
building IoU           = 0.5040
low_vegetation IoU     = 0.0132
tree IoU               = 0.1053
car IoU                = 0.7392
foreground mIoU        = 0.2846
```

该实验早于背景映射修复，只作为工程历史记录。

### 5.5 显式 patch，默认 prompt，5 patch

```text
background IoU         = 0.0019
impervious_surface IoU = 0.000002
building IoU           = 0.6300
low_vegetation IoU     = 0.0000
tree IoU               = 0.1370
car IoU                = 0.8701
mIoU                    = 0.2732
foreground mIoU        = 0.3274
```

### 5.6 显式 patch，Prompt4，5 patch

```text
background IoU         = 0.0034
impervious_surface IoU = 0.5954
building IoU           = 0.6302
low_vegetation IoU     = 0.4063
tree IoU               = 0.4858
car IoU                = 0.8818
mIoU                    = 0.5005
foreground mIoU        = 0.5999
```

Prompt4 对地表、低矮植被和树木的改善明显，但样本只有 5 个 patch。

### 5.7 Prompt4，256 patch，硬背景补全

```text
background IoU         = 0.1766
impervious_surface IoU = 0.4130
building IoU           = 0.5807
low_vegetation IoU     = 0.2298
tree IoU               = 0.1395
car IoU                = 0.7696
mIoU                    = 0.3849
foreground mIoU        = 0.4265
```

硬背景补全会把所有 SAM3 未覆盖区域直接当作背景，噪声较大。

### 5.8 Prompt4，256 patch，未覆盖区域设为 255

严格指标将 `255` 当作未预测区域：

```text
foreground mIoU = 0.4255
mIoU            = 0.3546
```

只评价已标注区域：

```text
labeled foreground mIoU = 0.6213
labeled mIoU            = 0.5178
labeled coverage        = 0.5133
```

已标注区域各类 IoU：

```text
impervious_surface = 0.5016
building           = 0.8255
low_vegetation     = 0.5190
tree               = 0.3616
car                = 0.8988
```

核心结论：SAM3 已生成的建筑和车辆伪标签较准确，但只覆盖约 51.3% 像素。树木和低矮植被的覆盖与区分仍是主要瓶颈。

### 5.9 PromptBG 消融

在 5 个 patch 上，PromptBG 生成的有效背景掩码数量为 `0`。典型最大 score 约为：

```text
clutter and miscellaneous objects = 0.1267
unclassified background           = 0.000034
boundary clutter                  = 0.0153
miscellaneous non-target regions  = 0.0251
```

结论：抽象背景文本不适合作为主背景生成方法，后续改用 CAM 排除法。

### 5.10 CAM 分类器，256 patch，1 epoch smoke

```text
Git commit              = 43366c6
backbone                = ImageNet pretrained ResNet-50
GPU                     = 1 x NVIDIA 2080Ti
loss                    = 0.4549
micro F1                = 0.8283
impervious_surface F1   = 0.9082
building F1             = 0.8328
low_vegetation F1       = 0.9719
tree F1                 = 0.6038
car F1                  = 0.6481
checkpoint              = runs/cam_smoke_256/checkpoints/last.pt
checkpoint size         = 270 MB
```

这些是训练批次上的分类指标，不是独立验证结果，也不能代表 CAM 空间定位质量。下一步必须生成 CAM 热力图，人工确认目标定位后再进行融合。

首批 CAM overlay 已完成人工检查。CAM 形成了局部热点，但多数类别只有少量离散响应，未覆盖完整区域；low vegetation 在车辆、硬质地面和屋顶附近出现错误响应，tree 与 low vegetation 以及其他共现类别的 CAM 也较为相似。这说明单父图、1 epoch 分类器存在明显的共现捷径。当前 CAM 仅适合继续做 5 patch 融合闭环测试，不适合直接作为 256 patch 正式伪标签。

5 patch 融合闭环随后成功运行，但没有带来指标提升：strict mIoU 为 `0.4993`，strict foreground mIoU 为 `0.5990`，labeled coverage 为 `0.9062`。约 41.5 万像素被填成背景，而其中只有 323 个是真背景，background IoU 仅为 `0.00078`。这说明高覆盖率主要来自错误背景填充。当前 `background_threshold=0.2` 的 CAM 低响应排除策略判定失败，在重新设计背景门控前不得扩展到全部 256 patch。

同一 5 patch 的公平 SAM3 Ignore255 基线为：labeled foreground mIoU `0.9064`、coverage `0.5866`。融合后 labeled foreground mIoU 降至 `0.6462`，coverage 升至 `0.9062`。像素统计显示 CAM 仅补充 4,185 个前景像素，而背景规则填充了 414,923 个像素。由此确认 SAM3-only 当前是高精度、低覆盖的有效伪标签源，失败点主要在 CAM 背景排除，而不是 SAM3 前景掩码。

全部 256 patch 的背景阈值扫描也已完成。背景阈值从 `0.00` 增至 `0.20` 时，coverage 从 `0.5250` 增至 `0.8927`，但 labeled foreground mIoU 从 `0.6113` 降至 `0.4659`；strict foreground mIoU 始终约为 `0.4248`。`0.10` 虽得到最高总 mIoU `0.3858`，提升完全来自背景类别，不能据此认定 CAM 前景有效。当前只考虑 `0.00/0.01` 作为保守背景种子候选，并需进一步比较背景 precision/recall。

背景 precision/recall 检查后，最终保守候选确定为 `background_threshold=0.00`：背景 precision `0.8875`、recall `0.0485`，得到 450,472 个正确背景种子。阈值提高到 `0.01` 时 precision 已降至 `0.5338`，不可用于高置信伪标签。下一步固定背景为 `0.00` 并扫描 CAM 前景阈值。

前景阈值扫描显示，阈值从 `0.70` 升到 `1.00` 时，CAM 新增前景从 329,899 降至 531，labeled foreground mIoU 反而从 `0.6113` 恢复到 `0.6208`，接近 SAM3-only 的 `0.6213`。因此当前正式候选改为 `background_only`：SAM3 独占前景，CAM 只在所有正类响应恰为零的位置提供背景种子，其他区域仍为 `255`。full hybrid 保留为消融。

`background_only` 的 256 patch 最终结果为：mIoU `0.3626`、foreground mIoU `0.4255`、labeled mIoU `0.5386`、labeled foreground mIoU `0.6208`、coverage `0.5209`。与 SAM3 Ignore255 相比，foreground mIoU 完全不变，总 mIoU 提升 `0.0080`，labeled mIoU 提升 `0.0209`。新增 507,581 个背景种子，其中 450,472 个正确，precision 为 `0.8875`。这是当前最可信的伪标签策略，但仍只在单一父图上验证。

background-only student 双卡 smoke 也已完成。使用 CAM checkpoint 初始化 ResNet-50、SegFormer-style head、2 x 2080Ti DataParallel 和 AMP，两次独立 1 epoch 运行的 loss 分别为 `4.6935`、`4.6305`，最新 checkpoint 为 `runs/student_background_only_smoke/checkpoints/last.pt`，大小 `284 MB`。训练闭环正常，但没有独立验证指标。

完整 Potsdam 已清点：38 张 RGBIR、38 张对应标签，共 9.0 GB。正式父图清单为 17 train、6 val、14 test，并排除 `top_potsdam_7_10`。按 512 patch、128 overlap 预计生成 9,472 个 patch，其中 train 4,352、val 1,536、test 3,584。代码已支持 `--parent-split`，自动生成三份 split-specific image-label CSV，并拒绝父图重复、遗漏或未知 ID。

## 6. 关键代码入口

| 功能 | 文件/模块 |
| --- | --- |
| 配置读取 | `src/sam3_remote_wsss/config.py` |
| Potsdam 读取与标签映射 | `src/sam3_remote_wsss/potsdam.py` |
| 显式 patch 数据集 | `src/sam3_remote_wsss/prepare_potsdam_patches.py` |
| Prompt 模板 | `src/sam3_remote_wsss/prompts.py` |
| SAM3 后端 | `src/sam3_remote_wsss/sam3_backend.py` |
| SAM3 伪标签 | `src/sam3_remote_wsss/generate_pseudo_labels.py` |
| 伪标签评估 | `src/sam3_remote_wsss/evaluate_pseudo_labels.py` |
| CAM 模型 | `src/sam3_remote_wsss/cam/model.py` |
| CAM 数据集 | `src/sam3_remote_wsss/cam/dataset.py` |
| CAM/SAM3 融合规则 | `src/sam3_remote_wsss/cam/fusion.py` |
| CAM 训练 | `src/sam3_remote_wsss/train_cam.py` |
| CAM 生成 | `src/sam3_remote_wsss/generate_cams.py` |
| CAM/SAM3 融合入口 | `src/sam3_remote_wsss/fuse_cam_sam.py` |
| Student 训练 | `src/sam3_remote_wsss/train_student.py` |
| SegFormer head | `src/sam3_remote_wsss/student/segformer_head.py` |

## 7. 机器和路径

### Windows 本地仓库

```text
C:\Users\28457\Desktop\Sheung_project1\Sheung_project1_sam3_remote_wsss
```

### 实验室服务器

```text
仓库根目录:     /home/undergr/Sheungzhen_project_1
WSSS 工程:      /home/undergr/Sheungzhen_project_1/sam3_remote_wsss
SAM3 源码:      /home/undergr/Sheungzhen_project_1/sam3-main/sam3-main
SAM3 checkpoint:/home/undergr/Sheungzhen_project_1/checkpoints/sam3.pt
Potsdam 原图:   /home/undergr/remote_dataset/Postdam
当前 patch:     /home/undergr/remote_dataset/Postdam_patches_512
Conda 环境:     sam3_wsss
GPU:            2 x NVIDIA 2080Ti
```

当前 patch 配置：

```text
/home/undergr/remote_dataset/Postdam_patches_512/potsdam_patches_config_prompt4.json
```

## 8. 环境注意事项

已经遇到并解决的问题：

- `pip install -e .` 必须在含 `pyproject.toml` 的项目目录执行。
- SAM3 必须单独从 SAM3 仓库执行 editable install。
- 缺失依赖包括 `einops`、`pycocotools`、`psutil` 等。
- SAM3 要求 `numpy>=1.26,<2`。
- OpenCV 安装时可能自动把 NumPy 升级到 2.x，应使用 `--no-deps` 固定版本。
- 2080Ti 不支持 Ampere Flash Attention，相关 warning 正常。
- 2080Ti 上 BF16 输入与 FP32 权重会报 dtype mismatch，当前 SAM3 后端已有 FP32 fallback。

推荐兼容版本：

```bash
pip install --force-reinstall "numpy==1.26.4"
pip install --force-reinstall --no-deps "opencv-python==4.11.0.86"
```

`pkg_resources is deprecated` 和 Flash Attention disabled 属于 warning，不是运行失败。

## 9. 当前 Git 状态

当前本地分支为 `main`，上一个已提交节点为：

```text
6fbfe38 add prompted background pseudo-label fusion
```

CAM/SAM3 混合模块、README 更新、测试和构建产物清理目前位于本地工作区，尚未提交。提交前应再次执行测试。

本轮还删除了被误提交的 `*.egg-info` 构建产物，并将 `*.egg-info/`、`__pycache__/` 和 `*.py[cod]` 加入 `.gitignore`。更新到该提交后，服务器的 editable install 不应再反复污染 Git 状态。

## 10. 最近一次代码同步流程

Windows 提交：

```powershell
cd C:\Users\28457\Desktop\Sheung_project1\Sheung_project1_sam3_remote_wsss
python -m unittest discover -s sam3_remote_wsss\tests -v
git add .
git commit -m "add CAM and SAM3 hybrid pseudo-label pipeline"
git push origin main
```

服务器更新：

```bash
cd ~/Sheungzhen_project_1

# 仅当旧的 editable install 产物阻塞第一次 pull 时执行
git restore -- sam3_remote_wsss/src/sam3_remote_wsss.egg-info/SOURCES.txt

git pull --ff-only origin main
cd sam3_remote_wsss
conda activate sam3_wsss
python -m pip install -e . --no-deps
```

验证新模块：

```bash
python - <<'PY'
from sam3_remote_wsss.cam.model import CAMClassifier
from sam3_remote_wsss.cam.fusion import fuse_cam_and_sam

print("CAM/SAM3 code available")
PY
```

## 11. 下一轮推荐实验

### 11.1 先跑 256 patch CAM smoke

```bash
cd ~/Sheungzhen_project_1/sam3_remote_wsss
conda activate sam3_wsss
export PATCH_ROOT=/home/undergr/remote_dataset/Postdam_patches_512

CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.train_cam \
  --config "$PATCH_ROOT/potsdam_patches_config_prompt4.json" \
  --labels-csv "$PATCH_ROOT/image_level_labels.csv" \
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

`--pretrained-backbone` 第一次运行可能需要下载 TorchVision ImageNet 权重。服务器无法联网时，应先在本地下载并传到 Torch 缓存；不要直接用一个父图从零训练并将结果解释为有效 CAM。

### 11.2 生成 5 个 CAM 并检查可视化

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_cams \
  --config "$PATCH_ROOT/potsdam_patches_config_prompt4.json" \
  --labels-csv "$PATCH_ROOT/image_level_labels.csv" \
  --checkpoint runs/cam_smoke_256/checkpoints/last.pt \
  --output-dir runs/cam_smoke_256/cams \
  --scales 1.0 \
  --visualize-limit 5 \
  --limit 5 \
  --amp
```

先检查 `runs/cam_smoke_256/cams/visualizations`。重点观察 car 是否集中在小目标、building 是否落在屋顶、vegetation/tree 是否大面积混淆。

### 11.3 融合 CAM 和已有 SAM3 Ignore255 伪标签

```bash
python -m sam3_remote_wsss.fuse_cam_sam \
  --config "$PATCH_ROOT/potsdam_patches_config_prompt4.json" \
  --labels-csv "$PATCH_ROOT/image_level_labels.csv" \
  --sam-pseudo-label-dir runs/sam3_prompt4_256patches_ignore255/pseudo_labels \
  --cam-dir runs/cam_smoke_256/cams \
  --output-dir runs/cam_sam_fused_smoke \
  --background-threshold 0.2 \
  --foreground-threshold 0.7 \
  --cam-support-threshold 0.3
```

只有生成全部 256 个 CAM 后才能对 256 个 patch 做完整融合。上一步使用 `--limit 5` 时，本步也只能融合相同的 5 个 patch。

### 11.4 评估融合伪标签

```bash
python -m sam3_remote_wsss.evaluate_pseudo_labels \
  --config "$PATCH_ROOT/potsdam_patches_config_prompt4.json" \
  --pseudo-label-dir runs/cam_sam_fused_smoke/pseudo_labels \
  --output runs/cam_sam_fused_smoke/pseudo_metrics.json
```

至少记录：

- strict mIoU 和 foreground mIoU。
- labeled mIoU 和 labeled foreground mIoU。
- labeled coverage。
- 每类 IoU 和每类覆盖率。
- 背景像素、CAM 补充像素、SAM 保留像素、冲突像素和 ignore 像素。

### 11.5 训练融合标签 Student

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sam3_remote_wsss.train_student \
  --config "$PATCH_ROOT/potsdam_patches_config_prompt4.json" \
  --pseudo-label-dir runs/cam_sam_fused/pseudo_labels \
  --cam-checkpoint runs/cam_resnet50/checkpoints/last.pt \
  --output-dir runs/student_cam_sam_segformer \
  --epochs 20 \
  --batch-size 4 \
  --crop-size 512 \
  --backbone resnet50 \
  --output-stride 16 \
  --head segformer \
  --segformer-embed-dim 256 \
  --samples-per-image 16 \
  --cat-max-ratio 0.75 \
  --min-component-area 16 \
  --ignore-boundary-width 1 \
  --data-parallel \
  --amp
```

## 12. 正式实验矩阵

正式数据准备完成后，建议固定父图划分并至少比较：

| 实验 | 伪标签来源 | 背景策略 | Student 初始化 |
| --- | --- | --- | --- |
| A | CAM-only | CAM 排除 | CAM encoder |
| B | SAM3 Prompt1 | 255 ignore | ImageNet/random |
| C | SAM3 Prompt4 | 255 ignore | ImageNet/random |
| D | SAM3 Prompt4 | 硬背景补全 | ImageNet/random |
| E | SAM3 Prompt4 + PromptBG | 背景 prompt | ImageNet/random |
| F | CAM + SAM3 Prompt4 | CAM 排除与冲突过滤 | CAM encoder |
| G | CAM + SAM3 Prompt4 | CAM 排除与冲突过滤 | 不加载 CAM encoder |

主要方法应以 F 为主，A/B/C/D/E/G 用于证明各模块贡献。

## 13. 尚未完成

- 尚未把完整 Potsdam 数据集切成 patch。
- 已建立 17/6/14 父图划分清单，尚未在服务器完成全量 patch 生成。
- 已完成单父图 256 patch 的 1 epoch CAM smoke，尚未在完整父图划分上正式训练。
- 已完成单父图 CAM/SAM3 hybrid 和 background-only 指标，尚未在正式多父图划分上验证。
- 尚未完成 student 独立验证集推理、patch 拼接和最终 mIoU 闭环。
- 已在单父图 256 patch 上扫描背景和前景阈值，正式数据仍需复核。
- 尚未完成完整 Prompt1/Prompt4/RemoteCLIP 排序消融。
- 尚未统计两张 2080Ti 上完整实验的运行时间和显存。

## 14. 交接给新 Codex 任务的文本

新开任务时发送下面这段即可：

```text
请先阅读 sam3_remote_wsss/docs/handoff.md 和 README.md，再继续当前项目。

这是一个 Potsdam image-level WSSS 工程。当前方法用 RemoteCLIP/B2C 风格
Prompt4 驱动 SAM3 生成高精度但稀疏的前景伪标签，再用多标签 CAM 补充前景、
通过低前景响应排除出背景，并将 CAM/SAM3 冲突置为 255，最终训练 SegFormer。

SAM3-only 的 256 patch 已标注区域前景 mIoU 为 0.6213，覆盖率为 0.5133；
但这 256 个 patch 只来自一个父图，只能视为 smoke。PromptBG 已验证失败。
CAM/SAM3 代码、CAM 热力图检查、阈值扫描、background-only 伪标签评估和双卡 student smoke 均已完成。当前最佳策略是 SAM3 负责前景、CAM exact-zero 负责高精度背景、其余像素为 255。
完整 Potsdam 已清点为 38 对父图/标签，正式 split 为 17 train、6 val、14 test、排除 7_10；下一步是生成 /home/undergr/remote_dataset/Postdam_patches_512_full。

请基于现有实现继续，不要重新设计。先检查 Git 状态和服务器是否拉取最新提交，
然后运行 256 patch CAM smoke、检查 CAM 可视化，再决定是否生成完整 Potsdam patch
数据集并建立按父图划分的正式实验。
```
