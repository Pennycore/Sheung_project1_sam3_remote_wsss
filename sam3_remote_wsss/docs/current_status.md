# 项目当前状态

本文档是项目的长期交接页。新开 Codex 任务、换电脑或换服务器时，先让协作者阅读本页，再阅读 `environment_setup.md` 和 `runbook.md`。

## 1. 研究目标

目标是把 SAM3 用于 Potsdam 遥感数据集的 image-level 弱监督语义分割（WSSS）：

```text
图像级类别标签
-> RemoteCLIP/B2C 风格文本提示词
-> SAM3 在重叠小块上生成实例掩码
-> 过滤、冲突处理与拼接
-> 语义伪标签
-> SegFormer-style student 分割网络
```

目前路线刻意保留了 “SAM3-only” 基线：不先训练分类网络，而是直接用图像级正类生成文本 prompt。RemoteCLIP 模块目前只负责对候选 prompt 排序，不负责生成新的类别标签。

## 2026-08-07 最新状态

当前正式协议已完成17/6/14父图互斥划分、512 patch训练、拼接test评估和三随机种子
复现。主方法 stitched test mIoU/foreground mIoU 为 `0.5244 +/- 0.0014` 和
`0.5964 +/- 0.0017`；SAM3-only为 `0.5081 +/- 0.0035` 和
`0.6097 +/- 0.0042`；同结构全监督结果为 `0.7318/0.8113`。历史Prompt4已更正命名
为Manual4领域提示集成；固定256-patch提示消融中，Manual4 foreground mIoU
`0.4560`，优于Prompt1 `0.2427`、B2C4 `0.2763`、CLIP-Top4 `0.3609` 和
RemoteCLIP-Top4 `0.2788`。

当前下一阶段是伪标签语义校正，不再继续搜索分割头和背景损失权重。固定256-patch缓存
已完成，共2,493个候选，且重建伪标签与Manual4基线完全一致。只读
`analyze_candidate_quality` 已在服务器完成：候选像素加权纯度为 `0.7796`，主导类别
一致率为 `0.8556`；背景污染仅 `0.0175`，其他前景污染为 `0.2029`。主要错误是
impervious surface 与 building/low vegetation、low vegetation 与 tree、car 与 tree 之间的
语义混淆。统一 `support>=2` 会把总体 recall 从 `0.5289` 降到 `0.2432`，因此不采用硬提示
一致性门槛。代码现新增只读 `analyze_candidate_cam`，下一步使用已生成的正式 CAM 文件离线
复核候选语义，不重新运行 SAM3，也不立即训练新 student。旧章节中的早期“尚未完成”描述
保留作历史记录，本节为最新状态。

从本次更新开始，所有像素级伪标签、student validation、patch test 和 stitched test 统一至少
报告六类宏平均 `mIoU`、六类宏平均 `mF1` 与总体像素准确率 `OA`，并继续报告逐类 IoU/F1、
`foreground_mIoU`、`foreground_mF1`。伪标签另分 strict 与 labeled-only 两套指标并同时报告
coverage；`pixel_accuracy` 仅作为 `OA` 的兼容别名保留。候选实例诊断不是完整语义图，因此仍
使用 purity、precision、recall、拒绝率等候选级指标，生成最终伪标签后再计算 mF1/OA/mIoU。

CAM 候选复核已完成。top20 的过滤后主导正确率/像素加权纯度为 `0.9262/0.8846`，mean 为
`0.9216/0.8745`；但 mean 在 impervious surface 和 low vegetation 上分别多保留约
`8.39/7.82` 个百分点的正确候选。考虑覆盖不足，下一组探索性规则冻结为 mean CAM 且只过滤
这两类。新增无 GT 的 `rebuild_candidate_pseudo_labels`，下一步离线重建 baseline、all-class
mean 和 selective mean 三组完整伪标签并比较 mIoU/mF1/OA，不立即训练 student。

固定256-patch重建评估已完成。baseline/all-class/selective mean 的 mIoU 为
`0.3800/0.3397/0.3922`，mF1 为 `0.4912/0.4590/0.5028`，OA 为
`0.4713/0.4028/0.4670`，coverage 为 `0.6063/0.4717/0.5437`。全类别过滤失败；选择性
过滤将 mIoU/mF1/foreground mIoU 提高 `0.0122/0.0116/0.0147`，但 OA/coverage 下降
`0.0043/0.0627`。下一步在父图互斥 validation 子集原样复现该规则，不再使用当前训练子集
GT 调整类别或参数。

父图互斥 validation 机制子集已冻结为 `data/candidate_validation_256.csv`：共256个 patch，
覆盖全部6张 validation 父图，每张42或43个；五类正样本数为 `221/193/249/230/110`。
下一步用冻结的训练 CAM checkpoint 和 Manual4 在该CSV上生成 CAM 与SAM3候选，随后只比较
baseline 和既定 selective mean 规则。

validation 256 的 CAM 与 Manual4 候选缓存已完整生成，共2,540个候选；score/面积中位数为
`0.7149/3188.5`，与训练诊断子集接近。下一步执行无 GT baseline/selective mean 重建并用
validation GT 一次性比较 mIoU/mF1/OA，不再生成新的规则分支。

validation 的最后一次2x2类别分解已经完成。baseline、impervious-only、low-vegetation-only、
两类联合的 mIoU 为 `0.3494/0.3629/0.3474/0.3609`，mF1 为
`0.4599/0.4733/0.4575/0.4710`。impervious-only 是唯一同时提高 mIoU、mF1 及两项前景指标
的规则，增量分别为 `+0.0135/+0.0134/+0.0162/+0.0161`；low-vegetation-only 的 mIoU/mF1
反而下降 `0.0020/0.0024`。最终候选校正规则冻结为 `mean CAM + impervious_surface-only
reject`，停止类别组合搜索。它的代价是 OA 从 `0.3935` 降至 `0.3795`、coverage 从
`0.5336` 降至 `0.4421`，论文中必须同时报告。下一步先按六张 validation 父图检查逐图增益和
父图宏平均稳定性，通过后才扩展到完整4,352训练 patch。

逐父图稳定性检查已经通过：impervious-only 的父图宏平均 mIoU 从 `0.3406 +/- 0.0450`
提高到 `0.3544 +/- 0.0380`，六张父图中五张的 mIoU 和 mF1 同时提高。因此候选校正规则保持
冻结，不再调参。随后重新审视公开对比协议，决定不替换 Potsdam，而是保留它作为主数据集并
计划增加 LoveDA 泛化实验；当前先建立与主流 Potsdam WSSS 工作一致的独立256协议。

代码已加入 `edge_mode=pad`、`--ignore-background-labels` 和
`configs/potsdam_parent_split_23_0_14_paper.json`。论文对齐协议采用23张训练父图、14张官方
评估父图、排除 `top_potsdam_7_10`、256非重叠网格及 clutter ignore。每张6000x6000父图
生成24x24=576个固定大小 patch，边缘采用 padding；全量预期为13,248个训练 patch、8,064个
测试 patch。下一步先在服务器生成单父图576-patch smoke 并核验配置，再生成全量数据。

论文交接文档的候选修复主线已经接入。新增
`analyze_candidate_recoverability` 离线诊断工具，在现有 Train256 候选缓存上比较原始融合、
Oracle Reject 和受 image-level 正类约束的 Oracle Relabel；同时计算 Geometric Recall、
Semantic Recall、Recoverable Semantic Gap、两种 SAM-to-GT confusion 以及各前景错配对的 CAM
纠正率和 margin 分布。该工具只读取 GT 做上限诊断，不写训练伪标签，也不据此直接设正式阈值。
当前下一步是在服务器现有2,493个候选上运行该诊断，判断“保留几何并修正语义”是否具有足够
上限，再决定 Candidate Reconciliation V1 的实现。

Train256 Candidate Recoverability 已完成（256 images / 2,493 candidates / 0 skipped）。Baseline、
Oracle Reject、Oracle Relabel 的 mIoU 为 `0.3800/0.4188/0.4623`，foreground mIoU 为
`0.4560/0.5026/0.5547`，coverage 为 `0.6063/0.5377/0.6058`。Relabel 相对 Reject 提高
`+0.0435 mIoU`、`+0.0522 foreground mIoU`、`+0.0558 OA`，并恢复 `+0.0681 coverage`；
这确认“保留候选几何并修正语义”具有明显上限，正式进入 Candidate Reconciliation V1。

宏平均 Geometric/Semantic Recall 为 `0.6675/0.5589`，Recoverable Semantic Gap 为
`0.1086`。低矮植被和树的 gap 最大，分别为 `0.2238/0.2372`；impervious/building/car 为
`0.0354/0.0357/0.0111`。Mean CAM 对主要错配的纠正率为 imp->lowveg `0.7465`、
imp->building `0.4328`、lowveg->tree `0.5135`，但 car->tree 仅 `0.0804`。下一步只使用
Train CAM top1-top2 margin 的无 GT 分布做 per-source 自适应校准，再冻结应用到 Validation256。

Candidate Reconciliation V1 已实现为两个严格分离的阶段。训练校准工具只读取 image-level
标签、SAM3 候选缓存和 CAM，使用 source/CAM 分歧候选的 top1-top2 margin 拟合两分量模型；
样本不足或不可分的类别回退到全局训练分布。应用工具读取冻结的校准 JSON：CAM 与 source
一致时 Keep，高 margin 分歧时 Relabel 到 image-level 正类约束下的 CAM top1，低 margin
分歧时 Ignore。两阶段均不读取像素 GT；输出记录逐类动作数、校准文件 SHA-256 和
`pixel_gt_used=false`。下一步先在 Train256 生成并冻结 calibration JSON，检查自动阈值后
原样应用到 Validation256，最终统一报告 mIoU、mF1、OA、前景指标和 coverage。

V1 已在 Validation256 原样运行，但未通过：mIoU/mF1/OA 从 baseline 的
`0.3494/0.4599/0.3935` 降至 `0.3133/0.4370/0.3618`，foreground mIoU 从 `0.4193`
降至 `0.3760`，coverage 从 `0.5336` 降至 `0.4414`。虽然 labeled mIoU 提高到
`0.5466`，但509个 Ignore 和341个 Relabel 造成过度稀疏及语义破坏。Oracle Relabel 在同一
Validation256 仍达到 `0.4328 mIoU/0.5471 mF1/0.4710 OA`，比 baseline 高约
`+0.0834/+0.0872/+0.0775`，且 recoverable gap 与 Train256 一致，因此失败来自 CAM margin
校准而非候选几何上限。代码现支持用冻结 calibration 做离线 GT 动作审计；下一步先量化逐类
有益/破坏性 Relabel 和 Ignore 误删率，不训练 student，也不据 validation GT 重拟合阈值。

动作审计显示 V1 的 Train/Validation 行为几乎一致：Relabel 正确率仅 `0.2291/0.2346`，
破坏性 Relabel 比例为 `0.6218/0.6305`，Ignore 删除原类别正确候选的比例为
`0.7371/0.7387`。主要失败对 `car->impervious` 的正确率仅 `0.0252/0.0194`；稳定的潜在
有效对为 `impervious->low vegetation`（`0.6939/0.8298`）、`impervious->building`
（`0.8182/0.5294`，像素纯度 `0.9059/0.7955`）和 `low vegetation->tree`
（`0.9000/0.8333`）。现已实现通用 relabel-pair allowlist 与 `keep` fallback。下一实验 V2-a
仅重标这三个方向，其他分歧保留 SAM3 source，不产生候选级 Ignore；它用于验证可靠语义修复
能否带来收益，类别 allowlist 仍视为数据集特定消融，不能作为最终通用贡献。

## 2. 已经完成

- 已实现 Potsdam 数据读取和像素标签到 image-level CSV 的转换。
- 已实现 RemoteCLIP/B2C 风格 prompt 模板。
- 已实现大图重叠切片、SAM3 文本推理、掩码过滤、语义融合和整图拼接。
- 已实现伪标签 PNG、overlay JPG、metadata JSON 输出。
- 已实现伪标签的 class IoU、mIoU 和 foreground mIoU 评估。
- 已修复 Potsdam 红色 clutter/background 的 GT 映射：背景为 `0`，未知颜色为 `255`。
- 已实现显式 patch 数据集生成器，可输出 RGBIR patch、隔离的评估 GT、patch-level image labels、坐标统计和新配置。
- 已实现 ResNet 多尺度编码器 + SegFormer-style decoder head 的 student。
- 已加入随机尺度裁剪、类别比例约束、翻转、90 度旋转、颜色扰动、模糊、小连通域清理和边界 ignore。
- 已保留 ToCo LargeFOV/ASPP head 作为消融选项。
- 已在单张 2080Ti 上跑通真实 SAM3 伪标签生成。
- 已在单张 2080Ti 上跑通 student 的 1 epoch smoke training。
- 仓库内 SAM3 的 `sam3/perflib/fused.py` 已包含 Turing/2080Ti 的 FP32 fallback，避免 BF16/FP32 dtype 报错。

## 3. 已有实验结果

修正 5 张图各自的 image-level 标签后，曾得到以下伪标签结果：

| 类别 | IoU |
| --- | ---: |
| impervious_surface | 0.0610 |
| building | 0.5040 |
| low_vegetation | 0.0132 |
| tree | 0.1053 |
| car | 0.7392 |
| foreground mIoU | **0.2846** |

这说明当前原型有可行性，特别是 building 和 car；impervious surface、low vegetation、tree 仍然是主要瓶颈。但这组指标产生于 background 映射修复之前，clutter 区域当时被忽略，因此必须在新代码上重算，不能直接作为最终结果。完整背景记录在 `experiment_log.md`。

## 4. 当前已知问题

1. Potsdam 的 `top_potsdam_2_10` 一类文件通常是约 `6000 x 6000` 的官方大幅 tile，不是已经准备好的 `512 x 512` 训练 patch。
2. 原始大图工作流仍会让每个运行时 tile 继承整图标签；新加入的 `prepare_potsdam_patches.py` 可以解决这个问题，但服务器上尚未生成和重跑 patch 基线。
3. patch-level image labels 当前由 Potsdam pixel GT 生成，用于模拟弱监督。pixel GT 只能用于标签派生和离线评估，不能输入 SAM3 或 student 训练。
4. 当前只实现了伪标签评估；student 的独立验证集推理、拼接和 mIoU 评估尚未形成完整闭环。
5. RemoteCLIP prompt 排序代码已经存在，但尚未完成正式对比实验。
6. `runs/`、`data/`、生成的 patch 数据集和权重文件不会随 GitHub 仓库自动同步。
7. 后续训练/验证划分必须按 Potsdam 父图进行，不能随机划分重叠 patch，否则会产生空间泄漏。
8. 从 pixel GT 派生 patch 标签会把数据样本重新定义为 patch，监督比原始父图标签更局部。论文中必须明确报告这一构造协议，不能声称只使用未经处理的父图级标签。

## 5. 推荐的下一阶段

按下面顺序推进：

1. 在服务器上用 `prepare_potsdam_patches.py --limit 1` 生成 patch smoke 数据集。
2. 使用修复后的 background 映射重跑 5 个 patch，建立新的可信基线。
3. 在固定 patch 子集上做 prompt、类别阈值和融合参数消融，优先改善三类低 IoU 类别。
4. 加入 student validation/inference 脚本，完成 pseudo label -> training -> validation mIoU 闭环。
5. 最后扩大规模，并比较 SAM3-only、RemoteCLIP+SAM3、CAM+SAM3 三条路线。

## 6. 已知路径

当前实验室服务器路径：

```text
工程:      /home/undergr/1/sam3_remote_wsss
SAM3:      /home/undergr/1/sam3-main/sam3-main
Potsdam:   /home/undergr/remote_dataset/Postdam
SAM3 权重: /home/undergr/1/checkpoints/sam3.pt
Conda 环境: sam3_wsss
GPU:       2 x NVIDIA 2080Ti
```

当前 Windows GitHub 拉取路径：

```text
C:\Users\28457\Desktop\Sheung_project1\Sheung_project1_sam3_remote_wsss
```

Windows 仓库根目录同时包含 `sam3-main/sam3-main` 和 `sam3_remote_wsss`。路径不是模型的一部分，换机器后按 `environment_setup.md` 修改配置即可。

## 7. 新任务交接文本

新开 Codex 任务时可以直接发送：

```text
请先阅读本工程的 docs/current_status.md、docs/environment_setup.md、
docs/experiment_log.md 和 docs/runbook.md，再继续开发。

当前目标是 SAM3 + RemoteCLIP 风格 prompt 的 Potsdam image-level WSSS。
真实 SAM3 伪标签和 SegFormer-style student smoke test 已跑通。
patch-level 数据准备和 background 映射已经实现但尚未在服务器重跑。
下一步先生成 patch smoke 数据集和新基线，再补齐 student 验证与 mIoU 评估闭环。
请以工程中的现有实现和文档记录为准，不要重新从零设计。
```
