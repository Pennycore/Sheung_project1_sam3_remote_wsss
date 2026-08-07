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
