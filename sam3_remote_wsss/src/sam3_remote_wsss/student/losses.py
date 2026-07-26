from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def safe_cross_entropy(pred: Tensor, label: Tensor, ignore_index: int = 255) -> Tensor:
    if torch.any(label != ignore_index):
        return F.cross_entropy(pred, label.long(), ignore_index=ignore_index)
    return pred.sum() * 0.0


def toco_seg_loss(pred: Tensor, label: Tensor, ignore_index: int = 255) -> Tensor:
    """ToCo-style balanced background/foreground segmentation loss.

    Original ToCo computes CE for background pixels and foreground pixels
    separately, then averages them. This version avoids NaNs when a crop has
    only foreground, only background, or only ignored pixels.
    """

    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index

    losses = []
    if torch.any(bg_label != ignore_index):
        losses.append(safe_cross_entropy(pred, bg_label, ignore_index))
    if torch.any(fg_label != ignore_index):
        losses.append(safe_cross_entropy(pred, fg_label, ignore_index))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
