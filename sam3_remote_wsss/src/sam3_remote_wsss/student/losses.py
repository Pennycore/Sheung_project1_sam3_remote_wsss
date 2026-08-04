from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def safe_cross_entropy(pred: Tensor, label: Tensor, ignore_index: int = 255) -> Tensor:
    if torch.any(label != ignore_index):
        return F.cross_entropy(pred, label.long(), ignore_index=ignore_index)
    return pred.sum() * 0.0


def toco_seg_loss(
    pred: Tensor,
    label: Tensor,
    ignore_index: int = 255,
    background_weight: float = 1.0,
    foreground_weight: float = 1.0,
) -> Tensor:
    """ToCo-style balanced background/foreground segmentation loss.

    Original ToCo computes CE for background pixels and foreground pixels
    separately, then averages them. This version avoids NaNs when a crop has
    only foreground, only background, or only ignored pixels.
    """

    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index

    weighted_losses: list[tuple[Tensor, float]] = []
    if background_weight > 0 and torch.any(bg_label != ignore_index):
        weighted_losses.append(
            (
                safe_cross_entropy(pred, bg_label, ignore_index),
                float(background_weight),
            )
        )
    if foreground_weight > 0 and torch.any(fg_label != ignore_index):
        weighted_losses.append(
            (
                safe_cross_entropy(pred, fg_label, ignore_index),
                float(foreground_weight),
            )
        )
    if not weighted_losses:
        return pred.sum() * 0.0
    total_weight = sum(weight for _loss, weight in weighted_losses)
    return sum(loss * weight for loss, weight in weighted_losses) / total_weight


def decomposed_background_seg_loss(
    pred: Tensor,
    label: Tensor,
    ignore_index: int = 255,
    background_weight: float = 1.0,
    foreground_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> Tensor:
    """Separate background evidence from conditional foreground semantics.

    Background and foreground seeds train a binary objective using the class-0
    logit against the aggregate foreground logits. Foreground seeds separately
    train the semantic classifier over classes 1..C-1. Standard argmax inference
    remains unchanged.
    """

    if pred.shape[1] < 2:
        raise ValueError("Decomposed segmentation requires at least two classes")

    valid = label != ignore_index
    foreground = valid & (label > 0)
    background = valid & (label == 0)
    foreground_score = torch.logsumexp(pred[:, 1:], dim=1)
    binary_logits = torch.stack((foreground_score, pred[:, 0]), dim=1)

    weighted_losses: list[tuple[Tensor, float]] = []
    if background_weight > 0 and torch.any(background):
        background_target = torch.full_like(label, ignore_index)
        background_target[background] = 1
        weighted_losses.append(
            (
                safe_cross_entropy(
                    binary_logits,
                    background_target,
                    ignore_index,
                ),
                float(background_weight),
            )
        )
    if foreground_weight > 0 and torch.any(foreground):
        foreground_target = torch.full_like(label, ignore_index)
        foreground_target[foreground] = 0
        weighted_losses.append(
            (
                safe_cross_entropy(
                    binary_logits,
                    foreground_target,
                    ignore_index,
                ),
                float(foreground_weight),
            )
        )
    if semantic_weight > 0 and torch.any(foreground):
        semantic_target = torch.full_like(label, ignore_index)
        semantic_target[foreground] = label[foreground] - 1
        weighted_losses.append(
            (
                safe_cross_entropy(
                    pred[:, 1:],
                    semantic_target,
                    ignore_index,
                ),
                float(semantic_weight),
            )
        )
    if not weighted_losses:
        return pred.sum() * 0.0
    total_weight = sum(weight for _loss, weight in weighted_losses)
    return sum(loss * weight for loss, weight in weighted_losses) / total_weight
