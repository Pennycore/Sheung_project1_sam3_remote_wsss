from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusionCanvas:
    height: int
    width: int
    ignore_index: int = 255
    conflict_margin: float = 0.03
    labels: np.ndarray = field(init=False)
    scores: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.labels = np.zeros((self.height, self.width), dtype=np.uint8)
        self.scores = np.zeros((self.height, self.width), dtype=np.float32)

    def add_mask(self, mask: np.ndarray, class_id: int, score: float, x0: int, y0: int) -> None:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return

        h, w = mask_bool.shape
        region_labels = self.labels[y0 : y0 + h, x0 : x0 + w]
        region_scores = self.scores[y0 : y0 + h, x0 : x0 + w]

        better = mask_bool & (score > region_scores + self.conflict_margin)
        close_conflict = (
            mask_bool
            & (region_labels != 0)
            & (region_labels != class_id)
            & (np.abs(score - region_scores) <= self.conflict_margin)
        )

        region_labels[better] = class_id
        region_scores[better] = score
        region_labels[close_conflict] = self.ignore_index

    def result(self) -> np.ndarray:
        return self.labels.copy()


def filter_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    min_area: int,
    max_area_ratio: float,
) -> list[tuple[np.ndarray, float]]:
    if masks.size == 0:
        return []
    masks = _normalize_mask_shape(masks)
    scores = np.asarray(scores).reshape(-1)
    kept: list[tuple[np.ndarray, float]] = []
    image_area = masks.shape[-2] * masks.shape[-1]
    for mask, score in zip(masks, scores):
        area = int(mask.astype(bool).sum())
        if float(score) < score_threshold:
            continue
        if area < min_area:
            continue
        if area > image_area * max_area_ratio:
            continue
        kept.append((mask.astype(bool), float(score)))
    return kept


def _normalize_mask_shape(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape NxHxW, got {masks.shape}")
    return masks

