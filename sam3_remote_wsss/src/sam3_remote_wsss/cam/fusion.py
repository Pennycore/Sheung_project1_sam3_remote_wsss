from __future__ import annotations

import numpy as np


def normalize_cams(cams: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cams = np.maximum(np.asarray(cams, dtype=np.float32), 0.0)
    if cams.ndim != 3:
        raise ValueError(f"Expected CAMs with shape CxHxW, got {cams.shape}")
    maxima = cams.reshape(cams.shape[0], -1).max(axis=1)
    output = np.zeros_like(cams, dtype=np.float32)
    valid = maxima > eps
    output[valid] = cams[valid] / maxima[valid, None, None]
    return np.clip(output, 0.0, 1.0)


def fuse_cam_and_sam(
    sam_label: np.ndarray,
    cams: np.ndarray,
    class_ids: np.ndarray | list[int] | tuple[int, ...],
    positive_class_ids: set[int] | None = None,
    background_threshold: float = 0.2,
    foreground_threshold: float = 0.7,
    cam_support_threshold: float = 0.3,
    ignore_index: int = 255,
    background_only: bool = False,
) -> np.ndarray:
    labels, _stats = fuse_cam_and_sam_with_stats(
        sam_label=sam_label,
        cams=cams,
        class_ids=class_ids,
        positive_class_ids=positive_class_ids,
        background_threshold=background_threshold,
        foreground_threshold=foreground_threshold,
        cam_support_threshold=cam_support_threshold,
        ignore_index=ignore_index,
        background_only=background_only,
    )
    return labels


def fuse_cam_and_sam_with_stats(
    sam_label: np.ndarray,
    cams: np.ndarray,
    class_ids: np.ndarray | list[int] | tuple[int, ...],
    positive_class_ids: set[int] | None = None,
    background_threshold: float = 0.2,
    foreground_threshold: float = 0.7,
    cam_support_threshold: float = 0.3,
    ignore_index: int = 255,
    background_only: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    sam_label = np.asarray(sam_label, dtype=np.uint8)
    cams = np.nan_to_num(np.asarray(cams, dtype=np.float32), nan=0.0)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if cams.ndim != 3:
        raise ValueError(f"Expected CAMs with shape CxHxW, got {cams.shape}")
    if cams.shape[0] != len(class_ids):
        raise ValueError("CAM channel count does not match class_ids")
    if cams.shape[1:] != sam_label.shape:
        raise ValueError(
            f"CAM/SAM size mismatch: {cams.shape[1:]} vs {sam_label.shape}"
        )
    if len(np.unique(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be unique")
    if not 0 <= background_threshold < foreground_threshold <= 1:
        raise ValueError("Require 0 <= background_threshold < foreground_threshold <= 1")
    if not 0 <= cam_support_threshold <= 1:
        raise ValueError("cam_support_threshold must be in [0, 1]")

    cams = np.clip(cams, 0.0, 1.0).copy()
    positive_ids = (
        set(int(value) for value in class_ids)
        if positive_class_ids is None
        else set(int(value) for value in positive_class_ids)
    )
    active_channels = np.asarray(
        [int(class_id) in positive_ids for class_id in class_ids],
        dtype=bool,
    )
    cams[~active_channels] = 0.0

    max_scores = cams.max(axis=0)
    sam_foreground = np.isin(sam_label, list(positive_ids))
    fill_background = ~sam_foreground & (max_scores <= background_threshold)

    if background_only:
        conflicts = np.zeros(sam_label.shape, dtype=bool)
        keep_sam = sam_foreground
        fill_cam = np.zeros(sam_label.shape, dtype=bool)
        top_classes = np.zeros(sam_label.shape, dtype=np.int64)
    else:
        top_indices = cams.argmax(axis=0)
        top_classes = class_ids[top_indices]
        sam_cam_scores = np.zeros(sam_label.shape, dtype=np.float32)
        for channel, class_id in enumerate(class_ids):
            pixels = sam_label == class_id
            sam_cam_scores[pixels] = cams[channel][pixels]

        confident_cam = max_scores >= foreground_threshold
        cam_disagrees = top_classes != sam_label
        conflicts = (
            sam_foreground
            & confident_cam
            & cam_disagrees
            & (sam_cam_scores < cam_support_threshold)
        )
        keep_sam = sam_foreground & ~conflicts
        fill_cam = ~sam_foreground & confident_cam

    result = np.full(sam_label.shape, ignore_index, dtype=np.uint8)
    result[fill_background] = 0
    result[fill_cam] = top_classes[fill_cam].astype(np.uint8)
    result[keep_sam] = sam_label[keep_sam]

    stats = {
        "background_pixels": int(np.count_nonzero(fill_background)),
        "sam_foreground_pixels": int(np.count_nonzero(keep_sam)),
        "cam_foreground_pixels": int(np.count_nonzero(fill_cam)),
        "conflict_pixels": int(np.count_nonzero(conflicts)),
        "ignored_pixels": int(np.count_nonzero(result == ignore_index)),
        "total_pixels": int(result.size),
    }
    return result, stats
