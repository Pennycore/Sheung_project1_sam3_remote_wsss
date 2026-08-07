from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .candidate_cache import candidate_cache_paths


REGION_SCORE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CandidateRegionViews:
    context: np.ndarray
    masked: np.ndarray
    crop_box: tuple[int, int, int, int]
    mask_fraction: float


def candidate_cache_fingerprint(cache_dir: str | Path, image_id: str) -> str:
    digest = hashlib.sha256()
    for path in candidate_cache_paths(cache_dir, image_id):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def make_candidate_region_views(
    image_rgb: np.ndarray,
    candidate,
    context_ratio: float = 0.25,
    min_crop_size: int = 48,
    background_retain: float = 0.25,
) -> CandidateRegionViews:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
    if context_ratio < 0:
        raise ValueError("context_ratio must be non-negative")
    if min_crop_size <= 0:
        raise ValueError("min_crop_size must be positive")
    if not 0 <= background_retain <= 1:
        raise ValueError("background_retain must be in [0, 1]")

    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Candidate mask must be a non-empty 2D array")
    height, width = mask.shape
    image_height, image_width = image_rgb.shape[:2]
    x0 = int(candidate.x0)
    y0 = int(candidate.y0)
    if x0 < 0 or y0 < 0 or x0 + width > image_width or y0 + height > image_height:
        raise ValueError("Candidate bounds exceed image dimensions")

    ys, xs = np.nonzero(mask)
    left = x0 + int(xs.min())
    top = y0 + int(ys.min())
    right = x0 + int(xs.max()) + 1
    bottom = y0 + int(ys.max()) + 1
    target_size = max(right - left, bottom - top)
    crop_size = max(min_crop_size, int(np.ceil(target_size * (1 + 2 * context_ratio))))
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left, crop_right = _centered_bounds(center_x, crop_size, image_width)
    crop_top, crop_bottom = _centered_bounds(center_y, crop_size, image_height)

    crop = np.ascontiguousarray(
        image_rgb[crop_top:crop_bottom, crop_left:crop_right]
    )
    crop_mask = np.zeros(crop.shape[:2], dtype=bool)
    intersect_left = max(crop_left, x0)
    intersect_top = max(crop_top, y0)
    intersect_right = min(crop_right, x0 + width)
    intersect_bottom = min(crop_bottom, y0 + height)
    if intersect_left < intersect_right and intersect_top < intersect_bottom:
        crop_mask[
            intersect_top - crop_top : intersect_bottom - crop_top,
            intersect_left - crop_left : intersect_right - crop_left,
        ] = mask[
            intersect_top - y0 : intersect_bottom - y0,
            intersect_left - x0 : intersect_right - x0,
        ]
    if not crop_mask.any():
        raise ValueError("Candidate mask does not intersect its computed crop")

    masked = crop.astype(np.float32)
    masked[~crop_mask] *= background_retain
    masked = np.rint(masked).clip(0, 255).astype(np.uint8)
    return CandidateRegionViews(
        context=crop,
        masked=np.ascontiguousarray(masked),
        crop_box=(crop_left, crop_top, crop_right, crop_bottom),
        mask_fraction=float(crop_mask.mean()),
    )


def build_class_text_prototypes(
    encoder,
    class_specs: Sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    class_ids = np.asarray([int(spec.id) for spec in class_specs], dtype=np.int16)
    prompt_groups = {
        str(spec.name): [str(prompt) for prompt in spec.prompts]
        for spec in class_specs
    }
    if any(not prompts for prompts in prompt_groups.values()):
        raise ValueError("Every class must have at least one text prompt")
    flat_prompts = [
        prompt for spec in class_specs for prompt in prompt_groups[str(spec.name)]
    ]
    text_features = encoder.encode_texts(flat_prompts).astype(np.float32, copy=False)
    prototypes = []
    offset = 0
    for spec in class_specs:
        count = len(prompt_groups[str(spec.name)])
        prototype = text_features[offset : offset + count].mean(axis=0)
        norm = float(np.linalg.norm(prototype))
        if norm == 0:
            raise ValueError(f"Zero text prototype for class {spec.name}")
        prototypes.append(prototype / norm)
        offset += count
    return class_ids, np.stack(prototypes), prompt_groups


def score_candidate_regions(
    image_rgb: np.ndarray,
    candidates: Sequence,
    encoder,
    class_prototypes: np.ndarray,
    batch_size: int = 32,
    context_ratio: float = 0.25,
    min_crop_size: int = 48,
    background_retain: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not candidates:
        return (
            np.empty((0, class_prototypes.shape[0]), dtype=np.float32),
            np.empty((0, 4), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    images = []
    boxes = []
    mask_fractions = []
    for candidate in candidates:
        views = make_candidate_region_views(
            image_rgb=image_rgb,
            candidate=candidate,
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
            background_retain=background_retain,
        )
        images.extend((views.context, views.masked))
        boxes.append(views.crop_box)
        mask_fractions.append(views.mask_fraction)

    features = encoder.encode_images(images, batch_size=batch_size).astype(
        np.float32, copy=False
    )
    if features.shape[0] != 2 * len(candidates):
        raise ValueError("Image encoder returned an unexpected feature count")
    fused = features.reshape(len(candidates), 2, -1).mean(axis=1)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    fused = fused / np.maximum(norms, 1e-12)
    scores = fused @ class_prototypes.astype(np.float32, copy=False).T
    return (
        scores.astype(np.float32, copy=False),
        np.asarray(boxes, dtype=np.int32),
        np.asarray(mask_fractions, dtype=np.float32),
    )


def active_region_decisions(
    scores: np.ndarray,
    class_ids: np.ndarray,
    active_class_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    class_to_column = {
        int(class_id): index for index, class_id in enumerate(class_ids)
    }
    missing = [
        int(class_id)
        for class_id in active_class_ids
        if int(class_id) not in class_to_column
    ]
    if missing:
        raise ValueError(f"Active classes missing from region scores: {missing}")
    active_ids = np.asarray(active_class_ids, dtype=np.int16)
    columns = np.asarray(
        [class_to_column[int(class_id)] for class_id in active_ids], dtype=np.int64
    )
    active_scores = scores[:, columns]
    best = np.argmax(active_scores, axis=1)
    predicted = active_ids[best]
    if active_scores.shape[1] == 1:
        margins = np.full((scores.shape[0],), np.nan, dtype=np.float32)
    else:
        ordered = np.partition(active_scores, -2, axis=1)
        margins = (ordered[:, -1] - ordered[:, -2]).astype(np.float32)
    return predicted, margins


def save_region_score_cache(
    output_dir: str | Path,
    image_id: str,
    scores: np.ndarray,
    class_ids: np.ndarray,
    active_class_ids: Sequence[int],
    crop_boxes: np.ndarray,
    mask_fractions: np.ndarray,
    candidate_fingerprint: str,
    metadata: dict,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    predicted, margins = active_region_decisions(
        scores, class_ids, active_class_ids
    )
    data_path = output_root / f"{image_id}.npz"
    metadata_path = output_root / f"{image_id}.json"
    with data_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray([REGION_SCORE_FORMAT_VERSION], dtype=np.int16),
            candidate_indices=np.arange(scores.shape[0], dtype=np.int32),
            scores=scores.astype(np.float32, copy=False),
            class_ids=np.asarray(class_ids, dtype=np.int16),
            active_class_ids=np.asarray(active_class_ids, dtype=np.int16),
            predicted_class_ids=predicted.astype(np.int16, copy=False),
            margins=margins,
            crop_boxes=np.asarray(crop_boxes, dtype=np.int32),
            mask_fractions=np.asarray(mask_fractions, dtype=np.float32),
        )
    payload = {
        "format_version": REGION_SCORE_FORMAT_VERSION,
        "image_id": image_id,
        "candidate_count": int(scores.shape[0]),
        "candidate_cache_sha256": candidate_fingerprint,
        **metadata,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_region_score_cache(
    score_dir: str | Path,
    image_id: str,
    candidate_dir: str | Path | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    root = Path(score_dir)
    metadata = json.loads((root / f"{image_id}.json").read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != REGION_SCORE_FORMAT_VERSION:
        raise ValueError("Unsupported candidate region score format")
    with np.load(root / f"{image_id}.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    count = int(metadata["candidate_count"])
    if arrays["scores"].shape[0] != count:
        raise ValueError("Candidate region score count mismatch")
    if not np.array_equal(arrays["candidate_indices"], np.arange(count)):
        raise ValueError("Candidate region score ordering is invalid")
    if candidate_dir is not None:
        current = candidate_cache_fingerprint(candidate_dir, image_id)
        if current != metadata["candidate_cache_sha256"]:
            raise ValueError(
                f"Candidate cache changed after region scoring for {image_id}"
            )
    return metadata, arrays


def _centered_bounds(center: float, size: int, limit: int) -> tuple[int, int]:
    size = min(int(size), int(limit))
    start = int(np.floor(center - size / 2))
    start = max(0, min(start, limit - size))
    return start, start + size
