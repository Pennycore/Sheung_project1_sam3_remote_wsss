from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VISUAL_PROTOTYPE_FORMAT_VERSION = 1


def normalize_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Visual features must have non-zero norm")
    return values / norms


def robust_visual_prototype(
    features: np.ndarray,
    keep_fraction: float = 0.7,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    values = normalize_features(features)
    if values.shape[0] == 0:
        raise ValueError("Cannot build a prototype without visual features")
    keep_count = max(1, int(np.ceil(values.shape[0] * keep_fraction)))
    retained = np.arange(values.shape[0], dtype=np.int64)
    prototype = _normalized_mean(values)
    for _ in range(iterations):
        similarities = np.einsum(
            "nd,d->n", values, prototype, optimize=False
        )
        retained = np.argsort(similarities)[-keep_count:]
        prototype = _normalized_mean(values[retained])
    similarities = np.einsum(
        "nd,d->n", values, prototype, optimize=False
    )
    return prototype, retained, similarities


def save_visual_prototype_calibration(
    output_path: str | Path,
    class_ids: np.ndarray,
    prototypes: np.ndarray,
    metadata: dict,
) -> None:
    json_path = Path(output_path)
    data_path = json_path.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(
                [VISUAL_PROTOTYPE_FORMAT_VERSION], dtype=np.int16
            ),
            class_ids=np.asarray(class_ids, dtype=np.int16),
            prototypes=normalize_features(prototypes).astype(np.float32),
        )
    payload = {
        "format_version": VISUAL_PROTOTYPE_FORMAT_VERSION,
        "data_file": data_path.name,
        **metadata,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_visual_prototype_calibration(
    calibration_path: str | Path,
) -> tuple[dict, np.ndarray, np.ndarray]:
    json_path = Path(calibration_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != VISUAL_PROTOTYPE_FORMAT_VERSION:
        raise ValueError("Unsupported visual prototype calibration format")
    data_path = json_path.parent / metadata["data_file"]
    with np.load(data_path, allow_pickle=False) as data:
        version = int(data["format_version"][0])
        class_ids = data["class_ids"].astype(np.int16)
        prototypes = data["prototypes"].astype(np.float32)
    if version != VISUAL_PROTOTYPE_FORMAT_VERSION:
        raise ValueError("Visual prototype data/metadata version mismatch")
    if prototypes.ndim != 2 or prototypes.shape[0] != class_ids.shape[0]:
        raise ValueError("Invalid visual prototype calibration dimensions")
    return metadata, class_ids, normalize_features(prototypes)


def _normalized_mean(features: np.ndarray) -> np.ndarray:
    mean = np.asarray(features, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        raise ValueError("Visual prototype mean has zero norm")
    return mean / norm
