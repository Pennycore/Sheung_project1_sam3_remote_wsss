from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import ClassSpec


def save_overlay(
    image_rgb: np.ndarray,
    label_ids: np.ndarray,
    classes: tuple[ClassSpec, ...],
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    palette = {spec.id: np.array(spec.label_color, dtype=np.float32) for spec in classes}
    overlay = image_rgb.astype(np.float32).copy()
    for class_id, color in palette.items():
        mask = label_ids == class_id
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def save_label_png(label_ids: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_ids.astype(np.uint8), mode="L").save(output_path)

