from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .config import ClassSpec, ProjectConfig


IMAGE_RE = re.compile(r"^(top_potsdam_\d+_\d+)_RGBIR\.tif$", re.IGNORECASE)
LABEL_RE = re.compile(r"^(top_potsdam_\d+_\d+)_label\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class PotsdamItem:
    image_id: str
    image_path: Path
    label_path: Path | None


def discover_potsdam_items(config: ProjectConfig) -> list[PotsdamItem]:
    image_root = config.dataset_root / config.image_dir
    label_root = config.dataset_root / config.label_dir
    labels = {}
    if label_root.exists():
        for label_path in label_root.glob("*.tif"):
            match = LABEL_RE.match(label_path.name)
            if match:
                labels[match.group(1)] = label_path

    items: list[PotsdamItem] = []
    for image_path in sorted(image_root.glob("*.tif")):
        match = IMAGE_RE.match(image_path.name)
        if not match:
            continue
        image_id = match.group(1)
        items.append(PotsdamItem(image_id=image_id, image_path=image_path, label_path=labels.get(image_id)))
    return items


def read_rgbir_as_rgb(path: str | Path, rgb_band_indices: tuple[int, int, int]) -> np.ndarray:
    """Read a Potsdam RGBIR TIFF and return an RGB uint8 array.

    Potsdam files in this workspace are compressed TIFFs. `tifffile.imread`
    reads the whole image, which is acceptable for 6000 x 6000 uint8 data
    during offline pseudo-label generation.
    """
    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWxC RGBIR image, got {arr.shape} from {path}")
    rgb = arr[:, :, list(rgb_band_indices)]
    if rgb.dtype != np.uint8:
        rgb = normalize_to_uint8(rgb)
    return np.ascontiguousarray(rgb)


def read_tiff_size(path: str | Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        return int(page.imagewidth), int(page.imagelength)


def read_label_rgb(path: str | Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 label image, got {arr.shape} from {path}")
    return np.ascontiguousarray(arr[:, :, :3])


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def label_rgb_to_ids(label_rgb: np.ndarray, classes: tuple[ClassSpec, ...], ignore_index: int) -> np.ndarray:
    label_ids = np.full(label_rgb.shape[:2], ignore_index, dtype=np.uint8)
    for spec in classes:
        color = np.array(spec.label_color, dtype=np.uint8)
        label_ids[np.all(label_rgb == color, axis=-1)] = spec.id
    return label_ids


def image_level_from_label(label_rgb: np.ndarray, classes: tuple[ClassSpec, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for spec in classes:
        color = np.array(spec.label_color, dtype=np.uint8)
        result[spec.name] = int(np.any(np.all(label_rgb == color, axis=-1)))
    return result


def write_image_level_csv(items: list[PotsdamItem], config: ProjectConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", *[spec.name for spec in config.classes]]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            if item.label_path is None:
                continue
            weak = image_level_from_label(read_label_rgb(item.label_path), config.classes)
            writer.writerow({"image_id": item.image_id, **weak})


def read_image_level_csv(path: str | Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        class_names = [name for name in reader.fieldnames or [] if name != "image_id"]
        for row in reader:
            image_id = row["image_id"]
            positives = {name for name in class_names if str(row.get(name, "0")).strip() in {"1", "true", "True"}}
            result[image_id] = positives
    return result
