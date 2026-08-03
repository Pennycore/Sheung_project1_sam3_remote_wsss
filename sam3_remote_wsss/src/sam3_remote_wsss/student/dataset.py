from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config import ProjectConfig
from ..potsdam import (
    discover_potsdam_items,
    label_rgb_to_ids,
    read_image_level_csv,
    read_label_rgb,
    read_rgbir_as_rgb,
)


_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PseudoLabelItem:
    image_id: str
    image_path: Path
    pseudo_label_path: Path


@dataclass(frozen=True)
class GroundTruthItem:
    image_id: str
    image_path: Path
    label_path: Path


def discover_pseudo_label_items(
    config: ProjectConfig,
    pseudo_label_dir: str | Path,
) -> list[PseudoLabelItem]:
    image_by_id = {item.image_id: item.image_path for item in discover_potsdam_items(config)}
    items: list[PseudoLabelItem] = []
    for label_path in sorted(Path(pseudo_label_dir).glob("*.png")):
        image_id = label_path.stem
        image_path = image_by_id.get(image_id)
        if image_path is None:
            continue
        items.append(
            PseudoLabelItem(
                image_id=image_id,
                image_path=image_path,
                pseudo_label_path=label_path,
            )
        )
    return items


class PotsdamGroundTruthSegDataset:
    def __init__(
        self,
        config: ProjectConfig,
        labels_csv: str | Path,
        image_size: int = 512,
        limit: int | None = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch before validating the student.") from exc

        selected_ids = set(read_image_level_csv(labels_csv))
        discovered = {
            item.image_id: item for item in discover_potsdam_items(config)
        }
        missing_images = sorted(selected_ids - discovered.keys())
        missing_labels = sorted(
            image_id
            for image_id in selected_ids & discovered.keys()
            if discovered[image_id].label_path is None
        )
        if missing_images or missing_labels:
            details = []
            if missing_images:
                details.append(
                    f"missing images={len(missing_images)} ({', '.join(missing_images[:5])})"
                )
            if missing_labels:
                details.append(
                    f"missing labels={len(missing_labels)} ({', '.join(missing_labels[:5])})"
                )
            raise FileNotFoundError(
                f"Validation CSV {labels_csv} is incomplete: {'; '.join(details)}"
            )
        self.items = [
            GroundTruthItem(
                image_id,
                discovered[image_id].image_path,
                discovered[image_id].label_path,
            )
            for image_id in sorted(selected_ids)
        ]
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError(f"No ground-truth items matched {labels_csv}")

        self.config = config
        self.image_size = int(image_size)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self._torch = torch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        image = read_rgbir_as_rgb(item.image_path, self.config.rgb_band_indices)
        label = label_rgb_to_ids(
            read_label_rgb(item.label_path),
            self.config.classes,
            self.config.ignore_index,
            background_colors=self.config.background_colors,
        )
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/label size mismatch for {item.image_id}: "
                f"{image.shape[:2]} vs {label.shape[:2]}"
            )
        if self.image_size > 0 and image.shape[:2] != (
            self.image_size,
            self.image_size,
        ):
            image_pil = Image.fromarray(image)
            label_pil = Image.fromarray(label, mode="L")
            image = np.asarray(
                image_pil.resize(
                    (self.image_size, self.image_size),
                    _RESAMPLING.BILINEAR,
                ),
                dtype=np.uint8,
            )
            label = np.asarray(
                label_pil.resize(
                    (self.image_size, self.image_size),
                    _RESAMPLING.NEAREST,
                ),
                dtype=np.uint8,
            )
        image_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_tensor = (image_tensor - self.mean) / self.std
        return {
            "image": self._torch.from_numpy(np.ascontiguousarray(image_tensor)),
            "label": self._torch.from_numpy(label.astype(np.int64)),
            "image_id": item.image_id,
        }


class PotsdamPseudoSegDataset:
    def __init__(
        self,
        config: ProjectConfig,
        pseudo_label_dir: str | Path,
        crop_size: int = 512,
        samples_per_image: int = 1,
        random_crop: bool = True,
        limit: int | None = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        cat_max_ratio: float = 0.75,
        max_crop_attempts: int = 10,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        photometric_prob: float = 0.5,
        blur_prob: float = 0.1,
        min_valid_ratio: float = 0.05,
        min_foreground_ratio: float = 0.001,
        min_component_area: int = 16,
        ignore_boundary_width: int = 1,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch before using PotsdamPseudoSegDataset.") from exc
        try:
            from scipy import ndimage as scipy_ndimage
        except ImportError:
            scipy_ndimage = None

        self.config = config
        self.items = discover_pseudo_label_items(config, pseudo_label_dir)
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError(f"No pseudo labels found in {pseudo_label_dir}")
        self.crop_size = int(crop_size)
        self.samples_per_image = max(1, int(samples_per_image))
        self.random_crop = bool(random_crop)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.augment = bool(augment)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.cat_max_ratio = float(cat_max_ratio)
        self.max_crop_attempts = max(1, int(max_crop_attempts))
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.photometric_prob = float(photometric_prob)
        self.blur_prob = float(blur_prob)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.min_component_area = int(min_component_area)
        self.ignore_boundary_width = int(ignore_boundary_width)
        self.num_classes = max(spec.id for spec in config.classes) + 1
        self._torch = torch
        self._ndi = scipy_ndimage

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_image

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index % len(self.items)]
        image = read_rgbir_as_rgb(item.image_path, self.config.rgb_band_indices)
        label = np.asarray(Image.open(item.pseudo_label_path).convert("L"), dtype=np.uint8)
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/label size mismatch for {item.image_id}: "
                f"{image.shape[:2]} vs {label.shape[:2]}"
            )
        image, label = self._prepare_pair(image, label)
        image_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_tensor = (image_tensor - self.mean) / self.std
        return {
            "image": self._torch.from_numpy(np.ascontiguousarray(image_tensor)),
            "label": self._torch.from_numpy(label.astype(np.int64)),
            "image_id": item.image_id,
        }

    def _prepare_pair(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, label = self._crop_pair(image, label)
        if self.augment:
            image, label = self._geometric_augment(image, label)
        label = self._clean_pseudo_label(label)
        if self.augment:
            image = self._photometric_augment(image)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def _crop_pair(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size <= 0:
            return image, label
        scale = random.uniform(*self.scale_range) if self.augment else 1.0
        source_h = max(1, int(round(self.crop_size / scale)))
        source_w = max(1, int(round(self.crop_size / scale)))
        image, label = self._pad_if_needed(image, label, source_h, source_w)

        last_image = image[:source_h, :source_w]
        last_label = label[:source_h, :source_w]
        h, w = label.shape
        for _ in range(self.max_crop_attempts):
            if self.random_crop:
                y0 = random.randint(0, h - source_h)
                x0 = random.randint(0, w - source_w)
            else:
                y0 = max(0, (h - source_h) // 2)
                x0 = max(0, (w - source_w) // 2)
            y1 = y0 + source_h
            x1 = x0 + source_w
            crop_image = image[y0:y1, x0:x1]
            crop_label = label[y0:y1, x0:x1]
            last_image, last_label = crop_image, crop_label
            if self._crop_is_useful(crop_label):
                break

        if (source_h, source_w) != (self.crop_size, self.crop_size):
            last_image, last_label = self._resize_pair(last_image, last_label, self.crop_size, self.crop_size)
        return last_image, last_label

    def _crop_is_useful(self, label: np.ndarray) -> bool:
        valid = label != self.config.ignore_index
        valid_ratio = float(valid.mean())
        if valid_ratio < self.min_valid_ratio:
            return False
        foreground_ratio = float((valid & (label != 0)).mean())
        if foreground_ratio < self.min_foreground_ratio:
            return False
        if self.cat_max_ratio > 0 and np.any(valid):
            _values, counts = np.unique(label[valid], return_counts=True)
            if float(counts.max()) / float(counts.sum()) > self.cat_max_ratio:
                return False
        return True

    def _resize_pair(
        self,
        image: np.ndarray,
        label: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_pil = Image.fromarray(image)
        label_pil = Image.fromarray(label.astype(np.uint8), mode="L")
        image = np.asarray(image_pil.resize((width, height), _RESAMPLING.BILINEAR), dtype=np.uint8)
        label = np.asarray(label_pil.resize((width, height), _RESAMPLING.NEAREST), dtype=np.uint8)
        return image, label

    def _geometric_augment(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.hflip_prob:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        if random.random() < self.vflip_prob:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)
        if random.random() < self.rotate90_prob:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k, axes=(0, 1))
            label = np.rot90(label, k=k, axes=(0, 1))
        return image, label

    def _photometric_augment(self, image: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(image)
        if random.random() < self.photometric_prob:
            transforms = [
                lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
                lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
            ]
            random.shuffle(transforms)
            for transform in transforms:
                image_pil = transform(image_pil)
        if random.random() < self.blur_prob:
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return np.asarray(image_pil, dtype=np.uint8)

    def _clean_pseudo_label(self, label: np.ndarray) -> np.ndarray:
        label = label.copy()
        invalid = (label != self.config.ignore_index) & (label >= self.num_classes)
        label[invalid] = self.config.ignore_index
        if self.min_component_area > 0:
            label = self._remove_small_components(label)
        if self.ignore_boundary_width > 0:
            label = self._ignore_label_boundaries(label)
        return label

    def _pad_if_needed(
        self,
        image: np.ndarray,
        label: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = label.shape
        pad_h = max(0, height - h)
        pad_w = max(0, width - w)
        if pad_h == 0 and pad_w == 0:
            return image, label
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        label = np.pad(
            label,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=self.config.ignore_index,
        )
        return image, label

    def _remove_small_components(self, label: np.ndarray) -> np.ndarray:
        output = label.copy()
        for class_id in range(1, self.num_classes):
            mask = output == class_id
            if not np.any(mask):
                continue
            if self._ndi is not None:
                components, count = self._ndi.label(mask)
                if count == 0:
                    continue
                areas = np.bincount(components.reshape(-1))
                small_ids = np.flatnonzero(areas < self.min_component_area)
                small_ids = small_ids[small_ids != 0]
                if small_ids.size:
                    output[np.isin(components, small_ids)] = self.config.ignore_index
            else:
                output = self._remove_small_components_fallback(output, class_id)
        return output

    def _remove_small_components_fallback(self, label: np.ndarray, class_id: int) -> np.ndarray:
        mask = label == class_id
        visited = np.zeros(mask.shape, dtype=bool)
        h, w = mask.shape
        output = label.copy()
        for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if len(component) < self.min_component_area:
                ys, xs = zip(*component)
                output[np.asarray(ys), np.asarray(xs)] = self.config.ignore_index
        return output

    def _ignore_label_boundaries(self, label: np.ndarray) -> np.ndarray:
        valid = label != self.config.ignore_index
        boundary = np.zeros(label.shape, dtype=bool)

        diff_y = valid[:-1, :] & valid[1:, :] & (label[:-1, :] != label[1:, :])
        boundary[:-1, :] |= diff_y
        boundary[1:, :] |= diff_y

        diff_x = valid[:, :-1] & valid[:, 1:] & (label[:, :-1] != label[:, 1:])
        boundary[:, :-1] |= diff_x
        boundary[:, 1:] |= diff_x

        for _ in range(max(0, self.ignore_boundary_width - 1)):
            boundary = self._dilate_mask_4n(boundary)
        output = label.copy()
        output[boundary] = self.config.ignore_index
        return output

    @staticmethod
    def _dilate_mask_4n(mask: np.ndarray) -> np.ndarray:
        result = mask.copy()
        result[:-1, :] |= mask[1:, :]
        result[1:, :] |= mask[:-1, :]
        result[:, :-1] |= mask[:, 1:]
        result[:, 1:] |= mask[:, :-1]
        return result
