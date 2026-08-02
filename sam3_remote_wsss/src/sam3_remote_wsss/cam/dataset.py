from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config import ProjectConfig
from ..potsdam import discover_potsdam_items, read_image_level_csv, read_rgbir_as_rgb


_RESAMPLING = getattr(Image, "Resampling", Image)


class PotsdamImageLevelDataset:
    def __init__(
        self,
        config: ProjectConfig,
        labels_csv: str | Path,
        image_size: int = 512,
        limit: int | None = None,
        augment: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        photometric_prob: float = 0.5,
        blur_prob: float = 0.1,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch before training the CAM classifier.") from exc

        self.config = config
        self.class_specs = tuple(sorted(config.classes, key=lambda spec: spec.id))
        self.class_names = tuple(spec.name for spec in self.class_specs)
        self.class_ids = tuple(spec.id for spec in self.class_specs)
        image_level = read_image_level_csv(labels_csv)
        self.items = [
            item
            for item in discover_potsdam_items(config)
            if item.image_id in image_level
        ]
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError(f"No dataset images matched image-level labels in {labels_csv}")

        self.targets = np.asarray(
            [
                [float(name in image_level[item.image_id]) for name in self.class_names]
                for item in self.items
            ],
            dtype=np.float32,
        )
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.photometric_prob = float(photometric_prob)
        self.blur_prob = float(blur_prob)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self._torch = torch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        image = read_rgbir_as_rgb(item.image_path, self.config.rgb_band_indices)
        image = self._prepare_image(image)
        image_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_tensor = (image_tensor - self.mean) / self.std
        return {
            "image": self._torch.from_numpy(np.ascontiguousarray(image_tensor)),
            "target": self._torch.from_numpy(self.targets[index].copy()),
            "image_id": item.image_id,
        }

    def positive_weights(self, maximum: float = 10.0) -> np.ndarray:
        positives = self.targets.sum(axis=0)
        negatives = len(self.targets) - positives
        weights = np.ones_like(positives, dtype=np.float32)
        present = positives > 0
        weights[present] = negatives[present] / positives[present]
        return np.clip(weights, 0.25, float(maximum))

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if self.augment:
            if random.random() < self.hflip_prob:
                image = np.flip(image, axis=1)
            if random.random() < self.vflip_prob:
                image = np.flip(image, axis=0)
            if random.random() < self.rotate90_prob:
                image = np.rot90(image, k=random.randint(0, 3), axes=(0, 1))

        image_pil = Image.fromarray(np.ascontiguousarray(image))
        if self.image_size > 0 and image_pil.size != (self.image_size, self.image_size):
            image_pil = image_pil.resize(
                (self.image_size, self.image_size),
                _RESAMPLING.BILINEAR,
            )
        if self.augment and random.random() < self.photometric_prob:
            transforms = [
                lambda value: ImageEnhance.Brightness(value).enhance(random.uniform(0.7, 1.3)),
                lambda value: ImageEnhance.Contrast(value).enhance(random.uniform(0.7, 1.3)),
                lambda value: ImageEnhance.Color(value).enhance(random.uniform(0.7, 1.3)),
            ]
            random.shuffle(transforms)
            for transform in transforms:
                image_pil = transform(image_pil)
        if self.augment and random.random() < self.blur_prob:
            image_pil = image_pil.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5))
            )
        return np.asarray(image_pil, dtype=np.uint8)
