from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    label_color: tuple[int, int, int]
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class RemoteCLIPConfig:
    enabled: bool
    model_name: str
    checkpoint_path: Path | None
    device: str
    top_k_per_class: int
    min_score: float | None


@dataclass(frozen=True)
class PromptingConfig:
    style: str
    include_manual_prompts: bool
    max_prompts_per_class: int | None


@dataclass(frozen=True)
class BackgroundPromptingConfig:
    enabled: bool
    prompts: tuple[str, ...]
    score_threshold: float
    min_mask_area: int
    max_mask_area_ratio: float
    conflict_margin: float


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    image_dir: str
    label_dir: str
    sam3_repo: Path
    checkpoint_path: Path | None
    device: str
    tile_size: int
    tile_overlap: int
    score_threshold: float
    mask_threshold: float
    min_mask_area: int
    max_mask_area_ratio: float
    conflict_margin: float
    ignore_index: int
    uncovered_label: int
    rgb_band_indices: tuple[int, int, int]
    classes: tuple[ClassSpec, ...]
    background_colors: tuple[tuple[int, int, int], ...]
    remoteclip: RemoteCLIPConfig
    prompting: PromptingConfig
    background_prompting: BackgroundPromptingConfig


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ProjectConfig:
    classes = tuple(
        ClassSpec(
            id=int(item["id"]),
            name=str(item["name"]),
            label_color=tuple(int(v) for v in item["label_color"]),
            prompts=tuple(str(prompt) for prompt in item["prompts"]),
        )
        for item in raw["classes"]
    )
    checkpoint_raw = raw.get("checkpoint_path")
    remoteclip_raw = raw.get("remoteclip", {})
    remoteclip_ckpt = remoteclip_raw.get("checkpoint_path")
    prompting_raw = raw.get("prompting", {})
    background_prompting_raw = raw.get("background_prompting", {})
    ignore_index = int(raw.get("ignore_index", 255))
    background_prompting_enabled = bool(background_prompting_raw.get("enabled", False))
    background_prompts = tuple(
        str(prompt).strip()
        for prompt in background_prompting_raw.get("prompts", [])
        if str(prompt).strip()
    )
    if background_prompting_enabled and not background_prompts:
        raise ValueError("background_prompting.prompts must not be empty when enabled")
    return ProjectConfig(
        dataset_root=Path(raw["dataset_root"]),
        image_dir=str(raw["image_dir"]),
        label_dir=str(raw["label_dir"]),
        sam3_repo=Path(raw["sam3_repo"]),
        checkpoint_path=Path(checkpoint_raw) if checkpoint_raw else None,
        device=str(raw.get("device", "cuda")),
        tile_size=int(raw.get("tile_size", 1024)),
        tile_overlap=int(raw.get("tile_overlap", 256)),
        score_threshold=float(raw.get("score_threshold", 0.5)),
        mask_threshold=float(raw.get("mask_threshold", 0.5)),
        min_mask_area=int(raw.get("min_mask_area", 64)),
        max_mask_area_ratio=float(raw.get("max_mask_area_ratio", 0.95)),
        conflict_margin=float(raw.get("conflict_margin", 0.03)),
        ignore_index=ignore_index,
        uncovered_label=int(raw.get("uncovered_label", ignore_index)),
        rgb_band_indices=tuple(int(v) for v in raw.get("rgb_band_indices", [0, 1, 2])),
        classes=classes,
        background_colors=tuple(
            tuple(int(v) for v in color) for color in raw.get("background_colors", [])
        ),
        remoteclip=RemoteCLIPConfig(
            enabled=bool(remoteclip_raw.get("enabled", False)),
            model_name=str(remoteclip_raw.get("model_name", "ViT-B-32")),
            checkpoint_path=Path(remoteclip_ckpt) if remoteclip_ckpt else None,
            device=str(remoteclip_raw.get("device", raw.get("device", "cuda"))),
            top_k_per_class=int(remoteclip_raw.get("top_k_per_class", 2)),
            min_score=(
                float(remoteclip_raw["min_score"])
                if remoteclip_raw.get("min_score") is not None
                else None
            ),
        ),
        prompting=PromptingConfig(
            style=str(prompting_raw.get("style", "manual")),
            include_manual_prompts=bool(prompting_raw.get("include_manual_prompts", True)),
            max_prompts_per_class=(
                int(prompting_raw["max_prompts_per_class"])
                if prompting_raw.get("max_prompts_per_class") is not None
                else None
            ),
        ),
        background_prompting=BackgroundPromptingConfig(
            enabled=background_prompting_enabled,
            prompts=background_prompts,
            score_threshold=float(
                background_prompting_raw.get(
                    "score_threshold",
                    raw.get("score_threshold", 0.5),
                )
            ),
            min_mask_area=int(
                background_prompting_raw.get(
                    "min_mask_area",
                    raw.get("min_mask_area", 64),
                )
            ),
            max_mask_area_ratio=float(
                background_prompting_raw.get("max_mask_area_ratio", 0.5)
            ),
            conflict_margin=float(
                background_prompting_raw.get(
                    "conflict_margin",
                    raw.get("conflict_margin", 0.03),
                )
            ),
        ),
    )
