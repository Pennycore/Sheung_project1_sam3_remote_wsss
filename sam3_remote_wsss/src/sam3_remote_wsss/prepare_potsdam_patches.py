from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from .config import ClassSpec, ProjectConfig, load_config
from .potsdam import (
    discover_potsdam_items,
    image_level_from_label,
    read_label_rgb,
)
from .tiling import crop_tile, generate_tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicit Potsdam patch dataset with per-patch image-level "
            "labels for WSSS. Pixel labels are copied only for offline evaluation."
        )
    )
    parser.add_argument("--config", required=True, help="Source Potsdam JSON config.")
    parser.add_argument("--output-root", required=True, help="Output patch dataset root.")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-overlap", type=int, default=128)
    parser.add_argument(
        "--edge-mode",
        choices=("shift", "pad"),
        default="shift",
        help=(
            "shift keeps full patches by moving the final window inward; pad uses "
            "a fixed stride grid and pads right/bottom edge patches."
        ),
    )
    parser.add_argument(
        "--min-class-pixels",
        type=int,
        default=16,
        help="Minimum labeled pixels required to mark a class present in a patch.",
    )
    parser.add_argument(
        "--min-class-ratio",
        type=float,
        default=0.0,
        help="Optional minimum patch-area ratio required for every class.",
    )
    parser.add_argument(
        "--class-min-pixels",
        action="append",
        default=[],
        metavar="CLASS=COUNT",
        help="Per-class threshold override. Can be repeated, for example car=4.",
    )
    parser.add_argument("--compression", choices=["none", "deflate"], default="deflate")
    parser.add_argument(
        "--ignore-background-labels",
        action="store_true",
        help=(
            "Write an output config with background_colors empty so Potsdam "
            "clutter/background pixels map to ignore_index during evaluation."
        ),
    )
    parser.add_argument(
        "--parent-split",
        default=None,
        help=(
            "Optional JSON manifest containing train/val/test/exclude parent IDs. "
            "Split-specific image-level CSV files are written when provided."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional source-image limit.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite patch TIFFs that already exist; CSV files are rebuilt.",
    )
    return parser.parse_args()


def parse_class_thresholds(values: list[str], classes: tuple[ClassSpec, ...]) -> dict[str, int]:
    valid_names = {spec.name for spec in classes}
    thresholds: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CLASS=COUNT, got: {value}")
        name, raw_count = value.split("=", 1)
        name = name.strip()
        if name not in valid_names:
            raise ValueError(f"Unknown class in --class-min-pixels: {name}")
        count = int(raw_count)
        if count < 1:
            raise ValueError("Class pixel thresholds must be at least 1")
        thresholds[name] = count
    return thresholds


def load_parent_split(
    path: str | Path,
    available_parent_ids: set[str],
) -> tuple[dict[str, str], set[str], dict]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    split_names = ("train", "val", "test")
    split_by_parent: dict[str, str] = {}
    excluded = set(str(value) for value in raw.get("exclude", []))

    for split_name in split_names:
        values = raw.get(split_name)
        if not isinstance(values, list):
            raise ValueError(f"Parent split manifest must contain a {split_name} list")
        for value in values:
            parent_id = str(value)
            if parent_id in split_by_parent or parent_id in excluded:
                raise ValueError(f"Parent ID appears in multiple split groups: {parent_id}")
            split_by_parent[parent_id] = split_name

    assigned = set(split_by_parent) | excluded
    unknown = sorted(assigned - available_parent_ids)
    missing = sorted(available_parent_ids - assigned)
    if unknown:
        raise ValueError(f"Parent split contains unknown IDs: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Parent split leaves IDs unassigned: {', '.join(missing)}")
    return split_by_parent, excluded, raw


def prepare_patches(
    config_path: str | Path,
    output_root: str | Path,
    patch_size: int = 512,
    patch_overlap: int = 128,
    min_class_pixels: int = 16,
    min_class_ratio: float = 0.0,
    class_min_pixels: dict[str, int] | None = None,
    compression: str = "deflate",
    parent_split: str | Path | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    edge_mode: str = "shift",
    ignore_background_labels: bool = False,
) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    output_root = Path(output_root)

    if output_root.resolve() == config.dataset_root.resolve():
        raise ValueError("--output-root must differ from the source dataset_root")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError("patch_overlap must be in [0, patch_size)")
    if edge_mode not in {"shift", "pad"}:
        raise ValueError("edge_mode must be 'shift' or 'pad'")
    if min_class_pixels < 1:
        raise ValueError("min_class_pixels must be at least 1")
    if min_class_ratio < 0.0 or min_class_ratio > 1.0:
        raise ValueError("min_class_ratio must be in [0, 1]")
    if compression not in {"none", "deflate"}:
        raise ValueError("compression must be 'none' or 'deflate'")

    all_items = discover_potsdam_items(config)
    split_by_parent: dict[str, str] = {}
    excluded_parent_ids: set[str] = set()
    split_manifest: dict | None = None
    if parent_split is not None:
        split_by_parent, excluded_parent_ids, split_manifest = load_parent_split(
            parent_split,
            {item.image_id for item in all_items},
        )
        items = [item for item in all_items if item.image_id in split_by_parent]
    else:
        items = all_items
    if limit is not None:
        items = items[:limit]
    missing_labels = [item.image_id for item in items if item.label_path is None]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise FileNotFoundError(f"Pixel labels are required to derive patch tags: {preview}")

    image_root = output_root / config.image_dir
    label_root = output_root / config.label_dir
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    class_names = [spec.name for spec in config.classes]
    label_rows: list[dict[str, int | str]] = []
    split_label_rows: dict[str, list[dict[str, int | str]]] = {
        split_name: [] for split_name in ("train", "val", "test")
    }
    metadata_rows: list[dict[str, int | float | str]] = []
    written_patches = 0
    compression_value = None if compression == "none" else compression

    for item in tqdm(items, desc="patching Potsdam"):
        image = tifffile.imread(item.image_path)
        label = read_label_rgb(item.label_path)  # type: ignore[arg-type]
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected HxWxC source image, got {image.shape}: {item.image_path}")
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/label shape mismatch for {item.image_id}: "
                f"{image.shape[:2]} vs {label.shape[:2]}"
            )

        height, width = image.shape[:2]
        tiles = generate_tiles(
            width,
            height,
            patch_size,
            patch_overlap,
            edge_mode=edge_mode,
        )
        for tile in tiles:
            patch_id = f"{item.image_id}_x{tile.x0:04d}_y{tile.y0:04d}"
            image_name = f"{patch_id}_RGBIR.tif"
            label_name = f"{patch_id}_label.tif"
            image_path = image_root / image_name
            label_path = label_root / label_name

            image_patch = _pad_patch(crop_tile(image, tile), tile.height, tile.width)
            label_patch = _pad_patch(crop_tile(label, tile), tile.height, tile.width)
            weak_labels = image_level_from_label(
                label_patch,
                config.classes,
                min_class_pixels=min_class_pixels,
                min_class_ratio=min_class_ratio,
                class_min_pixels=class_min_pixels,
            )

            if not skip_existing or not image_path.exists():
                _write_tiff(image_path, image_patch, compression_value)
            if not skip_existing or not label_path.exists():
                _write_tiff(label_path, label_patch, compression_value)

            label_rows.append({"image_id": patch_id, **weak_labels})
            split_name = split_by_parent.get(item.image_id, "all")
            if split_name in split_label_rows:
                split_label_rows[split_name].append(
                    {"image_id": patch_id, **weak_labels}
                )
            metadata = {
                "image_id": patch_id,
                "parent_image_id": item.image_id,
                "split": split_name,
                "image_path": (Path(config.image_dir) / image_name).as_posix(),
                "label_path": (Path(config.label_dir) / label_name).as_posix(),
                "x0": tile.x0,
                "y0": tile.y0,
                "x1": tile.x1,
                "y1": tile.y1,
                "width": tile.width,
                "height": tile.height,
                "content_width": min(tile.x1, width) - tile.x0,
                "content_height": min(tile.y1, height) - tile.y0,
            }
            for spec in config.classes:
                class_mask = np.all(
                    label_patch == np.asarray(spec.label_color, dtype=np.uint8),
                    axis=-1,
                )
                count = int(np.count_nonzero(class_mask))
                metadata[f"{spec.name}_pixels"] = count
                metadata[f"{spec.name}_ratio"] = count / float(tile.width * tile.height)
            metadata_rows.append(metadata)
            written_patches += 1

    labels_csv = output_root / "image_level_labels.csv"
    metadata_csv = output_root / "patches.csv"
    _write_csv_atomic(labels_csv, ["image_id", *class_names], label_rows)
    split_csvs: dict[str, str] = {}
    if split_manifest is not None:
        for split_name, rows in split_label_rows.items():
            split_csv = output_root / f"image_level_labels_{split_name}.csv"
            _write_csv_atomic(split_csv, ["image_id", *class_names], rows)
            split_csvs[split_name] = str(split_csv)
    metadata_fields = list(metadata_rows[0]) if metadata_rows else ["image_id"]
    _write_csv_atomic(metadata_csv, metadata_fields, metadata_rows)

    output_config = output_root / "potsdam_patches_config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["dataset_root"] = str(output_root.resolve())
    raw_config["tile_size"] = patch_size
    raw_config["tile_overlap"] = 0
    if ignore_background_labels:
        raw_config["background_colors"] = []
    if split_manifest is not None:
        split_copy = output_root / "parent_split.json"
        split_copy.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
        raw_config["parent_split_manifest"] = str(split_copy.resolve())
    output_config.write_text(json.dumps(raw_config, indent=2), encoding="utf-8")

    split_parent_counts = {
        split_name: sum(value == split_name for value in split_by_parent.values())
        for split_name in ("train", "val", "test")
    }
    split_patch_counts = {
        split_name: len(rows) for split_name, rows in split_label_rows.items()
    }

    summary = {
        "source_dataset_root": str(config.dataset_root),
        "output_root": str(output_root.resolve()),
        "source_images": len(items),
        "available_source_images": len(all_items),
        "excluded_parent_images": sorted(excluded_parent_ids),
        "split_parent_images": split_parent_counts,
        "split_patches": split_patch_counts,
        "patches": written_patches,
        "patch_size": patch_size,
        "patch_overlap": patch_overlap,
        "edge_mode": edge_mode,
        "ignore_background_labels": ignore_background_labels,
        "min_class_pixels": min_class_pixels,
        "min_class_ratio": min_class_ratio,
        "class_min_pixels": class_min_pixels or {},
        "image_level_labels": str(labels_csv),
        "split_image_level_labels": split_csvs,
        "patch_metadata": str(metadata_csv),
        "config": str(output_config),
    }
    (output_root / "patch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_tiff(path: Path, array: np.ndarray, compression: str | None) -> None:
    kwargs = {"photometric": "rgb"}
    if compression is not None:
        kwargs["compression"] = compression
    tifffile.imwrite(path, array, **kwargs)


def _pad_patch(array: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_height = height - array.shape[0]
    pad_width = width - array.shape[1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            f"Patch shape {array.shape[:2]} exceeds target {(height, width)}"
        )
    if pad_height or pad_width:
        pad_widths = [(0, pad_height), (0, pad_width)]
        pad_widths.extend((0, 0) for _ in range(array.ndim - 2))
        array = np.pad(array, pad_widths, mode="constant", constant_values=0)
    return np.ascontiguousarray(array)


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    thresholds = parse_class_thresholds(args.class_min_pixels, config.classes)
    summary = prepare_patches(
        config_path=args.config,
        output_root=args.output_root,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        edge_mode=args.edge_mode,
        min_class_pixels=args.min_class_pixels,
        min_class_ratio=args.min_class_ratio,
        class_min_pixels=thresholds,
        compression=args.compression,
        ignore_background_labels=args.ignore_background_labels,
        parent_split=args.parent_split,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
