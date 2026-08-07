from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .potsdam import read_image_level_csv
from .visualize import save_label_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge corrected foreground pseudo labels with background seeds. "
            "Non-background labels from the seed source are never copied."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--foreground-pseudo-label-dir", required=True)
    parser.add_argument("--background-seed-label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def merge_background_seed_dataset(
    config_path: str | Path,
    labels_csv: str | Path,
    foreground_pseudo_label_dir: str | Path,
    background_seed_label_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    image_ids = sorted(read_image_level_csv(labels_csv))
    if limit is not None:
        image_ids = image_ids[:limit]
    foreground_dir = Path(foreground_pseudo_label_dir)
    background_dir = Path(background_seed_label_dir)
    output_dir = Path(output_dir)
    _prepare_output(output_dir)
    foreground_ids = np.asarray([spec.id for spec in config.classes])
    skipped = Counter()
    totals = Counter()
    summaries = []

    for image_id in tqdm(image_ids, desc="merging-background-seeds"):
        foreground_path = foreground_dir / f"{image_id}.png"
        background_path = background_dir / f"{image_id}.png"
        if not foreground_path.exists():
            skipped["missing_foreground"] += 1
            continue
        if not background_path.exists():
            skipped["missing_background_seed"] += 1
            continue
        foreground = np.asarray(Image.open(foreground_path), dtype=np.uint8)
        background_source = np.asarray(
            Image.open(background_path), dtype=np.uint8
        )
        if foreground.shape != background_source.shape:
            raise ValueError(f"Input label shape mismatch for {image_id}")

        foreground_mask = np.isin(foreground, foreground_ids)
        background_mask = background_source == 0
        conflict_mask = foreground_mask & background_mask
        merged = np.full(foreground.shape, config.ignore_index, dtype=np.uint8)
        merged[foreground_mask] = foreground[foreground_mask]
        merged[background_mask & ~foreground_mask] = 0
        save_label_png(merged, output_dir / "pseudo_labels" / f"{image_id}.png")

        counts = {
            "foreground_pixels": int(foreground_mask.sum()),
            "background_pixels": int((merged == 0).sum()),
            "ignored_pixels": int((merged == config.ignore_index).sum()),
            "foreground_background_conflicts": int(conflict_mask.sum()),
            "discarded_old_foreground_pixels": int(
                ((background_source != 0) & (background_source != config.ignore_index)).sum()
            ),
        }
        totals.update(counts)
        summaries.append({"image_id": image_id, **counts})

    report = {
        "protocol": {
            "method": "corrected foreground plus background seeds",
            "pixel_gt_used": False,
            "foreground_rule": "copy configured foreground class IDs",
            "background_rule": (
                "copy only class 0 where corrected foreground is absent"
            ),
            "conflict_rule": "corrected foreground wins over background seed",
            "old_foreground_rule": (
                "discard every non-background class from background source"
            ),
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "foreground_pseudo_label_dir": str(foreground_dir.resolve()),
            "background_seed_label_dir": str(background_dir.resolve()),
        },
        "input_images": len(image_ids),
        "processed_images": len(summaries),
        "skipped_images": dict(sorted(skipped.items())),
        "pixel_totals": dict(sorted(totals.items())),
        "items": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _prepare_output(output_dir: Path) -> None:
    artifacts = [output_dir / "pseudo_labels", output_dir / "summary.json"]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "Background merge output already contains artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    report = merge_background_seed_dataset(
        config_path=args.config,
        labels_csv=args.labels_csv,
        foreground_pseudo_label_dir=args.foreground_pseudo_label_dir,
        background_seed_label_dir=args.background_seed_label_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))
    skipped = sum(report["skipped_images"].values())
    if args.require_all and skipped:
        raise RuntimeError(
            f"Could not merge {skipped}/{report['input_images']} images"
        )


if __name__ == "__main__":
    main()
