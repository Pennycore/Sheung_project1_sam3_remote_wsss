from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import ClassSpec, load_config
from .fusion import FusionCanvas, filter_masks
from .potsdam import discover_potsdam_items, read_image_level_csv, read_rgbir_as_rgb, read_tiff_size
from .prompts import prompts_for_class
from .remoteclip_backend import RemoteCLIPPromptSelector
from .sam3_backend import SAM3ImageBackend
from .tiling import crop_tile, generate_tiles
from .visualize import save_label_png, save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM3 pseudo labels for remote-sensing WSSS.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--labels-csv", required=True, help="Image-level label CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to process.")
    parser.add_argument("--image-id", action="append", default=None, help="Process only this image id. Can repeat.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of dataset shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose pseudo-label PNG already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned work without loading SAM3.")
    return parser.parse_args()


class SAM3WSSSPseudoLabeler:
    def __init__(self, config_path: str | Path, labels_csv: str | Path, output_dir: str | Path, dry_run: bool = False) -> None:
        self.config = load_config(config_path)
        self.image_level = read_image_level_csv(labels_csv)
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.class_by_name = {spec.name: spec for spec in self.config.classes}
        self.backend = None
        self.prompt_selector = None
        if not dry_run:
            self.backend = SAM3ImageBackend(
                sam3_repo=self.config.sam3_repo,
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                confidence_threshold=self.config.score_threshold,
            )
            if self.config.remoteclip.enabled:
                self.prompt_selector = RemoteCLIPPromptSelector(self.config.remoteclip)

    def process_item(self, image_id: str, image_path: Path) -> dict:
        positives = sorted(self.image_level.get(image_id, set()))
        specs = [self.class_by_name[name] for name in positives if name in self.class_by_name]
        if self.dry_run:
            width, height = read_tiff_size(image_path)
            tiles = generate_tiles(width, height, self.config.tile_size, self.config.tile_overlap)
            metadata = {
                "image_id": image_id,
                "image_path": str(image_path),
                "positive_classes": positives,
                "tile_count": len(tiles),
                "kept_masks": 0,
                "prompts": {spec.name: prompts_for_class(spec, self.config.prompting) for spec in specs},
                "remoteclip_enabled": self.config.remoteclip.enabled,
            }
            return metadata

        image = read_rgbir_as_rgb(image_path, self.config.rgb_band_indices)
        height, width = image.shape[:2]
        tiles = generate_tiles(width, height, self.config.tile_size, self.config.tile_overlap)

        metadata = {
            "image_id": image_id,
            "image_path": str(image_path),
            "positive_classes": positives,
            "tile_count": len(tiles),
            "kept_masks": 0,
            "prompts": {},
            "remoteclip_enabled": self.config.remoteclip.enabled,
            "remoteclip_selected_prompts": {},
        }

        canvas = FusionCanvas(
            height=height,
            width=width,
            ignore_index=self.config.ignore_index,
            uncovered_label=self.config.uncovered_label,
            conflict_margin=self.config.conflict_margin,
        )

        for tile in tiles:
            tile_image = crop_tile(image, tile)
            prompt_jobs: list[tuple[ClassSpec, str]] = []
            for spec in specs:
                metadata["prompts"].setdefault(
                    spec.name,
                    prompts_for_class(spec, self.config.prompting),
                )
                for prompt in metadata["prompts"][spec.name]:
                    prompt_jobs.append((spec, prompt))
            if not prompt_jobs:
                continue
            if self.prompt_selector is not None:
                selected = self.prompt_selector.select(tile_image, prompt_jobs)
                prompt_jobs = [(spec, prompt) for spec, prompt, _score in selected]
                for spec, prompt, score in selected:
                    metadata["remoteclip_selected_prompts"].setdefault(spec.name, []).append(
                        {
                            "prompt": prompt,
                            "score": score,
                            "tile": [tile.x0, tile.y0, tile.x1, tile.y1],
                        }
                    )
                if not prompt_jobs:
                    continue
            outputs = self.backend.predict_texts(  # type: ignore[union-attr]
                tile_image, [prompt for _, prompt in prompt_jobs]
            )
            for (spec, _prompt), output in zip(prompt_jobs, outputs):
                kept = filter_masks(
                    output["masks"],
                    output["scores"],
                    score_threshold=self.config.score_threshold,
                    min_area=self.config.min_mask_area,
                    max_area_ratio=self.config.max_mask_area_ratio,
                )
                for mask, score in kept:
                    canvas.add_mask(mask=mask, class_id=spec.id, score=score, x0=tile.x0, y0=tile.y0)
                metadata["kept_masks"] += len(kept)

        label = canvas.result()
        self._save_outputs(image_id, image, label, metadata)
        return metadata

    def _save_outputs(self, image_id: str, image, label, metadata: dict) -> None:
        save_label_png(label, self.output_dir / "pseudo_labels" / f"{image_id}.png")
        save_overlay(image, label, self.config.classes, self.output_dir / "overlays" / f"{image_id}.jpg")
        meta_path = self.output_dir / "metadata" / f"{image_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    labeler = SAM3WSSSPseudoLabeler(
        config_path=args.config,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    items = discover_potsdam_items(labeler.config)
    if args.image_id:
        wanted = set(args.image_id)
        items = [item for item in items if item.image_id in wanted]
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        items = [
            item
            for idx, item in enumerate(items)
            if idx % args.num_shards == args.shard_index
        ]
    if args.limit is not None:
        items = items[: args.limit]
    if args.skip_existing:
        output_dir = Path(args.output_dir)
        items = [
            item
            for item in items
            if not (output_dir / "pseudo_labels" / f"{item.image_id}.png").exists()
        ]

    summaries = []
    for item in tqdm(items, desc="pseudo-labeling"):
        summaries.append(labeler.process_item(item.image_id, item.image_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Processed {len(items)} images. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
