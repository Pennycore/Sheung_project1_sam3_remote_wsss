from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .cam.fusion import fuse_cam_and_sam_with_stats
from .config import load_config
from .potsdam import discover_potsdam_items, read_image_level_csv, read_rgbir_as_rgb
from .visualize import save_label_png, save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse class-aware SAM3 masks with CAM foreground/background seeds."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--sam-pseudo-label-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--background-threshold", type=float, default=0.2)
    parser.add_argument("--foreground-threshold", type=float, default=0.7)
    parser.add_argument("--cam-support-threshold", type=float, default=0.3)
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="Use CAMs only for background seeds; preserve all SAM3 foreground.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail before fusion if any labeled image lacks a SAM3 PNG or CAM NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_level = read_image_level_csv(args.labels_csv)
    class_id_by_name = {spec.name: spec.id for spec in config.classes}
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    expected_ids = set(image_level) & set(item_by_id)
    sam_dir = Path(args.sam_pseudo_label_dir)
    cam_dir = Path(args.cam_dir)
    if args.require_complete:
        _require_complete_inputs(expected_ids, sam_dir, cam_dir)
    sam_paths = sorted(
        path for path in sam_dir.glob("*.png") if path.stem in expected_ids
    )
    if args.limit is not None:
        sam_paths = sam_paths[: args.limit]
    output_dir = Path(args.output_dir)
    if args.skip_existing:
        sam_paths = [
            path
            for path in sam_paths
            if not (
                (output_dir / "pseudo_labels" / path.name).exists()
                and (output_dir / "overlays" / f"{path.stem}.jpg").exists()
                and (output_dir / "metadata" / f"{path.stem}.json").exists()
            )
        ]

    summaries = []
    for sam_path in tqdm(sam_paths, desc="fusing CAM and SAM3"):
        image_id = sam_path.stem
        item = item_by_id.get(image_id)
        cam_path = cam_dir / f"{image_id}.npz"
        if item is None or image_id not in image_level or not cam_path.exists():
            continue
        sam_label = np.asarray(Image.open(sam_path).convert("L"), dtype=np.uint8)
        with np.load(cam_path) as archive:
            cams = archive["cams"].astype(np.float32)
            class_ids = archive["class_ids"].astype(np.int64)
        positive_ids = {
            class_id_by_name[name]
            for name in image_level[image_id]
            if name in class_id_by_name
        }
        fused, stats = fuse_cam_and_sam_with_stats(
            sam_label=sam_label,
            cams=cams,
            class_ids=class_ids,
            positive_class_ids=positive_ids,
            background_threshold=args.background_threshold,
            foreground_threshold=args.foreground_threshold,
            cam_support_threshold=args.cam_support_threshold,
            ignore_index=config.ignore_index,
            background_only=args.background_only,
        )
        image = read_rgbir_as_rgb(item.image_path, config.rgb_band_indices)
        save_label_png(fused, output_dir / "pseudo_labels" / f"{image_id}.png")
        save_overlay(
            image,
            fused,
            config.classes,
            output_dir / "overlays" / f"{image_id}.jpg",
        )
        ids, counts = np.unique(fused, return_counts=True)
        metadata = {
            "image_id": image_id,
            "fusion_mode": "background_only" if args.background_only else "hybrid",
            "positive_classes": sorted(image_level[image_id]),
            "thresholds": {
                "background": args.background_threshold,
                "foreground": args.foreground_threshold,
                "cam_support": args.cam_support_threshold,
            },
            **stats,
            "pseudo_label_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(ids, counts)
            },
        }
        metadata_path = output_dir / "metadata" / f"{image_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summaries.append(metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "metadata").glob("*.json"))
        if path.stem in expected_ids
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2),
        encoding="utf-8",
    )
    print(
        f"Fused {len(summaries)} images this run; "
        f"summary contains {len(all_summaries)} images. Outputs: {output_dir}"
    )


def _require_complete_inputs(
    expected_ids: set[str],
    sam_dir: Path,
    cam_dir: Path,
) -> None:
    missing_sam = sorted(
        image_id
        for image_id in expected_ids
        if not (sam_dir / f"{image_id}.png").exists()
    )
    missing_cam = sorted(
        image_id
        for image_id in expected_ids
        if not (cam_dir / f"{image_id}.npz").exists()
    )
    if missing_sam or missing_cam:
        raise FileNotFoundError(
            "Incomplete fusion inputs: "
            f"missing SAM3={len(missing_sam)} {missing_sam[:5]}, "
            f"missing CAM={len(missing_cam)} {missing_cam[:5]}"
        )


if __name__ == "__main__":
    main()
