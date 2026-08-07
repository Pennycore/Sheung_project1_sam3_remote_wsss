from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from tqdm import tqdm

from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .candidate_region_scores import (
    build_class_text_prototypes,
    candidate_cache_fingerprint,
    save_region_score_cache,
    score_candidate_regions,
)
from .config import load_config
from .potsdam import (
    discover_potsdam_items,
    read_image_level_csv,
    read_rgbir_as_rgb,
)
from .remoteclip_backend import RemoteCLIPPromptSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache CLIP/RemoteCLIP semantics for each SAM3 candidate using "
            "context and mask-emphasized region views."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-ratio", type=float, default=0.25)
    parser.add_argument("--min-crop-size", type=int, default=48)
    parser.add_argument("--background-retain", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def score_candidate_region_dataset(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else config.remoteclip.checkpoint_path
    )
    encoder_config = replace(
        config.remoteclip,
        enabled=True,
        model_name=args.model_name or config.remoteclip.model_name,
        checkpoint_path=checkpoint_path,
        device=args.device or config.remoteclip.device,
    )
    encoder = RemoteCLIPPromptSelector(encoder_config)
    class_ids, prototypes, prompt_groups = build_class_text_prototypes(
        encoder, config.classes
    )
    image_level = read_image_level_csv(args.labels_csv)
    image_ids = sorted(image_level)
    if args.image_id is not None:
        if args.image_id not in image_level:
            raise KeyError(f"Image ID is not present in labels CSV: {args.image_id}")
        image_ids = [args.image_id]
    elif args.limit is not None:
        image_ids = image_ids[: args.limit]
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    class_by_name = {spec.name: spec for spec in config.classes}
    output_dir = Path(args.output_dir)
    skipped = Counter()
    processed = 0
    candidates_scored = 0

    for image_id in tqdm(image_ids, desc="candidate-region-scoring"):
        output_npz = output_dir / f"{image_id}.npz"
        output_json = output_dir / f"{image_id}.json"
        if args.skip_existing and output_npz.exists() and output_json.exists():
            skipped["existing"] += 1
            continue
        if not candidate_cache_exists(args.candidate_dir, image_id):
            skipped["missing_candidate_cache"] += 1
            continue
        item = item_by_id.get(image_id)
        if item is None:
            skipped["missing_item"] += 1
            continue
        candidate_metadata, candidates = load_candidate_cache(
            args.candidate_dir, image_id
        )
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        if not active_class_ids:
            skipped["no_positive_classes"] += 1
            continue
        image_rgb = read_rgbir_as_rgb(item.image_path, config.rgb_band_indices)
        scores, crop_boxes, mask_fractions = score_candidate_regions(
            image_rgb=image_rgb,
            candidates=candidates,
            encoder=encoder,
            class_prototypes=prototypes,
            batch_size=args.batch_size,
            context_ratio=args.context_ratio,
            min_crop_size=args.min_crop_size,
            background_retain=args.background_retain,
        )
        save_region_score_cache(
            output_dir=output_dir,
            image_id=image_id,
            scores=scores,
            class_ids=class_ids,
            active_class_ids=active_class_ids,
            crop_boxes=crop_boxes,
            mask_fractions=mask_fractions,
            candidate_fingerprint=candidate_cache_fingerprint(
                args.candidate_dir, image_id
            ),
            metadata={
                "candidate_cache": str(Path(args.candidate_dir).resolve()),
                "candidate_image_shape": candidate_metadata["image_shape"],
                "model_name": encoder_config.model_name,
                "weights_source": encoder.weights_source,
                "device": encoder_config.device,
                "view_fusion": "normalized mean of context and mask-emphasized features",
                "context_ratio": args.context_ratio,
                "min_crop_size": args.min_crop_size,
                "background_retain": args.background_retain,
                "class_ids": class_ids.tolist(),
                "active_class_ids": active_class_ids,
                "class_prompt_prototypes": prompt_groups,
                "pixel_gt_used": False,
            },
        )
        processed += 1
        candidates_scored += len(candidates)

    summary = {
        "input_images": len(image_ids),
        "processed_images": processed,
        "candidates_scored": candidates_scored,
        "skipped": dict(sorted(skipped.items())),
        "output_dir": str(output_dir),
        "model_name": encoder_config.model_name,
        "weights_source": encoder.weights_source,
        "pixel_gt_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = score_candidate_region_dataset(args)
    print(json.dumps(summary, indent=2))
    missing = sum(
        count for reason, count in summary["skipped"].items() if reason != "existing"
    )
    if args.require_all and missing:
        raise RuntimeError(
            f"Could not score {missing}/{summary['input_images']} requested images"
        )


if __name__ == "__main__":
    main()
