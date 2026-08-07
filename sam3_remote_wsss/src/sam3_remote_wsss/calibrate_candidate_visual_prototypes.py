from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .candidate_region_scores import load_region_score_cache
from .candidate_visual_prototypes import (
    normalize_features,
    robust_visual_prototype,
    save_visual_prototype_calibration,
)
from .config import load_config
from .potsdam import read_image_level_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build robust per-class visual prototypes from candidates whose "
            "SAM3 source class agrees with CAM. Pixel GT is never read."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--region-score-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cam-method", choices=("mean", "top20"), default="mean")
    parser.add_argument("--keep-fraction", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-seeds-per-class", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def calibrate_candidate_visual_prototypes(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    region_score_dir: str | Path,
    cam_dir: str | Path,
    output_path: str | Path,
    cam_method: str = "mean",
    keep_fraction: float = 0.7,
    iterations: int = 3,
    min_seeds_per_class: int = 10,
    limit: int | None = None,
) -> dict:
    if min_seeds_per_class <= 0:
        raise ValueError("min_seeds_per_class must be positive")
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    class_by_name = {spec.name: spec for spec in config.classes}
    id_to_name = {spec.id: spec.name for spec in config.classes}
    candidate_dir = Path(candidate_dir)
    region_score_dir = Path(region_score_dir)
    cam_dir = Path(cam_dir)
    seeds: dict[int, list[np.ndarray]] = defaultdict(list)
    source_candidates = Counter()
    skipped = Counter()
    model_identity = None
    evaluated_images = 0

    for image_id in tqdm(image_ids, desc="calibrating-visual-prototypes"):
        if not candidate_cache_exists(candidate_dir, image_id):
            skipped["missing_candidate_cache"] += 1
            continue
        cam_path = cam_dir / f"{image_id}.npz"
        if not cam_path.exists():
            skipped["missing_cam"] += 1
            continue
        if not (region_score_dir / f"{image_id}.npz").exists():
            skipped["missing_region_scores"] += 1
            continue

        _candidate_metadata, candidates = load_candidate_cache(
            candidate_dir, image_id
        )
        score_metadata, score_data = load_region_score_cache(
            region_score_dir, image_id, candidate_dir=candidate_dir
        )
        if "region_features" not in score_data:
            raise ValueError(
                "Region cache does not contain visual features; rerun "
                "score_candidate_regions into a new output directory"
            )
        identity = (
            score_metadata.get("model_name"),
            score_metadata.get("weights_source"),
            int(score_data["region_features"].shape[1]),
        )
        if model_identity is None:
            model_identity = identity
        elif identity != model_identity:
            raise ValueError("Region score directory mixes model identities")
        features = normalize_features(score_data["region_features"])
        if features.shape[0] != len(candidates):
            raise ValueError(f"Candidate/feature count mismatch for {image_id}")
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        for index, candidate in enumerate(candidates):
            source_id = int(candidate.class_id)
            source_candidates[source_id] += 1
            cam = score_candidate_cams(
                candidate=candidate,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
            )[cam_method]
            if int(cam["predicted_class_id"]) == source_id:
                seeds[source_id].append(features[index])
        evaluated_images += 1

    if model_identity is None:
        raise RuntimeError("No region feature cache was available for calibration")
    class_ids = np.asarray([spec.id for spec in config.classes], dtype=np.int16)
    prototypes = []
    class_summary = {}
    for class_id in class_ids:
        class_id = int(class_id)
        values = np.asarray(seeds[class_id], dtype=np.float32)
        if values.shape[0] < min_seeds_per_class:
            raise ValueError(
                f"Class {id_to_name[class_id]} has only {values.shape[0]} "
                f"CAM-consistent seeds; need {min_seeds_per_class}"
            )
        prototype, retained, similarities = robust_visual_prototype(
            values,
            keep_fraction=keep_fraction,
            iterations=iterations,
        )
        prototypes.append(prototype)
        class_summary[id_to_name[class_id]] = {
            "class_id": class_id,
            "source_candidates": int(source_candidates[class_id]),
            "cam_consistent_seeds": int(values.shape[0]),
            "retained_seeds": int(retained.size),
            "retained_fraction": float(retained.size / values.shape[0]),
            "similarity_percentiles": _percentiles(similarities),
            "retained_similarity_percentiles": _percentiles(
                similarities[retained]
            ),
        }

    summary = {
        "protocol": {
            "method": "robust CAM-consistent candidate visual prototypes",
            "pixel_gt_used": False,
            "seed_rule": "SAM3 source class equals active-class CAM top1",
            "outlier_rule": (
                "iteratively retain candidates with the highest cosine "
                "similarity to their source-class mean"
            ),
            "cam_method": cam_method,
            "keep_fraction": keep_fraction,
            "iterations": iterations,
            "min_seeds_per_class": min_seeds_per_class,
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_dir.resolve()),
            "region_score_dir": str(region_score_dir.resolve()),
            "cam_dir": str(cam_dir.resolve()),
        },
        "input_images": len(image_ids),
        "evaluated_images": evaluated_images,
        "skipped_images": dict(sorted(skipped.items())),
        "model_name": model_identity[0],
        "weights_source": model_identity[1],
        "feature_dimension": model_identity[2],
        "classes": class_summary,
    }
    save_visual_prototype_calibration(
        output_path=output_path,
        class_ids=class_ids,
        prototypes=np.stack(prototypes),
        metadata=summary,
    )
    return summary


def _percentiles(values: np.ndarray) -> dict[str, float]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(values, levels))
    }


def main() -> None:
    args = parse_args()
    summary = calibrate_candidate_visual_prototypes(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        region_score_dir=args.region_score_dir,
        cam_dir=args.cam_dir,
        output_path=args.output,
        cam_method=args.cam_method,
        keep_fraction=args.keep_fraction,
        iterations=args.iterations,
        min_seeds_per_class=args.min_seeds_per_class,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    skipped = sum(summary["skipped_images"].values())
    if args.require_all and skipped:
        raise RuntimeError(
            f"Could not calibrate {skipped}/{summary['input_images']} images"
        )


if __name__ == "__main__":
    main()
