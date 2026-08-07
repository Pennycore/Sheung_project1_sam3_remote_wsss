from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .candidate_region_scores import (
    active_region_decisions,
    load_region_score_cache,
)
from .candidate_visual_prototypes import (
    load_visual_prototype_calibration,
    normalize_features,
    validate_visual_prototype_model,
)
from .config import load_config
from .potsdam import read_image_level_csv
from .rebuild_candidate_pseudo_labels import fuse_candidate_assignments
from .visualize import save_label_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export pseudo labels using frozen candidate visual prototypes "
            "and CAM consensus. Pixel ground truth is never read."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--region-score-dir", required=True)
    parser.add_argument("--prototype-calibration", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cam-method", choices=("mean", "top20"), default="mean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def reconcile_candidate_visual_prototypes(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    region_score_dir: str | Path,
    prototype_calibration: str | Path,
    cam_dir: str | Path,
    output_dir: str | Path,
    cam_method: str = "mean",
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    class_by_name = {spec.name: spec for spec in config.classes}
    candidate_dir = Path(candidate_dir)
    region_score_dir = Path(region_score_dir)
    calibration_path = Path(prototype_calibration)
    cam_dir = Path(cam_dir)
    output_dir = Path(output_dir)
    _prepare_output(output_dir)
    prototype_metadata, prototype_class_ids, prototypes = (
        load_visual_prototype_calibration(calibration_path)
    )
    actions = Counter()
    class_actions: dict[str, Counter] = defaultdict(Counter)
    skipped = Counter()
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    summaries = []

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(image_ids, desc="reconciling-visual-prototypes"):
        if not candidate_cache_exists(candidate_dir, image_id):
            skip("missing_candidate_cache", image_id)
            continue
        if not (region_score_dir / f"{image_id}.npz").exists() or not (
            region_score_dir / f"{image_id}.json"
        ).exists():
            skip("missing_region_scores", image_id)
            continue
        cam_path = cam_dir / f"{image_id}.npz"
        if not cam_path.exists():
            skip("missing_cam", image_id)
            continue

        cache_metadata, candidates = load_candidate_cache(candidate_dir, image_id)
        score_metadata, score_data = load_region_score_cache(
            region_score_dir, image_id, candidate_dir=candidate_dir
        )
        if "region_features" not in score_data:
            raise ValueError(
                "Visual prototype reconciliation requires region_features"
            )
        if len(candidates) != int(score_metadata["candidate_count"]):
            raise ValueError(f"Candidate/region score mismatch for {image_id}")
        validate_visual_prototype_model(
            prototype_metadata,
            score_metadata,
            score_data["region_features"].shape[1],
        )
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        semantic_scores = np.einsum(
            "nd,cd->nc",
            normalize_features(score_data["region_features"]),
            prototypes,
            optimize=False,
        )
        region_predicted, _margins = active_region_decisions(
            semantic_scores, prototype_class_ids, active_class_ids
        )
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        image_shape = tuple(int(value) for value in cache_metadata["image_shape"])
        if tuple(cams.shape[1:]) != image_shape:
            raise ValueError(f"CAM/candidate shape mismatch for {image_id}")

        assignments = []
        image_actions = Counter()
        for index, candidate in enumerate(candidates):
            source_id = int(candidate.class_id)
            region_id = int(region_predicted[index])
            cam_id = int(
                score_candidate_cams(
                    candidate=candidate,
                    cams=cams,
                    cam_class_ids=cam_class_ids,
                    active_class_ids=active_class_ids,
                )[cam_method]["predicted_class_id"]
            )
            target_id = region_id if region_id == cam_id else source_id
            action = "keep" if target_id == source_id else "relabel"
            assignments.append(target_id)
            actions[action] += 1
            image_actions[action] += 1
            class_actions[candidate.class_name][action] += 1

        label = fuse_candidate_assignments(
            image_shape=image_shape,
            candidates=candidates,
            class_assignments=assignments,
            ignore_index=config.ignore_index,
            uncovered_label=config.uncovered_label,
            conflict_margin=config.conflict_margin,
        )
        save_label_png(label, output_dir / "pseudo_labels" / f"{image_id}.png")
        ids, pixel_counts = np.unique(label, return_counts=True)
        item_summary = {
            "image_id": image_id,
            "candidate_count": len(candidates),
            "actions": dict(sorted(image_actions.items())),
            "pseudo_label_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(ids, pixel_counts)
            },
        }
        metadata_path = output_dir / "metadata" / f"{image_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(item_summary, indent=2), encoding="utf-8")
        summaries.append(item_summary)

    calibration_data = calibration_path.parent / prototype_metadata["data_file"]
    report = {
        "protocol": {
            "method": "frozen visual prototype and CAM consensus",
            "pixel_gt_used": False,
            "states": ["keep", "relabel"],
            "consensus_rule": (
                "relabel only when CAM and visual prototype predict the same "
                "active image-level class; otherwise keep SAM3 source class"
            ),
            "cam_method": cam_method,
            "prototype_calibration": str(calibration_path.resolve()),
            "prototype_calibration_sha256": _sha256(calibration_path),
            "prototype_data_sha256": _sha256(calibration_data),
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_dir.resolve()),
            "region_score_dir": str(region_score_dir.resolve()),
            "cam_dir": str(cam_dir.resolve()),
        },
        "input_images": len(image_ids),
        "processed_images": len(summaries),
        "skipped_images": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "candidate_actions": dict(sorted(actions.items())),
        "class_actions": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(class_actions.items())
        },
        "items": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _prepare_output(output_dir: Path) -> None:
    artifacts = [
        output_dir / "pseudo_labels",
        output_dir / "metadata",
        output_dir / "summary.json",
    ]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "Visual reconciliation output already contains artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    report = reconcile_candidate_visual_prototypes(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        region_score_dir=args.region_score_dir,
        prototype_calibration=args.prototype_calibration,
        cam_dir=args.cam_dir,
        output_dir=args.output_dir,
        cam_method=args.cam_method,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))
    skipped = sum(report["skipped_images"].values())
    if args.require_all and skipped:
        raise RuntimeError(
            f"Could not reconcile {skipped}/{report['input_images']} images"
        )


if __name__ == "__main__":
    main()
