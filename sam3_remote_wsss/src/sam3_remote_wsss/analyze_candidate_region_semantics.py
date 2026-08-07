from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .analyze_candidate_quality import candidate_quality_record
from .analyze_candidate_recoverability import (
    accumulate_prediction,
    empty_metric_state,
)
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
from .evaluate_pseudo_labels import compute_evaluation_metrics
from .potsdam import (
    discover_potsdam_items,
    label_rgb_to_ids,
    read_image_level_csv,
    read_label_rgb,
)
from .rebuild_candidate_pseudo_labels import fuse_candidate_assignments


POLICIES = ("baseline", "region_relabel", "cam_region_consensus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose candidate-region CLIP semantics and CAM-region consensus. "
            "Pixel GT is used only for offline evaluation."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--region-score-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument(
        "--prototype-calibration",
        default=None,
        help=(
            "Optional frozen CAM-consistent visual prototype calibration. "
            "When omitted, use Manual4 text-prototype region scores."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output", default=None)
    parser.add_argument("--cam-method", choices=("mean", "top20"), default="mean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def summarize_assignment_records(
    records: list[dict],
    assignment_name: str,
    id_to_name: dict[int, str],
) -> dict:
    assigned = [
        (record, int(record["assignments"][assignment_name]))
        for record in records
    ]
    relabeled = [
        (record, target)
        for record, target in assigned
        if target != int(record["class_id"])
    ]
    beneficial = [
        (record, target)
        for record, target in relabeled
        if int(record["class_id"]) != int(record["dominant_class_id"])
        and target == int(record["dominant_class_id"])
    ]
    destructive = [
        (record, target)
        for record, target in relabeled
        if int(record["class_id"]) == int(record["dominant_class_id"])
    ]
    unresolved = [
        (record, target)
        for record, target in relabeled
        if int(record["class_id"]) != int(record["dominant_class_id"])
        and target != int(record["dominant_class_id"])
    ]
    pairs: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for record, target in relabeled:
        pair = f"{record['class_name']}->{id_to_name[target]}"
        pairs[pair].append((record, target))

    return {
        "candidates": len(records),
        "kept": len(records) - len(relabeled),
        "relabeled": len(relabeled),
        "assigned_dominant_match_rate": _rate(
            sum(target == int(record["dominant_class_id"]) for record, target in assigned),
            len(assigned),
        ),
        "pixel_weighted_assigned_purity": _assigned_pixel_purity(
            assigned, id_to_name
        ),
        "beneficial_relabels": len(beneficial),
        "beneficial_relabel_rate": _rate(len(beneficial), len(relabeled)),
        "destructive_relabels": len(destructive),
        "destructive_relabel_rate": _rate(len(destructive), len(relabeled)),
        "unresolved_relabels": len(unresolved),
        "unresolved_relabel_rate": _rate(len(unresolved), len(relabeled)),
        "per_relabel_pair": {
            pair: {
                "candidates": len(group),
                "beneficial_rate": _rate(
                    sum(
                        target == int(record["dominant_class_id"])
                        and int(record["class_id"])
                        != int(record["dominant_class_id"])
                        for record, target in group
                    ),
                    len(group),
                ),
                "destructive_rate": _rate(
                    sum(
                        int(record["class_id"])
                        == int(record["dominant_class_id"])
                        for record, _target in group
                    ),
                    len(group),
                ),
                "pixel_weighted_assigned_purity": _assigned_pixel_purity(
                    group, id_to_name
                ),
            }
            for pair, group in sorted(pairs.items())
        },
    }


def analyze_candidate_region_semantics(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    region_score_dir: str | Path,
    cam_dir: str | Path,
    prototype_calibration: str | Path | None = None,
    cam_method: str = "mean",
    limit: int | None = None,
) -> tuple[dict, list[dict]]:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    class_by_name = {spec.name: spec for spec in config.classes}
    id_to_name = {0: "background", **{spec.id: spec.name for spec in config.classes}}
    num_classes = max(id_to_name) + 1
    candidate_dir = Path(candidate_dir)
    region_score_dir = Path(region_score_dir)
    cam_dir = Path(cam_dir)
    prototype_metadata = None
    prototype_class_ids = None
    visual_prototypes = None
    if prototype_calibration is not None:
        (
            prototype_metadata,
            prototype_class_ids,
            visual_prototypes,
        ) = load_visual_prototype_calibration(prototype_calibration)
    states = {policy: empty_metric_state(num_classes) for policy in POLICIES}
    records = []
    skipped = Counter()
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    evaluated_images = 0

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(image_ids, desc="candidate-region-diagnostic"):
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
        item = item_by_id.get(image_id)
        if item is None:
            skip("missing_item", image_id)
            continue
        if item.label_path is None:
            skip("missing_label", image_id)
            continue

        candidate_metadata, candidates = load_candidate_cache(
            candidate_dir, image_id
        )
        score_metadata, score_data = load_region_score_cache(
            region_score_dir, image_id, candidate_dir=candidate_dir
        )
        if len(candidates) != int(score_metadata["candidate_count"]):
            raise ValueError(f"Candidate/region score mismatch for {image_id}")
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        if visual_prototypes is None:
            semantic_scores = score_data["scores"].astype(np.float32)
            semantic_class_ids = score_data["class_ids"]
        else:
            if "region_features" not in score_data:
                raise ValueError(
                    "Visual prototype analysis requires region_features; rerun "
                    "score_candidate_regions into a new output directory"
                )
            validate_visual_prototype_model(
                prototype_metadata,
                score_metadata,
                score_data["region_features"].shape[1],
            )
            semantic_scores = np.einsum(
                "nd,cd->nc",
                normalize_features(score_data["region_features"]),
                visual_prototypes,
                optimize=False,
            )
            semantic_class_ids = prototype_class_ids
        region_predicted, region_margins = active_region_decisions(
            semantic_scores, semantic_class_ids, active_class_ids
        )
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        gt = label_rgb_to_ids(
            read_label_rgb(item.label_path),
            config.classes,
            config.ignore_index,
            background_colors=config.background_colors,
        )
        if list(gt.shape) != list(candidate_metadata["image_shape"]):
            raise ValueError(f"Candidate/GT shape mismatch for {image_id}")
        if tuple(cams.shape[1:]) != tuple(gt.shape):
            raise ValueError(f"CAM/GT shape mismatch for {image_id}")

        assignments = {policy: [] for policy in POLICIES}
        for index, candidate in enumerate(candidates):
            source_id = int(candidate.class_id)
            region_id = int(region_predicted[index])
            cam = score_candidate_cams(
                candidate=candidate,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
            )[cam_method]
            cam_id = int(cam["predicted_class_id"])
            consensus_id = region_id if region_id == cam_id else source_id
            assignments["baseline"].append(source_id)
            assignments["region_relabel"].append(region_id)
            assignments["cam_region_consensus"].append(consensus_id)

            quality = candidate_quality_record(
                image_id=image_id,
                candidate_index=index,
                candidate=candidate,
                gt=gt,
                num_classes=num_classes,
                id_to_name=id_to_name,
                ignore_index=config.ignore_index,
            )
            if quality is None:
                continue
            records.append(
                {
                    **quality,
                    "region_predicted_class_id": region_id,
                    "region_predicted_class_name": id_to_name[region_id],
                    "region_margin": _finite_or_none(region_margins[index]),
                    "cam_predicted_class_id": cam_id,
                    "cam_predicted_class_name": id_to_name[cam_id],
                    "cam_expected_margin": cam["expected_margin"],
                    "cam_region_agree": cam_id == region_id,
                    "assignments": {
                        policy: int(values[-1])
                        for policy, values in assignments.items()
                    },
                }
            )

        for policy, values in assignments.items():
            prediction = fuse_candidate_assignments(
                image_shape=tuple(candidate_metadata["image_shape"]),
                candidates=candidates,
                class_assignments=values,
                ignore_index=config.ignore_index,
                uncovered_label=config.uncovered_label,
                conflict_margin=config.conflict_margin,
            )
            accumulate_prediction(
                states[policy], prediction, gt, config.ignore_index, num_classes
            )
        evaluated_images += 1

    policy_metrics = {
        policy: compute_evaluation_metrics(
            state["confusion"],
            state["gt_pixel_count"],
            state["labeled_gt_pixel_count"],
            config,
        )
        for policy, state in states.items()
    }
    report = {
        "protocol": {
            "purpose": "offline region-semantic diagnostic",
            "pixel_gt_used_for_scoring_or_assignment": False,
            "pixel_gt_used_for_evaluation_only": True,
            "region_classes_restricted_to_image_level_positives": True,
            "region_view": "mean of context and mask-emphasized CLIP features",
            "region_semantics": (
                "mean normalized embedding of Manual4 prompts per class"
                if visual_prototypes is None
                else "frozen robust CAM-consistent visual prototypes"
            ),
            "cam_method": cam_method,
            "consensus_policy": (
                "relabel only when CAM and region model predict the same active "
                "class; otherwise keep the SAM3 prompted class"
            ),
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_dir.resolve()),
            "region_score_dir": str(region_score_dir.resolve()),
            "cam_dir": str(cam_dir.resolve()),
            "prototype_calibration": (
                None
                if prototype_calibration is None
                else str(Path(prototype_calibration).resolve())
            ),
        },
        "input_images": len(image_ids),
        "evaluated_images": evaluated_images,
        "candidate_records": len(records),
        "skipped_images": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "cam_region_agreement_rate": _rate(
            sum(record["cam_region_agree"] for record in records), len(records)
        ),
        "metrics": policy_metrics,
        "candidate_assignment_audit": {
            policy: summarize_assignment_records(records, policy, id_to_name)
            for policy in POLICIES
        },
        "region_margin_percentiles": _percentiles(
            [record["region_margin"] for record in records]
        ),
    }
    return report, records


def _assigned_pixel_purity(
    records_and_targets: list[tuple[dict, int]],
    id_to_name: dict[int, str],
) -> float | None:
    valid = sum(int(record["valid_pixels"]) for record, _target in records_and_targets)
    if valid == 0:
        return None
    correct = sum(
        int(record["gt_distribution"].get(id_to_name[target], 0))
        for record, target in records_and_targets
    )
    return correct / valid


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _percentiles(values: list[float | None]) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    clean = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if clean.size == 0:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(clean, levels))
    }


def main() -> None:
    args = parse_args()
    report, records = analyze_candidate_region_semantics(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        region_score_dir=args.region_score_dir,
        cam_dir=args.cam_dir,
        prototype_calibration=args.prototype_calibration,
        cam_method=args.cam_method,
        limit=args.limit,
    )
    output_path = Path(args.output)
    records_path = (
        Path(args.records_output)
        if args.records_output is not None
        else output_path.with_name(f"{output_path.stem}_records.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Per-candidate records: {records_path}")
    skipped = sum(report["skipped_images"].values())
    if args.require_all and skipped:
        raise RuntimeError(
            f"Could not analyze {skipped}/{report['input_images']} requested images"
        )


if __name__ == "__main__":
    main()
