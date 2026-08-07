from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .analyze_candidate_quality import candidate_quality_record
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .evaluate_pseudo_labels import compute_evaluation_metrics
from .potsdam import (
    discover_potsdam_items,
    label_rgb_to_ids,
    read_image_level_csv,
    read_label_rgb,
)
from .rebuild_candidate_pseudo_labels import fuse_candidate_assignments


POLICIES = ("baseline", "oracle_reject", "oracle_relabel")
COMPARISON_METRICS = (
    "miou",
    "mf1",
    "oa",
    "foreground_miou",
    "foreground_mf1",
    "labeled_miou",
    "labeled_mf1",
    "labeled_oa",
    "labeled_coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether semantically wrong SAM3 candidates retain useful "
            "geometry and can be recovered by relabeling."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--cam-dir",
        default=None,
        help="Optional CAM NPZ directory for confusion-pair correction analysis.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested image lacks cache, CAM, item, or GT.",
    )
    return parser.parse_args()


def oracle_assignments(
    candidates: list,
    quality_by_index: dict[int, dict],
    active_class_ids: set[int],
) -> tuple[dict[str, list[int | None]], dict[str, dict[str, int]]]:
    assignments: dict[str, list[int | None]] = {
        "baseline": [int(candidate.class_id) for candidate in candidates],
        "oracle_reject": [],
        "oracle_relabel": [],
    }
    actions = {
        "baseline": Counter(),
        "oracle_reject": Counter(),
        "oracle_relabel": Counter(),
    }

    for index, candidate in enumerate(candidates):
        source_id = int(candidate.class_id)
        quality = quality_by_index.get(index)
        actions["baseline"]["kept"] += 1

        if quality is None:
            assignments["oracle_reject"].append(None)
            assignments["oracle_relabel"].append(None)
            actions["oracle_reject"]["rejected_no_valid_gt"] += 1
            actions["oracle_relabel"]["rejected_no_valid_gt"] += 1
            continue

        dominant_id = int(quality["dominant_class_id"])
        if dominant_id == source_id:
            assignments["oracle_reject"].append(source_id)
            actions["oracle_reject"]["kept"] += 1
        else:
            assignments["oracle_reject"].append(None)
            actions["oracle_reject"]["rejected_semantic_mismatch"] += 1

        if dominant_id == 0:
            assignments["oracle_relabel"].append(None)
            actions["oracle_relabel"]["rejected_background_dominant"] += 1
        elif dominant_id not in active_class_ids:
            assignments["oracle_relabel"].append(None)
            actions["oracle_relabel"]["rejected_nonpositive_target"] += 1
        else:
            assignments["oracle_relabel"].append(dominant_id)
            action = "kept" if dominant_id == source_id else "relabeled"
            actions["oracle_relabel"][action] += 1

    return assignments, {
        policy: dict(sorted(counts.items()))
        for policy, counts in actions.items()
    }


def empty_metric_state(num_classes: int) -> dict[str, np.ndarray]:
    return {
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
        "gt_pixel_count": np.zeros(num_classes, dtype=np.int64),
        "labeled_gt_pixel_count": np.zeros(num_classes, dtype=np.int64),
    }


def accumulate_prediction(
    state: dict[str, np.ndarray],
    prediction: np.ndarray,
    gt: np.ndarray,
    ignore_index: int,
    num_classes: int,
) -> None:
    if prediction.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {prediction.shape} vs {gt.shape}"
        )
    valid_gt = (gt != ignore_index) & (gt < num_classes)
    state["gt_pixel_count"] += np.bincount(
        gt[valid_gt], minlength=num_classes
    )
    labeled = valid_gt & (prediction != ignore_index) & (prediction < num_classes)
    state["labeled_gt_pixel_count"] += np.bincount(
        gt[labeled], minlength=num_classes
    )
    if not np.any(labeled):
        return
    values = (
        num_classes * gt[labeled].astype(np.int64)
        + prediction[labeled].astype(np.int64)
    )
    state["confusion"] += np.bincount(
        values,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def candidate_coverage_counts(
    candidates: list,
    gt: np.ndarray,
    foreground_class_ids: list[int],
) -> dict[int, dict[str, int]]:
    geometric = np.zeros(gt.shape, dtype=bool)
    semantic = {
        class_id: np.zeros(gt.shape, dtype=bool)
        for class_id in foreground_class_ids
    }
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        height, width = mask.shape
        geometric[
            candidate.y0 : candidate.y0 + height,
            candidate.x0 : candidate.x0 + width,
        ] |= mask
        if int(candidate.class_id) in semantic:
            semantic_region = semantic[int(candidate.class_id)][
                candidate.y0 : candidate.y0 + height,
                candidate.x0 : candidate.x0 + width,
            ]
            semantic_region |= mask

    counts = {}
    for class_id in foreground_class_ids:
        target = gt == class_id
        counts[class_id] = {
            "gt_pixels": int(target.sum()),
            "geometric_covered_pixels": int((target & geometric).sum()),
            "semantic_covered_pixels": int((target & semantic[class_id]).sum()),
        }
    return counts


def merge_coverage_counts(
    target: dict[int, dict[str, int]],
    source: dict[int, dict[str, int]],
) -> None:
    for class_id, values in source.items():
        bucket = target.setdefault(
            class_id,
            {
                "gt_pixels": 0,
                "geometric_covered_pixels": 0,
                "semantic_covered_pixels": 0,
            },
        )
        for key, value in values.items():
            bucket[key] += int(value)


def finalize_coverage(
    counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
) -> dict:
    per_class = {}
    geometric_recalls = []
    semantic_recalls = []
    gaps = []
    total_gt = 0
    total_geometric = 0
    total_semantic = 0
    for class_id in sorted(counts):
        values = counts[class_id]
        gt_pixels = int(values["gt_pixels"])
        geometric_pixels = int(values["geometric_covered_pixels"])
        semantic_pixels = int(values["semantic_covered_pixels"])
        geometric_recall = None if gt_pixels == 0 else geometric_pixels / gt_pixels
        semantic_recall = None if gt_pixels == 0 else semantic_pixels / gt_pixels
        gap = (
            None
            if geometric_recall is None or semantic_recall is None
            else geometric_recall - semantic_recall
        )
        per_class[id_to_name[class_id]] = {
            **values,
            "geometric_recall": geometric_recall,
            "semantic_recall": semantic_recall,
            "recoverable_semantic_gap": gap,
        }
        total_gt += gt_pixels
        total_geometric += geometric_pixels
        total_semantic += semantic_pixels
        if gap is not None:
            geometric_recalls.append(geometric_recall)
            semantic_recalls.append(semantic_recall)
            gaps.append(gap)

    return {
        "definition": {
            "geometric_recall": (
                "fraction of each GT foreground class covered by any SAM3 candidate"
            ),
            "semantic_recall": (
                "fraction covered by candidates prompted as that same class"
            ),
            "recoverable_semantic_gap": "geometric_recall - semantic_recall",
        },
        "per_class": per_class,
        "macro": {
            "geometric_recall": _mean(geometric_recalls),
            "semantic_recall": _mean(semantic_recalls),
            "recoverable_semantic_gap": _mean(gaps),
        },
        "micro": {
            "gt_pixels": total_gt,
            "geometric_covered_pixels": total_geometric,
            "semantic_covered_pixels": total_semantic,
            "geometric_recall": _ratio(total_geometric, total_gt),
            "semantic_recall": _ratio(total_semantic, total_gt),
            "recoverable_semantic_gap": _ratio(
                total_geometric - total_semantic,
                total_gt,
            ),
        },
    }


def finalize_candidate_confusion(
    count_confusion: np.ndarray,
    pixel_confusion: np.ndarray,
    source_class_ids: list[int],
    id_to_name: dict[int, str],
) -> dict:
    target_class_ids = sorted(id_to_name)

    def named_rows(matrix: np.ndarray) -> dict:
        return {
            id_to_name[source_id]: {
                id_to_name[target_id]: int(matrix[source_id, target_id])
                for target_id in target_class_ids
            }
            for source_id in source_class_ids
        }

    def normalized_rows(matrix: np.ndarray) -> dict:
        rows = {}
        for source_id in source_class_ids:
            total = int(matrix[source_id].sum())
            rows[id_to_name[source_id]] = {
                id_to_name[target_id]: (
                    None if total == 0 else float(matrix[source_id, target_id] / total)
                )
                for target_id in target_class_ids
            }
        return rows

    return {
        "candidate_count_weighted": {
            "definition": "one vote per candidate to its dominant GT class",
            "counts": named_rows(count_confusion),
            "row_normalized": normalized_rows(count_confusion),
        },
        "pixel_area_weighted": {
            "definition": (
                "all valid candidate-mask pixels accumulated by source and GT class; "
                "overlapping candidates are intentionally counted repeatedly"
            ),
            "counts": named_rows(pixel_confusion),
            "row_normalized": normalized_rows(pixel_confusion),
        },
    }


def summarize_cam_pairs(records: list[dict], id_to_name: dict[int, str]) -> dict:
    methods = {}
    for method in ("mean", "top20"):
        by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for record in records:
            source_id = int(record["class_id"])
            target_id = int(record["dominant_class_id"])
            if target_id != 0 and target_id != source_id:
                by_pair[(source_id, target_id)].append(record)

        pair_results = {}
        for (source_id, target_id), pair_records in sorted(by_pair.items()):
            decisions = [record["cam"][method] for record in pair_records]
            corrected = [
                decision for decision in decisions
                if int(decision["predicted_class_id"]) == target_id
            ]
            uncorrected = [
                decision for decision in decisions
                if int(decision["predicted_class_id"]) != target_id
            ]
            stayed_source = sum(
                int(decision["predicted_class_id"]) == source_id
                for decision in decisions
            )
            pair_name = f"{id_to_name[source_id]}->{id_to_name[target_id]}"
            pair_results[pair_name] = {
                "candidates": len(pair_records),
                "cam_corrected_to_dominant": len(corrected),
                "cam_correction_rate": _ratio(len(corrected), len(pair_records)),
                "cam_stayed_at_source": stayed_source,
                "cam_stayed_at_source_rate": _ratio(
                    stayed_source, len(pair_records)
                ),
                "pixel_weighted_correction_rate": _ratio(
                    sum(
                        int(record["valid_pixels"])
                        for record in pair_records
                        if int(record["cam"][method]["predicted_class_id"])
                        == target_id
                    ),
                    sum(int(record["valid_pixels"]) for record in pair_records),
                ),
                "top1_margin_percentiles": _percentiles(
                    [_top1_margin(decision) for decision in decisions]
                ),
                "corrected_margin_percentiles": _percentiles(
                    [_top1_margin(decision) for decision in corrected]
                ),
                "uncorrected_margin_percentiles": _percentiles(
                    [_top1_margin(decision) for decision in uncorrected]
                ),
            }

        all_wrong = [
            record
            for pair_records in by_pair.values()
            for record in pair_records
        ]
        methods[method] = {
            "foreground_mismatch_candidates": len(all_wrong),
            "cam_corrected_to_dominant": sum(
                int(record["cam"][method]["predicted_class_id"])
                == int(record["dominant_class_id"])
                for record in all_wrong
            ),
            "cam_correction_rate": _ratio(
                sum(
                    int(record["cam"][method]["predicted_class_id"])
                    == int(record["dominant_class_id"])
                    for record in all_wrong
                ),
                len(all_wrong),
            ),
            "per_pair": pair_results,
        }
    return methods


def analyze_candidate_recoverability(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    cam_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    foreground_class_ids = sorted(spec.id for spec in config.classes)
    id_to_name = {
        0: "background",
        **{spec.id: spec.name for spec in config.classes},
    }
    name_to_id = {name: class_id for class_id, name in id_to_name.items()}
    class_by_name = {spec.name: spec for spec in config.classes}
    num_classes = max(id_to_name) + 1
    candidate_dir = Path(candidate_dir)
    cam_root = None if cam_dir is None else Path(cam_dir)

    metric_states = {
        policy: empty_metric_state(num_classes)
        for policy in POLICIES
    }
    coverage_counts: dict[int, dict[str, int]] = {}
    count_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    pixel_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    action_counts = {policy: Counter() for policy in POLICIES}
    cam_records: list[dict] = []
    skipped = Counter()
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    candidates_without_valid_gt = 0
    evaluated_images = 0

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(image_ids, desc="candidate-recoverability"):
        if not candidate_cache_exists(candidate_dir, image_id):
            skip("missing_cache", image_id)
            continue
        item = item_by_id.get(image_id)
        if item is None:
            skip("missing_item", image_id)
            continue
        if item.label_path is None:
            skip("missing_label", image_id)
            continue
        cam_path = None if cam_root is None else cam_root / f"{image_id}.npz"
        if cam_root is not None and (cam_path is None or not cam_path.exists()):
            skip("missing_cam", image_id)
            continue

        metadata, candidates = load_candidate_cache(candidate_dir, image_id)
        gt = label_rgb_to_ids(
            read_label_rgb(item.label_path),
            config.classes,
            config.ignore_index,
            background_colors=config.background_colors,
        )
        if list(gt.shape) != list(metadata["image_shape"]):
            raise ValueError(
                f"Image shape mismatch for {image_id}: "
                f"cache={metadata['image_shape']}, gt={list(gt.shape)}"
            )

        active_class_ids = {
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        }
        quality_by_index = {}
        for index, candidate in enumerate(candidates):
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
                candidates_without_valid_gt += 1
                continue
            quality_by_index[index] = quality
            source_id = int(quality["class_id"])
            dominant_id = int(quality["dominant_class_id"])
            count_confusion[source_id, dominant_id] += 1
            for target_name, pixels in quality["gt_distribution"].items():
                target_id = name_to_id[target_name]
                pixel_confusion[source_id, target_id] += int(pixels)

        assignments, image_actions = oracle_assignments(
            candidates,
            quality_by_index,
            active_class_ids,
        )
        for policy, counts in image_actions.items():
            action_counts[policy].update(counts)
            prediction = fuse_candidate_assignments(
                image_shape=tuple(int(value) for value in metadata["image_shape"]),
                candidates=candidates,
                class_assignments=assignments[policy],
                ignore_index=config.ignore_index,
                uncovered_label=config.uncovered_label,
                conflict_margin=config.conflict_margin,
            )
            accumulate_prediction(
                metric_states[policy],
                prediction,
                gt,
                config.ignore_index,
                num_classes,
            )

        merge_coverage_counts(
            coverage_counts,
            candidate_coverage_counts(candidates, gt, foreground_class_ids),
        )

        if cam_path is not None:
            with np.load(cam_path, allow_pickle=False) as data:
                cams = data["cams"].astype(np.float32)
                cam_class_ids = data["class_ids"].astype(np.int64)
            if tuple(cams.shape[1:]) != tuple(gt.shape):
                raise ValueError(
                    f"CAM shape mismatch for {image_id}: "
                    f"cams={cams.shape}, gt={gt.shape}"
                )
            ordered_active_ids = sorted(active_class_ids)
            for index, quality in quality_by_index.items():
                quality = dict(quality)
                quality["cam"] = score_candidate_cams(
                    candidate=candidates[index],
                    cams=cams,
                    cam_class_ids=cam_class_ids,
                    active_class_ids=ordered_active_ids,
                )
                cam_records.append(quality)
        evaluated_images += 1

    policy_metrics = {
        policy: {
            **compute_evaluation_metrics(
                state["confusion"],
                state["gt_pixel_count"],
                state["labeled_gt_pixel_count"],
                config,
            ),
            "candidate_actions": dict(sorted(action_counts[policy].items())),
        }
        for policy, state in metric_states.items()
    }
    report = {
        "protocol": {
            "purpose": "offline-oracle-diagnostic-only",
            "gt_used_for_generation_or_training": False,
            "oracle_relabel_targets": (
                "dominant foreground GT class only when it is an image-level "
                "positive class; geometry and SAM score remain unchanged"
            ),
            "oracle_reject": (
                "keep only candidates whose source class equals dominant GT class"
            ),
            "cam_role": (
                None
                if cam_root is None
                else "diagnostic correction analysis; no pseudo label is changed"
            ),
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_dir.resolve()),
            "cam_dir": None if cam_root is None else str(cam_root.resolve()),
            "limit": limit,
        },
        "input_images": len(image_ids),
        "evaluated_images": evaluated_images,
        "candidates": int(count_confusion.sum()),
        "candidates_without_valid_gt": candidates_without_valid_gt,
        "skipped_images": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "policies": policy_metrics,
        "policy_comparisons": {
            "oracle_reject_minus_baseline": metric_deltas(
                policy_metrics["oracle_reject"],
                policy_metrics["baseline"],
            ),
            "oracle_relabel_minus_baseline": metric_deltas(
                policy_metrics["oracle_relabel"],
                policy_metrics["baseline"],
            ),
            "oracle_relabel_minus_oracle_reject": metric_deltas(
                policy_metrics["oracle_relabel"],
                policy_metrics["oracle_reject"],
            ),
        },
        "candidate_coverage": finalize_coverage(coverage_counts, id_to_name),
        "sam_to_gt_confusion": finalize_candidate_confusion(
            count_confusion,
            pixel_confusion,
            foreground_class_ids,
            id_to_name,
        ),
        "cam_confusion_pair_recoverability": (
            None if cam_root is None else summarize_cam_pairs(cam_records, id_to_name)
        ),
    }
    return report


def metric_deltas(current: dict, reference: dict) -> dict[str, float | None]:
    deltas = {}
    for metric in COMPARISON_METRICS:
        current_value = current.get(metric)
        reference_value = reference.get(metric)
        deltas[metric] = (
            None
            if current_value is None or reference_value is None
            else float(current_value - reference_value)
        )
    return deltas


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _top1_margin(decision: dict) -> float | None:
    second_score = decision.get("second_score")
    if second_score is None:
        return None
    return float(decision["predicted_score"] - second_score)


def _percentiles(values: list[float | None]) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    clean = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(clean, levels))
    }


def main() -> None:
    args = parse_args()
    report = analyze_candidate_recoverability(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        cam_dir=args.cam_dir,
        limit=args.limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    skipped_total = sum(report["skipped_images"].values())
    if args.require_all and skipped_total:
        raise RuntimeError(
            f"Could not analyze {skipped_total}/{report['input_images']} images: "
            f"{report['skipped_images']}"
        )


if __name__ == "__main__":
    main()
