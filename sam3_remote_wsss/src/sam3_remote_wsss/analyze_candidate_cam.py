from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_quality import candidate_quality_record
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .potsdam import (
    discover_potsdam_items,
    label_rgb_to_ids,
    read_image_level_csv,
    read_label_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CAM semantic consistency on cached SAM3 candidates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL path for per-candidate CAM records.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested image lacks a cache, CAM, item, or GT label.",
    )
    return parser.parse_args()


def score_candidate_cams(
    candidate,
    cams: np.ndarray,
    cam_class_ids: np.ndarray,
    active_class_ids: list[int],
    top_fraction: float = 0.2,
) -> dict:
    if cams.ndim != 3:
        raise ValueError(f"CAM array must be CxHxW, got {cams.shape}")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    class_to_channel = {
        int(class_id): index for index, class_id in enumerate(cam_class_ids)
    }
    missing = [class_id for class_id in active_class_ids if class_id not in class_to_channel]
    if missing:
        raise ValueError(f"Active classes missing from CAM file: {missing}")

    mask = np.asarray(candidate.mask, dtype=bool)
    height, width = mask.shape
    region = cams[
        :,
        candidate.y0 : candidate.y0 + height,
        candidate.x0 : candidate.x0 + width,
    ]
    if region.shape[1:] != mask.shape:
        raise ValueError(
            "Candidate bounds exceed CAM shape: "
            f"origin=({candidate.x0}, {candidate.y0}), "
            f"mask={mask.shape}, cams={cams.shape}"
        )
    if not mask.any():
        raise ValueError("Cannot score an empty candidate mask")

    channels = np.asarray(
        [class_to_channel[class_id] for class_id in active_class_ids],
        dtype=np.int64,
    )
    values = region[channels][:, mask].astype(np.float32, copy=False)
    mean_scores = values.mean(axis=1)
    top_count = max(1, int(np.ceil(values.shape[1] * top_fraction)))
    top_scores = np.partition(values, values.shape[1] - top_count, axis=1)[
        :, -top_count:
    ].mean(axis=1)

    return {
        "mean": _cam_decision(candidate.class_id, active_class_ids, mean_scores),
        "top20": _cam_decision(candidate.class_id, active_class_ids, top_scores),
    }


def _cam_decision(
    expected_class_id: int,
    active_class_ids: list[int],
    scores: np.ndarray,
) -> dict:
    best_index = int(np.argmax(scores))
    predicted_class_id = int(active_class_ids[best_index])
    expected_index = active_class_ids.index(int(expected_class_id))
    expected_score = float(scores[expected_index])
    if len(scores) == 1:
        expected_margin = None
        second_score = None
    else:
        other_scores = np.delete(scores, expected_index)
        expected_margin = float(expected_score - np.max(other_scores))
        second_score = float(np.partition(scores, -2)[-2])
    return {
        "predicted_class_id": predicted_class_id,
        "predicted_score": float(scores[best_index]),
        "second_score": second_score,
        "expected_score": expected_score,
        "expected_margin": expected_margin,
        "agrees_with_candidate": predicted_class_id == int(expected_class_id),
        "scores": {
            str(class_id): float(score)
            for class_id, score in zip(active_class_ids, scores)
        },
    }


def summarize_cam_records(records: list[dict], method: str) -> dict:
    if not records:
        return {
            "candidates": 0,
            "agreement_rate": None,
            "dominant_gt_accuracy": None,
            "foreground_dominant_gt_accuracy": None,
            "wrong_candidate_correction_rate": None,
            "agreement_filter": _empty_filter_summary(),
            "expected_margin_percentiles": _percentiles([]),
        }

    decisions = [record["cam"][method] for record in records]
    kept = [
        record
        for record, decision in zip(records, decisions)
        if decision["agrees_with_candidate"]
    ]
    correct = [record for record in records if record["expected_is_dominant"]]
    wrong = [record for record in records if not record["expected_is_dominant"]]
    wrong_foreground = [record for record in wrong if record["dominant_class_id"] != 0]
    foreground_dominant = [
        record for record in records if record["dominant_class_id"] != 0
    ]

    def predicted_matches(record: dict) -> bool:
        return (
            record["cam"][method]["predicted_class_id"]
            == record["dominant_class_id"]
        )

    margins = [
        decision["expected_margin"]
        for decision in decisions
        if decision["expected_margin"] is not None
    ]
    return {
        "candidates": len(records),
        "agreement_rate": len(kept) / len(records),
        "dominant_gt_accuracy": sum(map(predicted_matches, records)) / len(records),
        "foreground_dominant_gt_accuracy": _rate(
            sum(map(predicted_matches, foreground_dominant)),
            len(foreground_dominant),
        ),
        "wrong_candidate_correction_rate": _rate(
            sum(map(predicted_matches, wrong)),
            len(wrong),
        ),
        "wrong_foreground_candidate_correction_rate": _rate(
            sum(map(predicted_matches, wrong_foreground)),
            len(wrong_foreground),
        ),
        "agreement_filter": {
            "kept_candidates": len(kept),
            "kept_fraction": len(kept) / len(records),
            "dominant_match_rate_before": len(correct) / len(records),
            "dominant_match_rate_after": _rate(
                sum(record["expected_is_dominant"] for record in kept),
                len(kept),
            ),
            "mean_purity_after": _mean(
                [record["expected_purity"] for record in kept]
            ),
            "pixel_weighted_purity_after": _weighted_purity(kept),
            "correct_candidate_recall": _rate(
                sum(record["expected_is_dominant"] for record in kept),
                len(correct),
            ),
            "wrong_candidate_rejection_rate": _rate(
                sum(
                    not record["cam"][method]["agrees_with_candidate"]
                    for record in wrong
                ),
                len(wrong),
            ),
        },
        "expected_margin_percentiles": _percentiles(margins),
    }


def _empty_filter_summary() -> dict:
    return {
        "kept_candidates": 0,
        "kept_fraction": None,
        "dominant_match_rate_before": None,
        "dominant_match_rate_after": None,
        "mean_purity_after": None,
        "pixel_weighted_purity_after": None,
        "correct_candidate_recall": None,
        "wrong_candidate_rejection_rate": None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _weighted_purity(records: list[dict]) -> float | None:
    valid_pixels = sum(int(record["valid_pixels"]) for record in records)
    if valid_pixels == 0:
        return None
    expected_pixels = sum(int(record["expected_pixels"]) for record in records)
    return expected_pixels / valid_pixels


def _percentiles(values: list[float]) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    if not values:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(values, levels))
    }


def analyze_candidate_cams(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    cam_dir: str | Path,
) -> tuple[dict, list[dict]]:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    class_by_name = {spec.name: spec for spec in config.classes}
    id_to_name = {0: "background", **{spec.id: spec.name for spec in config.classes}}
    num_classes = max(id_to_name) + 1
    cam_dir = Path(cam_dir)

    records: list[dict] = []
    skipped_images: dict[str, int] = defaultdict(int)
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    candidates_without_valid_gt = 0
    evaluated_images = 0

    def skip(reason: str, image_id: str) -> None:
        skipped_images[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(sorted(image_level), desc="candidate-CAM"):
        if not candidate_cache_exists(candidate_dir, image_id):
            skip("missing_cache", image_id)
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

        metadata, candidates = load_candidate_cache(candidate_dir, image_id)
        gt = label_rgb_to_ids(
            read_label_rgb(item.label_path),
            config.classes,
            config.ignore_index,
            background_colors=config.background_colors,
        )
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        if list(gt.shape) != list(metadata["image_shape"]):
            raise ValueError(
                f"Image shape mismatch for {image_id}: "
                f"cache={metadata['image_shape']}, gt={list(gt.shape)}"
            )
        if tuple(cams.shape[1:]) != tuple(gt.shape):
            raise ValueError(
                f"CAM shape mismatch for {image_id}: cams={cams.shape}, gt={gt.shape}"
            )

        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        evaluated_images += 1
        for candidate_index, candidate in enumerate(candidates):
            quality = candidate_quality_record(
                image_id=image_id,
                candidate_index=candidate_index,
                candidate=candidate,
                gt=gt,
                num_classes=num_classes,
                id_to_name=id_to_name,
                ignore_index=config.ignore_index,
            )
            if quality is None:
                candidates_without_valid_gt += 1
                continue
            decisions = score_candidate_cams(
                candidate=candidate,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
            )
            for decision in decisions.values():
                decision["predicted_class_name"] = id_to_name[
                    decision["predicted_class_id"]
                ]
            quality["active_class_count"] = len(active_class_ids)
            quality["active_class_names"] = [id_to_name[value] for value in active_class_ids]
            quality["cam"] = decisions
            records.append(quality)

    records_by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_class[record["class_name"]].append(record)
    multi_class_records = [
        record for record in records if record["active_class_count"] > 1
    ]

    summary = {
        "protocol": {
            "purpose": "offline-diagnostic-only",
            "gt_used_for_generation_or_training": False,
            "cam_classes_restricted_to_image_level_positive_labels": True,
            "mean": "mean CAM response inside each candidate mask",
            "top20": "mean of the strongest 20 percent CAM pixels inside the mask",
            "agreement_filter": "keep only when CAM top-1 equals the SAM3 prompted class",
            "candidate_pixel_counts_include_overlap": True,
        },
        "input_images": len(image_level),
        "evaluated_images": evaluated_images,
        "candidate_records": len(records),
        "multi_positive_candidate_records": len(multi_class_records),
        "candidates_without_valid_gt": candidates_without_valid_gt,
        "skipped_images": dict(sorted(skipped_images.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "methods": {
            method: {
                "overall": summarize_cam_records(records, method),
                "multi_positive_only": summarize_cam_records(
                    multi_class_records, method
                ),
                "per_class": {
                    spec.name: summarize_cam_records(
                        records_by_class[spec.name], method
                    )
                    for spec in config.classes
                },
            }
            for method in ("mean", "top20")
        },
    }
    return summary, records


def main() -> None:
    args = parse_args()
    summary, records = analyze_candidate_cams(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        cam_dir=args.cam_dir,
    )
    skipped_total = sum(summary["skipped_images"].values())
    output_path = Path(args.output)
    records_path = (
        Path(args.records_output)
        if args.records_output is not None
        else output_path.with_name(f"{output_path.stem}_records.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Per-candidate records: {records_path}")
    if args.require_all and skipped_total:
        raise RuntimeError(
            f"Could not analyze {skipped_total}/{summary['input_images']} images: "
            f"{summary['skipped_images']}"
        )


if __name__ == "__main__":
    main()
