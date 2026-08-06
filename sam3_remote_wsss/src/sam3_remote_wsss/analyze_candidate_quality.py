from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .potsdam import (
    discover_potsdam_items,
    label_rgb_to_ids,
    read_image_level_csv,
    read_label_rgb,
)
from .prompts import prompts_for_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SAM3 candidate semantic purity and prompt agreement."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--records-output",
        default=None,
        help="Optional JSONL path for per-candidate records.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any image in labels-csv lacks a cache, item, or GT label.",
    )
    return parser.parse_args()


def candidate_quality_record(
    image_id: str,
    candidate_index: int,
    candidate,
    gt: np.ndarray,
    num_classes: int,
    id_to_name: dict[int, str],
    ignore_index: int,
) -> dict | None:
    mask = np.asarray(candidate.mask, dtype=bool)
    height, width = mask.shape
    gt_region = gt[
        candidate.y0 : candidate.y0 + height,
        candidate.x0 : candidate.x0 + width,
    ]
    if gt_region.shape != mask.shape:
        raise ValueError(
            f"Candidate {candidate_index} for {image_id} exceeds GT bounds: "
            f"mask={mask.shape}, gt_region={gt_region.shape}"
        )

    valid = mask & (gt_region != ignore_index) & (gt_region < num_classes)
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        return None

    counts = np.bincount(gt_region[valid], minlength=num_classes).astype(np.int64)
    expected_pixels = int(counts[candidate.class_id])
    dominant_class_id = int(np.argmax(counts))
    background_pixels = int(counts[0])
    other_foreground_pixels = valid_pixels - expected_pixels - background_pixels
    purity = expected_pixels / valid_pixels
    dominant_fraction = float(counts[dominant_class_id] / valid_pixels)
    distribution = {
        id_to_name[class_id]: int(count)
        for class_id, count in enumerate(counts)
        if count > 0 and class_id in id_to_name
    }
    return {
        "image_id": image_id,
        "candidate_index": candidate_index,
        "class_id": int(candidate.class_id),
        "class_name": str(candidate.class_name),
        "prompt": str(candidate.prompt),
        "sam_score": float(candidate.score),
        "area": int(mask.sum()),
        "valid_pixels": valid_pixels,
        "expected_pixels": expected_pixels,
        "expected_purity": purity,
        "background_pixels": background_pixels,
        "other_foreground_pixels": other_foreground_pixels,
        "dominant_class_id": dominant_class_id,
        "dominant_class_name": id_to_name[dominant_class_id],
        "dominant_fraction": dominant_fraction,
        "expected_is_dominant": dominant_class_id == candidate.class_id,
        "gt_distribution": distribution,
    }


def summarize_quality_records(records: list[dict]) -> dict:
    if not records:
        return {
            "candidates": 0,
            "candidate_purity": _percentiles(np.empty((0,), dtype=np.float64)),
            "mean_candidate_purity": None,
            "pixel_weighted_purity": None,
            "dominant_match_rate": None,
            "purity_at_least": {"0.50": None, "0.75": None, "0.90": None},
            "background_contamination": None,
            "other_foreground_contamination": None,
            "score_purity_correlation": None,
        }

    purities = np.asarray([record["expected_purity"] for record in records], dtype=np.float64)
    scores = np.asarray([record["sam_score"] for record in records], dtype=np.float64)
    valid_pixels = sum(int(record["valid_pixels"]) for record in records)
    expected_pixels = sum(int(record["expected_pixels"]) for record in records)
    background_pixels = sum(int(record["background_pixels"]) for record in records)
    other_foreground_pixels = sum(
        int(record["other_foreground_pixels"]) for record in records
    )
    dominant_matches = sum(bool(record["expected_is_dominant"]) for record in records)
    correlation = None
    if len(records) > 1 and np.std(scores) > 0 and np.std(purities) > 0:
        correlation = float(np.corrcoef(scores, purities)[0, 1])
    return {
        "candidates": len(records),
        "candidate_purity": _percentiles(purities),
        "mean_candidate_purity": float(np.mean(purities)),
        "pixel_weighted_purity": expected_pixels / valid_pixels,
        "dominant_match_rate": dominant_matches / len(records),
        "purity_at_least": {
            "0.50": float(np.mean(purities >= 0.50)),
            "0.75": float(np.mean(purities >= 0.75)),
            "0.90": float(np.mean(purities >= 0.90)),
        },
        "background_contamination": background_pixels / valid_pixels,
        "other_foreground_contamination": other_foreground_pixels / valid_pixels,
        "score_purity_correlation": correlation,
    }


def prompt_support_counts(
    support: np.ndarray,
    gt: np.ndarray,
    class_id: int,
    prompt_count: int,
    ignore_index: int,
) -> dict:
    valid = (gt != ignore_index)
    expected = valid & (gt == class_id)
    counts = {
        "positive_images": 1,
        "gt_pixels": int(expected.sum()),
        "thresholds": {},
        "exact_support": {},
    }
    for threshold in range(1, prompt_count + 1):
        predicted = valid & (support >= threshold)
        counts["thresholds"][str(threshold)] = {
            "true_positive": int((predicted & expected).sum()),
            "false_positive": int((predicted & ~expected).sum()),
            "predicted_pixels": int(predicted.sum()),
        }
    for level in range(prompt_count + 1):
        selected = valid & (support == level)
        counts["exact_support"][str(level)] = {
            "pixels": int(selected.sum()),
            "correct_pixels": int((selected & expected).sum()),
        }
    return counts


def merge_support_counts(target: dict, source: dict) -> None:
    target["positive_images"] += int(source["positive_images"])
    target["gt_pixels"] += int(source["gt_pixels"])
    for threshold, values in source["thresholds"].items():
        bucket = target["thresholds"].setdefault(
            threshold,
            {"true_positive": 0, "false_positive": 0, "predicted_pixels": 0},
        )
        for key, value in values.items():
            bucket[key] += int(value)
    for level, values in source["exact_support"].items():
        bucket = target["exact_support"].setdefault(
            level,
            {"pixels": 0, "correct_pixels": 0},
        )
        for key, value in values.items():
            bucket[key] += int(value)


def finalize_support_counts(counts: dict) -> dict:
    gt_pixels = int(counts["gt_pixels"])
    thresholds = {}
    for threshold, values in sorted(
        counts["thresholds"].items(), key=lambda item: int(item[0])
    ):
        tp = int(values["true_positive"])
        fp = int(values["false_positive"])
        predicted = int(values["predicted_pixels"])
        fn = gt_pixels - tp
        union = tp + fp + fn
        thresholds[threshold] = {
            **values,
            "false_negative": fn,
            "precision": None if predicted == 0 else tp / predicted,
            "recall": None if gt_pixels == 0 else tp / gt_pixels,
            "binary_iou": None if union == 0 else tp / union,
        }

    exact_support = {}
    for level, values in sorted(
        counts["exact_support"].items(), key=lambda item: int(item[0])
    ):
        pixels = int(values["pixels"])
        correct = int(values["correct_pixels"])
        exact_support[level] = {
            **values,
            "precision": None if pixels == 0 else correct / pixels,
            "gt_fraction": None if gt_pixels == 0 else correct / gt_pixels,
        }
    return {
        "positive_images": int(counts["positive_images"]),
        "gt_pixels": gt_pixels,
        "thresholds": thresholds,
        "exact_support": exact_support,
    }


def _empty_support_counts() -> dict:
    return {
        "positive_images": 0,
        "gt_pixels": 0,
        "thresholds": {},
        "exact_support": {},
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    if values.size == 0:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(values, levels))
    }


def analyze_candidate_quality(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
) -> tuple[dict, list[dict]]:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    class_by_name = {spec.name: spec for spec in config.classes}
    id_to_name = {0: "background", **{spec.id: spec.name for spec in config.classes}}
    num_classes = max(id_to_name) + 1

    records: list[dict] = []
    skipped_candidates_no_valid_gt = 0
    skipped_images = defaultdict(int)
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    support_by_class = {
        spec.name: _empty_support_counts()
        for spec in config.classes
    }
    support_overall = _empty_support_counts()
    evaluated_images = 0

    def skip(reason: str, image_id: str) -> None:
        skipped_images[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(sorted(image_level), desc="candidate-quality"):
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
        evaluated_images += 1

        for candidate_index, candidate in enumerate(candidates):
            record = candidate_quality_record(
                image_id=image_id,
                candidate_index=candidate_index,
                candidate=candidate,
                gt=gt,
                num_classes=num_classes,
                id_to_name=id_to_name,
                ignore_index=config.ignore_index,
            )
            if record is None:
                skipped_candidates_no_valid_gt += 1
            else:
                records.append(record)

        candidates_by_class_prompt: dict[tuple[int, str], list] = defaultdict(list)
        for candidate in candidates:
            candidates_by_class_prompt[(candidate.class_id, candidate.prompt)].append(candidate)

        for class_name in sorted(image_level[image_id]):
            spec = class_by_name.get(class_name)
            if spec is None:
                continue
            prompts = prompts_for_class(spec, config.prompting)
            support = np.zeros(gt.shape, dtype=np.uint8)
            for prompt in prompts:
                prompt_union = np.zeros(gt.shape, dtype=bool)
                for candidate in candidates_by_class_prompt.get((spec.id, prompt), []):
                    height, width = candidate.mask.shape
                    region = prompt_union[
                        candidate.y0 : candidate.y0 + height,
                        candidate.x0 : candidate.x0 + width,
                    ]
                    region |= np.asarray(candidate.mask, dtype=bool)
                support += prompt_union
            image_counts = prompt_support_counts(
                support=support,
                gt=gt,
                class_id=spec.id,
                prompt_count=len(prompts),
                ignore_index=config.ignore_index,
            )
            merge_support_counts(support_by_class[spec.name], image_counts)
            merge_support_counts(support_overall, image_counts)

    records_by_class: dict[str, list[dict]] = defaultdict(list)
    records_by_prompt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    dominant_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pixel_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        records_by_class[record["class_name"]].append(record)
        records_by_prompt[(record["class_name"], record["prompt"])].append(record)
        dominant_confusion[record["class_name"]][record["dominant_class_name"]] += 1
        for gt_name, count in record["gt_distribution"].items():
            pixel_confusion[record["class_name"]][gt_name] += int(count)

    summary = {
        "protocol": {
            "purpose": "offline-diagnostic-only",
            "gt_used_for_generation_or_training": False,
            "candidate_purity": "fraction of valid mask pixels matching the prompted class",
            "prompt_support": "number of distinct same-class prompts covering each pixel",
            "candidate_pixel_counts_include_overlap": True,
        },
        "input_images": len(image_level),
        "evaluated_images": evaluated_images,
        "candidate_records": len(records),
        "candidates_without_valid_gt": skipped_candidates_no_valid_gt,
        "skipped_images": dict(sorted(skipped_images.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "candidate_quality": {
            "overall": summarize_quality_records(records),
            "per_class": {
                spec.name: summarize_quality_records(records_by_class[spec.name])
                for spec in config.classes
            },
            "per_prompt": {
                spec.name: {
                    prompt: summarize_quality_records(
                        records_by_prompt[(spec.name, prompt)]
                    )
                    for prompt in prompts_for_class(spec, config.prompting)
                }
                for spec in config.classes
            },
            "candidate_dominant_confusion": {
                spec.name: {
                    gt_name: int(dominant_confusion[spec.name].get(gt_name, 0))
                    for gt_name in id_to_name.values()
                }
                for spec in config.classes
            },
            "candidate_pixel_confusion": {
                spec.name: {
                    gt_name: int(pixel_confusion[spec.name].get(gt_name, 0))
                    for gt_name in id_to_name.values()
                }
                for spec in config.classes
            },
        },
        "prompt_support": {
            "overall": finalize_support_counts(support_overall),
            "per_class": {
                class_name: finalize_support_counts(counts)
                for class_name, counts in support_by_class.items()
            },
        },
    }
    return summary, records


def main() -> None:
    args = parse_args()
    summary, records = analyze_candidate_quality(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
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
