from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .potsdam import read_image_level_csv


CALIBRATION_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate class-adaptive CAM margins for SAM3 candidate "
            "Keep/Relabel/Ignore decisions without pixel GT."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cam-method", choices=("mean", "top20"), default="mean")
    parser.add_argument("--min-class-samples", type=int, default=20)
    parser.add_argument("--min-separation", type=float, default=1.0)
    parser.add_argument("--fallback-quantile", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def fit_two_component_gmm(
    values: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
    variance_floor: float = 1e-6,
) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or float(np.ptp(values)) <= 1e-8:
        return None

    means = np.asarray(np.quantile(values, [0.25, 0.75]), dtype=np.float64)
    variance = max(float(np.var(values)), variance_floor)
    variances = np.asarray([variance, variance], dtype=np.float64)
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    previous_log_likelihood = -np.inf

    for iteration in range(max_iterations):
        log_probabilities = np.stack(
            [
                np.log(max(weights[index], 1e-12))
                + _normal_log_pdf(values, means[index], variances[index])
                for index in range(2)
            ],
            axis=1,
        )
        row_max = np.max(log_probabilities, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities - row_max)
        normalizer = np.sum(probabilities, axis=1, keepdims=True)
        responsibilities = probabilities / np.maximum(normalizer, 1e-12)
        log_likelihood = float(
            np.sum(row_max[:, 0] + np.log(np.maximum(normalizer[:, 0], 1e-12)))
        )

        component_mass = np.sum(responsibilities, axis=0)
        weights = component_mass / values.size
        means = np.sum(responsibilities * values[:, None], axis=0) / np.maximum(
            component_mass,
            1e-12,
        )
        centered = values[:, None] - means[None, :]
        variances = np.sum(responsibilities * centered**2, axis=0) / np.maximum(
            component_mass,
            1e-12,
        )
        variances = np.maximum(variances, variance_floor)

        if abs(log_likelihood - previous_log_likelihood) <= tolerance:
            break
        previous_log_likelihood = log_likelihood

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    pooled_std = float(np.sqrt(0.5 * (variances[0] + variances[1])))
    separation = float((means[1] - means[0]) / max(pooled_std, 1e-12))
    threshold = _posterior_boundary(weights, means, variances)
    return {
        "samples": int(values.size),
        "iterations": iteration + 1,
        "weights": weights.tolist(),
        "means": means.tolist(),
        "variances": variances.tolist(),
        "separation": separation,
        "posterior_boundary": threshold,
        "log_likelihood": log_likelihood,
    }


def choose_margin_calibration(
    values: list[float],
    min_samples: int,
    min_separation: float,
    fallback_quantile: float,
    fallback: dict | None = None,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    fit = fit_two_component_gmm(array)
    fit_is_usable = (
        array.size >= min_samples
        and fit is not None
        and fit["separation"] >= min_separation
        and min(fit["weights"]) >= 0.05
    )
    if fit_is_usable:
        threshold = float(fit["posterior_boundary"])
        source = "class_gmm"
    elif fallback is not None:
        threshold = float(fallback["threshold"])
        source = f"global_{fallback['threshold_source']}"
    elif array.size:
        threshold = float(np.quantile(array, fallback_quantile))
        source = "quantile"
    else:
        threshold = 1.0
        source = "no_disagreement_relabel_disabled"

    return {
        "samples": int(array.size),
        "threshold": threshold,
        "threshold_source": source,
        "accepted_samples": int(np.count_nonzero(array >= threshold)),
        "accepted_fraction": (
            None if array.size == 0 else float(np.mean(array >= threshold))
        ),
        "margin_percentiles": _percentiles(array),
        "gmm": fit,
    }


def calibrate_candidate_reconciliation(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    cam_dir: str | Path,
    cam_method: str = "mean",
    min_class_samples: int = 20,
    min_separation: float = 1.0,
    fallback_quantile: float = 0.75,
    limit: int | None = None,
) -> dict:
    if min_class_samples < 2:
        raise ValueError("min_class_samples must be at least 2")
    if min_separation < 0:
        raise ValueError("min_separation must be non-negative")
    if not 0 < fallback_quantile < 1:
        raise ValueError("fallback_quantile must be in (0, 1)")

    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    candidate_root = Path(candidate_dir)
    cam_root = Path(cam_dir)
    class_by_name = {spec.name: spec for spec in config.classes}

    margins_by_source: dict[str, list[float]] = {
        spec.name: [] for spec in config.classes
    }
    source_counts: dict[str, Counter] = {
        spec.name: Counter() for spec in config.classes
    }
    target_counts: dict[str, Counter] = {
        spec.name: Counter() for spec in config.classes
    }
    skipped = Counter()
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    evaluated_images = 0

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(image_ids, desc="calibrating-reconciliation"):
        if not candidate_cache_exists(candidate_root, image_id):
            skip("missing_cache", image_id)
            continue
        cam_path = cam_root / f"{image_id}.npz"
        if not cam_path.exists():
            skip("missing_cam", image_id)
            continue

        _metadata, candidates = load_candidate_cache(candidate_root, image_id)
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        id_to_name = {spec.id: spec.name for spec in config.classes}
        for candidate in candidates:
            source_name = str(candidate.class_name)
            source_counts[source_name]["candidates"] += 1
            decision = score_candidate_cams(
                candidate=candidate,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
            )[cam_method]
            if decision["agrees_with_candidate"]:
                source_counts[source_name]["agreements"] += 1
                continue
            margin = _top1_margin(decision)
            if margin is None:
                source_counts[source_name]["disagreement_without_margin"] += 1
                continue
            source_counts[source_name]["disagreements"] += 1
            margins_by_source[source_name].append(margin)
            target_name = id_to_name[int(decision["predicted_class_id"])]
            target_counts[source_name][target_name] += 1
        evaluated_images += 1

    all_margins = [
        margin
        for margins in margins_by_source.values()
        for margin in margins
    ]
    global_calibration = choose_margin_calibration(
        all_margins,
        min_samples=min_class_samples,
        min_separation=min_separation,
        fallback_quantile=fallback_quantile,
    )
    per_source = {}
    for spec in config.classes:
        calibration = choose_margin_calibration(
            margins_by_source[spec.name],
            min_samples=min_class_samples,
            min_separation=min_separation,
            fallback_quantile=fallback_quantile,
            fallback=global_calibration,
        )
        counts = source_counts[spec.name]
        candidates = int(counts["candidates"])
        disagreements = int(counts["disagreements"])
        per_source[spec.name] = {
            **calibration,
            "class_id": spec.id,
            "candidates": candidates,
            "agreements": int(counts["agreements"]),
            "disagreements": disagreements,
            "disagreement_without_margin": int(
                counts["disagreement_without_margin"]
            ),
            "disagreement_rate": _ratio(disagreements, candidates),
            "predicted_target_counts": dict(sorted(target_counts[spec.name].items())),
        }

    return {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "protocol": {
            "purpose": "weak-evidence-only-candidate-reconciliation-calibration",
            "pixel_gt_used": False,
            "feature": "CAM top1 score minus CAM top2 score on disagreement candidates",
            "decision": (
                "CAM agreement keeps source class; disagreement at or above the "
                "source threshold relabels to CAM top1; lower margin is ignored"
            ),
            "sam_score_used_for_semantic_confidence": False,
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_root.resolve()),
            "cam_dir": str(cam_root.resolve()),
            "limit": limit,
        },
        "cam_method": cam_method,
        "hyperparameters": {
            "min_class_samples": min_class_samples,
            "min_separation": min_separation,
            "minimum_component_weight": 0.05,
            "fallback_quantile": fallback_quantile,
        },
        "input_images": len(image_ids),
        "evaluated_images": evaluated_images,
        "skipped_images": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "global": global_calibration,
        "per_source": per_source,
    }


def _normal_log_pdf(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * np.pi * variance)
        + ((values - mean) ** 2) / variance
    )


def _posterior_boundary(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> float:
    low = float(means[0])
    high = float(means[1])
    if high <= low:
        return low
    grid = np.linspace(low, high, 4097, dtype=np.float64)
    log_low = np.log(max(float(weights[0]), 1e-12)) + _normal_log_pdf(
        grid, means[0], variances[0]
    )
    log_high = np.log(max(float(weights[1]), 1e-12)) + _normal_log_pdf(
        grid, means[1], variances[1]
    )
    return float(grid[int(np.argmin(np.abs(log_low - log_high)))])


def _top1_margin(decision: dict) -> float | None:
    second_score = decision.get("second_score")
    if second_score is None:
        return None
    return float(decision["predicted_score"] - second_score)


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    if values.size == 0:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(values, levels))
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> None:
    args = parse_args()
    report = calibrate_candidate_reconciliation(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        cam_dir=args.cam_dir,
        cam_method=args.cam_method,
        min_class_samples=args.min_class_samples,
        min_separation=args.min_separation,
        fallback_quantile=args.fallback_quantile,
        limit=args.limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    skipped_total = sum(report["skipped_images"].values())
    if args.require_all and skipped_total:
        raise RuntimeError(
            f"Could not calibrate {skipped_total}/{report['input_images']} images: "
            f"{report['skipped_images']}"
        )


if __name__ == "__main__":
    main()
