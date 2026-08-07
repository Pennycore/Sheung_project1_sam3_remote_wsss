from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .calibrate_candidate_reconciliation import CALIBRATION_FORMAT_VERSION
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .potsdam import read_image_level_csv
from .rebuild_candidate_pseudo_labels import fuse_candidate_assignments
from .visualize import save_label_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild pseudo labels with frozen CAM-based candidate "
            "Keep/Relabel/Ignore decisions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def reconcile_assignments(
    candidates: list,
    cams: np.ndarray,
    cam_class_ids: np.ndarray,
    active_class_ids: list[int],
    calibration: dict,
) -> tuple[list[int | None], dict[str, dict[str, int]]]:
    cam_method = str(calibration["cam_method"])
    per_source = calibration["per_source"]
    assignments: list[int | None] = []
    counts: dict[str, Counter] = defaultdict(Counter)

    for candidate in candidates:
        source_name = str(candidate.class_name)
        if source_name not in per_source:
            raise ValueError(f"Calibration missing source class: {source_name}")
        decision = score_candidate_cams(
            candidate=candidate,
            cams=cams,
            cam_class_ids=cam_class_ids,
            active_class_ids=active_class_ids,
        )[cam_method]
        bucket = counts[source_name]
        bucket["candidates"] += 1
        if decision["agrees_with_candidate"]:
            assignments.append(int(candidate.class_id))
            bucket["kept"] += 1
            continue

        second_score = decision.get("second_score")
        if second_score is None:
            assignments.append(None)
            bucket["ignored_no_margin"] += 1
            continue
        margin = float(decision["predicted_score"] - second_score)
        threshold = float(per_source[source_name]["threshold"])
        if margin >= threshold:
            assignments.append(int(decision["predicted_class_id"]))
            bucket["relabeled"] += 1
        else:
            assignments.append(None)
            bucket["ignored_low_margin"] += 1

    return assignments, {
        class_name: dict(sorted(class_counts.items()))
        for class_name, class_counts in sorted(counts.items())
    }


def reconcile_candidate_pseudo_labels(
    config_path: str | Path,
    labels_csv: str | Path,
    candidate_dir: str | Path,
    cam_dir: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    image_level = read_image_level_csv(labels_csv)
    image_ids = sorted(image_level)
    if limit is not None:
        image_ids = image_ids[:limit]
    candidate_root = Path(candidate_dir)
    cam_root = Path(cam_dir)
    calibration_path = Path(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _validate_calibration(calibration, config)
    output_root = Path(output_dir)
    _prepare_output(output_root)
    class_by_name = {spec.name: spec for spec in config.classes}

    skipped = Counter()
    skipped_examples: dict[str, list[str]] = defaultdict(list)
    totals = Counter()
    per_source_totals: dict[str, Counter] = defaultdict(Counter)
    summaries = []

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        if len(skipped_examples[reason]) < 5:
            skipped_examples[reason].append(image_id)

    for image_id in tqdm(image_ids, desc="reconciling-candidates"):
        if not candidate_cache_exists(candidate_root, image_id):
            skip("missing_cache", image_id)
            continue
        cam_path = cam_root / f"{image_id}.npz"
        if not cam_path.exists():
            skip("missing_cam", image_id)
            continue

        cache_metadata, candidates = load_candidate_cache(candidate_root, image_id)
        with np.load(cam_path, allow_pickle=False) as data:
            cams = data["cams"].astype(np.float32)
            cam_class_ids = data["class_ids"].astype(np.int64)
        image_shape = tuple(int(value) for value in cache_metadata["image_shape"])
        if tuple(cams.shape[1:]) != image_shape:
            raise ValueError(
                f"CAM shape mismatch for {image_id}: "
                f"cams={cams.shape}, cache={image_shape}"
            )
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        assignments, class_counts = reconcile_assignments(
            candidates=candidates,
            cams=cams,
            cam_class_ids=cam_class_ids,
            active_class_ids=active_class_ids,
            calibration=calibration,
        )
        label = fuse_candidate_assignments(
            image_shape=image_shape,
            candidates=candidates,
            class_assignments=assignments,
            ignore_index=config.ignore_index,
            uncovered_label=config.uncovered_label,
            conflict_margin=config.conflict_margin,
        )
        save_label_png(label, output_root / "pseudo_labels" / f"{image_id}.png")
        ids, pixel_counts = np.unique(label, return_counts=True)

        actions = Counter(
            "ignored" if assignment is None
            else (
                "kept"
                if assignment == int(candidate.class_id)
                else "relabeled"
            )
            for candidate, assignment in zip(candidates, assignments)
        )
        totals.update(actions)
        totals["candidates"] += len(candidates)
        for class_name, counts in class_counts.items():
            per_source_totals[class_name].update(counts)
        item_summary = {
            "image_id": image_id,
            "positive_classes": sorted(image_level[image_id]),
            "candidate_count": len(candidates),
            "actions": dict(sorted(actions.items())),
            "per_source_actions": class_counts,
            "pseudo_label_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(ids, pixel_counts)
            },
        }
        metadata_path = output_root / "metadata" / f"{image_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(item_summary, indent=2), encoding="utf-8")
        summaries.append(item_summary)

    report = {
        "protocol": {
            "method": "candidate-reconciliation-v1",
            "states": ["keep", "relabel", "ignore"],
            "pixel_gt_used": False,
            "cam_method": calibration["cam_method"],
            "semantic_confidence": "CAM top1 minus top2 margin",
            "thresholds_frozen_from_calibration": True,
            "sam_score_used_for_semantic_confidence": False,
        },
        "inputs": {
            "config": str(Path(config_path).resolve()),
            "labels_csv": str(Path(labels_csv).resolve()),
            "candidate_dir": str(candidate_root.resolve()),
            "cam_dir": str(cam_root.resolve()),
            "calibration": str(calibration_path.resolve()),
            "calibration_sha256": _sha256(calibration_path),
            "limit": limit,
        },
        "input_images": len(image_ids),
        "processed_images": len(summaries),
        "skipped_images": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "candidate_actions": dict(sorted(totals.items())),
        "per_source_actions": {
            class_name: dict(sorted(counts.items()))
            for class_name, counts in sorted(per_source_totals.items())
        },
        "items": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _validate_calibration(calibration: dict, config) -> None:
    if int(calibration.get("format_version", -1)) != CALIBRATION_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported calibration format: {calibration.get('format_version')}"
        )
    if calibration.get("cam_method") not in {"mean", "top20"}:
        raise ValueError(f"Unsupported CAM method: {calibration.get('cam_method')}")
    if calibration.get("protocol", {}).get("pixel_gt_used") is not False:
        raise ValueError("Calibration must explicitly declare pixel_gt_used=false")
    expected = {spec.name: spec.id for spec in config.classes}
    calibrated = {
        name: int(values["class_id"])
        for name, values in calibration.get("per_source", {}).items()
    }
    if calibrated != expected:
        raise ValueError(
            f"Calibration/config class mismatch: calibration={calibrated}, "
            f"config={expected}"
        )


def _prepare_output(output_dir: Path) -> None:
    artifacts = [
        output_dir / "pseudo_labels",
        output_dir / "metadata",
        output_dir / "summary.json",
    ]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "Reconciliation output already contains artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Use a new --output-dir."
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
    report = reconcile_candidate_pseudo_labels(
        config_path=args.config,
        labels_csv=args.labels_csv,
        candidate_dir=args.candidate_dir,
        cam_dir=args.cam_dir,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))
    skipped_total = sum(report["skipped_images"].values())
    if args.require_all and skipped_total:
        raise RuntimeError(
            f"Could not reconcile {skipped_total}/{report['input_images']} images: "
            f"{report['skipped_images']}"
        )


if __name__ == "__main__":
    main()
