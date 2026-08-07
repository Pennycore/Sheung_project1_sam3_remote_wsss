from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .analyze_candidate_cam import score_candidate_cams
from .candidate_cache import candidate_cache_exists, load_candidate_cache
from .config import load_config
from .fusion import FusionCanvas
from .potsdam import read_image_level_csv
from .visualize import save_label_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild pseudo labels from cached SAM3 candidates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cam-dir",
        default=None,
        help="CAM NPZ directory; required when --reject-classes is non-empty.",
    )
    parser.add_argument(
        "--cam-method",
        choices=("mean", "top20"),
        default="mean",
    )
    parser.add_argument(
        "--reject-classes",
        default="",
        help="Comma-separated classes whose candidates require CAM agreement.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested image lacks a candidate cache or CAM file.",
    )
    return parser.parse_args()


def fuse_cached_candidates(
    image_shape: tuple[int, int],
    candidates: list,
    keep: list[bool],
    ignore_index: int,
    uncovered_label: int,
    conflict_margin: float,
) -> np.ndarray:
    if len(candidates) != len(keep):
        raise ValueError("Candidate and keep-decision counts differ")
    assignments = [
        int(candidate.class_id) if should_keep else None
        for candidate, should_keep in zip(candidates, keep)
    ]
    return fuse_candidate_assignments(
        image_shape=image_shape,
        candidates=candidates,
        class_assignments=assignments,
        ignore_index=ignore_index,
        uncovered_label=uncovered_label,
        conflict_margin=conflict_margin,
    )


def fuse_candidate_assignments(
    image_shape: tuple[int, int],
    candidates: list,
    class_assignments: list[int | None],
    ignore_index: int,
    uncovered_label: int,
    conflict_margin: float,
) -> np.ndarray:
    if len(candidates) != len(class_assignments):
        raise ValueError("Candidate and assignment counts differ")
    canvas = FusionCanvas(
        height=int(image_shape[0]),
        width=int(image_shape[1]),
        ignore_index=ignore_index,
        uncovered_label=uncovered_label,
        conflict_margin=conflict_margin,
    )
    for candidate, class_id in zip(candidates, class_assignments):
        if class_id is None:
            continue
        canvas.add_mask(
            mask=candidate.mask,
            class_id=int(class_id),
            score=candidate.score,
            x0=candidate.x0,
            y0=candidate.y0,
        )
    return canvas.result()


def cam_keep_decisions(
    candidates: list,
    cams: np.ndarray,
    cam_class_ids: np.ndarray,
    active_class_ids: list[int],
    reject_classes: set[str],
    cam_method: str,
) -> tuple[list[bool], dict[str, dict[str, int]]]:
    keep = []
    counts: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        bucket = counts.setdefault(
            candidate.class_name,
            {"total": 0, "checked": 0, "kept": 0, "rejected": 0},
        )
        bucket["total"] += 1
        should_keep = True
        if candidate.class_name in reject_classes:
            bucket["checked"] += 1
            decision = score_candidate_cams(
                candidate=candidate,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
            )[cam_method]
            should_keep = bool(decision["agrees_with_candidate"])
        if should_keep:
            bucket["kept"] += 1
        else:
            bucket["rejected"] += 1
        keep.append(should_keep)
    return keep, counts


def _parse_reject_classes(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _prepare_output(output_dir: Path) -> None:
    artifacts = [
        output_dir / "pseudo_labels",
        output_dir / "metadata",
        output_dir / "summary.json",
    ]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "Candidate rebuild output already contains artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Use a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_level = read_image_level_csv(args.labels_csv)
    image_ids = sorted(image_level)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    reject_classes = _parse_reject_classes(args.reject_classes)
    known_classes = {spec.name for spec in config.classes}
    unknown = sorted(reject_classes - known_classes)
    if unknown:
        raise ValueError(f"Unknown reject classes: {', '.join(unknown)}")
    if reject_classes and args.cam_dir is None:
        raise ValueError("--cam-dir is required when --reject-classes is used")

    candidate_dir = Path(args.candidate_dir)
    cam_dir = Path(args.cam_dir) if args.cam_dir is not None else None
    output_dir = Path(args.output_dir)
    _prepare_output(output_dir)
    class_by_name = {spec.name: spec for spec in config.classes}

    summaries = []
    skipped: Counter[str] = Counter()
    skipped_examples: dict[str, list[str]] = {}

    def skip(reason: str, image_id: str) -> None:
        skipped[reason] += 1
        examples = skipped_examples.setdefault(reason, [])
        if len(examples) < 5:
            examples.append(image_id)

    for image_id in tqdm(image_ids, desc="rebuilding candidates"):
        if not candidate_cache_exists(candidate_dir, image_id):
            skip("missing_cache", image_id)
            continue
        cam_path = None if cam_dir is None else cam_dir / f"{image_id}.npz"
        if reject_classes and (cam_path is None or not cam_path.exists()):
            skip("missing_cam", image_id)
            continue

        cache_metadata, candidates = load_candidate_cache(candidate_dir, image_id)
        active_class_ids = sorted(
            class_by_name[name].id
            for name in image_level[image_id]
            if name in class_by_name
        )
        if reject_classes:
            with np.load(cam_path, allow_pickle=False) as data:  # type: ignore[arg-type]
                cams = data["cams"].astype(np.float32)
                cam_class_ids = data["class_ids"].astype(np.int64)
            keep, class_counts = cam_keep_decisions(
                candidates=candidates,
                cams=cams,
                cam_class_ids=cam_class_ids,
                active_class_ids=active_class_ids,
                reject_classes=reject_classes,
                cam_method=args.cam_method,
            )
        else:
            keep = [True] * len(candidates)
            class_counts = {}
            for candidate in candidates:
                bucket = class_counts.setdefault(
                    candidate.class_name,
                    {"total": 0, "checked": 0, "kept": 0, "rejected": 0},
                )
                bucket["total"] += 1
                bucket["kept"] += 1

        image_shape = tuple(int(value) for value in cache_metadata["image_shape"])
        label = fuse_cached_candidates(
            image_shape=image_shape,
            candidates=candidates,
            keep=keep,
            ignore_index=config.ignore_index,
            uncovered_label=config.uncovered_label,
            conflict_margin=config.conflict_margin,
        )
        save_label_png(label, output_dir / "pseudo_labels" / f"{image_id}.png")
        ids, pixel_counts = np.unique(label, return_counts=True)
        item_summary = {
            "image_id": image_id,
            "candidate_count": len(candidates),
            "kept_candidates": int(sum(keep)),
            "rejected_candidates": int(len(keep) - sum(keep)),
            "class_candidate_counts": class_counts,
            "pseudo_label_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(ids, pixel_counts)
            },
        }
        metadata_path = output_dir / "metadata" / f"{image_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(item_summary, indent=2), encoding="utf-8")
        summaries.append(item_summary)

    report = {
        "protocol": {
            "candidate_dir": str(candidate_dir.resolve()),
            "cam_dir": None if cam_dir is None else str(cam_dir.resolve()),
            "cam_method": args.cam_method,
            "reject_classes": sorted(reject_classes),
            "cam_role": "reject-only; no relabeling",
            "gt_used": False,
        },
        "input_images": len(image_ids),
        "processed_images": len(summaries),
        "skipped_images": int(sum(skipped.values())),
        "skipped_reasons": dict(sorted(skipped.items())),
        "skipped_examples": dict(sorted(skipped_examples.items())),
        "total_candidates": sum(item["candidate_count"] for item in summaries),
        "kept_candidates": sum(item["kept_candidates"] for item in summaries),
        "rejected_candidates": sum(item["rejected_candidates"] for item in summaries),
        "items": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))
    if args.require_all and skipped:
        raise RuntimeError(
            f"Could not rebuild {sum(skipped.values())}/{len(image_ids)} images: "
            f"{dict(skipped)}"
        )


if __name__ == "__main__":
    main()
