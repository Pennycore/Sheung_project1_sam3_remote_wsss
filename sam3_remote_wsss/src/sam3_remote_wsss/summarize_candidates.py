from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize compressed SAM3 candidate-mask caches."
    )
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output. Defaults to <candidate-dir>/summary.json.",
    )
    return parser.parse_args()


def summarize_candidate_dir(candidate_dir: str | Path) -> dict:
    root = Path(candidate_dir)
    metadata_paths = sorted(
        path
        for path in root.glob("*.json")
        if path.name != "summary.json"
    )
    class_candidate_counts: Counter[str] = Counter()
    class_image_counts: Counter[str] = Counter()
    prompt_candidate_counts: Counter[tuple[str, str]] = Counter()
    all_scores: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    total_candidates = 0
    disk_bytes = 0

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "image_id" not in metadata or "candidate_count" not in metadata:
            continue
        data_path = root / metadata["data_file"]
        if not data_path.exists():
            raise FileNotFoundError(
                f"Candidate data is missing for {metadata['image_id']}: {data_path}"
            )

        with np.load(data_path, allow_pickle=False) as data:
            scores = data["scores"].astype(np.float64, copy=False)
            areas = data["areas"].astype(np.int64, copy=False)
            prompt_ids = data["prompt_ids"].astype(np.int64, copy=False)
        expected = int(metadata["candidate_count"])
        if len(scores) != expected or len(areas) != expected or len(prompt_ids) != expected:
            raise ValueError(
                f"Candidate count mismatch for {metadata['image_id']}: expected {expected}"
            )

        prompt_table = {
            int(item["id"]): item
            for item in metadata["prompts"]
        }
        for prompt_id in prompt_ids:
            prompt_record = prompt_table[int(prompt_id)]
            prompt_candidate_counts[
                (str(prompt_record["class_name"]), str(prompt_record["prompt"]))
            ] += 1

        per_class = {
            str(name): int(count)
            for name, count in metadata["class_candidate_counts"].items()
        }
        class_candidate_counts.update(per_class)
        class_image_counts.update(name for name, count in per_class.items() if count > 0)
        total_candidates += expected
        all_scores.append(scores)
        all_areas.append(areas)
        disk_bytes += data_path.stat().st_size + metadata_path.stat().st_size

    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float64)
    areas = np.concatenate(all_areas) if all_areas else np.empty((0,), dtype=np.int64)
    return {
        "candidate_dir": str(root),
        "images": len(metadata_paths),
        "candidates": total_candidates,
        "disk_bytes": disk_bytes,
        "class_candidate_counts": dict(sorted(class_candidate_counts.items())),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "prompt_candidate_counts": {
            class_name: {
                prompt: int(count)
                for (name, prompt), count in sorted(prompt_candidate_counts.items())
                if name == class_name
            }
            for class_name in sorted({key[0] for key in prompt_candidate_counts})
        },
        "score_percentiles": _percentiles(scores),
        "area_percentiles": _percentiles(areas),
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    levels = (0, 25, 50, 75, 90, 95, 100)
    if values.size == 0:
        return {str(level): None for level in levels}
    return {
        str(level): float(value)
        for level, value in zip(levels, np.percentile(values, levels))
    }


def main() -> None:
    args = parse_args()
    summary = summarize_candidate_dir(args.candidate_dir)
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(args.candidate_dir) / "summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
