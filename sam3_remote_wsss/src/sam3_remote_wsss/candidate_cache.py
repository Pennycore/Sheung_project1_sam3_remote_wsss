from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CandidateMask:
    class_id: int
    class_name: str
    prompt: str
    score: float
    mask: np.ndarray
    x0: int = 0
    y0: int = 0


def candidate_cache_paths(cache_dir: str | Path, image_id: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{image_id}.npz", root / f"{image_id}.json"


def candidate_cache_exists(cache_dir: str | Path, image_id: str) -> bool:
    data_path, metadata_path = candidate_cache_paths(cache_dir, image_id)
    return data_path.exists() and metadata_path.exists()


def save_candidate_cache(
    cache_dir: str | Path,
    image_id: str,
    image_shape: tuple[int, int],
    candidates: list[CandidateMask],
    provenance: dict | None = None,
) -> dict:
    image_height, image_width = (int(value) for value in image_shape)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_shape must contain positive height and width")

    prompt_keys: dict[tuple[int, str], int] = {}
    prompt_table: list[dict] = []
    packed_parts: list[np.ndarray] = []
    offsets = [0]
    shapes: list[tuple[int, int]] = []
    origins: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    areas: list[int] = []
    scores: list[float] = []
    class_ids: list[int] = []
    prompt_ids: list[int] = []
    class_counts: dict[str, int] = {}

    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"Candidate mask must be 2D, got {mask.shape}")
        height, width = mask.shape
        x0 = int(candidate.x0)
        y0 = int(candidate.y0)
        if x0 < 0 or y0 < 0 or x0 + width > image_width or y0 + height > image_height:
            raise ValueError(
                "Candidate mask bounds exceed image shape: "
                f"origin=({x0}, {y0}), mask={mask.shape}, image={image_shape}"
            )

        packed = np.packbits(mask.reshape(-1), bitorder="little")
        packed_parts.append(packed)
        offsets.append(offsets[-1] + packed.size)
        shapes.append((height, width))
        origins.append((x0, y0))

        ys, xs = np.nonzero(mask)
        if xs.size:
            box = (
                x0 + int(xs.min()),
                y0 + int(ys.min()),
                x0 + int(xs.max()) + 1,
                y0 + int(ys.max()) + 1,
            )
        else:
            box = (x0, y0, x0, y0)
        boxes.append(box)

        prompt_key = (int(candidate.class_id), str(candidate.prompt))
        if prompt_key not in prompt_keys:
            prompt_keys[prompt_key] = len(prompt_table)
            prompt_table.append(
                {
                    "id": len(prompt_table),
                    "class_id": int(candidate.class_id),
                    "class_name": str(candidate.class_name),
                    "prompt": str(candidate.prompt),
                }
            )

        areas.append(int(mask.sum()))
        scores.append(float(candidate.score))
        class_ids.append(int(candidate.class_id))
        prompt_ids.append(prompt_keys[prompt_key])
        class_counts[candidate.class_name] = class_counts.get(candidate.class_name, 0) + 1

    packed_masks = (
        np.concatenate(packed_parts).astype(np.uint8, copy=False)
        if packed_parts
        else np.empty((0,), dtype=np.uint8)
    )
    candidate_count = len(candidates)
    data_path, metadata_path = candidate_cache_paths(cache_dir, image_id)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray([CACHE_FORMAT_VERSION], dtype=np.int16),
            image_shape=np.asarray([image_height, image_width], dtype=np.int32),
            packed_masks=packed_masks,
            offsets=np.asarray(offsets, dtype=np.int64),
            shapes=np.asarray(shapes, dtype=np.int32).reshape(candidate_count, 2),
            origins=np.asarray(origins, dtype=np.int32).reshape(candidate_count, 2),
            boxes=np.asarray(boxes, dtype=np.int32).reshape(candidate_count, 4),
            areas=np.asarray(areas, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
            class_ids=np.asarray(class_ids, dtype=np.int16),
            prompt_ids=np.asarray(prompt_ids, dtype=np.int16),
        )

    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "image_id": image_id,
        "image_shape": [image_height, image_width],
        "candidate_count": candidate_count,
        "foreground_only": True,
        "mask_encoding": "flattened-packbits-little",
        "data_file": data_path.name,
        "prompts": prompt_table,
        "class_candidate_counts": dict(sorted(class_counts.items())),
        "provenance": dict(provenance or {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_candidate_cache(
    cache_dir: str | Path,
    image_id: str,
) -> tuple[dict, list[CandidateMask]]:
    data_path, metadata_path = candidate_cache_paths(cache_dir, image_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported candidate cache version: {metadata.get('format_version')}")

    with np.load(data_path, allow_pickle=False) as data:
        version = int(data["format_version"][0])
        if version != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported candidate data version: {version}")
        packed_masks = data["packed_masks"]
        offsets = data["offsets"]
        shapes = data["shapes"]
        origins = data["origins"]
        scores = data["scores"]
        class_ids = data["class_ids"]
        prompt_ids = data["prompt_ids"]

        prompt_table = {
            int(item["id"]): item
            for item in metadata["prompts"]
        }
        candidates = []
        for index in range(len(scores)):
            height, width = (int(value) for value in shapes[index])
            start = int(offsets[index])
            end = int(offsets[index + 1])
            flat = np.unpackbits(
                packed_masks[start:end],
                bitorder="little",
                count=height * width,
            )
            prompt_record = prompt_table[int(prompt_ids[index])]
            x0, y0 = (int(value) for value in origins[index])
            candidates.append(
                CandidateMask(
                    class_id=int(class_ids[index]),
                    class_name=str(prompt_record["class_name"]),
                    prompt=str(prompt_record["prompt"]),
                    score=float(scores[index]),
                    mask=flat.reshape(height, width).astype(bool, copy=False),
                    x0=x0,
                    y0=y0,
                )
            )

    expected_count = int(metadata["candidate_count"])
    if len(candidates) != expected_count:
        raise ValueError(
            f"Candidate count mismatch: metadata={expected_count}, data={len(candidates)}"
        )
    return metadata, candidates
