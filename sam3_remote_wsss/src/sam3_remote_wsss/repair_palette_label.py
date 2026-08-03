from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a noisy RGB label to the nearest configured class color."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--compression", choices=["none", "deflate"], default="deflate")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.background_colors:
        raise ValueError("The project config must define at least one background color")

    names = ["background", *(spec.name for spec in config.classes)]
    colors = np.asarray(
        [config.background_colors[0], *(spec.label_color for spec in config.classes)],
        dtype=np.uint8,
    )
    report = repair_palette_label(
        input_path=args.input,
        output_path=args.output,
        names=names,
        colors=colors,
        max_distance=args.max_distance,
        chunk_rows=args.chunk_rows,
        compression=args.compression,
        overwrite_output=args.overwrite_output,
    )
    print(json.dumps(report, indent=2))


def repair_palette_label(
    input_path: str | Path,
    output_path: str | Path,
    names: list[str] | tuple[str, ...],
    colors: np.ndarray,
    max_distance: float = 80.0,
    chunk_rows: int = 256,
    compression: str = "deflate",
    overwrite_output: bool = False,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    names = tuple(names)
    colors = np.asarray(colors, dtype=np.uint8)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Repair output must differ from the input label")
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(
            f"Repair output already exists: {output_path}. "
            "Use --overwrite-output only for intentional replacement."
        )
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(names):
        raise ValueError("Palette names and colors must define matching RGB entries")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if compression not in {"none", "deflate"}:
        raise ValueError("compression must be 'none' or 'deflate'")

    image = tifffile.imread(input_path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB label, got {image.shape}: {input_path}")
    rgb = np.asarray(image[..., :3], dtype=np.uint8)
    repaired = np.empty_like(rgb)
    palette = colors.astype(np.int32)
    counts = np.zeros(len(colors), dtype=np.int64)
    maximum_distance_squared = 0
    threshold_squared = float(max_distance) ** 2
    pixels_over_threshold = 0

    for y0 in range(0, rgb.shape[0], chunk_rows):
        y1 = min(rgb.shape[0], y0 + chunk_rows)
        chunk = rgb[y0:y1].astype(np.int32)
        best_distance_squared = np.full(
            chunk.shape[:2],
            np.iinfo(np.int64).max,
            dtype=np.int64,
        )
        best_index = np.zeros(chunk.shape[:2], dtype=np.uint8)

        for index, color in enumerate(palette):
            difference = chunk - color
            distance_squared = np.sum(
                difference * difference,
                axis=-1,
                dtype=np.int64,
            )
            closer = distance_squared < best_distance_squared
            best_distance_squared[closer] = distance_squared[closer]
            best_index[closer] = index

        repaired[y0:y1] = colors[best_index]
        counts += np.bincount(best_index.ravel(), minlength=len(colors))
        maximum_distance_squared = max(
            maximum_distance_squared,
            int(best_distance_squared.max()),
        )
        pixels_over_threshold += int(
            np.count_nonzero(best_distance_squared > threshold_squared)
        )

    report: dict[str, object] = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "shape": list(rgb.shape),
        "max_allowed_distance": float(max_distance),
        "max_observed_distance": float(np.sqrt(maximum_distance_squared)),
        "pixels_over_threshold": pixels_over_threshold,
        "pixel_counts": {
            name: int(count) for name, count in zip(names, counts)
        },
    }
    if pixels_over_threshold:
        raise ValueError(
            f"Refusing palette repair: {pixels_over_threshold} pixels exceed "
            f"max distance {max_distance}; max observed distance is "
            f"{report['max_observed_distance']:.2f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    kwargs = {"photometric": "rgb"}
    if compression != "none":
        kwargs["compression"] = compression
    tifffile.imwrite(temporary_path, repaired, **kwargs)
    temporary_path.replace(output_path)

    report_path = output_path.with_suffix(output_path.suffix + ".repair.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
