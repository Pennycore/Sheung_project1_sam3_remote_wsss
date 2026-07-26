from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .potsdam import discover_potsdam_items, write_image_level_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-level labels from Potsdam pixel labels.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    items = discover_potsdam_items(config)
    if args.limit is not None:
        items = items[: args.limit]
    write_image_level_csv(items, config, Path(args.output))
    print(f"Wrote image-level labels for {len(items)} images to {args.output}")


if __name__ == "__main__":
    main()
