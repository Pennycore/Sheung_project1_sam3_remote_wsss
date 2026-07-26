from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .potsdam import discover_potsdam_items, label_rgb_to_ids, read_label_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 pseudo labels against Potsdam pixel labels.")
    parser.add_argument("--config", required=True, help="Path to project JSON config.")
    parser.add_argument("--pseudo-label-dir", required=True, help="Directory containing pseudo-label PNGs.")
    parser.add_argument("--output", required=True, help="Output JSON metrics file.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    item_by_id = {item.image_id: item for item in discover_potsdam_items(config)}
    pseudo_paths = sorted(Path(args.pseudo_label_dir).glob("*.png"))
    if args.limit is not None:
        pseudo_paths = pseudo_paths[: args.limit]

    num_classes = max(spec.id for spec in config.classes) + 1
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    evaluated = 0
    for pseudo_path in tqdm(pseudo_paths, desc="evaluating"):
        item = item_by_id.get(pseudo_path.stem)
        if item is None or item.label_path is None:
            continue
        pred = np.asarray(Image.open(pseudo_path).convert("L"), dtype=np.uint8)
        gt = label_rgb_to_ids(
            read_label_rgb(item.label_path),
            config.classes,
            config.ignore_index,
            background_colors=config.background_colors,
        )
        valid = (gt != config.ignore_index) & (pred != config.ignore_index)
        valid &= (gt < num_classes) & (pred < num_classes)
        if not np.any(valid):
            continue
        bincount = np.bincount(
            num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
            minlength=num_classes * num_classes,
        )
        confusion += bincount.reshape(num_classes, num_classes)
        evaluated += 1

    metrics = compute_iou(confusion, config)
    metrics["evaluated_images"] = evaluated
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def compute_iou(confusion: np.ndarray, config) -> dict:
    class_iou = {}
    valid_ious = []
    foreground_ious = []
    id_to_name = {0: "background", **{spec.id: spec.name for spec in config.classes}}
    for class_id, name in id_to_name.items():
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - confusion[class_id, class_id])
        fn = float(confusion[class_id, :].sum() - confusion[class_id, class_id])
        denom = tp + fp + fn
        iou = None if denom == 0 else tp / denom
        class_iou[name] = iou
        if iou is not None:
            valid_ious.append(iou)
            if class_id != 0:
                foreground_ious.append(iou)
    return {
        "class_iou": class_iou,
        "miou": None if not valid_ious else float(np.mean(valid_ious)),
        "foreground_miou": None if not foreground_ious else float(np.mean(foreground_ious)),
        "confusion": confusion.tolist(),
    }


if __name__ == "__main__":
    main()
