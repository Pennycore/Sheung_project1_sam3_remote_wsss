from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import ProjectConfig, load_config
from .potsdam import label_rgb_to_ids, read_image_level_csv, read_label_rgb
from .student.dataset import PotsdamGroundTruthSegDataset
from .student.model import StudentSegmentor
from .train_student import segmentation_metrics


@dataclass(frozen=True)
class PatchRecord:
    image_id: str
    parent_image_id: str
    label_path: Path
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained segmentation student and stitch overlapping "
            "Potsdam patches back to parent tiles."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-metadata", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--parent-limit",
        type=int,
        default=None,
        help="Evaluate complete patches from only the first N parents for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    metadata_path = (
        Path(args.patch_metadata)
        if args.patch_metadata
        else config.dataset_root / "patches.csv"
    )
    selected_ids = set(read_image_level_csv(args.labels_csv))
    records_by_parent = read_patch_records(
        metadata_path,
        config.dataset_root,
        selected_ids,
    )
    parent_ids = sorted(records_by_parent)
    if args.parent_limit is not None:
        if args.parent_limit < 1:
            raise ValueError("--parent-limit must be at least 1")
        parent_ids = parent_ids[: args.parent_limit]
        records_by_parent = {
            parent_id: records_by_parent[parent_id] for parent_id in parent_ids
        }
        selected_ids = {
            record.image_id
            for records in records_by_parent.values()
            for record in records
        }

    output_dir = prepare_evaluation_output(Path(args.output_dir))
    patch_prediction_dir = output_dir / "patch_predictions"
    stitched_prediction_dir = output_dir / "stitched_predictions"
    patch_prediction_dir.mkdir()
    stitched_prediction_dir.mkdir()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, checkpoint = load_student_checkpoint(
        args.checkpoint,
        config,
        device,
    )
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    dataset = PotsdamGroundTruthSegDataset(
        config=config,
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        image_ids=selected_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    num_classes = max(spec.id for spec in config.classes) + 1
    class_names = ("background", *(spec.name for spec in config.classes))
    patch_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="student inference"):
            images = batch["image"].to(device, non_blocking=True)
            with autocast(
                device_type=amp_device,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            labels = batch["label"].numpy().astype(np.int64)
            for image_id, prediction, label in zip(
                batch["image_id"], predictions, labels
            ):
                Image.fromarray(prediction, mode="L").save(
                    patch_prediction_dir / f"{image_id}.png"
                )
                update_confusion(
                    patch_confusion,
                    label,
                    prediction,
                    num_classes,
                    config.ignore_index,
                )

    stitched_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_parent: dict[str, dict[str, object]] = {}
    for parent_id in tqdm(parent_ids, desc="stitching parents"):
        prediction, label = stitch_parent(
            records_by_parent[parent_id],
            patch_prediction_dir,
            config,
        )
        Image.fromarray(prediction, mode="L").save(
            stitched_prediction_dir / f"{parent_id}.png"
        )
        parent_confusion = np.zeros_like(stitched_confusion)
        update_confusion(
            parent_confusion,
            label,
            prediction,
            num_classes,
            config.ignore_index,
        )
        stitched_confusion += parent_confusion
        per_parent[parent_id] = segmentation_metrics(
            parent_confusion,
            class_names,
        )

    patch_metrics = segmentation_metrics(patch_confusion, class_names)
    patch_metrics["evaluated_images"] = len(dataset)
    stitched_metrics = segmentation_metrics(stitched_confusion, class_names)
    stitched_metrics["evaluated_parents"] = len(parent_ids)
    stitched_metrics["per_parent"] = per_parent
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "patch_metadata": str(metadata_path.resolve()),
        "patch_metrics": patch_metrics,
        "stitched_metrics": stitched_metrics,
    }
    output_path = output_dir / "student_metrics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def prepare_evaluation_output(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Student evaluation output is not empty: {output_dir}. "
            "Use a new --output-dir to protect existing predictions and metrics."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_patch_records(
    metadata_path: str | Path,
    dataset_root: str | Path,
    selected_ids: set[str],
) -> dict[str, list[PatchRecord]]:
    metadata_path = Path(metadata_path)
    dataset_root = Path(dataset_root)
    records_by_parent: dict[str, list[PatchRecord]] = {}
    found_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if image_id not in selected_ids:
                continue
            if image_id in found_ids:
                raise ValueError(f"Duplicate patch metadata for {image_id}")
            found_ids.add(image_id)
            record = PatchRecord(
                image_id=image_id,
                parent_image_id=row["parent_image_id"],
                label_path=dataset_root / row["label_path"],
                x0=int(row["x0"]),
                y0=int(row["y0"]),
                x1=int(row["x1"]),
                y1=int(row["y1"]),
            )
            records_by_parent.setdefault(record.parent_image_id, []).append(record)
    missing_ids = sorted(selected_ids - found_ids)
    if missing_ids:
        raise FileNotFoundError(
            f"Patch metadata is missing {len(missing_ids)} selected IDs: "
            f"{', '.join(missing_ids[:5])}"
        )
    for records in records_by_parent.values():
        records.sort(key=lambda record: (record.y0, record.x0, record.image_id))
    return records_by_parent


def load_student_checkpoint(
    checkpoint_path: str | Path,
    config: ProjectConfig,
    device,
):
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("model_args", {})
    model = StudentSegmentor(
        num_classes=max(spec.id for spec in config.classes) + 1,
        backbone=model_args.get("backbone", "resnet50"),
        head=model_args.get("head", "segformer"),
        output_stride=int(model_args.get("output_stride", 16)),
        segformer_embed_dim=int(model_args.get("segformer_embed_dim", 256)),
        dropout_ratio=float(model_args.get("dropout_ratio", 0.1)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def stitch_parent(
    records: list[PatchRecord],
    prediction_dir: str | Path,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot stitch an empty parent record list")
    parent_width = max(record.x1 for record in records)
    parent_height = max(record.y1 for record in records)
    prediction = np.zeros((parent_height, parent_width), dtype=np.uint8)
    best_weight = np.zeros((parent_height, parent_width), dtype=np.uint16)
    label = np.full(
        (parent_height, parent_width),
        config.ignore_index,
        dtype=np.uint8,
    )
    assigned = np.zeros((parent_height, parent_width), dtype=bool)
    prediction_dir = Path(prediction_dir)

    for record in records:
        patch_prediction = np.asarray(
            Image.open(prediction_dir / f"{record.image_id}.png").convert("L"),
            dtype=np.uint8,
        )
        patch_label = label_rgb_to_ids(
            read_label_rgb(record.label_path),
            config.classes,
            config.ignore_index,
            background_colors=config.background_colors,
        )
        expected_shape = (record.height, record.width)
        if patch_prediction.shape != expected_shape or patch_label.shape != expected_shape:
            raise ValueError(
                f"Patch shape mismatch for {record.image_id}: expected "
                f"{expected_shape}, prediction={patch_prediction.shape}, "
                f"label={patch_label.shape}"
            )

        ys = slice(record.y0, record.y1)
        xs = slice(record.x0, record.x1)
        region_assigned = assigned[ys, xs]
        region_label = label[ys, xs]
        conflicts = region_assigned & (region_label != patch_label)
        if np.any(conflicts):
            raise ValueError(
                f"Overlapping GT patches disagree for parent {record.parent_image_id}"
            )
        region_label[~region_assigned] = patch_label[~region_assigned]
        region_assigned[...] = True

        weight = center_weight(record, parent_width, parent_height)
        region_weight = best_weight[ys, xs]
        region_prediction = prediction[ys, xs]
        replace = weight > region_weight
        region_prediction[replace] = patch_prediction[replace]
        region_weight[replace] = weight[replace]

    if not np.all(assigned):
        raise ValueError(
            f"Patch metadata does not fully cover parent {records[0].parent_image_id}"
        )
    return prediction, label


def center_weight(
    record: PatchRecord,
    parent_width: int,
    parent_height: int,
) -> np.ndarray:
    x = np.arange(record.width, dtype=np.uint16)
    y = np.arange(record.height, dtype=np.uint16)
    left = x + 1
    right = record.width - x
    top = y + 1
    bottom = record.height - y
    if record.x0 == 0:
        left = np.full_like(left, record.width)
    if record.x1 == parent_width:
        right = np.full_like(right, record.width)
    if record.y0 == 0:
        top = np.full_like(top, record.height)
    if record.y1 == parent_height:
        bottom = np.full_like(bottom, record.height)
    horizontal = np.minimum(left, right)
    vertical = np.minimum(top, bottom)
    return np.minimum(vertical[:, None], horizontal[None, :])


def update_confusion(
    confusion: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> None:
    valid = (label != ignore_index) & (label >= 0) & (label < num_classes)
    encoded = (
        num_classes * label[valid].astype(np.int64)
        + prediction[valid].astype(np.int64)
    )
    confusion += np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


if __name__ == "__main__":
    main()
