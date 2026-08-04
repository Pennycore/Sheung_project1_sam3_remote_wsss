from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from .config import load_config
from .cam.model import load_encoder_from_cam_checkpoint
from .student.dataset import (
    PotsdamGroundTruthSegDataset,
    PotsdamGroundTruthTrainDataset,
    PotsdamPseudoSegDataset,
)
from .student.losses import (
    decomposed_background_seg_loss,
    safe_cross_entropy,
    toco_seg_loss,
)
from .student.model import StudentSegmentor
from .training_output import prepare_training_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a SegFormer-head segmentor on WSSS pseudo labels or pixel GT."
        )
    )
    parser.add_argument("--config", required=True, help="Path to project JSON config.")
    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument(
        "--pseudo-label-dir",
        help="Directory containing WSSS pseudo-label PNGs.",
    )
    train_source.add_argument(
        "--train-labels-csv",
        help="Training split CSV whose corresponding pixel GT supplies an upper bound.",
    )
    parser.add_argument(
        "--val-labels-csv",
        default=None,
        help="Optional parent-disjoint validation CSV with pixel GT available.",
    )
    parser.add_argument("--output-dir", required=True, help="Training output directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--head", choices=["segformer", "large_fov", "aspp"], default="segformer")
    parser.add_argument("--segformer-embed-dim", type=int, default=256)
    parser.add_argument("--dropout-ratio", type=float, default=0.1)
    parser.add_argument("--output-stride", type=int, choices=[16, 32], default=16)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--cam-checkpoint",
        default=None,
        help="Optional CAM checkpoint used to initialize the student encoder.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA devices are visible.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of source images for a smoke test.")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time data augmentation.")
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--cat-max-ratio", type=float, default=0.75)
    parser.add_argument("--max-crop-attempts", type=int, default=10)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--vflip-prob", type=float, default=0.5)
    parser.add_argument("--rotate90-prob", type=float, default=0.5)
    parser.add_argument("--photometric-prob", type=float, default=0.5)
    parser.add_argument("--blur-prob", type=float, default=0.1)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.001)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--ignore-boundary-width", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=["auto", "cross_entropy", "toco", "decomposed"],
        default="auto",
        help=(
            "Training loss. auto uses ToCo for pseudo labels and cross-entropy "
            "for pixel GT."
        ),
    )
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for labeled background seeds.",
    )
    parser.add_argument(
        "--foreground-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for labeled foreground seeds.",
    )
    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=1.0,
        help="Relative foreground semantic term for decomposed loss.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["miou", "foreground_miou"],
        default="miou",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly replace existing student logs and checkpoints.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    checkpoint_dir, log_path = prepare_training_output(
        output_dir,
        experiment_name="Student",
        overwrite=args.overwrite_output,
    )

    common_dataset_args = {
        "config": config,
        "crop_size": args.crop_size,
        "samples_per_image": args.samples_per_image,
        "random_crop": True,
        "limit": args.limit,
        "augment": not args.no_augment,
        "scale_range": (args.scale_min, args.scale_max),
        "cat_max_ratio": args.cat_max_ratio,
        "max_crop_attempts": args.max_crop_attempts,
        "hflip_prob": args.hflip_prob,
        "vflip_prob": args.vflip_prob,
        "rotate90_prob": args.rotate90_prob,
        "photometric_prob": args.photometric_prob,
        "blur_prob": args.blur_prob,
        "min_valid_ratio": args.min_valid_ratio,
        "min_foreground_ratio": args.min_foreground_ratio,
    }
    if args.train_labels_csv:
        dataset = PotsdamGroundTruthTrainDataset(
            labels_csv=args.train_labels_csv,
            **common_dataset_args,
        )
    else:
        dataset = PotsdamPseudoSegDataset(
            pseudo_label_dir=args.pseudo_label_dir,
            min_component_area=args.min_component_area,
            ignore_boundary_width=args.ignore_boundary_width,
            **common_dataset_args,
        )
    loss_name = _resolve_training_loss(args)
    _validate_loss_weights(args, loss_name)
    supervision = "ground_truth" if args.train_labels_csv else "pseudo_label"
    print(
        f"training supervision={supervision} loss={loss_name} "
        f"images={len(dataset.items)} samples={len(dataset)} "
        f"bg_loss_weight={args.background_loss_weight:g} "
        f"fg_loss_weight={args.foreground_loss_weight:g} "
        f"semantic_loss_weight={args.semantic_loss_weight:g}"
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(
            "No training batches were created. Lower --batch-size or increase --samples-per-image."
        )

    num_classes = max(spec.id for spec in config.classes) + 1
    model = StudentSegmentor(
        num_classes=num_classes,
        backbone=args.backbone,
        head=args.head,
        pretrained_backbone=args.pretrained_backbone,
        output_stride=args.output_stride,
        segformer_embed_dim=args.segformer_embed_dim,
        dropout_ratio=args.dropout_ratio,
    )
    if args.cam_checkpoint:
        checkpoint = load_encoder_from_cam_checkpoint(model.encoder, args.cam_checkpoint)
        print(
            "initialized student encoder from CAM checkpoint "
            f"epoch={checkpoint.get('epoch', 'unknown')}"
        )

    val_dataset = None
    val_loader = None
    if args.val_labels_csv:
        val_dataset = PotsdamGroundTruthSegDataset(
            config=config,
            labels_csv=args.val_labels_csv,
            image_size=args.crop_size,
            limit=args.val_limit,
        )
        _ensure_parent_disjoint(
            (item.image_id for item in dataset.items),
            (item.image_id for item in val_dataset.items),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    model = model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=args.amp and device.type == "cuda")
    total_steps = max(1, args.epochs * len(loader))
    global_step = 0
    best_score = float("-inf")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        last_lr = args.lr
        for step, batch in enumerate(loader, start=1):
            lr = compute_poly_lr(
                base_lr=args.lr,
                global_step=global_step,
                total_steps=total_steps,
                power=args.poly_power,
                warmup_iters=args.warmup_iters,
            )
            set_optimizer_lr(optimizer, lr)
            last_lr = lr
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device, enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                if loss_name == "cross_entropy":
                    loss = safe_cross_entropy(
                        logits,
                        labels.long(),
                        ignore_index=config.ignore_index,
                    )
                elif loss_name == "toco":
                    loss = toco_seg_loss(
                        logits,
                        labels,
                        ignore_index=config.ignore_index,
                        background_weight=args.background_loss_weight,
                        foreground_weight=args.foreground_loss_weight,
                    )
                else:
                    loss = decomposed_background_seg_loss(
                        logits,
                        labels,
                        ignore_index=config.ignore_index,
                        background_weight=args.background_loss_weight,
                        foreground_weight=args.foreground_loss_weight,
                        semantic_weight=args.semantic_loss_weight,
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_batch_size = int(labels.shape[0])
            running_loss += float(loss.detach().cpu()) * current_batch_size
            seen_samples += current_batch_size
            global_step += 1
            if step % args.log_interval == 0:
                avg = running_loss / max(1, seen_samples)
                print(f"epoch={epoch} step={step}/{len(loader)} loss={avg:.4f} lr={lr:.6g}")

        epoch_loss = running_loss / max(1, seen_samples)
        log_record = {"epoch": epoch, "loss": epoch_loss, "lr": last_lr}
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_student(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=num_classes,
                ignore_index=config.ignore_index,
                class_names=("background", *(spec.name for spec in config.classes)),
                amp_enabled=args.amp and device.type == "cuda",
                amp_device=amp_device,
            )
            log_record["validation"] = val_metrics
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            config_path=args.config,
            loss=epoch_loss,
            val_metrics=val_metrics,
        )
        if val_metrics is not None:
            raw_score = val_metrics[args.selection_metric]
            selection_score = (
                float(raw_score) if raw_score is not None else float("-inf")
            )
            selection_loss = float(val_metrics["loss"])
        else:
            selection_score = -epoch_loss
            selection_loss = epoch_loss
        is_best = selection_score > best_score or (
            selection_score == best_score and selection_loss < best_loss
        )
        if is_best:
            best_score = selection_score
            best_loss = selection_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                config_path=args.config,
                loss=epoch_loss,
                val_metrics=val_metrics,
            )
        val_message = ""
        if val_metrics is not None:
            val_message = (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_miou={_format_metric(val_metrics['miou'])} "
                "val_fg_miou="
                f"{_format_metric(val_metrics['foreground_miou'])}"
            )
        print(
            f"epoch={epoch} loss={epoch_loss:.4f}{val_message} "
            f"best={is_best} checkpoint={checkpoint_dir / 'last.pt'}"
        )


def compute_poly_lr(
    base_lr: float,
    global_step: int,
    total_steps: int,
    power: float = 0.9,
    warmup_iters: int = 0,
) -> float:
    if warmup_iters > 0 and global_step < warmup_iters:
        return base_lr * float(global_step + 1) / float(warmup_iters)
    if warmup_iters > 0:
        global_step -= warmup_iters
        total_steps = max(1, total_steps - warmup_iters)
    progress = min(1.0, float(global_step) / float(max(1, total_steps)))
    return base_lr * (1.0 - progress) ** power


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _resolve_training_loss(args: argparse.Namespace) -> str:
    if args.loss != "auto":
        return args.loss
    return "cross_entropy" if args.train_labels_csv else "toco"


def _validate_loss_weights(args: argparse.Namespace, loss_name: str) -> None:
    if (
        args.background_loss_weight < 0
        or args.foreground_loss_weight < 0
        or args.semantic_loss_weight < 0
    ):
        raise ValueError("Segmentation loss weights must be non-negative")
    if (
        loss_name == "toco"
        and args.background_loss_weight == 0
        and args.foreground_loss_weight == 0
    ):
        raise ValueError("At least one ToCo segmentation loss weight must be positive")
    if (
        loss_name == "decomposed"
        and args.background_loss_weight == 0
        and args.foreground_loss_weight == 0
        and args.semantic_loss_weight == 0
    ):
        raise ValueError("At least one decomposed loss weight must be positive")


_PATCH_SUFFIX = re.compile(r"_x\d+_y\d+$")


def _parent_image_id(image_id: str) -> str:
    return _PATCH_SUFFIX.sub("", image_id)


def _ensure_parent_disjoint(train_image_ids, val_image_ids) -> None:
    train_parents = {_parent_image_id(image_id) for image_id in train_image_ids}
    val_parents = {_parent_image_id(image_id) for image_id in val_image_ids}
    overlap = sorted(train_parents & val_parents)
    if overlap:
        raise ValueError(
            "Student train/validation splits share parent images: "
            f"{', '.join(overlap[:5])}"
        )


def segmentation_metrics(
    confusion: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    class_iou: dict[str, float | None] = {}
    valid_ious = []
    foreground_ious = []
    for class_id, name in enumerate(class_names):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - true_positive
        false_negative = int(confusion[class_id, :].sum()) - true_positive
        denominator = true_positive + false_positive + false_negative
        iou = None if denominator == 0 else true_positive / denominator
        class_iou[name] = iou
        if iou is not None:
            valid_ious.append(iou)
            if class_id != 0:
                foreground_ious.append(iou)
    total = int(confusion.sum())
    return {
        "class_iou": class_iou,
        "miou": None if not valid_ious else float(np.mean(valid_ious)),
        "foreground_miou": (
            None if not foreground_ious else float(np.mean(foreground_ious))
        ),
        "pixel_accuracy": (
            None if total == 0 else float(np.trace(confusion) / total)
        ),
        "confusion": confusion.tolist(),
        "valid_pixels": total,
    }


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def evaluate_student(
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int,
    class_names: tuple[str, ...],
    amp_enabled: bool,
    amp_device: str,
) -> dict[str, object]:
    import torch
    from torch.amp import autocast

    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0
    seen_samples = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(device_type=amp_device, enabled=amp_enabled):
                logits = model(images)
                loss = safe_cross_entropy(
                    logits,
                    labels.long(),
                    ignore_index,
                )
            batch_size = int(labels.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            seen_samples += batch_size

            predictions = logits.argmax(dim=1)
            valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
            encoded = (
                num_classes * labels[valid].to(torch.int64)
                + predictions[valid].to(torch.int64)
            )
            counts = torch.bincount(
                encoded,
                minlength=num_classes * num_classes,
            )
            confusion += counts.reshape(num_classes, num_classes).cpu().numpy()

    metrics = segmentation_metrics(confusion, class_names)
    metrics["loss"] = running_loss / max(1, seen_samples)
    metrics["evaluated_images"] = len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    config_path: str,
    loss: float,
    val_metrics: dict[str, object] | None,
) -> None:
    import torch

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "config_path": config_path,
            "loss": loss,
            "val_metrics": val_metrics,
            "training_supervision": (
                "ground_truth" if args.train_labels_csv else "pseudo_label"
            ),
            "training_loss": _resolve_training_loss(args),
            "model_args": {
                "backbone": args.backbone,
                "head": args.head,
                "output_stride": args.output_stride,
                "segformer_embed_dim": args.segformer_embed_dim,
                "dropout_ratio": args.dropout_ratio,
            },
        },
        path,
    )


if __name__ == "__main__":
    main()
