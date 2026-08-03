from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .cam.dataset import PotsdamImageLevelDataset
from .cam.model import CAMClassifier
from .config import load_config
from .train_student import compute_poly_lr, set_optimizer_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-label CAM classifier from image-level labels."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--val-labels-csv",
        default=None,
        help="Optional parent-disjoint validation image-level CSV.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--output-stride", type=int, choices=[16, 32], default=16)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly replace existing CAM logs and checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    checkpoint_dir, log_path = _prepare_training_output(
        output_dir,
        overwrite=args.overwrite_output,
    )

    dataset = PotsdamImageLevelDataset(
        config=config,
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        limit=args.limit,
        augment=not args.no_augment,
    )
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_dataset = None
    val_loader = None
    if args.val_labels_csv:
        val_dataset = PotsdamImageLevelDataset(
            config=config,
            labels_csv=args.val_labels_csv,
            image_size=args.image_size,
            limit=args.val_limit,
            augment=False,
        )
        if val_dataset.class_ids != dataset.class_ids:
            raise ValueError("Train and validation CAM classes do not match")
        _ensure_parent_disjoint(
            (item.image_id for item in dataset.items),
            (item.image_id for item in val_dataset.items),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = CAMClassifier(
        num_classes=len(dataset.class_ids),
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        output_stride=args.output_stride,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = torch.from_numpy(dataset.positive_weights()).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=args.amp and device.type == "cuda")
    total_steps = max(1, args.epochs * len(loader))
    global_step = 0
    best_macro_f1 = -1.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        tp = np.zeros(len(dataset.class_ids), dtype=np.int64)
        fp = np.zeros(len(dataset.class_ids), dtype=np.int64)
        fn = np.zeros(len(dataset.class_ids), dtype=np.int64)
        last_lr = args.lr
        for step, batch in enumerate(loader, start=1):
            lr = compute_poly_lr(
                args.lr,
                global_step,
                total_steps,
                power=args.poly_power,
            )
            set_optimizer_lr(optimizer, lr)
            last_lr = lr
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=amp_device,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                predicted = torch.sigmoid(logits) >= 0.5
                actual = targets >= 0.5
                tp += (predicted & actual).sum(dim=0).cpu().numpy()
                fp += (predicted & ~actual).sum(dim=0).cpu().numpy()
                fn += (~predicted & actual).sum(dim=0).cpu().numpy()
            batch_size = int(targets.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            seen_samples += batch_size
            global_step += 1
            if step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={running_loss / max(1, seen_samples):.4f} lr={lr:.6g}"
                )

        epoch_loss = running_loss / max(1, seen_samples)
        micro_f1, macro_f1, per_class_f1 = _f1_metrics(
            tp,
            fp,
            fn,
            dataset.class_names,
        )
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "lr": last_lr,
        }
        val_metrics = None
        if val_loader is not None and val_dataset is not None:
            val_metrics = _evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=args.amp and device.type == "cuda",
                amp_device=amp_device,
                class_names=val_dataset.class_names,
            )
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_per_class_f1": val_metrics["per_class_f1"],
                }
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            epoch_loss,
            micro_f1,
            macro_f1,
            per_class_f1,
            val_metrics,
            args,
            dataset,
        )
        selection_macro = (
            float(val_metrics["macro_f1"])
            if val_metrics is not None
            else macro_f1
        )
        selection_loss = (
            float(val_metrics["loss"])
            if val_metrics is not None
            else epoch_loss
        )
        is_best = selection_macro > best_macro_f1 or (
            selection_macro == best_macro_f1 and selection_loss < best_loss
        )
        if is_best:
            best_macro_f1 = selection_macro
            best_loss = selection_loss
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                epoch_loss,
                micro_f1,
                macro_f1,
                per_class_f1,
                val_metrics,
                args,
                dataset,
            )
        val_message = ""
        if val_metrics is not None:
            val_message = (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
        print(
            f"epoch={epoch} loss={epoch_loss:.4f} micro_f1={micro_f1:.4f} "
            f"macro_f1={macro_f1:.4f}{val_message} "
            f"best={is_best} checkpoint={checkpoint_dir / 'last.pt'}"
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_PATCH_SUFFIX = re.compile(r"_x\d+_y\d+$")


def _prepare_training_output(
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"
    managed_paths = [
        log_path,
        checkpoint_dir / "best.pt",
        checkpoint_dir / "last.pt",
    ]
    existing = [path for path in managed_paths if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "CAM output already contains training artifacts: "
            f"{paths}. Use a new --output-dir, or pass --overwrite-output "
            "only when replacement is intentional."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()
    log_path.write_text("", encoding="utf-8")
    return checkpoint_dir, log_path


def _parent_image_id(image_id: str) -> str:
    return _PATCH_SUFFIX.sub("", image_id)


def _ensure_parent_disjoint(
    train_image_ids: Iterable[str],
    val_image_ids: Iterable[str],
) -> None:
    train_parents = {_parent_image_id(image_id) for image_id in train_image_ids}
    val_parents = {_parent_image_id(image_id) for image_id in val_image_ids}
    overlap = sorted(train_parents & val_parents)
    if overlap:
        examples = ", ".join(overlap[:5])
        raise ValueError(
            "CAM train/validation splits share parent images: "
            f"{examples}. Split Potsdam before patch extraction."
        )


def _f1_metrics(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[float, float, dict[str, float]]:
    denominator = 2 * tp + fp + fn
    scores = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )
    micro_denominator = int(denominator.sum())
    micro_f1 = (
        0.0
        if micro_denominator == 0
        else float(2 * tp.sum() / micro_denominator)
    )
    per_class_f1 = {
        name: float(scores[index]) for index, name in enumerate(class_names)
    }
    return micro_f1, float(scores.mean()), per_class_f1


def _evaluate(
    model,
    loader,
    criterion,
    device,
    amp_enabled: bool,
    amp_device: str,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    import torch
    from torch.amp import autocast

    model.eval()
    running_loss = 0.0
    seen_samples = 0
    tp = np.zeros(len(class_names), dtype=np.int64)
    fp = np.zeros(len(class_names), dtype=np.int64)
    fn = np.zeros(len(class_names), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with autocast(device_type=amp_device, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            seen_samples += batch_size
            predicted = torch.sigmoid(logits) >= 0.5
            actual = targets >= 0.5
            tp += (predicted & actual).sum(dim=0).cpu().numpy()
            fp += (predicted & ~actual).sum(dim=0).cpu().numpy()
            fn += (~predicted & actual).sum(dim=0).cpu().numpy()
    micro_f1, macro_f1, per_class_f1 = _f1_metrics(
        tp,
        fp,
        fn,
        class_names,
    )
    return {
        "loss": running_loss / max(1, seen_samples),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    loss: float,
    micro_f1: float,
    macro_f1: float,
    per_class_f1: dict[str, float],
    val_metrics: dict[str, object] | None,
    args: argparse.Namespace,
    dataset: PotsdamImageLevelDataset,
) -> None:
    import torch

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "val_metrics": val_metrics,
            "class_ids": list(dataset.class_ids),
            "class_names": list(dataset.class_names),
            "model_args": {
                "backbone": args.backbone,
                "output_stride": args.output_stride,
            },
            "train_args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
