from __future__ import annotations

import argparse
import json
import random
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    log_path = output_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
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
            running_loss += float(loss.detach().cpu())
            global_step += 1
            if step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={running_loss / step:.4f} lr={lr:.6g}"
                )

        epoch_loss = running_loss / max(1, len(loader))
        micro_denom = int((2 * tp + fp + fn).sum())
        micro_f1 = 0.0 if micro_denom == 0 else float(2 * tp.sum() / micro_denom)
        per_class_f1 = {
            name: (
                0.0
                if 2 * tp[index] + fp[index] + fn[index] == 0
                else float(
                    2 * tp[index]
                    / (2 * tp[index] + fp[index] + fn[index])
                )
            )
            for index, name in enumerate(dataset.class_names)
        }
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "micro_f1": micro_f1,
            "per_class_f1": per_class_f1,
            "lr": last_lr,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            epoch_loss,
            micro_f1,
            per_class_f1,
            args,
            dataset,
        )
        print(
            f"epoch={epoch} loss={epoch_loss:.4f} micro_f1={micro_f1:.4f} "
            f"checkpoint={checkpoint_dir / 'last.pt'}"
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


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    loss: float,
    micro_f1: float,
    per_class_f1: dict[str, float],
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
            "per_class_f1": per_class_f1,
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
