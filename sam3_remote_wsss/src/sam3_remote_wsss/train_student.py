from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .config import load_config
from .cam.model import load_encoder_from_cam_checkpoint
from .student.dataset import PotsdamPseudoSegDataset
from .student.losses import toco_seg_loss
from .student.model import StudentSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SegFormer-head segmentation student on SAM3 pseudo labels.")
    parser.add_argument("--config", required=True, help="Path to project JSON config.")
    parser.add_argument("--pseudo-label-dir", required=True, help="Directory containing SAM3 pseudo-label PNGs.")
    parser.add_argument("--output-dir", required=True, help="Training output directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--samples-per-image", type=int, default=16)
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
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = PotsdamPseudoSegDataset(
        config=config,
        pseudo_label_dir=args.pseudo_label_dir,
        crop_size=args.crop_size,
        samples_per_image=args.samples_per_image,
        random_crop=True,
        limit=args.limit,
        augment=not args.no_augment,
        scale_range=(args.scale_min, args.scale_max),
        cat_max_ratio=args.cat_max_ratio,
        max_crop_attempts=args.max_crop_attempts,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate90_prob=args.rotate90_prob,
        photometric_prob=args.photometric_prob,
        blur_prob=args.blur_prob,
        min_valid_ratio=args.min_valid_ratio,
        min_foreground_ratio=args.min_foreground_ratio,
        min_component_area=args.min_component_area,
        ignore_boundary_width=args.ignore_boundary_width,
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
    model = model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=args.amp and device.type == "cuda")
    log_path = output_dir / "train_log.jsonl"
    total_steps = max(1, args.epochs * len(loader))
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
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
                loss = toco_seg_loss(logits, labels, ignore_index=config.ignore_index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())
            global_step += 1
            if step % args.log_interval == 0:
                avg = running_loss / step
                print(f"epoch={epoch} step={step}/{len(loader)} loss={avg:.4f} lr={lr:.6g}")

        epoch_loss = running_loss / max(1, len(loader))
        log_record = {"epoch": epoch, "loss": epoch_loss, "lr": last_lr}
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
        )
        print(f"epoch={epoch} loss={epoch_loss:.4f} checkpoint={checkpoint_dir / 'last.pt'}")


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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    config_path: str,
    loss: float,
) -> None:
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "config_path": config_path,
            "loss": loss,
        },
        path,
    )


if __name__ == "__main__":
    main()
