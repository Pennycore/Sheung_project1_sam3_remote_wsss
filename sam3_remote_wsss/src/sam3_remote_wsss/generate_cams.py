from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .cam.fusion import normalize_cams
from .cam.model import load_cam_checkpoint
from .config import load_config
from .potsdam import discover_potsdam_items, read_image_level_csv, read_rgbir_as_rgb


_RESAMPLING = getattr(Image, "Resampling", Image)
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normalized multi-scale CAMs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", default="0.5,1.0,1.5")
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=0,
        help="Save CAM heatmap overlays for the first N processed images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    config = load_config(args.config)
    image_level = read_image_level_csv(args.labels_csv)
    class_specs = tuple(sorted(config.classes, key=lambda spec: spec.id))
    class_ids = np.asarray([spec.id for spec in class_specs], dtype=np.int16)
    class_names = [spec.name for spec in class_specs]
    scales = _parse_scales(args.scales)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, checkpoint = load_cam_checkpoint(args.checkpoint, device=device)
    if list(checkpoint.get("class_ids", [])) != class_ids.tolist():
        raise ValueError("Checkpoint class_ids do not match the project config")

    items = [
        item
        for item in discover_potsdam_items(config)
        if item.image_id in image_level
    ]
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    items = [
        item
        for index, item in enumerate(items)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        items = items[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        items = [item for item in items if not (output_dir / f"{item.image_id}.npz").exists()]

    summaries = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for item in tqdm(items, desc="generating CAMs"):
        image = read_rgbir_as_rgb(item.image_path, config.rgb_band_indices)
        height, width = image.shape[:2]
        accumulated = []
        for scale in scales:
            scaled = _resize_image(image, scale)
            variants = [(scaled, False)]
            if not args.no_hflip:
                variants.append((np.flip(scaled, axis=1).copy(), True))
            for variant, flipped in variants:
                tensor = _image_tensor(variant, torch).to(device)
                with torch.inference_mode(), autocast(
                    device_type=amp_device,
                    enabled=args.amp and device.type == "cuda",
                ):
                    cams = model.forward_cam(tensor)
                    cams = torch.relu(cams)
                    cams = F.interpolate(
                        cams,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                cam_array = cams.float().cpu().numpy()
                if flipped:
                    cam_array = np.flip(cam_array, axis=2).copy()
                accumulated.append(normalize_cams(cam_array))

        cams = normalize_cams(np.mean(accumulated, axis=0))
        positives = image_level[item.image_id]
        active = np.asarray([name in positives for name in class_names], dtype=bool)
        cams[~active] = 0.0
        np.savez_compressed(
            output_dir / f"{item.image_id}.npz",
            cams=cams.astype(np.float16),
            class_ids=class_ids,
        )
        if len(summaries) < args.visualize_limit:
            for index, name in enumerate(class_names):
                if active[index]:
                    _save_cam_overlay(
                        image,
                        cams[index],
                        output_dir / "visualizations" / f"{item.image_id}_{name}.jpg",
                    )
        summaries.append(
            {
                "image_id": item.image_id,
                "positive_classes": sorted(positives),
                "active_cam_channels": int(active.sum()),
                "cam_maxima": {
                    name: float(cams[index].max())
                    for index, name in enumerate(class_names)
                },
            }
        )

    summary_path = output_dir / f"summary_shard{args.shard_index}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Generated {len(items)} CAM files in {output_dir}")


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must contain positive comma-separated values")
    return scales


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if size == (width, height):
        return image
    return np.asarray(
        Image.fromarray(image).resize(size, _RESAMPLING.BILINEAR),
        dtype=np.uint8,
    )


def _image_tensor(image: np.ndarray, torch_module):
    tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = (tensor - _MEAN) / _STD
    return torch_module.from_numpy(np.ascontiguousarray(tensor))[None]


def _save_cam_overlay(image: np.ndarray, cam: np.ndarray, output_path: Path) -> None:
    strength = np.clip(cam, 0.0, 1.0).astype(np.float32)
    color = np.zeros_like(image, dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = 180.0 * strength
    alpha = (0.55 * strength)[..., None]
    overlay = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


if __name__ == "__main__":
    main()
