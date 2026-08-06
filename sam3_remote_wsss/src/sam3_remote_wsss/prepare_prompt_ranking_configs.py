from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .config import parse_config
from .prompts import prompts_for_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired OpenAI CLIP and RemoteCLIP Top-K prompt-ranking configs."
        )
    )
    parser.add_argument("--base-config", required=True, help="Frozen Manual4 JSON config.")
    parser.add_argument(
        "--remoteclip-checkpoint",
        required=True,
        help="Path to RemoteCLIP OpenCLIP-format checkpoint.",
    )
    parser.add_argument(
        "--openai-checkpoint",
        default=None,
        help=(
            "Optional local OpenAI CLIP checkpoint. When omitted, OpenCLIP "
            "downloads the openai weights at runtime."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Config output directory. Defaults to the base config directory.",
    )
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--device", default=None, help="Selector device. Defaults to config device.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_prompt_ranking_configs(
    base: dict[str, Any],
    remoteclip_checkpoint: str | Path,
    *,
    openai_checkpoint: str | Path | None = None,
    model_name: str = "ViT-B-32",
    device: str | None = None,
    top_k: int = 4,
    min_score: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    ranked = copy.deepcopy(base)
    ranked["prompting"] = {
        "style": "remoteclip_b2c",
        "include_manual_prompts": True,
        "max_prompts_per_class": None,
    }

    selector = dict(ranked.get("remoteclip", {}))
    selector.update(
        {
            "enabled": True,
            "model_name": model_name,
            "device": device or str(ranked.get("device", "cuda")),
            "top_k_per_class": top_k,
            "min_score": min_score,
        }
    )

    clip_config = copy.deepcopy(ranked)
    clip_config["remoteclip"] = {
        **selector,
        "checkpoint_path": (
            str(openai_checkpoint) if openai_checkpoint is not None else None
        ),
    }

    remoteclip_config = copy.deepcopy(ranked)
    remoteclip_config["remoteclip"] = {
        **selector,
        "checkpoint_path": str(remoteclip_checkpoint),
    }

    parsed = parse_config(remoteclip_config)
    too_small = {
        spec.name: len(prompts_for_class(spec, parsed.prompting))
        for spec in parsed.classes
        if len(prompts_for_class(spec, parsed.prompting)) < top_k
    }
    if too_small:
        raise ValueError(
            f"Prompt candidate pools are smaller than top_k={top_k}: {too_small}"
        )
    return clip_config, remoteclip_config


def write_config(path: Path, config: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config)
    checkpoint_path = Path(args.remoteclip_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {checkpoint_path}")
    openai_checkpoint_path = (
        Path(args.openai_checkpoint) if args.openai_checkpoint else None
    )
    if openai_checkpoint_path is not None and not openai_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OpenAI CLIP checkpoint not found: {openai_checkpoint_path}"
        )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    clip_config, remoteclip_config = build_prompt_ranking_configs(
        base,
        checkpoint_path,
        openai_checkpoint=openai_checkpoint_path,
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    output_dir = Path(args.output_dir) if args.output_dir else base_path.parent
    clip_path = output_dir / f"potsdam_patches_config_clip_ranked{args.top_k}.json"
    remoteclip_path = (
        output_dir / f"potsdam_patches_config_remoteclip_ranked{args.top_k}.json"
    )
    write_config(clip_path, clip_config, args.overwrite)
    write_config(remoteclip_path, remoteclip_config, args.overwrite)

    parsed = parse_config(remoteclip_config)
    print("OpenAI CLIP config:", clip_path)
    print("RemoteCLIP config:", remoteclip_path)
    print("Top-K per class:", parsed.remoteclip.top_k_per_class)
    print("Candidate prompts:")
    for spec in parsed.classes:
        candidates = prompts_for_class(spec, parsed.prompting)
        print(f"  {spec.name}: {len(candidates)}")


if __name__ == "__main__":
    main()
