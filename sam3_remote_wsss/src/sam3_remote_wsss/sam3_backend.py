from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


def _cast_floating_inputs_to_dtype(obj, dtype):
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype) if torch.is_floating_point(obj) else obj
    if isinstance(obj, tuple):
        return tuple(_cast_floating_inputs_to_dtype(v, dtype) for v in obj)
    if isinstance(obj, list):
        return [_cast_floating_inputs_to_dtype(v, dtype) for v in obj]
    if isinstance(obj, dict):
        return {k: _cast_floating_inputs_to_dtype(v, dtype) for k, v in obj.items()}
    return obj


def _install_fp32_dtype_hooks(model):
    model.float()

    def hook(module, inputs):
        dtype = None
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and torch.is_floating_point(weight):
            dtype = weight.dtype
        in_proj_weight = getattr(module, "in_proj_weight", None)
        if dtype is None and torch.is_tensor(in_proj_weight) and torch.is_floating_point(in_proj_weight):
            dtype = in_proj_weight.dtype
        if dtype is None:
            return inputs
        return _cast_floating_inputs_to_dtype(inputs, dtype)

    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, torch.nn.LayerNorm)):
            module.register_forward_pre_hook(hook)



class SAM3ImageBackend:
    """Thin adapter around the original SAM3 image processor."""

    def __init__(
        self,
        sam3_repo: str | Path,
        checkpoint_path: str | Path | None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ) -> None:
        repo = Path(sam3_repo)
        if not repo.exists():
            raise FileNotFoundError(f"SAM3 repo not found: {repo}")
        sys.path.insert(0, str(repo))

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            device=device,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            load_from_HF=checkpoint_path is None,
            eval_mode=True,
        )
        model = model.float()
        self.model = model
        self.device = str(device)
        self.autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        self._wrap_backbone_outputs_as_fp32()

        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _wrap_backbone_outputs_as_fp32(self) -> None:
        def to_fp32(value):
            if torch.is_tensor(value):
                return value.float() if torch.is_floating_point(value) else value
            if isinstance(value, dict):
                return {k: to_fp32(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_fp32(v) for v in value]
            if isinstance(value, tuple):
                return tuple(to_fp32(v) for v in value)
            return value

        def wrap(fn):
            def wrapped(*args, **kwargs):
                return to_fp32(fn(*args, **kwargs))
            return wrapped

        self.model.backbone.forward_image = wrap(self.model.backbone.forward_image)
        self.model.backbone.forward_text = wrap(self.model.backbone.forward_text)

    def predict_text(self, image_rgb: np.ndarray, prompt: str) -> dict[str, Any]:
        return self.predict_texts(image_rgb, [prompt])[0]

    def predict_texts(self, image_rgb: np.ndarray, prompts: list[str]) -> list[dict[str, Any]]:
        pil_image = Image.fromarray(image_rgb, mode="RGB")
        with torch.inference_mode(), torch.amp.autocast(device_type=self.autocast_device, enabled=False):
            state = self.processor.set_image(pil_image)
            outputs = []
            for prompt in prompts:
                output = self.processor.set_text_prompt(prompt=prompt, state=state)
                outputs.append(
                    {
                        "prompt": prompt,
                        "masks": _to_numpy(output.get("masks")),
                        "masks_logits": _to_numpy(output.get("masks_logits")),
                        "boxes": _to_numpy(output.get("boxes")),
                        "scores": _to_numpy(output.get("scores")),
                    }
                )
            return outputs


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
