from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn
import torch

from ..student.model import ResNetFeatureExtractor


class CAMClassifier(nn.Module):
    """Multi-label classifier whose final 1x1 convolution produces CAMs."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained_backbone: bool = False,
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = ResNetFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained_backbone,
            output_stride=output_stride,
        )
        self.classifier = nn.Conv2d(
            self.encoder.out_channels,
            num_classes,
            kernel_size=1,
            bias=False,
        )
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)

    def forward_cam(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x)[-1])

    def forward(self, x: Tensor, return_cams: bool = False) -> Tensor | dict[str, Tensor]:
        cams = self.forward_cam(x)
        logits = cams.mean(dim=(-2, -1))
        if return_cams:
            return {"logits": logits, "cams": cams}
        return logits


def load_cam_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CAMClassifier, dict]:
    checkpoint = _torch_load(checkpoint_path, device)
    model_args = checkpoint.get("model_args", {})
    class_ids = checkpoint.get("class_ids")
    if not class_ids:
        raise ValueError("CAM checkpoint is missing class_ids")
    model = CAMClassifier(
        num_classes=len(class_ids),
        backbone=str(model_args.get("backbone", "resnet50")),
        pretrained_backbone=False,
        output_stride=int(model_args.get("output_stride", 16)),
    )
    state = checkpoint.get("model")
    if state is None:
        raise ValueError("CAM checkpoint is missing model weights")
    model.load_state_dict(_strip_module_prefix(state))
    model.to(device)
    model.eval()
    return model, checkpoint


def load_encoder_from_cam_checkpoint(
    encoder: nn.Module,
    checkpoint_path: str | Path,
) -> dict:
    checkpoint = _torch_load(checkpoint_path, "cpu")
    state = checkpoint.get("model")
    if state is None:
        raise ValueError("CAM checkpoint is missing model weights")
    state = _strip_module_prefix(state)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("CAM checkpoint does not contain encoder weights")
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "CAM encoder checkpoint is incompatible with the student encoder: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def _strip_module_prefix(state: dict[str, Tensor]) -> dict[str, Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _torch_load(path: str | Path, device: str | torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
