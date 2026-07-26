from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from .toco_head import ASPPHead, LargeFOVHead


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = False, output_stride: int = 16) -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise ImportError(
                "To train the student model, install torchvision in the SAM3 environment."
            ) from exc

        if not hasattr(models, backbone):
            raise ValueError(f"Unknown torchvision ResNet backbone: {backbone}")
        if output_stride not in {16, 32}:
            raise ValueError("output_stride must be 16 or 32")
        if output_stride == 16 and backbone in {"resnet18", "resnet34"}:
            raise ValueError("resnet18/resnet34 only support output_stride=32 in this prototype.")

        builder = getattr(models, backbone)
        replace_stride_with_dilation = [False, False, output_stride == 16]
        try:
            resnet = builder(
                weights="DEFAULT" if pretrained else None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        except TypeError as exc:
            if output_stride != 32:
                raise RuntimeError(
                    "This torchvision version does not support output_stride=16 for ResNet."
                ) from exc
            resnet = builder(pretrained=pretrained)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = int(resnet.fc.in_features)
        expansion = self.out_channels // 512
        self.feature_channels = (64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class StudentSegmentor(nn.Module):
    """A trainable segmentation student for SAM3 pseudo labels.

    SAM3 generates pseudo labels offline. This model then learns a dense
    semantic segmentation function from those pseudo labels. The default
    decoder is a SegFormer head; ToCo heads remain available for ablations.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        head: str = "large_fov",
        pretrained_backbone: bool = False,
        large_fov_dilation: int = 5,
        output_stride: int = 16,
        segformer_embed_dim: int = 256,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = head
        self.encoder = ResNetFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained_backbone,
            output_stride=output_stride,
        )
        if head == "large_fov":
            self.decoder = LargeFOVHead(
                in_planes=self.encoder.out_channels,
                out_planes=num_classes,
                dilation=large_fov_dilation,
            )
        elif head == "aspp":
            self.decoder = ASPPHead(in_planes=self.encoder.out_channels, out_planes=num_classes)
        elif head == "segformer":
            self.decoder = SegFormerHead(
                in_channels=self.encoder.feature_channels,
                num_classes=num_classes,
                embed_dim=segformer_embed_dim,
                dropout_ratio=dropout_ratio,
            )
        else:
            raise ValueError(f"Unsupported head: {head}. Choose 'segformer', 'large_fov', or 'aspp'.")

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | dict[str, object]:
        features = self.encoder(x)
        if self.head == "segformer":
            logits = self.decoder(features)
        else:
            logits = self.decoder(features[-1])
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if return_features:
            return {"logits": logits, "features": features}
        return logits


ToCoStudent = StudentSegmentor
