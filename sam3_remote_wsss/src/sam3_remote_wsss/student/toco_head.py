from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class LargeFOVHead(nn.Module):
    """ToCo LargeFOV segmentation head.

    This is adapted from `ToCo-main/model/decoder/conv_head.py`. The structure
    is intentionally kept the same while making initialization safe for
    bias-free convolutions.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        dilation: int = 5,
        embed_dim: int = 512,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dilation = dilation
        self.conv6 = conv3x3(in_planes, embed_dim, padding=dilation, dilation=dilation)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = conv3x3(embed_dim, embed_dim, padding=dilation, dilation=dilation)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = conv1x1(embed_dim, out_planes, padding=0)
        if init_weights:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return self.conv8(x)


class ASPPHead(nn.Module):
    """ToCo ASPP-style segmentation head."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        atrous_rates: Sequence[int] = (6, 12, 18, 24),
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=1,
                    padding=rate,
                    dilation=rate,
                    bias=True,
                )
                for rate in atrous_rates
            ]
        )
        if init_weights:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return sum(stage(x) for stage in self.stages)


LargeFOV = LargeFOVHead
ASPP = ASPPHead
