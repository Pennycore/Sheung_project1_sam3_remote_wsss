from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear embedding used by SegFormer decode heads."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SegFormerHead(nn.Module):
    """Lightweight SegFormer decoder head.

    Adapted from `SegFormer-master/mmseg/models/decode_heads/segformer_head.py`
    without mmcv/mmseg dependencies. It expects four feature maps ordered from
    high resolution to low resolution: C1, C2, C3, C4.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        embed_dim: int = 256,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"SegFormerHead expects four feature levels, got {len(in_channels)}")

        self.in_channels = tuple(int(c) for c in in_channels)
        self.embed_dim = int(embed_dim)
        self.align_corners = align_corners

        c1, c2, c3, c4 = self.in_channels
        self.linear_c4 = MLP(input_dim=c4, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embed_dim)
        self.linear_fuse = ConvBNReLU(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        if len(inputs) != 4:
            raise ValueError(f"SegFormerHead expects four feature maps, got {len(inputs)}")
        c1, c2, c3, c4 = inputs
        out_size = c1.shape[-2:]
        n = c1.shape[0]

        c4 = self._project_and_resize(self.linear_c4, c4, n, out_size)
        c3 = self._project_and_resize(self.linear_c3, c3, n, out_size)
        c2 = self._project_and_resize(self.linear_c2, c2, n, out_size)
        c1 = self._project_and_resize(self.linear_c1, c1, n, out_size)

        x = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        x = self.dropout(x)
        return self.linear_pred(x)

    def _project_and_resize(
        self,
        projector: MLP,
        feature: Tensor,
        batch_size: int,
        out_size: tuple[int, int],
    ) -> Tensor:
        h, w = feature.shape[-2:]
        x = projector(feature).permute(0, 2, 1).reshape(batch_size, self.embed_dim, h, w)
        if (h, w) != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=self.align_corners)
        return x
