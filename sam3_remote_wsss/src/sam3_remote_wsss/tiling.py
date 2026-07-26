from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_overlap must be in [0, tile_size)")

    stride = tile_size - overlap
    xs = _positions(width, tile_size, stride)
    ys = _positions(height, tile_size, stride)
    return [
        Tile(x0=x, y0=y, x1=min(x + tile_size, width), y1=min(y + tile_size, height))
        for y in ys
        for x in xs
    ]


def crop_tile(image: np.ndarray, tile: Tile) -> np.ndarray:
    return image[tile.y0 : tile.y1, tile.x0 : tile.x1]


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions

