"""Sliding-window origin generation for full-scene inference.

Convention everywhere:
  arrays are indexed [row=y, col=x]
  rasterio Window(col_off=x, row_off=y, width, height)

window_origins(length, tile, stride) -> list[int]
  Starts are in [0, length-tile], sorted, unique, first=0, last=length-tile.
  If length < tile, returns [0] (the caller pads the preprocessed array).

tile_grid(height, width, tile=512, stride=384) -> list[tuple[int,int]]
  Returns [(y0, x0), ...] for all (row, col) origin pairs, in row-major order.
"""
from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

logger = logging.getLogger(__name__)

TILE_SIZE: int = 512


# ---------------------------------------------------------------------------
# 1-D origin list
# ---------------------------------------------------------------------------

def window_origins(length: int, tile: int, stride: int) -> List[int]:
    """Sliding-window start positions for one spatial dimension.

    Parameters
    ----------
    length : scene dimension (pixels)
    tile   : tile size (pixels)
    stride : step between starts; must satisfy 1 <= stride <= tile

    Returns
    -------
    Sorted, unique list of start positions.
      - First element is always 0.
      - Last element is always ``length - tile`` (or 0 if length <= tile).
      - Every consecutive gap is <= stride (full coverage guaranteed).
      - If length <= tile, returns [0]; the caller is responsible for padding.
    """
    if stride < 1 or stride > tile:
        raise ValueError(
            f"stride must satisfy 1 <= stride <= tile, got stride={stride} tile={tile}"
        )
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")

    if length <= tile:
        # Scene fits in one window; caller handles padding if length < tile
        return [0]

    starts: List[int] = list(range(0, length - tile + 1, stride))

    # Ensure the last window ends exactly at length (no missed edge pixels)
    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    return starts


# ---------------------------------------------------------------------------
# 2-D origin grid
# ---------------------------------------------------------------------------

def tile_grid(
    height: int,
    width: int,
    tile: int = TILE_SIZE,
    stride: int = 384,
) -> List[Tuple[int, int]]:
    """Full 2-D grid of (y, x) window origins.

    Parameters
    ----------
    height, width : scene spatial dimensions (rows, cols)
    tile   : square tile side (pixels); default 512
    stride : step in both dimensions; default 384

    Returns
    -------
    List of (y, x) origin tuples in row-major order (y outer, x inner).
    Emits a warning if either scene dimension is < tile (padding required).
    """
    if stride < 1 or stride > tile:
        raise ValueError(
            f"stride must satisfy 1 <= stride <= tile, got stride={stride} tile={tile}"
        )
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")

    if height < tile or width < tile:
        warnings.warn(
            f"Scene ({height}x{width}) is smaller than tile ({tile}x{tile}); "
            "the preprocessed array will be padded with reflect padding before "
            "inference and cropped back to the original size.",
            stacklevel=2,
        )

    y_starts = window_origins(height, tile, stride)
    x_starts = window_origins(width, tile, stride)

    return [(y, x) for y in y_starts for x in x_starts]
