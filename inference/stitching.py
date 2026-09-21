"""Probability stitcher for full-scene inference.

ProbabilityStitcher accumulates per-tile float32 probability maps into a
full-scene map by uniform averaging (sum / count).

Rules enforced structurally:
  - Thresholding is NEVER performed here. Only finalize() returns probabilities;
    the binary mask is produced externally from the probability map.
  - No per-tile predictions are written to disk.
  - Shape and bounds violations raise immediately; they are never silently clipped.
  - finalize() raises if any pixel was not covered (count == 0).

Usage::

    stitcher = ProbabilityStitcher(H, W, tile=512)
    for y, x, prob_tile in generator:
        stitcher.add_tile(y, x, prob_tile)
    prob_map = stitcher.finalize()          # float32 (H, W), values in [0, 1]
    binary   = (prob_map >= threshold).astype(np.uint8)   # done externally

Convenience wrapper::

    prob_map = stitch(H, W, tile, iterable_of_(y, x, prob))
"""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


class ProbabilityStitcher:
    """Accumulate per-tile probability maps by uniform averaging.

    Parameters
    ----------
    height, width : full scene spatial dimensions (rows, cols)
    tile          : tile side length (pixels); tiles are square
    """

    def __init__(self, height: int, width: int, tile: int = 512) -> None:
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}")
        if tile <= 0:
            raise ValueError(f"tile must be positive, got {tile}")

        self._H = height
        self._W = width
        self._T = tile

        # float32 accumulator — no threshold applied here, ever
        self._sum_prob: np.ndarray = np.zeros((height, width), dtype=np.float32)
        # uint16 counter — supports up to 65535 overlapping tiles per pixel
        self._count: np.ndarray = np.zeros((height, width), dtype=np.uint16)

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def add_tile(self, y: int, x: int, prob: np.ndarray) -> None:
        """Add one tile's probability map at origin (y, x).

        Parameters
        ----------
        y, x : top-left corner of the tile in scene coordinates (row, col)
        prob : float32 array of shape (tile, tile) with values in [0, 1]

        Raises
        ------
        ValueError  if prob shape != (tile, tile)
        ValueError  if the tile footprint is wholly or partly outside the scene
        """
        t = self._T

        # Shape check
        if prob.shape != (t, t):
            raise ValueError(
                f"add_tile: expected prob shape ({t}, {t}), got {prob.shape}"
            )

        # Bounds check — raise, never clip
        if y < 0 or x < 0 or y + t > self._H or x + t > self._W:
            raise ValueError(
                f"add_tile: tile at (y={y}, x={x}) with size {t} falls outside "
                f"scene ({self._H}x{self._W}); "
                f"window [y:{y}:{y+t}, x:{x}:{x+t}]"
            )

        self._sum_prob[y : y + t, x : x + t] += prob.astype(np.float32)
        self._count[y : y + t, x : x + t] += 1

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> np.ndarray:
        """Compute and return the averaged probability map.

        Returns
        -------
        float32 ndarray of shape (H, W) with values in [0, 1].

        Raises
        ------
        AssertionError  if any pixel has count == 0 (coverage hole)
        AssertionError  if the result contains non-finite values
        AssertionError  if any value is outside [0, 1]
        """
        n_uncovered = int(np.sum(self._count == 0))
        if n_uncovered > 0:
            raise AssertionError(
                f"finalize: {n_uncovered} pixel(s) were never covered by any tile "
                f"(count == 0). Every pixel must receive at least one tile before "
                f"finalize() is called."
            )

        # Safe division — count.min() >= 1 guaranteed above
        prob = self._sum_prob / self._count.astype(np.float32)

        # Sanity assertions on the result
        if not np.all(np.isfinite(prob)):
            raise AssertionError(
                "finalize: probability map contains non-finite values (NaN/Inf); "
                "this indicates corrupted model output or a numerical overflow."
            )
        if prob.min() < 0.0 or prob.max() > 1.0:
            raise AssertionError(
                f"finalize: probability values outside [0, 1]: "
                f"min={prob.min():.6f}, max={prob.max():.6f}. "
                "Ensure sigmoid is applied exactly once before add_tile()."
            )
        if prob.shape != (self._H, self._W):
            raise AssertionError(
                f"finalize: output shape {prob.shape} != expected ({self._H}, {self._W})"
            )

        return prob

    # ------------------------------------------------------------------
    # Properties (read-only access for testing)
    # ------------------------------------------------------------------

    @property
    def count(self) -> np.ndarray:
        """The current uint16 coverage count array (read-only view)."""
        return self._count

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._H, self._W)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def stitch(
    height: int,
    width: int,
    tile: int,
    tiles: Iterable[Tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Stitch an iterable of (y, x, prob_tile) into a probability map.

    Parameters
    ----------
    height, width : full scene dimensions
    tile          : tile side length
    tiles         : iterable yielding (y, x, prob_tile[tile,tile] float32)

    Returns
    -------
    float32 probability map of shape (height, width)
    """
    stitcher = ProbabilityStitcher(height, width, tile)
    for y, x, prob in tiles:
        stitcher.add_tile(y, x, prob)
    return stitcher.finalize()
