"""Pure-NumPy morphological operations for binary masks.

scipy is NOT available in this project (not in requirements.txt), so all
operations are implemented using numpy only.  For a square structuring
element, erosion and dilation are performed efficiently with a 2-D sliding
window implemented via numpy.lib.stride_tricks.

Supported operations:
  binary_erode   -- remove pixels that don't fit the structuring element
  binary_dilate  -- add pixels adjacent to the structuring element
  binary_open    -- erosion → dilation (removes small bright noise)
  binary_close   -- dilation → erosion (fills small dark holes)
  morphological_cleanup -- configurable opening + closing

All functions accept / return uint8 arrays with values in {0, 1}.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided


# ---------------------------------------------------------------------------
# Core: sliding-window view for square structuring element
# ---------------------------------------------------------------------------

def _sliding_window(arr: np.ndarray, k: int) -> np.ndarray:
    """Return a view of shape (H-k+1, W-k+1, k, k) for a 2-D array.

    This is a zero-copy view.  The caller is responsible for not writing into
    the returned array.
    """
    H, W = arr.shape
    oh = H - k + 1
    ow = W - k + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f"Array ({H}x{W}) is too small for kernel size {k}x{k}."
        )
    shape = (oh, ow, k, k)
    strides = (arr.strides[0], arr.strides[1], arr.strides[0], arr.strides[1])
    return as_strided(arr, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# Binary erosion
# ---------------------------------------------------------------------------

def binary_erode(mask: np.ndarray, k: int) -> np.ndarray:
    """Erode a binary (uint8 {0,1}) mask with a k×k square structuring element.

    A foreground pixel survives only if ALL pixels in its k×k neighbourhood
    are foreground.

    Parameters
    ----------
    mask : uint8 array of shape (H, W) with values {0, 1}
    k    : kernel side length (positive odd integer)

    Returns
    -------
    uint8 array of shape (H, W)
    """
    if k < 1:
        raise ValueError(f"kernel size must be >= 1, got {k}")
    if k == 1:
        return mask.copy().astype(np.uint8)

    pad = k // 2
    # Pad with zeros (background) so that output has the same size as input
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
    windows = _sliding_window(padded, k)          # (H, W, k, k)
    # Erosion: foreground iff every pixel in window is foreground
    out = windows.min(axis=(2, 3)).astype(np.uint8)
    assert out.shape == mask.shape, f"erosion shape mismatch {out.shape} vs {mask.shape}"
    return out


# ---------------------------------------------------------------------------
# Binary dilation
# ---------------------------------------------------------------------------

def binary_dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Dilate a binary (uint8 {0,1}) mask with a k×k square structuring element.

    A background pixel becomes foreground if ANY pixel in its k×k neighbourhood
    is foreground.

    Parameters
    ----------
    mask : uint8 array of shape (H, W) with values {0, 1}
    k    : kernel side length (positive odd integer)

    Returns
    -------
    uint8 array of shape (H, W)
    """
    if k < 1:
        raise ValueError(f"kernel size must be >= 1, got {k}")
    if k == 1:
        return mask.copy().astype(np.uint8)

    pad = k // 2
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
    windows = _sliding_window(padded, k)          # (H, W, k, k)
    # Dilation: foreground iff any pixel in window is foreground
    out = windows.max(axis=(2, 3)).astype(np.uint8)
    assert out.shape == mask.shape, f"dilation shape mismatch {out.shape} vs {mask.shape}"
    return out


# ---------------------------------------------------------------------------
# Opening and closing
# ---------------------------------------------------------------------------

def binary_open(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary morphological opening: erosion → dilation.

    Removes isolated foreground pixels and small bright noise whose spatial
    extent is smaller than k pixels.
    """
    if k <= 1:
        return mask.copy().astype(np.uint8)
    return binary_dilate(binary_erode(mask, k), k)


def binary_close(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary morphological closing: dilation → erosion.

    Fills small holes in foreground regions whose spatial extent is smaller
    than k pixels.
    """
    if k <= 1:
        return mask.copy().astype(np.uint8)
    return binary_erode(binary_dilate(mask, k), k)


# ---------------------------------------------------------------------------
# Combined cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(
    binary_mask: np.ndarray,
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 3,
    enabled: bool = True,
) -> np.ndarray:
    """Apply opening then closing to a binary mask.

    Parameters
    ----------
    binary_mask          : uint8 array (H, W), values {0, 1}
    opening_kernel_size  : kernel side for opening (removes noise)
    closing_kernel_size  : kernel side for closing (fills holes)
    enabled              : if False, returns a copy of the input unchanged

    Returns
    -------
    uint8 array of shape (H, W)
    """
    arr = np.asarray(binary_mask, dtype=np.uint8)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("binary_mask must contain only {0, 1} values")

    if not enabled:
        return arr.copy()

    cleaned = binary_open(arr, opening_kernel_size)
    cleaned = binary_close(cleaned, closing_kernel_size)
    return cleaned
