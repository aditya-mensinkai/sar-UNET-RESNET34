"""Train-only geometric augmentation (numpy-only, interpolation-free).

Image and mask ALWAYS receive the identical transform. All ops are exact
(flips / rot90) so masks stay binary with plain nearest semantics - no
interpolation touches either array, hence no RGB-style or resampling artifacts.
Validation/test must call augment() with enable=False (identity).
Optional: controlled multiplicative speckle injection on the image only.
"""
from __future__ import annotations

import numpy as np


def _check_pair(image, mask):
    img = np.asarray(image)
    m = np.asarray(mask)
    if img.shape[-2:] != m.shape[-2:]:
        raise ValueError(f"image/mask spatial mismatch {img.shape} vs {m.shape}")
    u = np.unique(m)
    if not set(u.tolist()) <= {0, 1}:
        raise ValueError(f"mask must be binary {{0,1}}, got unique={u.tolist()}")
    return img, m


def augment(image, mask, rng, hflip=True, vflip=True, rot90=True,
            speckle=False, speckle_std=0.05, enable=True):
    """Return (image_aug, mask_aug). Deterministic given rng; identity if not enable."""
    img, m = _check_pair(image, mask)
    img, m = np.array(img, copy=True), np.array(m, copy=True)
    if not enable:
        return img, m
    if hflip and bool(rng.random() < 0.5):
        img, m = img[..., ::-1].copy(), m[..., ::-1].copy()
    if vflip and bool(rng.random() < 0.5):
        img, m = img[:, ::-1, :].copy(), m[::-1, :].copy()
    if rot90:
        k = int(rng.integers(0, 4))
        if k:
            img = np.rot90(img, k, axes=(-2, -1)).copy()
            m = np.rot90(m, k).copy()
    if speckle:
        n = rng.normal(1.0, speckle_std, size=img.shape)
        img = (img * n).astype(img.dtype, copy=False)
    u = np.unique(m)
    if not set(u.tolist()) <= {0, 1}:
        raise ValueError(f"augmentation broke mask binarity: {u.tolist()}")
    return img, m
