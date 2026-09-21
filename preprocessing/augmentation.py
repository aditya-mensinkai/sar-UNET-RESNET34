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


def sample_params(rng, hflip=True, vflip=True, rot90=True, speckle=False,
                  hflip_p=0.5, vflip_p=0.5, rot_choices=(0, 1, 2, 3)):
    """Sample ONE parameter set applied identically to image AND mask.

    Never sample transforms separately for image and mask. Returned dict is the
    single transformation object both arrays pass through in apply_transform().
    """
    if not (0.0 <= hflip_p <= 1.0 and 0.0 <= vflip_p <= 1.0):
        raise ValueError(f"flip probabilities must be in [0,1]: {hflip_p}, {vflip_p}")
    if rot90 and not set(rot_choices) <= {0, 1, 2, 3}:
        raise ValueError(f"rot_choices must be subset of {{0,1,2,3}}: {rot_choices}")
    return {"hflip": bool(hflip and rng.random() < hflip_p),
            "vflip": bool(vflip and rng.random() < vflip_p),
            "rot_k": int(rng.choice(list(rot_choices))) if rot90 else 0,
            "speckle": bool(speckle)}


def apply_transform(image, mask, params, speckle_std=0.05, rng=None):
    """Apply one sampled params dict to both arrays. Exact ops only (no interpolation)."""
    img, m = _check_pair(image, mask)
    img, m = np.array(img, copy=True), np.array(m, copy=True)
    if params.get("hflip"):
        img, m = img[..., ::-1].copy(), m[..., ::-1].copy()
    if params.get("vflip"):
        img, m = img[:, ::-1, :].copy(), m[::-1, :].copy()
    k = int(params.get("rot_k", 0))
    if k:
        img = np.rot90(img, k, axes=(-2, -1)).copy()
        m = np.rot90(m, k).copy()
    if params.get("speckle"):
        if rng is None:
            raise ValueError("speckle injection requires rng")
        n = rng.normal(1.0, speckle_std, size=img.shape)  # image ONLY, never mask
        img = (img * n).astype(img.dtype, copy=False)
    u = np.unique(m)
    if not set(u.tolist()) <= {0, 1}:
        raise ValueError(f"augmentation broke mask binarity: {u.tolist()} "
                         f"(params={params}) - quarantine this sample, do not round labels")
    return img, m


def augment(image, mask, rng, hflip=True, vflip=True, rot90=True,
            speckle=False, speckle_std=0.05, enable=True,
            hflip_p=0.5, vflip_p=0.5, rot_choices=(0, 1, 2, 3)):
    """Return (image_aug, mask_aug). Deterministic given rng; identity if not enable."""
    img, m = _check_pair(image, mask)
    if not enable:
        return np.array(img, copy=True), np.array(m, copy=True)
    params = sample_params(rng, hflip=hflip, vflip=vflip, rot90=rot90,
                           speckle=speckle, hflip_p=hflip_p, vflip_p=vflip_p,
                           rot_choices=rot_choices)
    return apply_transform(img, m, params, speckle_std=speckle_std, rng=rng)
