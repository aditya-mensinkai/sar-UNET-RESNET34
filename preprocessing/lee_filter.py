"""Lee despeckling for multiplicative speckle (numpy-only, memory-efficient).

Operates STRICTLY in the linear intensity/power domain. dB round-trip helpers
provided; filtering dB values directly is refused by design (filter() expects
linear >= 0 input).

  local_mean = box_mean(x, w); local_var = box_mean(x^2) - mean^2
  noise_var  = estimated (default: mean of local_var over homogeneous pixels,
               homogeneous = local_var <= median(local_var)) or user fixed value
  W = local_var * (local_var - noise_var) / (local_var^2 + eps), clipped [0, 1]
      (W -> 0 in flat areas = full smoothing; W -> 1 on edges/points = preserved)
  out = local_mean + W * (x - local_mean)
Box means use an integral image (reflect padding) -> O(1) per pixel per band.
"""
from __future__ import annotations

import numpy as np

ALLOWED_WINDOWS = (3, 5, 7)
DB_EPS = 1e-10


def db_to_linear(db, eps=DB_EPS):
    """Physical conversion linear = 10 ** (dB / 10). No additive floor.

    `eps` is retained in the signature for backward compatibility only and is
    intentionally NOT added: adding it biased the extreme low tail (real data
    reaches -128 dB). Zero-protection lives solely in linear_to_db().
    """
    db = np.asarray(db, dtype=np.float64)
    if not np.all(np.isfinite(db)):
        raise ValueError("non-finite dB input")
    return 10.0 ** (db / 10.0)


def linear_to_db(linear, eps=DB_EPS):
    lin = np.asarray(linear, dtype=np.float64)
    if not np.all(np.isfinite(lin)):
        raise ValueError("non-finite linear input")
    if np.any(lin < 0):
        raise ValueError("linear intensity must be >= 0")
    return 10.0 * np.log10(np.maximum(lin, eps))


def _box_mean(x, w):
    if w not in ALLOWED_WINDOWS:
        raise ValueError(f"invalid Lee window {w} (allowed 3/5/7)")
    p = w // 2
    xp = np.pad(x, p, mode="reflect")
    ii = np.zeros((xp.shape[0] + 1, xp.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    return (ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]) / (w * w)


def estimate_noise_var(local_var):
    """Homogeneous-pixel estimate: mean local variance below the median."""
    lv = np.asarray(local_var, dtype=np.float64).ravel()
    lv = lv[np.isfinite(lv)]
    if lv.size == 0:
        raise ValueError("empty input to noise estimation")
    homo = lv[lv <= np.median(lv)]
    return float(homo.mean()) if homo.size else float(lv.mean())


def lee_filter(linear, window_size=5, noise_var=None):
    """Filter one (H, W) or (C, H, W) linear-intensity array. Returns float64."""
    x = np.asarray(linear, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite linear input (NaN/Inf rejected)")
    if np.any(x < 0):
        raise ValueError("Lee filter requires linear intensity >= 0 (convert dB first)")
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"expected (H,W) or (C,H,W), got {x.shape}")
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        b = x[c]
        mean = _box_mean(b, window_size)
        var = np.maximum(_box_mean(b * b, window_size) - mean * mean, 0.0)
        nv = estimate_noise_var(var) if noise_var is None else float(noise_var)
        w = np.clip(var * (var - nv) / (var * var + 1e-12), 0.0, 1.0)
        out[c] = mean + w * (b - mean)
    return out[0] if squeeze else out
