"""Deterministic tiling / stitching with full geospatial bookkeeping.

Split-at-image-level FIRST (our split_index.csv is already image-level), then
tile each split independently -> no tile leakage by construction. Tiles carry
source id, row/col, pixel offsets, per-tile affine transform, CRS, and source
dims. Stitching averages overlapping probabilities, reconstructs full size,
and thresholds only after reconstruction. CRS/transform are required inputs -
never fabricated (ValueError if missing).
"""
from __future__ import annotations

import numpy as np


def tile_grid(h, w, size=256, stride=192):
    if size <= 0 or stride <= 0:
        raise ValueError("TILE_SIZE/TILE_STRIDE must be positive")
    if stride > size:
        raise ValueError("TILE_STRIDE > TILE_SIZE leaves gaps")
    rs = list(range(0, max(h - size + 1, 1), stride))
    cs = list(range(0, max(w - size + 1, 1), stride))
    if rs[-1] + size < h:
        rs.append(h - size)
    if cs[-1] + size < w:
        cs.append(w - size)
    return [(r, c) for r in rs for c in cs]


def tile_image(image, source_id, transform, crs, size=256, stride=192):
    """Yield dicts: array view + geospatial bookkeeping. transform/CRS required."""
    if transform is None or crs is None:
        raise ValueError("tiling requires georeferencing (transform+CRS); refusing to fake it")
    img = np.asarray(image)
    h, w = img.shape[-2:]
    px_w = abs(transform[0]) if len(transform) >= 6 else None
    px_h = abs(transform[4]) if len(transform) >= 6 else None
    for n, (r, c) in enumerate(tile_grid(h, w, size, stride)):
        sl = (..., slice(r, r + size), slice(c, c + size)) if img.ndim == 3 else (slice(r, r + size), slice(c, c + size))
        t = list(transform)
        t[2] = transform[2] + c * transform[0]  # xoff shift (affine a,e order assumed)
        t[5] = transform[5] + r * transform[4]
        yield {"source_id": source_id, "tile_index": n, "row": r // stride, "col": c // stride,
               "offset_y": r, "offset_x": c, "array": np.array(img[sl], copy=True),
               "transform": tuple(t), "crs": str(crs), "source_shape": (h, w)}


def stitch(tiles, shape, size=256, method="probability_average", threshold=0.5):
    """Reconstruct (H, W) probabilities by overlap averaging, then threshold."""
    if method != "probability_average":
        raise ValueError(f"unsupported stitch method {method!r}")
    h, w = shape
    acc = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    for t in tiles:
        r, c = t["offset_y"], t["offset_x"]
        p = np.asarray(t["prob"], dtype=np.float64)
        if p.shape != (size, size):
            raise ValueError(f"tile prob shape {p.shape} != {(size, size)}")
        acc[r:r + size, c:c + size] += p
        cnt[r:r + size, c:c + size] += 1.0
    if np.any(cnt == 0):
        raise ValueError("tiling left uncovered pixels (stride > size?)")
    prob = acc / cnt
    return {"probability": prob.astype(np.float32),
            "mask": (prob >= threshold).astype(np.uint8)}
