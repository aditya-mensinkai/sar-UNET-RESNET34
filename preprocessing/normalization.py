"""Normalization (masks must NEVER enter this module).

Robust percentile normalization on dB values (post-Lee):
    norm = clip((x - P_low) / (P_high - P_low), 0, 1)

Two roles (do not confuse):
  * fit_bounds(db, ...): single-scene bounds, for QC/display ONLY
    (preprocessing.visualize). NEVER for train/val/test/inference scaling.
  * fit_global_bounds(paths, ...): GLOBAL P1/P99 over the TRAIN SPLIT ONLY,
    via a deterministic seeded reservoir over post-Lee dB pixels
    (same chain as pipeline.process_scene). Saved to config as
    "normalization_bounds" and consumed frozen by normalization.apply().
"""
from __future__ import annotations

import numpy as np


def fit_bounds(db, low=1.0, high=99.0, stride=4):
    x = np.asarray(db, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite dB input")
    if not (0 <= low < high <= 100):
        raise ValueError(f"invalid percentile range low={low} high={high}")
    per_band = []
    for c in range(x.shape[0]):
        s = x[c][::stride, ::stride].ravel()
        lo, hi = float(np.percentile(s, low)), float(np.percentile(s, high))
        if not hi > lo:
            raise ValueError(f"degenerate percentile bounds band {c}: [{lo}, {hi}]")
        per_band.append((lo, hi))
    return {"method": "percentile", "low": low, "high": high, "bands": per_band}


def apply(db, bounds):
    x = np.asarray(db, dtype=np.float64)
    outs = []
    for c, (lo, hi) in enumerate(bounds["bands"]):
        outs.append(np.clip((x[c] - lo) / (hi - lo), 0.0, 1.0))
    return np.stack(outs).astype(np.float32)


# --- global training bounds (FIX 1) -------------------------------------------
# Meaning of "global P1/P99": quantiles of the POOLED train-split pixel
# distribution (post-Lee dB, per band), estimated from a deterministic seeded
# reservoir: every train scene contributes up to `per_file` uniformly sampled
# pixels (seeded by master seed + scene index); the pool is capped at `cap`
# pixels per band (seeded downsample). Single scene in RAM at a time.
# Documented approximation: sample quantiles of a uniform pool sample, NOT
# per-scene bounds, NOT validation/test data.

def collect_scene_samples(image_path, index, per_file=4000, stride=4,
                          seed=42, lee_window=5, lee_enabled=True, eps=1e-10):
    """Post-Lee dB reservoir contribution of ONE train scene. No masks involved."""
    import rasterio
    from . import calibration, lee_filter

    with rasterio.open(image_path) as src:
        if src.count != 2:
            raise ValueError(f"expected 2 SAR bands in {image_path}")
        raw = src.read().astype(np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"NaN/Inf in {image_path}")
    cal = calibration.apply(raw, source_units="sigma0_dB")["values"]
    lin = lee_filter.db_to_linear(cal)
    if lee_enabled:
        lin = lee_filter.lee_filter(lin, window_size=lee_window)
    db = lee_filter.linear_to_db(lin, eps=eps)
    rng = np.random.RandomState(seed + int(index))
    out = []
    for c in range(db.shape[0]):
        pool = db[c][::stride, ::stride].ravel()
        pool = pool[np.isfinite(pool)]
        if pool.size > per_file:
            pool = pool[rng.choice(pool.size, size=per_file, replace=False)]
        out.append(pool.astype(np.float64))
    return out


def global_bounds_from_samples(samples, low=1.0, high=99.0, seed=42, cap=4_000_000):
    """Merge per-scene pools (lists per band) -> frozen global bounds dict."""
    if not (0 <= low < high <= 100):
        raise ValueError(f"invalid percentile range low={low} high={high}")
    rng = np.random.RandomState(seed)
    bands = []
    for c, parts in enumerate(samples):
        pool = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
        if pool.size > cap:
            pool = pool[rng.choice(pool.size, size=cap, replace=False)]
        lo, hi = float(np.percentile(pool, low)), float(np.percentile(pool, high))
        if not hi > lo:
            raise ValueError(f"degenerate global bounds band {c}: [{lo}, {hi}]")
        bands.append((lo, hi))
    return {"method": "global-percentile-reservoir", "low": low, "high": high,
            "bands": bands}


def fit_global_bounds(image_paths, low=1.0, high=99.0, seed=42, per_file=4000,
                      cap=4_000_000, stride=4, lee_window=5, lee_enabled=True,
                      eps=1e-10):
    """GLOBAL P1/P99 over TRAIN scenes only. Returns (bounds, provenance).

    bounds format: {"band_0": {"p1","p99"}, "band_1": {...}} + "bands" list for
    normalization.apply(). provenance records split/method/scenes for the config.
    """
    paths = sorted(image_paths)
    per_band = [[], []]
    for i, p in enumerate(paths):
        for c, s in enumerate(collect_scene_samples(
                p, i, per_file=per_file, stride=stride, seed=seed,
                lee_window=lee_window, lee_enabled=lee_enabled, eps=eps)):
            per_band[c].append(s)
    raw = global_bounds_from_samples(per_band, low=low, high=high, seed=seed, cap=cap)
    bounds = {"method": raw["method"], "low": low, "high": high,
              "bands": raw["bands"],
              "band_0": {"p1": raw["bands"][0][0], "p99": raw["bands"][0][1]},
              "band_1": {"p1": raw["bands"][1][0], "p99": raw["bands"][1][1]},
              "provenance": {"split": "train", "n_scenes": len(paths),
                             "per_file": per_file, "cap_per_band": cap,
                             "stride": stride, "seed": seed,
                             "lee_enabled": lee_enabled, "lee_window": lee_window}}
    return bounds, per_band
