"""Shared preprocessing configuration (single source of truth).

Dataset facts (B1 + radiometry investigation, read-only):
  input = Sentinel-1 Sigma0 in dB, 2 bands (polarization set {VV, VH},
  per-band order UNKNOWN), float32, EPSG:4326, already calibrated by authors.
Consequence: calibration is SKIPPED by default (no double-calibration);
  Lee filtering runs in linear domain with dB round-trip.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "random_seed": 42,
    # --- radiometry: input already Sigma0 dB -> skip calibration ---
    "calibration_applied": False,
    "calibration_type": "none (input is author-calibrated Sigma0 dB; "
                        "see processed_dataset/reports/sar_radiometry_report.txt)",
    "source_units": "sigma0_dB",
    "output_units": "sigma0_dB",
    "db_epsilon": 1e-10,
    # --- Lee despeckling (linear domain) ---
    "LEE_FILTER_ENABLED": True,
    "LEE_WINDOW_SIZE": 5,          # allowed: 3, 5, 7
    "LEE_ALLOWED_WINDOWS": [3, 5, 7],
    "LEE_NOISE_VAR": None,         # None = estimate per-band (see lee_filter.py)
    # --- normalization (robust percentile, computed on dB AFTER Lee) ---
    "NORMALIZATION_ENABLED": True,
    "NORMALIZATION_METHOD": "percentile",
    "NORMALIZATION_LOW": 1,
    "NORMALIZATION_HIGH": 99,
    "NORMALIZATION_BOUNDS": None,  # frozen global train bounds (see fit_global_bounds);
                                   # None = unfitted; process_scene() refuses to run without them
    # --- channels: preserve source bands, never fabricate ---
    "channels": ["band1 (VV or VH - UNKNOWN order)", "band2 (VH or VV - UNKNOWN order)"],
    "n_channels": 2,
    # --- tiling ---
    "TILE_SIZE": 256,
    "TILE_STRIDE": 192,
    # --- tile selection: every oil tile retained; background kept at BG_KEEP_FRACTION
    # (deterministic seeded subsample when < 1.0); thresholds explicit, recorded per tile
    "MIN_OIL_FRACTION": 0.0,       # oil tiles are NEVER rejected for low fraction
    "BG_KEEP_FRACTION": 1.0,       # 1.0 = keep all background tiles
    "BG_SEED": 42,
    # --- augmentation (train only; geometric, mask = nearest/exact) ---
    "AUG_HFLIP": True,
    "AUG_VFLIP": True,
    "AUG_ROT90": True,             # 90/180/270 via exact rot90 (no interpolation)
    "AUG_HFLIP_P": 0.5,
    "AUG_VFLIP_P": 0.5,
    "AUG_ROT_CHOICES": [0, 1, 2, 3],
    "AUG_SMALL_ROTATIONS": False,  # off: would need interpolation (mask risk)
    "AUG_SPECKLE_INJECTION": False,
    "AUG_SPECKLE_STD": 0.05,
    # --- stitching ---
    "STITCH_OVERLAP": "probability_average",
    "PRED_THRESHOLD": 0.5,         # applied AFTER reconstruction
}


def validate(cfg):
    errs = []
    if cfg.get("LEE_WINDOW_SIZE") not in cfg.get("LEE_ALLOWED_WINDOWS", [3, 5, 7]):
        errs.append(f"invalid LEE_WINDOW_SIZE={cfg.get('LEE_WINDOW_SIZE')} (allowed 3/5/7)")
    lo, hi = cfg.get("NORMALIZATION_LOW"), cfg.get("NORMALIZATION_HIGH")
    if not (0 <= lo < hi <= 100):
        errs.append(f"invalid normalization percentiles low={lo} high={hi}")
    nb = cfg.get("NORMALIZATION_BOUNDS")
    if nb is not None:
        try:
            for b in ("band_0", "band_1"):
                p1, p99 = nb[b]["p1"], nb[b]["p99"]
                if not (np.isfinite(p1) and np.isfinite(p99) and p99 > p1):
                    raise KeyError
        except (KeyError, TypeError):
            errs.append("NORMALIZATION_BOUNDS must hold band_0/band_1 {p1,p99} with p99>p1")
    if cfg.get("TILE_SIZE", 0) <= 0 or cfg.get("TILE_STRIDE", 0) <= 0:
        errs.append("TILE_SIZE/TILE_STRIDE must be positive")
    if cfg.get("TILE_STRIDE", 0) > cfg.get("TILE_SIZE", 0):
        errs.append("TILE_STRIDE > TILE_SIZE leaves coverage gaps")
    if cfg.get("n_channels") != 2:
        errs.append("this dataset provides exactly 2 SAR bands; channel fabrication is forbidden")
    for k in ("AUG_HFLIP_P", "AUG_VFLIP_P"):
        p = cfg.get(k)
        if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            errs.append(f"{k} must be in [0,1], got {p!r}")
    if not set(cfg.get("AUG_ROT_CHOICES", [])) <= {0, 1, 2, 3}:
        errs.append(f"AUG_ROT_CHOICES must be subset of {{0,1,2,3}}")
    if not (0.0 <= cfg.get("MIN_OIL_FRACTION", -1) <= 1.0):
        errs.append("MIN_OIL_FRACTION must be in [0,1]")
    if not (0.0 < cfg.get("BG_KEEP_FRACTION", 0) <= 1.0):
        errs.append("BG_KEEP_FRACTION must be in (0,1]")
    if errs:
        raise ValueError("preprocessing config invalid: " + "; ".join(errs))
    return dict(cfg)


def with_overrides(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    return validate(cfg)


def ablation_configs():
    """Experiment A/B/C/D: identical except Lee window (None/3/5/7)."""
    out = {"A_no_lee": with_overrides(LEE_FILTER_ENABLED=False)}
    for w, tag in ((3, "B"), (5, "C"), (7, "D")):
        out[f"{tag}_lee{w}"] = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=w)
    return out
