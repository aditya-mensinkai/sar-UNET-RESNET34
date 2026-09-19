"""End-to-end preprocessing pipeline: inspect -> calibrate(gate) -> dB->linear ->
Lee -> linear->dB -> normalize -> (train-only augment) -> tile. Plus metrics
(IoU/Dice/Precision/Recall/F1) for the A/B/C/D Lee ablation.

Per-file streaming: one scene in RAM at a time (2048px float64 ~67 MB/band).
Saves preprocessing_config.json per run for reproducibility.
"""
from __future__ import annotations

import json
import os

import numpy as np
import rasterio

from . import calibration, lee_filter, normalization, augmentation, tiling
from . import validate as validate_cfg


def compute_metrics(pred, truth):
    """Binary IoU/Dice/Precision/Recall/F1. No predictions-as-truth anywhere."""
    p = np.asarray(pred).astype(bool)
    t = np.asarray(truth).astype(bool)
    if p.shape != t.shape:
        raise ValueError(f"pred/truth shape mismatch {p.shape} vs {t.shape}")
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    denom_iou = tp + fp + fn
    return {"IoU": tp / denom_iou if denom_iou else 1.0,
            "Dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "Precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "Recall": tp / (tp + fn) if (tp + fn) else 1.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "TP": tp, "FP": fp, "FN": fn}


def process_scene(image_path, mask_path=None, cfg=None, rng=None, split="train",
                  bounds=None):
    """Full chain for one scene. Returns dict of stage arrays + stats. Raises on bad data.

    Normalization uses FROZEN GLOBAL TRAINING bounds (normalization.apply with
    `bounds`); per-scene fitting is forbidden here. `bounds` is REQUIRED for
    every split - including train - and must come from
    normalization.fit_global_bounds over the train split (or the
    "normalization_bounds" config entry). Use fit_global_bounds() explicitly to
    (re)fit; never fit inside scene processing.
    """
    cfg = validate_cfg(cfg or {})
    if bounds is None:
        raise ValueError(
            f"frozen training normalization bounds are required (split={split!r}); "
            "compute them once with normalization.fit_global_bounds over the TRAIN "
            "split and pass bounds=... . Per-scene fit_bounds() is forbidden here.")
    if split in ("val", "test", "inference") and bounds is None:
        raise ValueError(  # unreachable guard (see above); kept explicit per spec
            f"split={split!r} must reuse frozen training bounds; bounds=None given.")
    with rasterio.open(image_path) as src:
        if src.count != cfg["n_channels"]:
            raise ValueError(f"unsupported channel count {src.count} in {image_path}")
        raw = src.read().astype(np.float64)
        meta = {"crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform) if src.transform else None,
                "width": src.width, "height": src.height}
    if raw.shape[0] != 2 or raw.dtype != np.float64:
        raise ValueError("unexpected input array")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"NaN/Inf in {image_path}")
    if raw.size == 0:
        raise ValueError(f"empty image {image_path}")
    cal = calibration.apply(raw, source_units=cfg["source_units"])
    lin = lee_filter.db_to_linear(cal["values"], eps=cfg["db_epsilon"])
    if cfg["LEE_FILTER_ENABLED"]:
        lin_f = lee_filter.lee_filter(lin, window_size=cfg["LEE_WINDOW_SIZE"],
                                      noise_var=cfg["LEE_NOISE_VAR"])
    else:
        lin_f = lin
    db_f = lee_filter.linear_to_db(lin_f, eps=cfg["db_epsilon"])
    norm = normalization.apply(db_f, bounds)
    mask = None
    if mask_path:
        with rasterio.open(mask_path) as m:
            mask = m.read(1)
        if mask.shape != raw.shape[1:]:
            raise ValueError("image/mask dimension mismatch")
        if not set(np.unique(mask).tolist()) <= {0, 1}:
            raise ValueError("mask not binary {0,1}")
    stats = {k: {"min": float(np.min(v)), "max": float(np.max(v)),
                 "mean": float(np.mean(v)), "std": float(np.std(v)),
                 "median": float(np.median(v)),
                 "P1": float(np.percentile(v, 1)), "P99": float(np.percentile(v, 99))}
             for k, v in (("linear", lin_f), ("db", db_f), ("normalized", norm))}
    out = {"calibrated_db": cal["values"], "linear": lin, "linear_filtered": lin_f,
           "db_filtered": db_f, "normalized": norm, "mask": mask,
           "bounds": bounds, "stats": stats, "meta": meta,
           "calibration": {k: cal[k] for k in ("calibration_applied", "calibration_type",
                                              "source_units", "output_units")}}
    if mask is not None and split == "train":
        if rng is None:
            raise ValueError("train split requires rng for augmentation")
        out["normalized"], out["mask"] = augmentation.augment(
            norm, mask, rng, hflip=cfg["AUG_HFLIP"], vflip=cfg["AUG_VFLIP"],
            rot90=cfg["AUG_ROT90"], speckle=cfg["AUG_SPECKLE_INJECTION"],
            speckle_std=cfg["AUG_SPECKLE_STD"], enable=True)
    return out


def save_config(cfg, path):
    cfg = validate_cfg(cfg)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path
