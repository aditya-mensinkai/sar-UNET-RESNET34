"""Preprocessing inspection utility (read-only wrt dataset/).

For a sample scene saves: original SAR, calibrated SAR, linear SAR,
Lee-filtered SAR, dB image, normalized image, ground-truth mask, SAR+overlay -
plus before/after-Lee statistics to catch over-smoothing, lost slick
boundaries, bad normalization, clipping, NaN/Inf, broken masks, channel errors.
Usage: python -m preprocessing.visualize --image <tif> --mask <tif> --out <png>
        [--lee-window 5] [--no-lee]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import lee_filter, normalization, calibration
from . import validate as validate_cfg


def run(image_path, mask_path, out_path, cfg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    cfg = validate_cfg(cfg or {})
    with rasterio.open(image_path) as src:
        raw = src.read().astype(np.float64)
    with rasterio.open(mask_path) as m:
        mask = m.read(1)
    cal = calibration.apply(raw, source_units=cfg["source_units"])["values"]
    lin = lee_filter.db_to_linear(cal, eps=cfg["db_epsilon"])
    lin_f = lee_filter.lee_filter(lin, window_size=cfg["LEE_WINDOW_SIZE"]) if cfg["LEE_FILTER_ENABLED"] else lin
    db_f = lee_filter.linear_to_db(lin_f, eps=cfg["db_epsilon"])
    norm = normalization.apply(db_f, normalization.fit_bounds(
        db_f, low=cfg["NORMALIZATION_LOW"], high=cfg["NORMALIZATION_HIGH"]))

    def stretch(x):
        f = x[np.isfinite(x)]
        lo, hi = np.percentile(f, (2, 98))
        return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)

    panels = [("1 original SAR b1", stretch(raw[0]), "gray"),
              ("2 calibrated SAR b1", stretch(cal[0]), "gray"),
              ("3 linear SAR b1", stretch(lin[0]), "gray"),
              ("4 Lee-filtered linear b1", stretch(lin_f[0]), "gray"),
              ("5 dB post-Lee b1", stretch(db_f[0]), "gray"),
              ("6 normalized b1", norm[0], "gray"),
              ("7 ground-truth mask", mask, "gray")]
    ov = np.dstack([stretch(db_f[0])] * 3)
    ov[mask > 0] = [1, 0, 0]
    panels.append(("8 SAR+mask overlay", ov, None))
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{os.path.basename(image_path)}  lee={cfg['LEE_WINDOW_SIZE'] if cfg['LEE_FILTER_ENABLED'] else 'OFF'}", fontsize=10)
    for a, (t, im, cm) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=cm) if cm else a.imshow(im)
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    stats = {}
    for k, v in (("linear_pre_lee_b1", lin[0]), ("linear_post_lee_b1", lin_f[0]),
                 ("db_post_lee_b1", db_f[0]), ("normalized_b1", norm[0].astype(np.float64))):
        stats[k] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                    "std": float(v.std()), "median": float(np.median(v)),
                    "P1": float(np.percentile(v, 1)), "P99": float(np.percentile(v, 99))}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lee-window", type=int, default=5)
    ap.add_argument("--no-lee", action="store_true")
    ns = ap.parse_args()
    from . import with_overrides
    cfg = with_overrides(LEE_WINDOW_SIZE=ns.lee_window, LEE_FILTER_ENABLED=not ns.no_lee)
    import json
    print(json.dumps(run(ns.image, ns.mask, ns.out, cfg), indent=1))


if __name__ == "__main__":
    main()
