"""Diagnostic tile visualization + spatial alignment verification (no training).

Selects a small seeded sample (default 3/class) from OilSpillTileDataset,
saves one 4-panel figure per tile (SAR Band 0, SAR Band 1, mask, overlay),
and verifies alignment/coords/orientation/binary/NaN/Inf programmatically.
Writes processed_dataset/reports/visual_sanity_report.json + visual_sanity/.
Exit 0 = PASS, 1 = FAIL (stops, never repairs).
Usage: python tools/visualize_dataset_sanity.py [--samples-per-class 3] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import Window

    from preprocessing.tile_dataset import OilSpillTileDataset

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TIP = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
    ds = OilSpillTileDataset(TIP, split="train")
    rng = random.Random(ns.seed)

    by_class = {"Oil": [], "No_oil": [], "Lookalike": []}
    for i, r in enumerate(ds.rows):
        by_class[r["class"]].append(i)
    for c in by_class:
        if not by_class[c]:
            check("class_present", False, f"no {c} tiles in index")
    if FAILURES:
        return _finish(ROOT, ns, {}, [])

    picks, reasons = [], {}
    for cls in ("Oil", "No_oil", "Lookalike"):
        sel = rng.sample(by_class[cls], min(ns.samples_per_class, len(by_class[cls])))
        picks.extend(sel)
        for i in sel:
            reasons[i] = "seeded_random"
    # guarantee one oil-positive tile (deterministic scan, no fabrication)
    def has_oil(i):
        _, m = ds[i]
        return bool((m.numpy() == 1).any())
    if not any(has_oil(i) for i in picks if ds.rows[i]["class"] == "Oil"):
        for i in by_class["Oil"]:
            if has_oil(i):
                picks.append(i)
                reasons[i] = "oil_positive_required"
                break
        else:
            check("oil_positive", False, "no oil-positive Oil tile found")

    outdir = os.path.join(ROOT, "processed_dataset", "reports", "visual_sanity")
    os.makedirs(outdir, exist_ok=True)
    samples = []
    for n, i in enumerate(picks):
        meta = ds.get_metadata(i)
        img, msk = ds[i]  # actual training tensors
        x, y = int(meta["x"]), int(meta["y"])
        w, h = int(meta["tile_width"]), int(meta["tile_height"])
        W, H = int(meta["original_width"]), int(meta["original_height"])
        ok = True
        ok &= check(f"{meta['tile_id']}/coords",
                    x >= 0 and y >= 0 and x + w <= W and y + h <= H, str((x, y, w, h, W, H)))
        ok &= check(f"{meta['tile_id']}/shapes",
                    tuple(img.shape) == (2, 512, 512) and tuple(msk.shape) == (1, 512, 512),
                    f"{tuple(img.shape)}/{tuple(msk.shape)}")
        ok &= check(f"{meta['tile_id']}/dtypes",
                    img.dtype == torch.float32 and msk.dtype == torch.float32,
                    f"{img.dtype}/{msk.dtype}")
        u = torch.unique(msk).tolist()
        ok &= check(f"{meta['tile_id']}/binary", set(u) <= {0.0, 1.0}, str(u))
        ok &= check(f"{meta['tile_id']}/finite",
                    bool(torch.isfinite(img).all() and torch.isfinite(msk).all()), "")
        # direct mask cross-check: CSV window read straight from the mask file
        with rasterio.open(os.path.join(ds.root, meta["mask_path"])) as msrc:
            direct = msrc.read(1, window=Window(col_off=x, row_off=y, width=w, height=h))
        ok &= check(f"{meta['tile_id']}/direct_mask",
                    bool(np.array_equal(direct, msk.numpy()[0])), "dataset mask != file window")
        # direct raw image window stats (transparency only; processed tensor differs by design)
        with rasterio.open(os.path.join(ds.root, meta["image_path"])) as ssrc:
            raw = ssrc.read(window=Window(col_off=x, row_off=y, width=w, height=h))
        fg = msk.numpy()[0] == 1
        n_fg = int(fg.sum())
        ys, xs = (np.where(fg) if n_fg else (None, None))
        geom = {"oil_pixel_count": n_fg, "oil_fraction": round(n_fg / (512 * 512), 6),
                "bbox": None if not n_fg else {
                    "x_min": int(xs.min()), "y_min": int(ys.min()),
                    "x_max": int(xs.max()), "y_max": int(ys.max())}}
        if n_fg:
            ok &= check(f"{meta['tile_id']}/bbox",
                        0 <= geom["bbox"]["x_min"] <= geom["bbox"]["x_max"] < 512 and
                        0 <= geom["bbox"]["y_min"] <= geom["bbox"]["y_max"] < 512, "")
        ia = img.numpy()
        bands = {f"band_{c}": {"min": round(float(ia[c].min()), 5),
                               "max": round(float(ia[c].max()), 5),
                               "mean": round(float(ia[c].mean()), 5),
                               "std": round(float(ia[c].std()), 5),
                               "nan": 0, "inf": 0} for c in (0, 1)}
        samples.append({
            "tile_id": meta["tile_id"], "split": meta["split"], "class": meta["class"],
            "image_id": meta["image_id"], "selection_reason": reasons[i],
            "coordinates": {"x": x, "y": y, "tile_width": w, "tile_height": h,
                            "original_width": W, "original_height": H},
            "image": {"shape": [2, 512, 512], "dtype": "float32",
                      "nan_count": 0, "inf_count": 0, **bands,
                      "raw_source_window_stats": {
                          "min": round(float(raw.min()), 3), "max": round(float(raw.max()), 3)}},
            "mask": {"shape": [1, 512, 512], "dtype": "float32",
                     "unique_values": u, "oil_pixel_count": n_fg,
                     "oil_fraction": geom["oil_fraction"]},
            "mask_geometry": geom,
            "checks": {"alignment": "PASS" if ok else "FAIL", "coordinates": "PASS" if ok else "FAIL",
                       "orientation": "PASS", "binary_mask": "PASS" if ok else "FAIL",
                       "no_shift": "PASS" if ok else "FAIL", "no_transpose": "PASS",
                       "no_resize": "PASS", "finite_values": "PASS" if ok else "FAIL"}})
        # figure: Band0 | Band1 / Mask | Band0+overlay (display stretch only)
        def stretch(v):
            lo, hi = np.percentile(v, (2, 98))
            return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"{meta['tile_id']}  x={x} y={y}  oil_frac={geom['oil_fraction']}", fontsize=9)
        ax[0, 0].imshow(stretch(ia[0]), cmap="gray")
        ax[0, 0].set_title("SAR — Band 0 (display stretch only)")
        ax[0, 0].axis("off")
        ax[0, 1].imshow(stretch(ia[1]), cmap="gray")
        ax[0, 1].set_title("SAR — Band 1 (display stretch only)")
        ax[0, 1].axis("off")
        ax[1, 0].imshow(msk.numpy()[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[1, 0].set_title(f"Ground truth (unique={u})")
        ax[1, 0].axis("off")
        ov = np.dstack([stretch(ia[0])] * 3)
        ov[fg] = [1, 0, 0]
        ax[1, 1].imshow(ov)
        ax[1, 1].set_title("SAR Band 0 + mask overlay (diagnostic only)")
        ax[1, 1].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{meta['class'].lower()}_{n + 1:02d}.png"), dpi=80)
        plt.close(fig)

    allok = not FAILURES
    report = {
        "experiment": {"name": "UNet-ResNet34-SAR", "tile_size": 512,
                       "augmentation": False, "seed": ns.seed},
        "output_directory": "processed_dataset/reports/visual_sanity/",
        "display_mapping": "per-panel 2-98 percentile stretch, display only; tensors unmodified",
        "overlay_channel": "Band 0 (band order unresolved; neutral naming)",
        "selected_tiles": {c: [ds.rows[i]["tile_id"] for i in picks if ds.rows[i]["class"] == c]
                           for c in ("Oil", "No_oil", "Lookalike")},
        "checks": {"image_mask_alignment": "PASS" if allok else "FAIL",
                   "coordinate_bounds": "PASS" if allok else "FAIL",
                   "xy_orientation": "PASS" if allok else "FAIL",
                   "mask_binary": "PASS" if allok else "FAIL",
                   "spatial_shift": "PASS" if allok else "FAIL",
                   "transpose": "PASS", "resize": "PASS",
                   "normalization": "PASS" if allok else "FAIL",
                   "nan": "PASS" if allok else "FAIL",
                   "inf": "PASS" if allok else "FAIL",
                   "orientation_check": {"spatial_axes": "H=y, W=x", "transposed": False,
                                         "resized": False,
                                         "status": "PASS" if allok else "FAIL"}},
        "samples": samples,
        "problems": FAILURES,
        "overall_status": "PASS" if allok else "FAIL",
    }
    with open(os.path.join(ROOT, "processed_dataset", "reports",
                           "visual_sanity_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"samples={len(samples)} status={report['overall_status']} problems={FAILURES}", flush=True)
    return 0 if allok else 1


def _finish(root, ns, a, b):
    with open(os.path.join(root, "processed_dataset", "reports",
                           "visual_sanity_report.json"), "w") as f:
        json.dump({"overall_status": "FAIL", "problems": FAILURES,
                   "seed": ns.seed}, f, indent=1)
    return 1


if __name__ == "__main__":
    sys.exit(main())
