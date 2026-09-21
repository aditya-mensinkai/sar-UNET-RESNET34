"""Scene-level dataset verification (read-only audit, no repairs).

Checks every split_index.csv pair: file existence, image<->mask pairing,
dimensions, channels, dtype, NaN/Inf (full read), mask values/binary validity,
split separation (no test/val source in train), duplicate sources.
Writes processed_dataset/reports/dataset_verification_report.json + .csv
(per-pair rows). Exit 0 = all valid; exit 2 = problems found (reported only).
Usage: python tools/verify_dataset.py [--dataset processed_dataset]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="processed_dataset")
    a = ap.parse_args()
    root = os.path.abspath(a.dataset)
    ds = os.path.join(os.path.dirname(root), "dataset")
    rows = [r for r in csv.DictReader(open(os.path.join(root, "metadata", "split_index.csv")))]
    live = [r for r in rows if r["excluded"] == "NO"]

    recs, problems = [], []
    seen_src, dup_src = {}, []
    for r in live:
        ip = os.path.join(ds, r["image_source"].replace("/", os.sep))
        mp = os.path.join(ds, r["mask_source"].replace("/", os.sep))
        rec = {"split": r["split"], "class": r["class"],
               "image_source": r["image_source"], "mask_source": r["mask_source"],
               "status": "valid", "issues": []}
        if r["image_source"] in seen_src and seen_src[r["image_source"]] != r["split"]:
            dup_src.append((r["image_source"], seen_src[r["image_source"]], r["split"]))
            rec["issues"].append("source_in_multiple_splits")
        seen_src.setdefault(r["image_source"], r["split"])
        if not os.path.exists(ip):
            rec["issues"].append("missing_image")
        if not os.path.exists(mp):
            rec["issues"].append("missing_mask")
        if rec["issues"]:
            rec["status"] = "invalid"
            recs.append(rec)
            continue
        try:
            with rasterio.open(ip) as s:
                img = s.read()
                rec.update(width=s.width, height=s.height, bands=s.count,
                           dtype="+".join(s.dtypes), crs=str(s.crs))
            with rasterio.open(mp) as m:
                msk = m.read(1)
                rec.update(mask_width=m.width, mask_height=m.height,
                           mask_unique=sorted(np.unique(msk).tolist()))
        except Exception as e:  # noqa: BLE001
            rec["issues"].append(f"unreadable: {type(e).__name__}")
            rec["status"] = "invalid"
            recs.append(rec)
            continue
        if (rec["width"], rec["height"]) != (rec["mask_width"], rec["mask_height"]):
            rec["issues"].append("dimension_mismatch")
        if img.size == 0:
            rec["issues"].append("empty_image")
        rec["nan"] = int(np.isnan(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0
        rec["posinf"] = int(np.isposinf(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0
        rec["neginf"] = int(np.isneginf(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0
        if rec["nan"] or rec["posinf"] or rec["neginf"]:
            rec["issues"].append("nan_or_inf")
        if not set(rec["mask_unique"]) <= {0, 1}:
            rec["issues"].append("non_binary_mask")
        if rec["issues"]:
            rec["status"] = "invalid"
            problems.append(rec)
        recs.append(rec)
        print(f"{r['split']}/{r['image_source']}: {rec['status']}", flush=True) \
            if rec["status"] == "invalid" else None

    n = len(live)
    summary = {
        "total_images": n,
        "valid_images": sum(1 for r in recs if r["status"] == "valid"),
        "invalid_images": sum(1 for r in recs if r["status"] == "invalid"),
        "missing_masks": sum(1 for r in recs if "missing_mask" in r["issues"]),
        "missing_images": sum(1 for r in recs if "missing_image" in r["issues"]),
        "dimension_mismatches": sum(1 for r in recs if "dimension_mismatch" in r["issues"]),
        "channel_counts": dict(Counter(str(r.get("bands")) for r in recs if "bands" in r)),
        "dtypes": dict(Counter(str(r.get("dtype")) for r in recs if "dtype" in r)),
        "dims": dict(Counter(f"{r.get('width')}x{r.get('height')}" for r in recs if "width" in r)),
        "mask_values_global": sorted({v for r in recs for v in r.get("mask_unique", [])}),
        "mask_empty_count": sum(1 for r in recs if r.get("mask_unique") == [0]),
        "nan_total": sum(r.get("nan", 0) for r in recs),
        "inf_total": sum(r.get("posinf", 0) + r.get("neginf", 0) for r in recs),
        "duplicate_sources_cross_split": [{"source": s, "splits": sorted(set([a, b]))}
                                          for s, a, b in dup_src],
        "split_counts": {sp: sum(1 for r in recs if r["split"] == sp)
                         for sp in ("train", "val", "test", "quarantined")},
        "problems": problems,
    }
    rep = os.path.join(root, "reports")
    json.dump(summary, open(os.path.join(rep, "dataset_verification_report.json"), "w"), indent=1)
    with open(os.path.join(rep, "dataset_verification_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "image_source", "mask_source",
                                          "width", "height", "bands", "dtype",
                                          "mask_width", "mask_height", "mask_unique",
                                          "nan", "posinf", "neginf", "status", "issues"])
        w.writeheader()
        for r in recs:
            w.writerow({k: ((";".join(r[k]) if k == "issues" else r.get(k, ""))) for k in
                        ["split", "class", "image_source", "mask_source", "width", "height",
                         "bands", "dtype", "mask_width", "mask_height", "mask_unique",
                         "nan", "posinf", "neginf", "status", "issues"]})
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
    return 2 if summary["invalid_images"] or dup_src else 0


if __name__ == "__main__":
    sys.exit(main())
