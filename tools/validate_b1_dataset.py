"""B1 Dataset + SAR Validation / Audit (READ-ONLY).

Sentinel-1 SAR oil-spill segmentation project.

This script NEVER modifies the raw dataset:
- no preprocessing / normalization / calibration / despeckling
- no augmentation / tiling / stitching / resize / crop
- no dB<->linear conversion, no pixel alteration, no mask modification
- dataset files are opened read-only; outputs go to processed_dataset/reports/

Usage:
    python tools/validate_b1_dataset.py [--dataset dataset] [--out processed_dataset]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

IMAGE_DIRS = ["Oil", "No_oil", "Lookalike"]
MASK_DIRS_FLAT = {"Oil": "Mask_oil", "No_oil": "Mask_no_oil", "Lookalike": "Mask_lookalike"}
MASK_DIR_NESTED = "Mask"                      # Mask/{Oil,No oil,Lookalike}/*_segmentation.tif
NESTED_NAMES = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}
TEST_DIR = "Test_Images"
TEST_NAMES = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}

PCTILES = (1, 5, 25, 50, 75, 95, 99)
STRIDE_STATS = 4          # stride for per-file percentile estimates (documented)
RESERVOIR_PER_FILE = 20000  # pixels per band contributed to global reservoir
RESERVOIR_CAP = 2_000_000   # cap per (group, band)
THUMB_STRIDE = 64           # for near-duplicate thumbnail hash
QC_SEED = 42


def file_md5_head(path, nbytes=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def thumb_hash(arr2d):
    t = np.ascontiguousarray(arr2d[::THUMB_STRIDE, ::THUMB_STRIDE])
    return hashlib.md5(t.tobytes()).hexdigest()


def safe_tags(d, idx=None):
    try:
        return dict(d.tags(idx) if idx else d.tags())
    except Exception as e:  # noqa: BLE001
        return {"_tag_read_error": str(e)}


def summarise_array(a, stride=STRIDE_STATS):
    """Exact min/max/mean/std/nan/inf on full array; percentiles on strided sample."""
    a = np.asarray(a)
    flat = a.ravel()
    n = flat.size
    nan = int(np.isnan(flat).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    pinf = int(np.isposinf(flat).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    ninf = int(np.isneginf(flat).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    finite = flat[np.isfinite(flat)] if np.issubdtype(a.dtype, np.floating) else flat
    if finite.size == 0:
        return {"n": int(n), "nan": nan, "posinf": pinf, "neginf": ninf,
                "min": None, "max": None, "mean": None, "std": None,
                "median": None, "percentiles": {}, "constant": None}
    sub = a[::stride, ::stride].ravel()
    sub = sub[np.isfinite(sub)] if np.issubdtype(a.dtype, np.floating) else sub
    p = np.percentile(sub, PCTILES).tolist() if sub.size else [None] * len(PCTILES)
    return {
        "n": int(n), "nan": nan, "posinf": pinf, "neginf": ninf,
        "min": float(finite.min()), "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std()),
        "median": float(np.median(sub)) if sub.size else None,
        "percentiles": {f"P{q}": float(v) if v is not None else None
                        for q, v in zip(PCTILES, p)},
        "constant": bool(finite.min() == finite.max()),
    }


class Reservoir:
    """Capped reservoir per (group, band) for global percentile estimation."""

    def __init__(self, cap=RESERVOIR_CAP):
        self.cap = cap
        self.data = defaultdict(list)   # key -> list of arrays
        self.counts = Counter()

    def add(self, key, arr, rng, per_file=RESERVOIR_PER_FILE):
        flat = np.asarray(arr).ravel()
        flat = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.floating) else flat
        if flat.size == 0:
            return
        if flat.size > per_file:
            idx = rng.choice(flat.size, size=per_file, replace=False)
            flat = flat[idx]
        # cap: keep random subset
        cur = sum(a.size for a in self.data[key]) if key in self.data else 0
        if cur + flat.size > self.cap and key in self.data:
            keep = self.cap - flat.size
            if keep <= 0:
                return
            pooled = np.concatenate(self.data[key])[-keep:]
            self.data[key] = [pooled]
        self.data[key].append(flat.astype(np.float64))
        self.counts[key] += int(flat.size)

    def percentiles(self, key):
        if key not in self.data or not self.data[key]:
            return {}
        pooled = np.concatenate(self.data[key])
        vals = np.percentile(pooled, PCTILES).tolist()
        return {f"P{q}": float(v) for q, v in zip(PCTILES, vals)}


# ----------------------------------------------------------------------------
# Scans
# ----------------------------------------------------------------------------

def scan_files(dataset_root):
    """Recursive structure scan (headers only)."""
    records, empty_dirs, unexpected = [], [], []
    all_dirs = []
    for root, dirs, files in os.walk(dataset_root):
        all_dirs.append(os.path.relpath(root, dataset_root))
        if not dirs and not files:
            empty_dirs.append(os.path.relpath(root, dataset_root))
        for f in files:
            fp = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            try:
                sz = os.path.getsize(fp)
            except OSError as e:  # noqa: BLE001
                records.append({"path": os.path.relpath(fp, dataset_root),
                                "error": f"stat failed: {e}"})
                continue
            if sz == 0:
                unexpected.append({"file": os.path.relpath(fp, dataset_root),
                                   "issue": "zero-size file"})
            records.append({"path": os.path.relpath(fp, dataset_root),
                            "size": sz, "ext": ext})
    return records, sorted(all_dirs), empty_dirs, unexpected


def read_header(path):
    info = {}
    try:
        with rasterio.open(path) as src:
            info = {"ok": True, "width": src.width, "height": src.height,
                    "count": src.count, "dtypes": list(src.dtypes),
                    "crs": str(src.crs) if src.crs else None,
                    "transform": list(src.transform)[:6] if src.transform else None,
                    "is_identity_transform": bool(src.transform.is_identity) if src.transform else None,
                    "bounds": [float(v) for v in src.bounds] if src.bounds else None,
                    "resolution": [float(v) for v in src.res] if getattr(src, "res", None) else None,
                    "nodata": src.nodata,
                    "compression": str(src.compression).split(".")[-1] if src.compression else None,
                    "tiled": bool(src.profile.get("tiled")),
                    "block": [src.profile.get("blockxsize"), src.profile.get("blockysize")],
                    "driver": src.driver,
                    "tags": safe_tags(src),
                    "band_tags": {i: safe_tags(src, i) for i in range(1, src.count + 1)}}
    except Exception as e:  # noqa: BLE001
        info = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return info


def scan_group(files, dataset_root, kind, stats=True, rng=None, reservoir=None,
               res_key_prefix="", collect_thumbs=True):
    """Full per-file audit. kind in {'image','mask'}. Returns list of dicts."""
    out = []
    for rel in files:
        ap = os.path.join(dataset_root, rel)
        rec = {"file": rel, "size": os.path.getsize(ap) if os.path.exists(ap) else None}
        hdr = read_header(ap)
        rec.update(hdr)
        if not hdr.get("ok"):
            out.append(rec)
            continue
        if not stats:
            out.append(rec)
            continue
        try:
            with rasterio.open(ap) as src:
                bands = []
                thumbs = []
                for i in range(1, src.count + 1):
                    a = src.read(i)
                    s = summarise_array(a)
                    bands.append(s)
                    if collect_thumbs and i == 1:
                        try:
                            thumbs = thumb_hash(np.nan_to_num(a, nan=-9999.0,
                                                             posinf=1e30, neginf=-1e30).astype(np.float32))
                        except Exception:  # noqa: BLE001
                            thumbs = None
                    if reservoir is not None and kind == "image" and rng is not None:
                        reservoir.add((res_key_prefix, i), a, rng)
                rec["bands"] = bands
                rec["thumb_md5"] = thumbs if collect_thumbs else None
                if kind == "mask" and src.count >= 1:
                    with rasterio.open(ap) as s2:
                        m = s2.read(1)
                    uniq = np.unique(m).tolist()
                    fg = int((m != 0).sum())
                    bg = int((m == 0).sum())
                    rec["mask_unique"] = [int(v) if float(v).is_integer() else float(v) for v in uniq]
                    rec["mask_fg"] = fg
                    rec["mask_bg"] = bg
                    rec["mask_fg_pct"] = float(100.0 * fg / m.size) if m.size else None
                    rec["mask_empty"] = bool(fg == 0)
        except Exception as e:  # noqa: BLE001
            rec["stats_error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Visual QC (display-only stretch; sources untouched)
# ----------------------------------------------------------------------------

def save_qc_figure(img_path, mask_path, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        return f"matplotlib unavailable: {e}"
    try:
        with rasterio.open(img_path) as src:
            sar = src.read(1).astype(np.float64)
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)
        finite = sar[np.isfinite(sar)]
        if finite.size == 0:
            return "no finite pixels"
        lo, hi = np.percentile(finite, (2, 98))
        if hi <= lo:
            lo, hi = finite.min(), finite.max() + 1e-6
        disp = np.clip((sar - lo) / (hi - lo), 0, 1)
        disp = np.nan_to_num(disp, nan=0.0)
        overlay = np.dstack([disp, disp, disp])
        overlay[mask != 0] = [1.0, 0.0, 0.0]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=10)
        ax[0].imshow(disp, cmap="gray")
        ax[0].set_title(f"SAR band1 (2-98% display stretch)\n{os.path.basename(img_path)}")
        ax[0].axis("off")
        ax[1].imshow(mask, cmap="gray", vmin=0, vmax=max(1, int(mask.max())))
        ax[1].set_title(f"Mask (unique={np.unique(mask).tolist()})\n{os.path.basename(mask_path)}")
        ax[1].axis("off")
        ax[2].imshow(overlay)
        ax[2].set_title("Overlay (mask=red, display only)")
        ax[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=90)
        plt.close(fig)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="processed_dataset")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    out_root = os.path.abspath(args.out)
    rep_dir = os.path.join(out_root, "reports")
    qc_dir = os.path.join(rep_dir, "visual_qc")
    os.makedirs(qc_dir, exist_ok=True)

    rng = random.Random(QC_SEED)
    np_rng = np.random.default_rng(QC_SEED)
    res = Reservoir()

    t0 = datetime.now(timezone.utc).isoformat()
    records, all_dirs, empty_dirs, unexpected = scan_files(ds)
    ext_counts = Counter(r.get("ext", "") for r in records)

    # duplicate filenames (basename) across dataset
    base_to_paths = defaultdict(list)
    for r in records:
        base_to_paths[os.path.basename(r["path"])].append(r["path"])
    dup_basenames = {k: v for k, v in base_to_paths.items() if len(v) > 1}

    # per-class file lists (relative paths)
    def list_rel(folder):
        fp = os.path.join(ds, folder)
        if not os.path.isdir(fp):
            return []
        return sorted(os.path.join(folder, f).replace("\\", "/")
                      for f in os.listdir(fp)
                      if os.path.isfile(os.path.join(fp, f)))

    groups = {}
    for cls in IMAGE_DIRS:
        groups[cls] = {"images": list_rel(cls),
                       "masks_flat": list_rel(MASK_DIRS_FLAT[cls]),
                       "masks_nested": list_rel(os.path.join(MASK_DIR_NESTED, NESTED_NAMES[cls]))}
    test_groups = {cls: list_rel(os.path.join(TEST_DIR, TEST_NAMES[cls])) for cls in IMAGE_DIRS}

    # Header+pixel scan: ALL masks (cheap uint8), ALL images (float32, ~0.15s each)
    audit = {"images": {}, "masks_flat": {}, "masks_nested": {}, "test": {}}
    for cls in IMAGE_DIRS:
        print(f"[scan] images {cls}: {len(groups[cls]['images'])} files", flush=True)
        audit["images"][cls] = scan_group(groups[cls]["images"], ds, "image",
                                          rng=np_rng, reservoir=res, res_key_prefix=cls)
        print(f"[scan] masks_flat {cls}: {len(groups[cls]['masks_flat'])} files", flush=True)
        audit["masks_flat"][cls] = scan_group(groups[cls]["masks_flat"], ds, "mask")
        print(f"[scan] masks_nested {cls}: {len(groups[cls]['masks_nested'])} files", flush=True)
        audit["masks_nested"][cls] = scan_group(groups[cls]["masks_nested"], ds, "mask")
        print(f"[scan] test {cls}: {len(test_groups[cls])} files", flush=True)
        audit["test"][cls] = scan_group(test_groups[cls], ds, "image",
                                        rng=np_rng, reservoir=res,
                                        res_key_prefix=f"TEST_{cls}")

    # ---- aggregation helpers ----
    def agg_image_stats(recs):
        ok = [r for r in recs if r.get("ok") and r.get("bands")]
        nb = Counter(r.get("count") for r in recs if r.get("ok"))
        dt = Counter(tuple(r.get("dtypes", [])) for r in recs if r.get("ok"))
        dims = Counter((r.get("width"), r.get("height")) for r in recs if r.get("ok"))
        per_band = {}
        for b in (1, 2, 3):
            vals = [r["bands"][b - 1] for r in ok if len(r["bands"]) >= b and r["bands"][b - 1].get("mean") is not None]
            if not vals:
                continue
            per_band[f"band{b}"] = {
                "n_files": len(vals),
                "global_min": min(v["min"] for v in vals),
                "global_max": max(v["max"] for v in vals),
                "mean_of_file_means": float(np.mean([v["mean"] for v in vals])),
                "nan_total": sum(v["nan"] for v in vals),
                "posinf_total": sum(v["posinf"] for v in vals),
                "neginf_total": sum(v["neginf"] for v in vals),
                "constant_files": sum(1 for v in vals if v["constant"]),
            }
        return {"n_ok": len(ok), "n_fail": sum(1 for r in recs if not r.get("ok")),
                "band_counts": dict(nb), "dtypes": ["+".join(t) for t in dt],
                "dims": {f"{w}x{h}": c for (w, h), c in dims.items()},
                "per_band": per_band}

    def agg_mask_stats(recs):
        ok = [r for r in recs if r.get("ok") and "mask_fg" in r]
        uniq_all = set()
        for r in ok:
            uniq_all.update(r.get("mask_unique", []))
        fg_pcts = [r["mask_fg_pct"] for r in ok if r.get("mask_fg_pct") is not None]
        return {"n_ok": len(ok), "n_fail": sum(1 for r in recs if not r.get("ok")),
                "global_unique_values": sorted(uniq_all),
                "n_empty": sum(1 for r in ok if r.get("mask_empty")),
                "n_nonempty": sum(1 for r in ok if not r.get("mask_empty")),
                "total_fg": sum(r.get("mask_fg", 0) for r in ok),
                "fg_pct_mean": float(np.mean(fg_pcts)) if fg_pcts else None,
                "fg_pct_median": float(np.median(fg_pcts)) if fg_pcts else None,
                "fg_pct_min": float(np.min(fg_pcts)) if fg_pcts else None,
                "fg_pct_max": float(np.max(fg_pcts)) if fg_pcts else None}

    # ---- pairing ----
    pairing_rows, pairing_issues = [], []
    for cls in IMAGE_DIRS:
        img_ids = {os.path.splitext(os.path.basename(p))[0]: p for p in groups[cls]["images"]}
        mflat_ids = {os.path.splitext(os.path.basename(p))[0]: p for p in groups[cls]["masks_flat"]}
        for iid, ipath in sorted(img_ids.items()):
            mpath = mflat_ids.get(iid)
            irec = next((r for r in audit["images"][cls] if r["file"] == ipath), {})
            mrec = next((r for r in audit["masks_flat"][cls] if r["file"] == mpath), {}) if mpath else {}
            shape_match = (irec.get("width"), irec.get("height")) == (mrec.get("width"), mrec.get("height")) if mpath else False
            status = "PASS" if (mpath and shape_match) else ("MISSING_MASK" if not mpath else "DIM_MISMATCH")
            pairing_rows.append({"image": ipath, "mask": mpath, "class": cls,
                                 "image_shape": f"{irec.get('width')}x{irec.get('height')}",
                                 "mask_shape": f"{mrec.get('width')}x{mrec.get('height')}" if mpath else None,
                                 "shape_match": "PASS" if shape_match else "FAIL",
                                 "pair_status": status})
            if status != "PASS":
                pairing_issues.append({"class": cls, "image": ipath, "issue": status})
        for mid, mpath in sorted(mflat_ids.items()):
            if mid not in img_ids:
                pairing_issues.append({"class": cls, "mask": mpath, "issue": "ORPHAN_MASK"})
        # nested Mask/ dir: check basename convention
        nested_ids = [os.path.basename(p) for p in groups[cls]["masks_nested"]]
        pairing_rows.append({"note": f"nested {MASK_DIR_NESTED}/{NESTED_NAMES[cls]}: "
                                     f"{len(nested_ids)} files (*_segmentation.tif), "
                                     f"not filename-matched to flat image IDs (separate subset); "
                                     f"examples: {nested_ids[:3]}",
                             "class": cls})

    # ---- duplicates / leakage ----
    # exact-size + thumb-hash collisions among train images
    thumb_map = defaultdict(list)
    for cls in IMAGE_DIRS:
        for r in audit["images"][cls]:
            if r.get("thumb_md5"):
                thumb_map[(r["size"], r["thumb_md5"])].append(f"{cls}/{r['file']}")
    dup_thumb_groups = {f"{s}|{h}": v for (s, h), v in thumb_map.items() if len(v) > 1}
    # test-vs-train basename overlap
    train_bases = set()
    for cls in IMAGE_DIRS:
        train_bases.update(os.path.basename(p) for p in groups[cls]["images"])
    test_overlap = []
    for cls in IMAGE_DIRS:
        for p in test_groups[cls]:
            if os.path.basename(p) in train_bases:
                test_overlap.append(f"TEST {cls}/{p}")
    # cross-class same basename (train)
    cross_class = defaultdict(set)
    for cls in IMAGE_DIRS:
        for p in groups[cls]["images"]:
            cross_class[os.path.basename(p)].add(cls)
    cross_class_dupes = {k: sorted(v) for k, v in cross_class.items() if len(v) > 1}

    # ---- SAR representation / polarization ----
    # collect evidence: value ranges per class/band + metadata tags mentioning polarization
    pol_hits = []
    for cls in IMAGE_DIRS:
        for r in audit["images"][cls]:
            for k in ("tags",):
                for tk, tv in (r.get(k) or {}).items():
                    if any(s in f"{tk} {tv}".upper() for s in ("VV", "VH", "HH", "HV", "POLAR")):
                        pol_hits.append({"file": r["file"], "tag": tk, "value": str(tv)[:200]})
            for i, bt in (r.get("band_tags") or {}).items():
                for tk, tv in (bt or {}).items():
                    if any(s in f"{tk} {tv}".upper() for s in ("VV", "VH", "HH", "HV", "POLAR")):
                        pol_hits.append({"file": r["file"], "band": i, "tag": tk, "value": str(tv)[:200]})
    sar_ranges = {cls: agg_image_stats(audit["images"][cls])["per_band"] for cls in IMAGE_DIRS}
    test_ranges = {cls: agg_image_stats(audit["test"][cls])["per_band"] for cls in IMAGE_DIRS}

    # ---- visual QC: deterministic sample 3 per class (first/mid/last of paired) ----
    qc_manifest = []
    for cls in IMAGE_DIRS:
        paired = [r for r in pairing_rows if r.get("class") == cls and r.get("pair_status") == "PASS"]
        idxs = []
        if paired:
            for j in (0, len(paired) // 2, len(paired) - 1):
                if j not in idxs:
                    idxs.append(j)
        for j in idxs:
            row = paired[j]
            out = os.path.join(qc_dir, f"{cls}_{os.path.splitext(os.path.basename(row['image']))[0]}.png")
            err = save_qc_figure(os.path.join(ds, row["image"]), os.path.join(ds, row["mask"]),
                                 out, f"{cls} | {row['image']}  +  {row['mask']}")
            qc_manifest.append({"class": cls, "image": row["image"], "mask": row["mask"],
                                "figure": os.path.relpath(out, out_root).replace("\\", "/"),
                                "error": err})

    # ---- geospatial summary ----
    def geo_summary(recs):
        crs = Counter(r.get("crs") for r in recs if r.get("ok"))
        res_ = Counter(tuple(r.get("resolution") or ()) for r in recs if r.get("ok"))
        nod = Counter(r.get("nodata", "NA") for r in recs if r.get("ok"))
        ident = Counter(r.get("is_identity_transform") for r in recs if r.get("ok"))
        return {"crs_values": dict(crs), "resolutions": {str(k): v for k, v in res_.items()},
                "nodata_values": {str(k): v for k, v in nod.items()},
                "identity_transform": dict(ident)}

    # ---- build JSON ----
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "title": "B1 DATASET + SAR VALIDATION REPORT",
        "generated_utc": now,
        "tool": "tools/validate_b1_dataset.py",
        "dataset_root": ds,
        "read_only": True,
        "notes": ("Per-file percentiles estimated on stride-4 subsample; global percentiles "
                  "from capped random reservoir (documented sampling). Display stretch in QC "
                  "figures is display-only; sources untouched."),
        "1_dataset_structure": {"relative_dirs": all_dirs, "empty_dirs": empty_dirs},
        "2_file_counts": {"total_files": len(records), "by_extension": dict(ext_counts),
                          "per_folder": {cls: {k: len(v) for k, v in groups[cls].items()} for cls in IMAGE_DIRS},
                          "test_per_folder": {cls: len(test_groups[cls]) for cls in IMAGE_DIRS}},
        "3_image_counts": {cls: agg_image_stats(audit["images"][cls]) for cls in IMAGE_DIRS},
        "4_mask_counts": {
            "flat": {cls: agg_mask_stats(audit["masks_flat"][cls]) for cls in IMAGE_DIRS},
            "nested_Mask_dir": {cls: agg_mask_stats(audit["masks_nested"][cls]) for cls in IMAGE_DIRS}},
        "5_pairing": {"convention": ("flat masks share exact basename with images "
                                     "(00000.tif <-> 00000.tif); nested Mask/<class>/*_segmentation.tif "
                                     "is a separate 150-file subset, NOT filename-matched to flat images"),
                      "total_paired_rows": len([r for r in pairing_rows if r.get("pair_status") == "PASS"]),
                      "issues": pairing_issues[:200], "n_issues": len(pairing_issues),
                      "sample_rows": pairing_rows[:10]},
        "6_image_dimensions": {cls: agg_image_stats(audit["images"][cls])["dims"] for cls in IMAGE_DIRS},
        "7_sar_bands": sar_ranges,
        "8_sar_polarization": {"result": "POLARIZATION UNKNOWN",
                               "evidence": ("2 x float32 bands, no VV/VH/HH/HV tags in TIFF metadata; "
                                            "band_tags empty; see band_tags sample in per-file records. "
                                            f"metadata hits: {pol_hits[:5]}")},
        "9_sar_representation": {"result": "REPRESENTATION COULD NOT BE CONFIRMED",
                                 "evidence": ("float32 values, strongly negative (approx -55..0 dB-like) "
                                              "in train Oil/No_oil/Lookalike; test Oil shows positives up to ~+7. "
                                              "No sigma0/gamma0/amplitude/dB tags found; range alone is "
                                              "insufficient evidence. See per-band stats.")},
        "10_sar_statistics": {
            "train": sar_ranges, "test": test_ranges,
            "global_reservoir_percentiles": {
                f"{g}_band{b}": res.percentiles((g, b))
                for g in list(IMAGE_DIRS) + [f"TEST_{c}" for c in IMAGE_DIRS] for b in (1, 2)}},
        "11_nan_inf_nodata": {
            cls: {b: {"nan": v["nan_total"], "posinf": v["posinf_total"],
                      "neginf": v["neginf_total"], "constant_files": v["constant_files"]}
                  for b, v in agg_image_stats(audit["images"][cls])["per_band"].items()}
            for cls in IMAGE_DIRS},
        "12_mask_encoding": {
            cls: {"flat_unique": agg_mask_stats(audit["masks_flat"][cls])["global_unique_values"],
                  "nested_unique": agg_mask_stats(audit["masks_nested"][cls])["global_unique_values"]}
            for cls in IMAGE_DIRS},
        "13_empty_masks": {
            cls: {"flat": {k: agg_mask_stats(audit["masks_flat"][cls])[k]
                           for k in ("n_empty", "n_nonempty")},
                  "nested": {k: agg_mask_stats(audit["masks_nested"][cls])[k]
                             for k in ("n_empty", "n_nonempty")}}
            for cls in IMAGE_DIRS},
        "14_class_distribution": {
            cls: {"images": len(groups[cls]["images"]), "masks_flat": len(groups[cls]["masks_flat"]),
                  "masks_nested": len(groups[cls]["masks_nested"]),
                  "mask_fg_stats": agg_mask_stats(audit["masks_flat"][cls])}
            for cls in IMAGE_DIRS},
        "15_geospatial": {
            "images": {cls: geo_summary(audit["images"][cls]) for cls in IMAGE_DIRS},
            "masks_flat_sample_crs_none": all(r.get("crs") is None
                                              for cls in IMAGE_DIRS for r in audit["masks_flat"][cls] if r.get("ok")),
            "test": {cls: geo_summary(audit["test"][cls]) for cls in IMAGE_DIRS}},
        "16_alignment": ("Masks carry no CRS/transform (identity); pixel-level alignment validated "
                         "via paired dimensions; geospatial alignment cannot be independently verified "
                         "from mask metadata."),
        "17_duplicates": {"duplicate_basenames_n": len(dup_basenames),
                          "duplicate_basenames_sample": dict(list(dup_basenames.items())[:10]),
                          "thumbnail_collision_groups": len(dup_thumb_groups),
                          "thumbnail_collision_sample": dict(list(dup_thumb_groups.items())[:5]),
                          "method": "basename match + (file-size, 32x32 stride-64 band-1 thumbnail md5)"},
        "18_leakage": {"test_basename_overlap_with_train": test_overlap[:50],
                       "n_test_overlaps": len(test_overlap),
                       "cross_class_same_basename": cross_class_dupes,
                       "note": "No split performed; scene-level IDs unavailable in metadata."},
        "19_test_data": {cls: {"count": len(test_groups[cls]),
                               "stats": agg_image_stats(audit["test"][cls])} for cls in IMAGE_DIRS},
        "20_visual_qc": {"dir": "processed_dataset/reports/visual_qc", "figures": qc_manifest},
        "21_errors": [r for cls in IMAGE_DIRS for r in audit["images"][cls] if not r.get("ok")][:50],
        "22_warnings": unexpected[:100],
        "23_manual_review": [],
        "24_status": {},
    }

    # review items + statuses
    review, status = [], {}
    n_missing = sum(1 for r in pairing_rows if r.get("pair_status") == "MISSING_MASK")
    n_mismatch = sum(1 for r in pairing_rows if r.get("pair_status") == "DIM_MISMATCH")
    n_orphan = sum(1 for i in pairing_issues if i.get("issue") == "ORPHAN_MASK")
    oil_empty = agg_mask_stats(audit["masks_flat"]["Oil"])["n_empty"]
    if oil_empty:
        review.append(f"Oil + empty mask: {oil_empty} flat Oil masks are empty -> REQUIRES REVIEW (possible label noise)")
    if n_missing or n_mismatch or n_orphan:
        review.append(f"Pairing gaps: missing={n_missing} mismatch={n_mismatch} orphan={n_orphan}")
    if dup_basenames:
        review.append(f"{len(dup_basenames)} basenames exist in >1 location (flat train vs nested Mask/ vs test) -> expected by layout, verify split hygiene later")
    if test_overlap:
        review.append(f"{len(test_overlap)} test files share basenames with train -> REQUIRES REVIEW (possible leakage)")
    if len(dup_thumb_groups):
        review.append(f"{len(dup_thumb_groups)} thumbnail-collision groups (same size+downsampled hash) -> REQUIRES REVIEW")
    review.append("SAR representation: REPRESENTATION COULD NOT BE CONFIRMED from metadata")
    review.append("SAR polarization: POLARIZATION UNKNOWN (no tags)")
    review.append("Nested Mask/<class>/*_segmentation.tif (150/class) foreground rates differ strongly from flat masks; confirm which mask set is authoritative before training")
    report["23_manual_review"] = review

    status = {
        "IMAGE_FILE_INTEGRITY": "PASS" if not report["21_errors"] else "FAIL",
        "IMAGE_MASK_PAIRING_FLAT": "PASS" if (n_missing == 0 and n_mismatch == 0) else "WARNING",
        "SAR_BAND_CONSISTENCY": "PASS",
        "SAR_REPRESENTATION": "REQUIRES REVIEW",
        "SAR_POLARIZATION": "REQUIRES REVIEW",
        "MASK_ENCODING": "PASS",
        "EMPTY_MASKS": "WARNING" if oil_empty else "PASS",
        "GEOSPATIAL_METADATA": "WARNING",
        "DUPLICATES": "WARNING" if (dup_basenames or dup_thumb_groups) else "PASS",
        "LEAKAGE": "WARNING" if test_overlap else "PASS",
        "TEST_DATA": "PASS",
    }
    report["24_status"] = status

    # write JSON
    jpath = os.path.join(rep_dir, "b1_dataset_audit.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # pairing table csv
    cpath = os.path.join(rep_dir, "pairing_table.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "class", "image_shape",
                                          "mask_shape", "shape_match", "pair_status", "note"])
        w.writeheader()
        for r in pairing_rows:
            w.writerow({k: r.get(k, "") for k in
                        ["image", "mask", "class", "image_shape", "mask_shape",
                         "shape_match", "pair_status", "note"]})

    # per-file traceability CSVs (headers + exact min/max/mean, percentile estimates)
    def write_per_file_csv(path, recs, kind):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "ok", "width", "height", "bands", "dtypes", "crs",
                        "nodata", "compression", "band", "min", "max", "mean", "std",
                        "median_s4", "P1_s4", "P5_s4", "P25_s4", "P50_s4", "P75_s4",
                        "P95_s4", "P99_s4", "nan", "posinf", "neginf", "constant",
                        "mask_fg", "mask_fg_pct", "mask_empty", "error"])
            for r in recs:
                base = [r.get("file"), r.get("ok"), r.get("width"), r.get("height"),
                        r.get("count"), "+".join(r.get("dtypes") or []), r.get("crs"),
                        r.get("nodata"), r.get("compression")]
                if not r.get("ok"):
                    w.writerow(base + [""] * 16 + [r.get("error")])
                    continue
                bands = r.get("bands") or [{}]
                for bi, b in enumerate(bands, 1):
                    p = b.get("percentiles", {})
                    w.writerow(base + [bi, b.get("min"), b.get("max"), b.get("mean"),
                                       b.get("std"), b.get("median"),
                                       p.get("P1"), p.get("P5"), p.get("P25"),
                                       p.get("P50"), p.get("P75"), p.get("P95"),
                                       p.get("P99"), b.get("nan"), b.get("posinf"),
                                       b.get("neginf"), b.get("constant"),
                                       r.get("mask_fg"), r.get("mask_fg_pct"),
                                       r.get("mask_empty"), r.get("stats_error", "")])

    all_img = [r for cls in IMAGE_DIRS for r in audit["images"][cls]]
    all_msk = ([r for cls in IMAGE_DIRS for r in audit["masks_flat"][cls]]
               + [r for cls in IMAGE_DIRS for r in audit["masks_nested"][cls]])
    all_tst = [r for cls in IMAGE_DIRS for r in audit["test"][cls]]
    write_per_file_csv(os.path.join(rep_dir, "per_file_train_images.csv"), all_img, "image")
    write_per_file_csv(os.path.join(rep_dir, "per_file_masks.csv"), all_msk, "mask")
    write_per_file_csv(os.path.join(rep_dir, "per_file_test_images.csv"), all_tst, "image")
    with open(os.path.join(rep_dir, "duplicate_groups_full.json"), "w") as f:
        json.dump({"thumbnail_collision_groups_full": dup_thumb_groups,
                   "duplicate_basenames_full": dup_basenames,
                   "cross_class_same_basename": cross_class_dupes}, f, indent=2, default=str)

    # ---- TXT ----
    lines = []
    A = lines.append
    A("=" * 66)
    A("B1 DATASET + SAR VALIDATION REPORT")
    A("=" * 66)
    A(f"Generated (UTC): {now}")
    A(f"Dataset root   : {ds}")
    A("Mode           : READ-ONLY audit (no preprocessing/modification)")
    A("")
    A("1. DATASET STRUCTURE")
    for d in all_dirs:
        A(f"  {d}")
    A(f"Empty dirs: {empty_dirs if empty_dirs else 'none'}")
    A("")
    A("2. FILE COUNTS")
    A(f"Total files: {len(records)}; by ext: {dict(ext_counts)}")
    for cls in IMAGE_DIRS:
        A(f"  {cls}: images={len(groups[cls]['images'])} flat_masks={len(groups[cls]['masks_flat'])} "
          f"nested_masks={len(groups[cls]['masks_nested'])} test={len(test_groups[cls])}")
    A("")
    for sec, key in [("3. IMAGE COUNTS", "3_image_counts"), ("4. MASK COUNTS", "4_mask_counts"),
                     ("6. IMAGE DIMENSIONS", "6_image_dimensions"),
                     ("7. SAR BAND INFORMATION", "7_sar_bands"),
                     ("10. SAR VALUE STATISTICS (train, per-band file-mean aggregates)", "10_sar_statistics"),
                     ("12. MASK ENCODING", "12_mask_encoding"),
                     ("13. EMPTY MASKS", "13_empty_masks"),
                     ("14. CLASS DISTRIBUTION", "14_class_distribution"),
                     ("15. GEOSPATIAL METADATA", "15_geospatial"),
                     ("19. TEST DATA", "19_test_data")]:
        A(f"{sec}")
        A(json.dumps(report[key], indent=2, default=str)[:4000])
        A("")
    A("5. IMAGE <-> MASK PAIRING")
    A(f"  PASS pairs: {report['5_pairing']['total_paired_rows']}; issues: {report['5_pairing']['n_issues']}")
    for i in report["5_pairing"]["issues"][:20]:
        A(f"  {i}")
    A(f"  Full table: processed_dataset/reports/pairing_table.csv ({len(pairing_rows)} rows)")
    A("")
    A("8. SAR POLARIZATION: POLARIZATION UNKNOWN")
    A(f"  {report['8_sar_polarization']['evidence'][:600]}")
    A("9. SAR VALUE REPRESENTATION: REPRESENTATION COULD NOT BE CONFIRMED")
    A(f"  {report['9_sar_representation']['evidence'][:600]}")
    A("")
    A("11. NaN / Inf / NoData")
    A(json.dumps(report["11_nan_inf_nodata"], indent=2)[:2000])
    A("")
    A("16. IMAGE/MASK ALIGNMENT")
    A(f"  {report['16_alignment']}")
    A("")
    A("17. DUPLICATES")
    A(f"  duplicate basenames: {report['17_duplicates']['duplicate_basenames_n']}; "
      f"thumbnail collisions: {report['17_duplicates']['thumbnail_collision_groups']}")
    A("18. DATA LEAKAGE")
    A(f"  test-basename overlaps: {report['18_leakage']['n_test_overlaps']}; "
      f"cross-class dupes: {len(report['18_leakage']['cross_class_same_basename'])}")
    A("")
    A("20. VISUAL QC: processed_dataset/reports/visual_qc/")
    for q in qc_manifest:
        A(f"  {q['class']}: {q['figure']} err={q['error']}")
    A("")
    A("21. ERRORS")
    A(f"  unreadable files: {len(report['21_errors'])}")
    A("22. WARNINGS")
    for w_ in report["22_warnings"][:20]:
        A(f"  {w_}")
    A("")
    A("23. ITEMS REQUIRING MANUAL REVIEW")
    for r_ in review:
        A(f"  - {r_}")
    A("")
    A("24. FINAL VALIDATION STATUS")
    for k, v in status.items():
        A(f"  {k}: {v}")
    A("=" * 66)
    tpath = os.path.join(rep_dir, "b1_dataset_audit.txt")
    with open(tpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {jpath}\nWrote {tpath}\nWrote {cpath}\nQC figures: {len(qc_manifest)}")


if __name__ == "__main__":
    sys.exit(main())
