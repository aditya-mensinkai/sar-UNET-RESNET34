"""Scene -> preprocess -> 256px tile-pair -> validate -> filter -> GeoTIFF.

Consumes the authoritative scene split (split_index.csv, image-level) and the
frozen NORMALIZATION_BOUNDS. Never re-splits, never fabricates. Tiles are NEW
derived GeoTIFFs (float32 SAR + uint8 mask) carrying per-tile CRS/transform.
Augmentation stays dynamic (train-time); materialized tiles are unaugmented.
Background subsampling (if BG_KEEP_FRACTION < 1) is deterministic and seeded;
every oil-containing tile is always retained.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random

import numpy as np
import rasterio
from rasterio.crs import CRS
from affine import Affine

from . import tiling, normalization
from . import validate as validate_cfg
from .pipeline import process_scene

TILE_FIELDS = ["tile_id", "scene_id", "split", "class", "image_path", "mask_path",
               "row_start", "col_start", "row_end", "col_end",
               "original_height", "original_width", "tile_height", "tile_width",
               "tile_size", "stride", "overlap",
               "padded", "valid_height", "valid_width",
               "oil_pixel_count", "oil_pixel_fraction", "has_oil", "valid_pixel_fraction",
               "filtering_status", "filtering_reason",
               "source_image", "source_mask"]


def write_tile(path, array, crs, transform, dtype, count):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = array.shape[-2:]
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=count,
                       dtype=dtype, crs=CRS.from_string(str(crs)),
                       transform=Affine(*tuple(transform)[:6]), tiled=True) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)


def _fail(quarantine, tile_id, scene_id, split, reason, source_image):
    quarantine.append({"tile_id": tile_id, "scene_id": scene_id, "split": split,
                       "reason": reason, "source_image": source_image})


def build_tiles(split_index_csv, out_root, cfg, frozen_bounds, splits=("train", "val", "test"),
                limit_scenes=None, classes=("Oil", "No_oil", "Lookalike")):
    """Full run over split_index.csv. Re-runnable: existing tiles are verified, never rewritten."""
    cfg = validate_cfg(cfg)
    size, stride = cfg["TILE_SIZE"], cfg["TILE_STRIDE"]
    if size != 256:
        raise ValueError(f"TILE_SIZE must be 256, got {size}")
    overlap = size - stride
    rows = [r for r in csv.DictReader(open(split_index_csv))
            if r["split"] in splits and r["class"] in classes and r["excluded"] == "NO"]
    if limit_scenes is not None:
        keep = {}
        for r in rows:
            keep.setdefault((r["split"], r["class"]), []).append(r)
        rows = [r for v in keep.values() for r in v[:limit_scenes]]
    bg_rng = random.Random(cfg.get("BG_SEED", 42))
    tile_rows, quarantine, stats = [], [], {"scenes": 0, "tiles": 0, "kept": 0,
                                            "filtered_bg": 0, "rejected": 0,
                                            "oil_tiles": 0, "bg_tiles": 0, "oil_px": 0}
    for r in sorted(rows, key=lambda x: (x["split"], x["class"], x["image_filename"])):
        scene_id = f"{r['class']}/{os.path.splitext(r['image_filename'])[0]}"
        base = out_root  # image/mask_destination are relative to processed_dataset/
        sp = os.path.join(base, r["image_destination"])
        mp = os.path.join(base, r["mask_destination"])
        try:
            out = process_scene(sp, mp, cfg, rng=None, split="val",
                                bounds=frozen_bounds, augment_override=False)
        except Exception as e:  # noqa: BLE001
            _fail(quarantine, f"{scene_id}:SCENE", scene_id, r["split"],
                  f"scene preprocessing failed: {type(e).__name__}: {e}", r["image_source"])
            stats["rejected"] += 1
            continue
        stats["scenes"] += 1
        img, msk = out["normalized"], out["mask"]
        meta = out["meta"]
        for t in tiling.tile_pair(img, msk, scene_id, meta["transform"], meta["crs"],
                                  size=size, stride=stride):
            oy, ox = t["offset_y"], t["offset_x"]
            # tile_id is split-qualified: bare scene IDs (e.g. 00000.tif) repeat
            # across train/val/test by dataset design; unqualified IDs collide
            # across splits and corrupt index merges.
            tid = (f"{r['split']}_{r['class']}_"
                   f"{os.path.splitext(r['image_filename'])[0]}_r{oy}_c{ox}")
            h, w = t["image"].shape[-2:]
            fg = int((t["mask"] != 0).sum())
            oil_frac = fg / (h * w)
            rec = {"tile_id": tid, "scene_id": scene_id, "split": r["split"], "class": r["class"],
                   "image_path": f"tiled/{r['split']}/images/{tid}.tif",
                   "mask_path": f"tiled/{r['split']}/masks/{tid}.tif",
                   "row_start": oy, "col_start": ox, "row_end": oy + h, "col_end": ox + w,
                   "original_height": meta["height"], "original_width": meta["width"],
                   "tile_height": h, "tile_width": w, "tile_size": size, "stride": stride,
                   "overlap": overlap, "padded": False, "valid_height": h, "valid_width": w,
                   "oil_pixel_count": fg, "oil_pixel_fraction": round(oil_frac, 6),
                   "has_oil": fg > 0, "valid_pixel_fraction": 1.0,
                   "filtering_status": "kept", "filtering_reason": "",
                   "source_image": r["image_source"], "source_mask": r["mask_source"]}
            # --- validation (image, mask, pair); fail -> quarantine, never silent ---
            ok, reason = True, ""
            ia, ma = t["image"], t["mask"]
            if ia.shape != (2, size, size):
                ok, reason = False, f"bad image tile shape {ia.shape}"
            elif ma.shape != (size, size):
                ok, reason = False, f"bad mask tile shape {ma.shape}"
            elif ia.dtype != np.float32:
                ok, reason = False, f"bad image dtype {ia.dtype}"
            elif ma.dtype != np.uint8:
                ok, reason = False, f"bad mask dtype {ma.dtype}"
            elif not np.all(np.isfinite(ia)):
                ok, reason = False, "non-finite SAR values"
            elif ia.min() < 0.0 or ia.max() > 1.0:
                ok, reason = False, f"SAR out of [0,1]: [{ia.min()}, {ia.max()}]"
            elif not set(np.unique(ma).tolist()) <= {0, 1}:
                ok, reason = False, f"non-binary mask {np.unique(ma).tolist()}"
            if not ok:
                rec["filtering_status"] = "rejected"
                rec["filtering_reason"] = reason
                _fail(quarantine, tid, scene_id, r["split"], reason, r["image_source"])
                stats["rejected"] += 1
                tile_rows.append(rec)
                continue
            # --- selection: every oil tile kept; background per BG_KEEP_FRACTION ---
            if fg == 0 and bg_rng.random() >= cfg.get("BG_KEEP_FRACTION", 1.0):
                rec["filtering_status"] = "filtered"
                rec["filtering_reason"] = (f"background subsample "
                                           f"(BG_KEEP_FRACTION={cfg.get('BG_KEEP_FRACTION')})")
                stats["filtered_bg"] += 1
                tile_rows.append(rec)
                continue
            ip = os.path.join(out_root, rec["image_path"])
            mp_ = os.path.join(out_root, rec["mask_path"])
            for path, arr, dt, n in ((ip, ia, "float32", 2), (mp_, ma, "uint8", 1)):
                if os.path.exists(path):
                    with rasterio.open(path) as s:
                        chk = (s.width == size and s.height == size and s.count == n
                               and s.dtypes[0] == dt)
                    if not chk:
                        _fail(quarantine, tid, scene_id, r["split"],
                              f"existing tile failed verification: {path}", r["image_source"])
                        rec["filtering_status"] = "rejected"
                        rec["filtering_reason"] = "existing tile corrupt"
                        stats["rejected"] += 1
                else:
                    write_tile(path, arr, t["crs"], t["transform"], dt, n)
            if rec["filtering_status"] == "kept":
                stats["tiles"] += 1
                stats["kept"] += 1
                stats["oil_px"] += fg
                if fg > 0:
                    stats["oil_tiles"] += 1
                else:
                    stats["bg_tiles"] += 1
            tile_rows.append(rec)
    return tile_rows, quarantine, stats
