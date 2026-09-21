"""Build metadata-only 512px tile index for on-the-fly training (no pixels saved).

Reuses preprocessing.tiling.tile_grid (edge-overlap: full-size shifted edge
windows; no partial windows, padding, or dropped pixels). Reads scene dims from
headers + mask scenes in memory for oil/background classification. Writes
processed_dataset/metadata/tile_index.csv (replacing the stale physical-tile
index) + reports/tile_index_report.json. Creates NO image files.
Usage: python tools/build_tile_index.py [--tile-size 512] [--stride 512]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio

from preprocessing.tiling import tile_grid

COLS = ["tile_id", "split", "class", "image_id", "image_path", "mask_path",
        "x", "y", "tile_width", "tile_height", "original_width", "original_height"]


def cover_1d(n, size, stride):
    covered = np.zeros(n, dtype=bool)
    for s in tile_grid(n, 1, size, stride):
        covered[s[0]:s[0] + size] = True
    return covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--dataset", default="processed_dataset")
    a = ap.parse_args()
    if a.tile_size != 512 or a.stride != 512:
        print(f"NOTE: non-default geometry size={a.tile_size} stride={a.stride} "
              f"(defaults required: 512/512)")
    root = os.path.abspath(a.dataset)
    rows = [r for r in csv.DictReader(open(os.path.join(root, "metadata", "split_index.csv")))
            if r["excluded"] == "NO"]
    rows.sort(key=lambda r: (r["split"], r["class"], r["image_filename"]))

    tiles, per_scene, skipped, invalid = [], {}, [], []
    counts = {"scenes": 0, "tiles": 0, "oil": 0, "bg": 0}
    for r in rows:
        iid = r["image_source"]
        ipath = os.path.join(root, r["image_destination"])
        mpath = os.path.join(root, r["mask_destination"])
        if not (os.path.isfile(ipath) and os.path.isfile(mpath)):
            invalid.append({"image_id": iid, "reason": "missing scene image or mask file"})
            continue
        try:
            with rasterio.open(ipath) as s:
                H, W = s.height, s.width
            with rasterio.open(mpath) as m:
                mk = m.read(1)
        except Exception as e:  # noqa: BLE001
            invalid.append({"image_id": iid, "reason": f"unreadable: {type(e).__name__}"})
            continue
        if mk.shape != (H, W):
            invalid.append({"image_id": iid, "reason": "mask/scene dimension mismatch"})
            continue
        if H < a.tile_size or W < a.tile_size:
            invalid.append({"image_id": iid, "reason": f"scene smaller than tile: {W}x{H}"})
            continue
        grid = tile_grid(H, W, a.tile_size, a.stride)
        # edge coverage audit (1D union per axis; overlap-shift policy => full cover)
        cov_r, cov_c = cover_1d(H, a.tile_size, a.stride), cover_1d(W, a.tile_size, a.stride)
        miss_r, miss_c = int((~cov_r).sum()), int((~cov_c).sum())
        if miss_r or miss_c:
            skipped.append({"image_id": iid, "original_width": W, "original_height": H,
                            "right_edge_pixels": miss_c, "bottom_edge_pixels": miss_r,
                            "reason": "existing_tiler_leaves_gap (unexpected)"})
        stem = os.path.splitext(r["image_filename"])[0]
        n_oil = 0
        for (y, x) in grid:
            assert x >= 0 and y >= 0 and x + a.tile_size <= W and y + a.tile_size <= H, iid
            tid = f"{r['split']}_{r['class']}_{stem}_x{x:04d}_y{y:04d}"
            oil = bool((mk[y:y + a.tile_size, x:x + a.tile_size] == 1).any())
            n_oil += oil
            tiles.append({"tile_id": tid, "split": r["split"], "class": r["class"],
                          "image_id": iid, "image_path": r["image_destination"],
                          "mask_path": r["mask_destination"], "x": x, "y": y,
                          "tile_width": a.tile_size, "tile_height": a.tile_size,
                          "original_width": W, "original_height": H})
        per_scene[iid] = {"split": r["split"], "class": r["class"],
                          "tile_count": len(grid), "empty_mask_tiles": len(grid) - n_oil,
                          "oil_containing_tiles": n_oil}
        counts["scenes"] += 1
        counts["tiles"] += len(grid)
        counts["oil"] += n_oil
        counts["bg"] += len(grid) - n_oil
    assert len({t["tile_id"] for t in tiles}) == len(tiles), "tile_id collision"

    tip = os.path.join(root, "metadata", "tile_index.csv")
    with open(tip, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(tiles)

    def agg(key, val):
        d = {}
        for t in tiles:
            d.setdefault(t[key], []).append(t)
        out = {}
        for k, v in sorted(d.items()):
            if val == "count":
                out[k] = len(v)
            else:
                out[k] = sum(1 for x in v if (int((x.get("oil")) if False else 0)))
        return out

    by_split = {}
    by_class = {}
    for sp in ("train", "val", "test"):
        by_split[sp] = sum(1 for t in tiles if t["split"] == sp)
        by_class[sp] = {c: sum(1 for t in tiles if t["split"] == sp and t["class"] == c)
                        for c in ("Oil", "No_oil", "Lookalike")}
    report = {
        "configuration": {"tile_size": a.tile_size, "train_stride": a.stride,
                          "edge_policy": "existing tile_grid overlap-shift: full-size edge "
                                         "windows, no partial windows, no padding, no dropped pixels",
                          "semantics": "coordinates index preprocessed scenes; loader calls "
                                       "process_scene() then crops (x,y,w,h); masks sliced, never resampled"},
        "scenes_per_split": {sp: sum(1 for r in rows if r["split"] == sp)
                             for sp in ("train", "val", "test")},
        "total_tiles_per_split": by_split,
        "tiles_per_class": by_class,
        "tiles_per_scene": per_scene,
        "skipped_edge_regions": skipped,
        "invalid_scenes": invalid,
        "empty_mask_tile_counts":
            {sp: sum(v["empty_mask_tiles"] for k, v in per_scene.items() if v["split"] == sp)
             for sp in ("train", "val", "test")},
        "oil_containing_tile_counts":
            {sp: sum(v["oil_containing_tiles"] for k, v in per_scene.items() if v["split"] == sp)
             for sp in ("train", "val", "test")},
        "background_only_tile_counts":
            {sp: sum(v["empty_mask_tiles"] for k, v in per_scene.items() if v["split"] == sp)
             for sp in ("train", "val", "test")},
    }
    rep = os.path.join(root, "reports", "tile_index_report.json")
    json.dump(report, open(rep, "w"), indent=1)
    h1 = hashlib.sha256(open(tip, "rb").read()).hexdigest()
    print(json.dumps({"tiles": len(tiles), "scenes": counts["scenes"],
                      "oil_tiles": counts["oil"], "bg_tiles": counts["bg"],
                      "skipped": skipped, "invalid": invalid,
                      "sha256": h1}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
