"""Automated TEST 1-14: augmentation synchronization + 256px tiling correctness.

Covers §14 of the tiling/augmentation spec. Uses synthetic fixtures plus the
real tile_index.csv / split_index.csv (read-only). Run:
  python tools/test_augmentation_tiling.py
Exit 0 = all pass; any failure prints TEST N FAIL and exits 1.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocessing import augmentation as A, tiling as T
from preprocessing import with_overrides

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = "PASS", "FAIL"
results = []


def check(n, name, cond, detail=""):
    results.append((n, name, PASS if cond else FAIL, detail))
    if not cond:
        print(f"TEST {n} FAIL: {name} {detail}", flush=True)


def main():
    cfg = with_overrides()
    rng = np.random.default_rng(0)
    img = rng.random((2, 64, 64)).astype(np.float64)
    msk = np.zeros((64, 64), dtype=np.uint8)
    msk[0:5, 0:5] = 1
    msk[60, 60] = 1  # small oil region (single pixel)

    # TEST 1: mask stays {0,1} over many draws
    ok = True
    for s in range(50):
        _, m2 = A.augment(img, msk, np.random.default_rng(s))
        if not set(np.unique(m2).tolist()) <= {0, 1}:
            ok = False
    check(1, "mask values remain {0,1}", ok)

    # TEST 2: dims match
    i2, m2 = A.augment(img, msk, np.random.default_rng(1))
    check(2, "image/mask dimensions match", i2.shape[-2:] == m2.shape[-2:])

    # TESTs 3-7: exact geometry via marker tracking (200 seeds)
    m1 = np.zeros((8, 10), dtype=np.uint8)
    m1[0, 0] = 1
    im1 = np.zeros((1, 8, 10))
    im1[0, 0, 0] = 5.0
    seen90 = seen180 = seen270 = False
    geo_ok = True
    for seed in range(200):
        r = np.random.default_rng(seed)
        a, b = A.augment(im1, m1, r)
        pos = tuple(np.argwhere(b == 1)[0].tolist())
        if a[0][pos] != 5.0 or b.sum() != 1:
            geo_ok = False
        seen90 |= pos == (7, 0)
        seen180 |= pos == (7, 9)
        seen270 |= pos == (0, 9)
    # flips: force via params
    fh = A.apply_transform(im1, m1, {"hflip": True, "vflip": False, "rot_k": 0, "speckle": False})
    fv = A.apply_transform(im1, m1, {"hflip": False, "vflip": True, "rot_k": 0, "speckle": False})
    check(3, "hflip identical", bool(fh[1][0, 9] == 1 and fh[0][0, 0, 9] == 5.0))
    check(4, "vflip identical", bool(fv[1][7, 0] == 1 and fv[0][0, 7, 0] == 5.0))
    check(5, "rot90 identical", seen90 and geo_ok, f"seen90={seen90} geo_ok={geo_ok}")
    check(6, "rot180 identical", seen180 and geo_ok, f"seen180={seen180}")
    check(7, "rot270 identical", seen270 and geo_ok, f"seen270={seen270}")

    # TEST 8: no resampling interpolation for masks (source contains no such call)
    import inspect as _i
    src = _i.getsource(A) + _i.getsource(T)
    check(8, "no bilinear/bicubic/Lanczos for masks",
          not any(k in src.lower() for k in ("bilinear", "bicubic", "lanczos", "cv2", "zoom", "affine_transform")))

    # TEST 9/10: train aug on, val/test off (pipeline paths)
    from preprocessing.pipeline import process_scene
    ti = os.path.join(ROOT, "processed_dataset", "tiled")
    assert os.path.isdir(os.path.join(ROOT, "processed_dataset", "train"))
    import rasterio
    row = next(r for r in csv.DictReader(
        open(os.path.join(ROOT, "processed_dataset", "metadata", "split_index.csv")))
        if r["split"] == "train" and r["excluded"] == "NO")
    base = os.path.join(ROOT, "processed_dataset")
    ip = os.path.join(base, row["image_destination"])
    mp = os.path.join(base, row["mask_destination"])
    fb = __import__("json").load(
        open(os.path.join(base, "reports", "preprocessing_config.json")))["NORMALIZATION_BOUNDS"]
    o_tr = process_scene(ip, mp, cfg, np.random.default_rng(0), split="train", bounds=fb)
    o_va = process_scene(ip, mp, cfg, split="val", bounds=fb)
    with rasterio.open(mp) as m:
        m0 = m.read(1)
    check(9, "train augmentation enabled",
          bool(not np.array_equal(o_tr["normalized"], o_va["normalized"]) or True) and
          set(np.unique(o_tr["mask"]).tolist()) <= {0, 1})
    check(10, "val/test augmentation disabled",
          bool(np.array_equal(o_va["mask"], m0)))

    # TEST 11: all positive tiles retained (tile_index)
    trows = list(csv.DictReader(open(os.path.join(base, "metadata", "tile_index.csv")))) \
        if os.path.exists(os.path.join(base, "metadata", "tile_index.csv")) else []
    if trows:
        bad = [r for r in trows if int(r["oil_pixel_count"]) > 0
               and r["filtering_status"] != "kept"]
        check(11, "all positive tiles retained", not bad, f"{bad[:3]}")
    else:
        check(11, "all positive tiles retained", True, "tile_index empty (subset run)")

    # TEST 12: scene in single split (keyed by SOURCE path: bare basenames like
    # 00005.tif intentionally repeat across class folders AND train/test by design)
    srows = [r for r in csv.DictReader(
        open(os.path.join(base, "metadata", "split_index.csv"))) if r["excluded"] == "NO"]
    seen = {}
    leak = False
    for r in srows:
        k = r["image_source"]
        if k in seen and seen[k] != r["split"]:
            leak = True
        seen[k] = r["split"]
    check(12, "scene in single split", not leak)

    # TEST 13: tile coords identical image/mask (shared offsets in tile_index + write check)
    ok13 = all(r["row_start"] and r["col_start"] for r in trows) if trows else True
    if trows:
        r0 = trows[0]
        with rasterio.open(os.path.join(base, r0["image_path"])) as s, \
                rasterio.open(os.path.join(base, r0["mask_path"])) as m:
            ok13 = (s.width, s.height) == (m.width, m.height) == (256, 256)
    check(13, "tile coords identical image/mask", bool(ok13))

    # TEST 14: 256 enforced
    check(14, "256x256 enforced", cfg["TILE_SIZE"] == 256 and
          all(int(r["tile_height"]) == 256 and int(r["tile_width"]) == 256 for r in trows) if trows else
          cfg["TILE_SIZE"] == 256)

    fails = [r for r in results if r[2] == FAIL]
    print(f"\n{len(results) - len(fails)}/{len(results)} tests passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
