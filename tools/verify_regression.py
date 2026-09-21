"""Numerical regression test for the optimized preprocessing pipeline.

Verifies that the compute_stats=False optimization does NOT change any tensor
values returned by OilSpillTileDataset.__getitem__(). The only change is that
process_scene()'s returned 'stats' dict is None when compute_stats=False; the
'normalized' array and 'mask' array must be bit-identical.

Test strategy:
1. Pick N representative tile IDs (fixed seed=42, across splits / classes).
2. Load each tile with the CURRENT pipeline (compute_stats=False, i.e. after fix).
3. Also call process_scene with compute_stats=True on the same scene and compare
   the normalized/mask arrays directly.
4. Report: shape, dtype, min, max, mean, std, mask unique values for each tile.
5. Assert numerical tolerance: images must match exactly (identical float32 arrays);
   masks must match exactly (binary).

Usage (from project root):
    .venv\\Scripts\\python.exe tools\\verify_regression.py
    .venv\\Scripts\\python.exe tools\\verify_regression.py --n-tiles 20 --split train
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILE_INDEX = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
PREP_CFG   = os.path.join(ROOT, "processed_dataset", "metadata", "preprocessing_config.json")


def pick_tiles(rows, n=10, seed=42):
    """Pick n tile indices deterministically across classes."""
    rng = np.random.RandomState(seed)
    classes = sorted({r["class"] for r in rows})
    chosen = []
    per_class = max(1, n // len(classes))
    for cls in classes:
        cls_rows = [i for i, r in enumerate(rows) if r["class"] == cls]
        k = min(per_class, len(cls_rows))
        chosen.extend(rng.choice(cls_rows, size=k, replace=False).tolist())
    # fill remaining if needed
    remaining = n - len(chosen)
    if remaining > 0:
        all_indices = list(range(len(rows)))
        extra = rng.choice(all_indices, size=remaining, replace=False).tolist()
        chosen.extend(extra)
    return sorted(set(chosen))[:n]


def verify_tile(ds, idx, verbose=True):
    """Compare compute_stats=False (fast) output vs compute_stats=True (reference)."""
    from preprocessing.pipeline import process_scene

    row = ds.rows[idx]
    ipath = os.path.join(ds.root, row["image_path"])
    mpath = os.path.join(ds.root, row["mask_path"])
    x = int(row["x"])
    y = int(row["y"])
    w = int(row["tile_width"])
    h = int(row["tile_height"])

    # Reference: compute_stats=True
    out_ref = process_scene(
        ipath, mpath, ds.cfg, rng=None, split=ds.split,
        bounds=ds.bounds, augment_override=False, compute_stats=True)
    norm_ref = np.asarray(out_ref["normalized"], dtype=np.float32)
    mask_ref = np.asarray(out_ref["mask"])

    # Optimized: compute_stats=False
    out_opt = process_scene(
        ipath, mpath, ds.cfg, rng=None, split=ds.split,
        bounds=ds.bounds, augment_override=False, compute_stats=False)
    norm_opt = np.asarray(out_opt["normalized"], dtype=np.float32)
    mask_opt = np.asarray(out_opt["mask"])

    # Check stats dict behavior
    assert out_ref["stats"] is not None, "Reference: stats should not be None"
    assert out_opt["stats"] is None,     "Optimized: stats should be None (skipped)"

    # Numerical comparison: must be exactly identical
    img_identical  = np.array_equal(norm_ref, norm_opt)
    mask_identical = np.array_equal(mask_ref, mask_opt)

    # Tile-level stats (from optimized path)
    tile_img  = norm_opt[:, y:y + h, x:x + w].astype(np.float32)
    tile_mask = mask_opt[y:y + h, x:x + w]

    result = {
        "tile_id": row["tile_id"],
        "split": row["split"],
        "class": row["class"],
        "image_id": row["image_id"],
        "image_shape": tuple(tile_img.shape),
        "mask_shape": tuple(tile_mask.shape),
        "image_dtype": str(tile_img.dtype),
        "mask_dtype": str(tile_mask.dtype),
        "image_min": float(np.min(tile_img)),
        "image_max": float(np.max(tile_img)),
        "image_mean": float(np.mean(tile_img)),
        "image_std": float(np.std(tile_img)),
        "mask_unique": sorted(np.unique(tile_mask).tolist()),
        "img_identical_to_ref": img_identical,
        "mask_identical_to_ref": mask_identical,
        "PASS": img_identical and mask_identical,
    }

    if verbose:
        status = "PASS" if result["PASS"] else "FAIL"
        print(f"  [{status}] {row['tile_id']}")
        print(f"         shape={result['image_shape']}  dtype={result['image_dtype']}")
        print(f"         img  min={result['image_min']:.4f}  max={result['image_max']:.4f}"
              f"  mean={result['image_mean']:.4f}  std={result['image_std']:.4f}")
        print(f"         mask dtype={result['mask_dtype']}  unique={result['mask_unique']}")
        if not img_identical:
            diff = np.abs(norm_ref - norm_opt)
            print(f"         !! IMAGE MISMATCH: max_diff={diff.max():.6e}  "
                  f"mean_diff={diff.mean():.6e}")
        if not mask_identical:
            print(f"         !! MASK MISMATCH")

    return result


def main():
    ap = argparse.ArgumentParser(description="Numerical regression test for optimized pipeline")
    ap.add_argument("--n-tiles", type=int, default=10,
                    help="Number of tiles to verify (default 10)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 60)
    print("NUMERICAL REGRESSION VERIFICATION")
    print("=" * 60)
    print(f"Split       : {args.split}")
    print(f"Tiles       : {args.n_tiles}")
    print(f"Seed        : {args.seed}")
    print(f"Tile index  : {TILE_INDEX}", flush=True)

    from preprocessing.tile_dataset import OilSpillTileDataset
    ds = OilSpillTileDataset(TILE_INDEX, split=args.split, scene_cache_size=2)
    print(f"Dataset rows: {len(ds.rows)}", flush=True)

    indices = pick_tiles(ds.rows, n=args.n_tiles, seed=args.seed)
    print(f"Selected indices: {indices}\n", flush=True)

    results = []
    t0 = time.perf_counter()
    for idx in indices:
        try:
            r = verify_tile(ds, idx, verbose=True)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] idx={idx}: {e}")
            import traceback; traceback.print_exc()
            results.append({"tile_id": f"idx_{idx}", "PASS": False, "error": str(e)})
    elapsed = time.perf_counter() - t0

    n_pass = sum(1 for r in results if r.get("PASS"))
    n_fail = len(results) - n_pass

    print("\n" + "=" * 60)
    print(f"RESULTS: {n_pass}/{len(results)} PASS  ({n_fail} FAIL)  "
          f"time={elapsed:.1f}s")
    print("=" * 60)

    # Verify expected shapes and dtypes
    print("\nExpected properties check:")
    for r in results:
        if not r.get("PASS"):
            continue
        ok = True
        errs = []
        if r.get("image_shape") != (2, 512, 512):
            ok = False; errs.append(f"shape={r['image_shape']} expected (2,512,512)")
        if r.get("mask_shape") != (512, 512):
            ok = False; errs.append(f"mask_shape={r['mask_shape']} expected (512,512)")
        if r.get("image_dtype") != "float32":
            ok = False; errs.append(f"dtype={r['image_dtype']} expected float32")
        img_min = r.get("image_min", 0)
        img_max = r.get("image_max", 1)
        if not (-0.01 <= img_min and img_max <= 1.01):
            ok = False; errs.append(f"values out of [0,1]: min={img_min:.4f} max={img_max:.4f}")
        mu = r.get("mask_unique", [])
        if not set(mu) <= {0, 1}:
            ok = False; errs.append(f"mask values {mu} not subset of {{0,1}}")
        tag = "OK" if ok else "WARN"
        msg = "  all expected properties correct" if ok else f"  {'; '.join(errs)}"
        print(f"  [{tag}] {r['tile_id']}{msg}")

    if n_fail > 0:
        print("\nFAILED tiles:")
        for r in results:
            if not r.get("PASS"):
                print(f"  {r.get('tile_id','?')}: {r.get('error','image or mask mismatch')}")
        sys.exit(1)
    else:
        print("\nAll tiles: image and mask arrays are BIT-IDENTICAL between "
              "compute_stats=True (reference) and compute_stats=False (optimized).")
        print("Optimization is numerically safe.")
        sys.exit(0)


if __name__ == "__main__":
    main()
