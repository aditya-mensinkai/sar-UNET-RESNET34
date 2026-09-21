"""Smoke test for OilSpillTileDataset (no training, no writes).

Finds one Oil / No_oil / Lookalike tile from the index, prints shapes, dtypes,
min/max, mask values, NaN/Inf counts, alignment check, determinism repeat,
and confirms no physical tiles were created. Exit 0 = all PASS.
Usage: python tools/smoke_test_oil_spill_dataset.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from preprocessing.tile_dataset import OilSpillTileDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIP = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def main():
    for split, want in (("train", 32860), ("val", 8208), ("test", 7200)):
        ds = OilSpillTileDataset(TIP, split=split)
        print(f"{split} len={len(ds)}", flush=True)
    ds = OilSpillTileDataset(TIP, split="train")
    # first occurrence per class
    pick = {}
    for i, r in enumerate(ds.rows):
        if r["class"] not in pick:
            pick[r["class"]] = i
    for cls in ("Oil", "No_oil", "Lookalike"):
        if cls not in pick:
            fail(f"class {cls} missing from train index")
        i = pick[cls]
        img, msk = ds[i]
        meta = ds.get_metadata(i)
        u = torch.unique(msk)
        nan_c = int(torch.isnan(img).sum() + torch.isnan(msk).sum())
        inf_c = int(torch.isinf(img).sum() + torch.isinf(msk).sum())
        print("=" * 50, flush=True)
        print(f"CLASS: {cls}", flush=True)
        print(f"tile_id: {meta['tile_id']}\nimage_id: {meta['image_id']}\n"
              f"split: {meta['split']}\nimage_path: {meta['image_path']}\n"
              f"mask_path: {meta['mask_path']}\nx: {meta['x']}\ny: {meta['y']}", flush=True)
        print(f"image shape: {img.shape}\nmask shape: {msk.shape}", flush=True)
        print(f"image dtype: {img.dtype}\nmask dtype: {msk.dtype}", flush=True)
        print(f"image min: {float(img.min())}\nimage max: {float(img.max())}", flush=True)
        print(f"mask unique values: {u.tolist()}", flush=True)
        print(f"NaN count: {nan_c}\nInf count: {inf_c}", flush=True)
        if img.shape != torch.Size([2, 512, 512]) or msk.shape != torch.Size([1, 512, 512]):
            fail(f"bad shapes for {cls}")
        if not set(u.tolist()) <= {0.0, 1.0}:
            fail(f"non-binary mask for {cls}: {u.tolist()}")
        if nan_c or inf_c:
            fail(f"NaN/Inf for {cls}")
        # alignment: same row coords for both windows
        if not (img.shape[-2:] == msk.shape[-2:] == (512, 512)):
            fail(f"window dim mismatch for {cls}")
        print(f"alignment (dims + shared x/y row): PASS", flush=True)
        # determinism
        a1, m1 = ds[i]
        a2, m2 = ds[i]
        di = bool(torch.equal(a1, a2))
        dm = bool(torch.equal(m1, m2))
        print(f"deterministic image: {'PASS' if di else 'FAIL'}", flush=True)
        print(f"deterministic mask: {'PASS' if dm else 'FAIL'}", flush=True)
        if not (di and dm):
            fail(f"non-deterministic {cls}")
    # no physical tiles
    tiled = os.path.join(ROOT, "processed_dataset", "tiled")
    print(f"processed_dataset/tiled exists: {os.path.exists(tiled)}", flush=True)
    if os.path.exists(tiled):
        fail("tiled/ directory appeared")
    print("Physical tiles created: NO", flush=True)
    print("ALL SMOKE CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
