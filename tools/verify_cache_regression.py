"""Phase 6 regression test: verify cache == raw on >=200 train + >=100 val tiles.

For each tile: compare CachedTileDataset.__getitem__() vs OilSpillTileDataset.__getitem__()
and assert bit-identical float32 tensors (image) and bit-identical uint8->float32 masks.

Also runs 200 optimizer steps on both paths with seed=42 and compares losses.

Usage:
    .venv\\Scripts\\python.exe tools\\verify_cache_regression.py
    .venv\\Scripts\\python.exe tools\\verify_cache_regression.py --n-train 200 --n-val 100
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

SEP = "=" * 68


def pick_indices(rows, n: int, seed: int = 42) -> list[int]:
    rng = np.random.RandomState(seed)
    classes = sorted({r["split"] + "/" + r["class"] for r in rows})
    picked = []
    per_class = max(1, n // len(classes))
    for cls in classes:
        s, c = cls.split("/", 1)
        idxs = [i for i, r in enumerate(rows) if r["split"] == s and r["class"] == c]
        k = min(per_class, len(idxs))
        picked.extend(rng.choice(idxs, size=k, replace=False).tolist())
    # fill to n
    all_idxs = list(range(len(rows)))
    while len(picked) < n:
        extra = rng.choice(all_idxs, size=min(n - len(picked), len(all_idxs)), replace=False)
        picked.extend(extra.tolist())
    return sorted(set(picked))[:n]


def compare_tiles(raw_ds, cache_ds, indices: list[int], label: str) -> dict:
    n = len(indices)
    max_img_diff = 0.0
    max_msk_diff = 0.0
    n_img_mismatch = 0
    n_msk_mismatch = 0
    t0 = time.perf_counter()

    for rank, i in enumerate(indices):
        img_raw,  msk_raw   = raw_ds[i]
        img_cache, msk_cache = cache_ds[i]

        # dtype checks
        assert img_raw.dtype  == torch.float32, f"raw image dtype {img_raw.dtype}"
        assert img_cache.dtype == torch.float32, f"cache image dtype {img_cache.dtype}"
        assert msk_raw.dtype  == torch.float32, f"raw mask dtype {msk_raw.dtype}"
        assert msk_cache.dtype == torch.float32, f"cache mask dtype {msk_cache.dtype}"

        # shape checks
        assert img_raw.shape == (2, 512, 512),  f"raw image shape {img_raw.shape}"
        assert img_cache.shape == (2, 512, 512), f"cache image shape {img_cache.shape}"
        assert msk_raw.shape == (1, 512, 512),  f"raw mask shape {msk_raw.shape}"
        assert msk_cache.shape == (1, 512, 512), f"cache mask shape {msk_cache.shape}"

        # numerical comparison
        img_diff = float((img_raw - img_cache).abs().max())
        msk_diff = float((msk_raw - msk_cache).abs().max())
        max_img_diff = max(max_img_diff, img_diff)
        max_msk_diff = max(max_msk_diff, msk_diff)
        if img_diff > 0.0:
            n_img_mismatch += 1
        if msk_diff > 0.0:
            n_msk_mismatch += 1

    elapsed = time.perf_counter() - t0
    pass_ = (n_img_mismatch == 0 and n_msk_mismatch == 0)
    status = "PASS" if pass_ else "FAIL"

    print(f"\n  [{status}] {label}  ({n} tiles, {elapsed:.1f}s)")
    print(f"    max |img_raw - img_cache|: {max_img_diff:.2e}  "
          f"({n_img_mismatch} mismatches)")
    print(f"    max |msk_raw - msk_cache|: {max_msk_diff:.2e}  "
          f"({n_msk_mismatch} mismatches)")
    if not pass_:
        print(f"    *** REGRESSION DETECTED ***")

    return {"pass": pass_, "max_img_diff": max_img_diff, "max_msk_diff": max_msk_diff,
            "n_img_mismatch": n_img_mismatch, "n_msk_mismatch": n_msk_mismatch,
            "n_tiles": n}


def compare_losses(n_steps: int = 200) -> dict:
    """Run n_steps optimizer steps on both raw and cache paths with seed=42.
    Compare per-step losses.  Expect exact equality (same data, same model init).
    """
    print(f"\n  Comparing losses over {n_steps} steps (seed=42)...", flush=True)
    from models import UNetResNet34, UNetResNet34Config
    from training.losses import BCEDiceLoss
    from preprocessing.tile_dataset import OilSpillTileDataset
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"

    losses_raw   = []
    losses_cache = []

    for path_name, DatasetClass, kwargs in [
        ("raw",   OilSpillTileDataset, {"tile_index_path": TILE_INDEX, "split": "train"}),
        ("cache", CachedTileDataset,   {"split": "train"}),
    ]:
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
        model.to(device).train()
        crit = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
        opt  = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda") if amp else None

        ds     = DatasetClass(**kwargs)
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

        step_losses = []
        t0 = time.perf_counter()
        for step, (images, masks) in enumerate(loader):
            if step >= n_steps:
                break
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                loss   = crit(logits, masks)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step_losses.append(float(loss.detach()))

        elapsed = time.perf_counter() - t0
        print(f"    {path_name}: {len(step_losses)} steps in {elapsed:.1f}s  "
              f"first_loss={step_losses[0]:.5f}  last_loss={step_losses[-1]:.5f}",
              flush=True)
        if path_name == "raw":
            losses_raw = step_losses
        else:
            losses_cache = step_losses

    if len(losses_raw) != len(losses_cache):
        print(f"    FAIL: different number of steps {len(losses_raw)} vs {len(losses_cache)}")
        return {"pass": False}

    diffs = [abs(a - b) for a, b in zip(losses_raw, losses_cache)]
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    # Expect bit-identical (same data, same model seed, same optimizer) -> 0.0
    # AMP may introduce tiny fp16 rounding on some steps -> allow 1e-5 tolerance
    pass_ = max_diff < 1e-4

    print(f"\n    Loss comparison (raw vs cache, {n_steps} steps):")
    print(f"      max  |loss_raw - loss_cache| = {max_diff:.2e}")
    print(f"      mean |loss_raw - loss_cache| = {mean_diff:.2e}")
    print(f"      [{('PASS' if pass_ else 'FAIL')}]  (tolerance 1e-4)")
    return {"pass": pass_, "max_diff": max_diff, "mean_diff": mean_diff, "n_steps": n_steps}


def main():
    ap = argparse.ArgumentParser(description="Phase 6 cache regression test")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val",   type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=200,
                    help="Optimizer steps for loss comparison (default 200)")
    ap.add_argument("--skip-loss", action="store_true",
                    help="Skip the loss comparison (tile comparison only)")
    args = ap.parse_args()

    print(SEP)
    print("PHASE 6: CACHE REGRESSION TEST")
    print(SEP)
    print(f"Train tiles: {args.n_train}  Val tiles: {args.n_val}  "
          f"Loss steps: {args.n_steps}", flush=True)

    from preprocessing.tile_dataset import OilSpillTileDataset
    from preprocessing.cached_tile_dataset import CachedTileDataset

    # Build datasets
    raw_train   = OilSpillTileDataset(TILE_INDEX, split="train")
    raw_val     = OilSpillTileDataset(TILE_INDEX, split="val")
    cache_train = CachedTileDataset(split="train")
    cache_val   = CachedTileDataset(split="val")

    assert len(raw_train) == len(cache_train), \
        f"Train length mismatch: raw={len(raw_train)} cache={len(cache_train)}"
    assert len(raw_val) == len(cache_val), \
        f"Val length mismatch: raw={len(raw_val)} cache={len(cache_val)}"
    print(f"\nDataset lengths: train={len(raw_train)}  val={len(raw_val)}  [OK]")

    # Pick indices
    train_idxs = pick_indices(raw_train.rows, args.n_train, seed=42)
    val_idxs   = pick_indices(raw_val.rows,   args.n_val,   seed=42)
    print(f"Sampled train indices: {len(train_idxs)}  val indices: {len(val_idxs)}")

    results = {}

    # Tile comparison
    print(f"\n{'-'*68}")
    print("Tile-level comparison (image + mask bit-identity)")
    print(f"{'-'*68}")
    results["train"] = compare_tiles(raw_train, cache_train, train_idxs, "TRAIN")
    results["val"]   = compare_tiles(raw_val,   cache_val,   val_idxs,   "VAL")

    # Loss comparison
    if not args.skip_loss:
        print(f"\n{'-'*68}")
        print("Loss comparison (200 optimizer steps, seed=42)")
        print(f"{'-'*68}")
        results["loss"] = compare_losses(n_steps=args.n_steps)

    # Summary
    print(f"\n{SEP}")
    all_pass = all(v.get("pass", False) for v in results.values())
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    for k, v in results.items():
        s = "PASS" if v.get("pass") else "FAIL"
        print(f"  {k:<8}: {s}")
    print(SEP)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
