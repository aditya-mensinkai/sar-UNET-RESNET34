"""Batch size throughput benchmark: 2, 4, 8, 16.

Protocol:
  - fused Adam + cudnn.benchmark + num_workers=1
  - 20 warmup batches + 240 measured batches per size
  - Reports: ms/batch, tiles/sec, peak GPU memory
  - Stops at first OOM
  - Gate: if tiles/sec at best size < 10% above bs=2 → report and exit
  - If gate passes: launch 3-epoch pilot at bs=8 lr=2e-4 vs bs=2 lr=1e-4
    using the real CachedTileDataset and compare val_loss + val_dice
    at the same wall-clock time.

Usage:
  python -m tools.batch_benchmark
  (must be run from the project root with .venv activated)
"""
from __future__ import annotations

import os
import sys
import time
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

WARMUP = 20
MEASURED = 240


def _make_fake_loader(n_batches, batch_size, device):
    class _FL:
        def __init__(self, n, bs, dev):
            self.n, self.bs, self.dev = n, bs, dev
            self.scene_sampler = None
        def __len__(self): return self.n
        def __iter__(self):
            for _ in range(self.n):
                imgs = torch.randn(self.bs, 2, 256, 256, device=self.dev)
                msks = (torch.rand(self.bs, 1, 256, 256, device=self.dev) > 0.97).float()
                yield imgs, msks
    return _FL(n_batches, batch_size, device)


def bench_one(batch_size, device):
    """Returns (mean_ms, tiles_sec, peak_vram_mb) or raises torch.cuda.OutOfMemoryError."""
    from models import UNetResNet34, UNetResNet34Config
    from training.losses import BCEDiceLoss

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device)
    model.train()

    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    crit = BCEDiceLoss(0.5, 0.5, 1.0)
    scaler = torch.amp.GradScaler("cuda")
    loader = _make_fake_loader(WARMUP + MEASURED, batch_size, device)

    times = []
    for bi, (imgs, msks) in enumerate(loader):
        if bi >= WARMUP + MEASURED:
            break
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(imgs)
            loss = crit(logits, msks)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if bi >= WARMUP:
            times.append(ms)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    mean_ms = float(np.mean(times))
    tiles_sec = (batch_size * 1000.0) / mean_ms
    del model, opt, crit, scaler
    torch.cuda.empty_cache()
    return mean_ms, tiles_sec, peak_mb


def run_mini_pilot(batch_size, lr, n_epochs, device, max_wall_sec):
    """
    Run a short pilot using CachedTileDataset (real data).
    Stops after n_epochs OR max_wall_sec, whichever is first.
    Returns dict with val_loss, val_dice, elapsed_sec, epochs_done.
    """
    from models import UNetResNet34, UNetResNet34Config
    from training.losses import BCEDiceLoss
    from training.metrics import ConfusionAccumulator
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device)

    try:
        opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    crit = BCEDiceLoss(0.5, 0.5, 1.0)
    scaler = torch.amp.GradScaler("cuda")

    ds_train = CachedTileDataset(split="train")
    ds_val   = CachedTileDataset(split="val")
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=1, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=1, pin_memory=True, persistent_workers=True)

    t_start = time.time()
    epochs_done = 0
    val_loss_last = None
    val_dice_last = None

    for ep in range(n_epochs):
        if time.time() - t_start > max_wall_sec:
            break
        model.train()
        for imgs, msks in train_loader:
            if time.time() - t_start > max_wall_sec:
                break
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(imgs)
                loss = crit(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # val
        model.eval()
        va_loss, va_acc, va_n = 0.0, ConfusionAccumulator(), 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(imgs)
                    va_loss += float(crit(logits, msks))
                va_acc.add(logits, msks)
                va_n += 1
        vm = va_acc.metrics()
        val_loss_last = va_loss / max(va_n, 1)
        val_dice_last = vm["Dice"]
        epochs_done += 1
        print(f"  [bs={batch_size} lr={lr}] epoch {ep+1}/{n_epochs}  "
              f"val_loss={val_loss_last:.5f}  val_dice={val_dice_last:.5f}  "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

    elapsed = time.time() - t_start
    return {"val_loss": val_loss_last, "val_dice": val_dice_last,
            "elapsed_sec": round(elapsed, 1), "epochs_done": epochs_done}


def main():
    print("=" * 60, flush=True)
    print("BATCH SIZE BENCHMARK  (bs=2,4,8,16)", flush=True)
    print(f"  fused Adam + cudnn.benchmark + num_workers=1", flush=True)
    print(f"  {WARMUP} warmup + {MEASURED} measured batches per size", flush=True)
    print("=" * 60, flush=True)

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device available", flush=True)
        return 1

    device = torch.device("cuda")
    batch_sizes = [2, 4, 8, 16]
    bench_results = {}

    for bs in batch_sizes:
        print(f"\n--- batch_size={bs} ---", flush=True)
        try:
            mean_ms, tiles_sec, peak_mb = bench_one(bs, device)
            bench_results[bs] = {"mean_ms": mean_ms, "tiles_sec": tiles_sec,
                                  "peak_vram_mb": round(peak_mb, 1)}
            print(f"  mean_ms={mean_ms:.1f}  tiles/sec={tiles_sec:.1f}  "
                  f"peak_vram={peak_mb:.0f} MB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs} — stopping sweep", flush=True)
            bench_results[bs] = {"oom": True}
            break

    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'bs':>4}  {'ms/batch':>10}  {'tiles/s':>10}  {'peak_VRAM_MB':>14}", flush=True)
    for bs in batch_sizes:
        r = bench_results.get(bs)
        if r is None:
            continue
        if r.get("oom"):
            print(f"{bs:>4}  {'OOM':>10}", flush=True)
        else:
            print(f"{bs:>4}  {r['mean_ms']:>10.1f}  {r['tiles_sec']:>10.1f}  "
                  f"{r['peak_vram_mb']:>14.0f}", flush=True)

    # Find best (non-OOM) size
    valid = {bs: r for bs, r in bench_results.items() if not r.get("oom")}
    if not valid:
        print("No valid results (all OOM)", flush=True)
        return 1

    best_bs = max(valid, key=lambda bs: valid[bs]["tiles_sec"])
    base_tiles = valid[2]["tiles_sec"] if 2 in valid else None
    best_tiles = valid[best_bs]["tiles_sec"]

    print(f"\nBest batch size: {best_bs}  ({best_tiles:.1f} tiles/s)", flush=True)
    if base_tiles is not None:
        gain_pct = (best_tiles - base_tiles) / base_tiles * 100
        print(f"Gain over bs=2: {gain_pct:+.1f}%", flush=True)

    # ── Gate check ──────────────────────────────────────────────────────────
    GATE_THRESHOLD = 0.10  # 10%
    if base_tiles is not None and gain_pct < GATE_THRESHOLD * 100:
        print(f"\n*** GATE FAILED: best tiles/sec gain = {gain_pct:.1f}% < 10% ***", flush=True)
        print("Larger batch size will NOT shorten training time. Stopping.", flush=True)
        print("Recommended: keep --batch-size 2", flush=True)
        return 0

    print(f"\n*** GATE PASSED: {gain_pct:.1f}% gain — proceeding to 3-epoch pilot ***", flush=True)

    # ── 3-epoch pilot: bs=8 lr=2e-4  vs  bs=2 lr=1e-4 ──────────────────────
    print("\n" + "=" * 60, flush=True)
    print("3-EPOCH PILOT: bs=8 lr=2e-4  vs  bs=2 lr=1e-4", flush=True)
    print("Comparing at same wall-clock budget (pilot wall time of faster run)", flush=True)
    print("=" * 60, flush=True)

    # First run bs=8 (3 epochs, record wall time)
    print("\n[bs=8 lr=2e-4]", flush=True)
    try:
        r8 = run_mini_pilot(8, 2e-4, 3, device, max_wall_sec=3600)
        wall_budget = r8["elapsed_sec"]
        print(f"  Finished: val_loss={r8['val_loss']:.5f}  val_dice={r8['val_dice']:.5f}  "
              f"time={wall_budget}s  epochs={r8['epochs_done']}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("  OOM at bs=8 during pilot", flush=True)
        print("Gate passed but bs=8 OOM in full training — recommend bs=4", flush=True)
        return 0

    # Then run bs=2 capped at the same wall time
    print(f"\n[bs=2 lr=1e-4]  (wall budget = {wall_budget}s)", flush=True)
    r2 = run_mini_pilot(2, 1e-4, 3, device, max_wall_sec=wall_budget)
    print(f"  Finished: val_loss={r2['val_loss']:.5f}  val_dice={r2['val_dice']:.5f}  "
          f"time={r2['elapsed_sec']}s  epochs={r2['epochs_done']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("TIME-TO-QUALITY COMPARISON", flush=True)
    print(f"  bs=8  lr=2e-4 : val_loss={r8['val_loss']:.5f}  val_dice={r8['val_dice']:.5f}  "
          f"time={r8['elapsed_sec']}s  epochs={r8['epochs_done']}", flush=True)
    print(f"  bs=2  lr=1e-4 : val_loss={r2['val_loss']:.5f}  val_dice={r2['val_dice']:.5f}  "
          f"time={r2['elapsed_sec']}s  epochs={r2['epochs_done']}", flush=True)
    if r8["val_loss"] is not None and r2["val_loss"] is not None:
        diff_loss = r8["val_loss"] - r2["val_loss"]
        diff_dice = r8["val_dice"] - r2["val_dice"]
        print(f"\n  Δval_loss (bs8−bs2) = {diff_loss:+.5f}", flush=True)
        print(f"  Δval_dice (bs8−bs2) = {diff_dice:+.5f}", flush=True)
        print(f"\nConclusion: {'bs=8 better or equal quality in same wall time'  if diff_loss <= 0 else 'bs=2 better quality in same wall time'}", flush=True)

    # Save results
    pilot_summary = {"bench": bench_results, "gate_passed": True,
                     "best_bs": best_bs, "gain_pct": round(gain_pct, 2),
                     "pilot_bs8": r8, "pilot_bs2": r2}
    out = os.path.join("processed_dataset", "reports", "batch_benchmark_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(pilot_summary, open(out, "w"), indent=2)
    print(f"\nResults saved to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
