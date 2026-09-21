"""Real-tile batch-size throughput benchmark (batch sizes 2, 4, 8, 16).

Why this exists
---------------
tools/batch_benchmark.py measures the training STEP with synthetic 256x256
tensors. Real tiles are 512x512 and come from the ~97 GB memmap cache through a
DataLoader worker, so the synthetic sweep cannot see the data-pipeline
bottleneck. This tool repeats the same protocol with the REAL pipeline:

  - fused Adam + cudnn.benchmark + num_workers=1
  - CachedTileDataset (processed_dataset/tile_cache), 512x512 float32
  - 20 warmup batches + 240 measured batches per batch size
  - Reports ms/batch two ways:
      * wall = iteration-to-iteration wall clock (data wait + H2D + step),
               i.e. what epoch time is actually made of
      * gpu  = optimizer step only, GPU-synchronized
    plus tiles/sec and peak GPU memory; stops at the first OOM.

Read-only with respect to the dataset. Writes nothing unless --out is given.

Usage:
  .venv\\Scripts\\python.exe tools\\bench_batch_real.py --batch-sizes 2 4 8 16
  .venv\\Scripts\\python.exe tools\\bench_batch_real.py --measured 300 --batch-sizes 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


def make_train_loader(batch_size, workers, split="train"):
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader

    ds = CachedTileDataset(split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=True)


def bench_one(batch_size, device, warmup, measured, workers=1):
    """Return dict of measurements, or raise torch.cuda.OutOfMemoryError."""
    from models import UNetResNet34, UNetResNet34Config
    from training.losses import BCEDiceLoss

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device)
    model.train()

    fused = True
    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    except (TypeError, RuntimeError):
        fused = False
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    crit = BCEDiceLoss(0.5, 0.5, 1.0)
    scaler = torch.amp.GradScaler("cuda")

    loader = make_train_loader(batch_size, workers)
    n_total = warmup + measured + 1          # +1 so the first measured wall delta exists
    wall_ms, gpu_ms = [], []
    prev = None
    try:
        for i, (imgs, msks) in enumerate(loader):
            now = time.perf_counter()
            if prev is not None and i > warmup:
                wall_ms.append((now - prev) * 1000.0)   # data wait + H2D + step
            prev = now
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(imgs)
                loss = crit(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            if i >= warmup:
                gpu_ms.append((time.perf_counter() - t0) * 1000.0)
            if i + 1 >= n_total:
                break
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    del model, opt, crit, scaler
    gc.collect()
    torch.cuda.empty_cache()

    wall_mean = float(np.mean(wall_ms))
    gpu_mean = float(np.mean(gpu_ms))
    return {
        "batch_size": batch_size,
        "n_measured": len(wall_ms),
        "wall_ms_mean": wall_mean,
        "wall_ms_p50": float(np.percentile(wall_ms, 50)),
        "wall_ms_p95": float(np.percentile(wall_ms, 95)),
        "gpu_ms_mean": gpu_mean,
        "tiles_sec_wall": (batch_size * 1000.0) / wall_mean,
        "tiles_sec_gpu": (batch_size * 1000.0) / gpu_mean,
        "peak_vram_mb": round(peak_mb, 1),
        "fused_adam": fused,
        "num_workers": workers,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-tile batch size benchmark")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measured", type=int, default=240)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ns = ap.parse_args(argv)

    print("=" * 60, flush=True)
    print("REAL-TILE BATCH SIZE BENCHMARK  (512x512 cached tiles)", flush=True)
    print(f"  fused Adam + cudnn.benchmark + num_workers={ns.workers}", flush=True)
    print(f"  {ns.warmup} warmup + {ns.measured} measured batches per size", flush=True)
    print("=" * 60, flush=True)

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device available", flush=True)
        return 1
    device = torch.device("cuda")

    results = {}
    for bs in ns.batch_sizes:
        print(f"\n--- batch_size={bs} ---", flush=True)
        try:
            r = bench_one(bs, device, ns.warmup, ns.measured, ns.workers)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch_size={bs} — stopping sweep", flush=True)
                results[bs] = {"oom": True}
                break
            raise
        results[bs] = r
        print(f"  wall {r['wall_ms_mean']:.1f} ms/batch (p50 {r['wall_ms_p50']:.1f}, "
              f"p95 {r['wall_ms_p95']:.1f})   gpu {r['gpu_ms_mean']:.1f} ms", flush=True)
        print(f"  tiles/sec wall {r['tiles_sec_wall']:.1f}   gpu {r['tiles_sec_gpu']:.1f}   "
              f"peak_vram {r['peak_vram_mb']:.0f} MB   fused={r['fused_adam']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY (real 512x512 tiles, num_workers={ns.workers})", flush=True)
    print("=" * 60, flush=True)
    print(f"{'bs':>4}  {'wall ms/b':>10}  {'gpu ms/b':>9}  {'tiles/s wall':>13}  {'peak MB':>9}",
          flush=True)
    for bs in ns.batch_sizes:
        r = results.get(bs)
        if r is None:
            continue
        if r.get("oom"):
            print(f"{bs:>4}  {'OOM':>10}", flush=True)
        else:
            print(f"{bs:>4}  {r['wall_ms_mean']:>10.1f}  {r['gpu_ms_mean']:>9.1f}  "
                  f"{r['tiles_sec_wall']:>13.1f}  {r['peak_vram_mb']:>9.0f}", flush=True)

    valid = {bs: r for bs, r in results.items() if not r.get("oom")}
    if valid and 2 in valid:
        best_bs = max(valid, key=lambda b: valid[b]["tiles_sec_wall"])
        gain = ((valid[best_bs]["tiles_sec_wall"] - valid[2]["tiles_sec_wall"])
                / valid[2]["tiles_sec_wall"] * 100)
        print(f"\nBest (wall) batch size: {best_bs}  "
              f"({valid[best_bs]['tiles_sec_wall']:.1f} tiles/s)", flush=True)
        print(f"Gain over bs=2: {gain:+.1f}%   "
              f"(gate: >= +10.0% required to justify larger batches)", flush=True)
        print(f"Gate: {'PASSED' if gain >= 10.0 else 'FAILED'}", flush=True)

    if ns.out:
        os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
        json.dump({"protocol": {"warmup": ns.warmup, "measured": ns.measured,
                                "workers": ns.workers, "tile_size": 512,
                                "data": "CachedTileDataset (real)",
                                "fused_adam": True, "cudnn_benchmark": True},
                   "results": results}, open(ns.out, "w"), indent=2)
        print(f"\nSaved: {ns.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
