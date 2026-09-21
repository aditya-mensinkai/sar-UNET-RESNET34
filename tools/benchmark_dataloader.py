"""Benchmark script for UNet+ResNet34 SAR training pipeline.

Measures separately:
  - Dataset __getitem__ latency (scene load vs tile slice)
  - DataLoader batch creation (num_workers comparison)
  - CPU -> CUDA transfer time
  - Model forward pass time
  - Backward pass time
  - Optimizer step time
  - Full batch iteration time

Reports: total time, avg time/batch, tiles/sec, GPU utilization estimate,
RAM, GPU memory.

Usage (from project root):
    .venv\\Scripts\\python.exe tools\\benchmark_dataloader.py
    .venv\\Scripts\\python.exe tools\\benchmark_dataloader.py --num-batches 100
    .venv\\Scripts\\python.exe tools\\benchmark_dataloader.py --num-batches 50 --workers 0 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

# ── resolve project root (this script lives in tools/) ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILE_INDEX = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
PREP_CFG   = os.path.join(ROOT, "processed_dataset", "metadata", "preprocessing_config.json")


# ── helpers ──────────────────────────────────────────────────────────────────
def _gb(x):
    return x / 1024 ** 3 if x else 0.0


def _mb(x):
    return x / 1024 ** 2 if x else 0.0


def _mem_python_gb():
    """Rough RSS estimate (Windows/Linux)."""
    try:
        import psutil
        return _gb(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def gpu_allocated_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1e6, 1)
    return None


def gpu_reserved_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 1e6, 1)
    return None


def hms(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h:
        return f"{h}h {m:02d}m {sec:.0f}s"
    if m:
        return f"{m}m {sec:.1f}s"
    return f"{sec:.3f}s"


# ── section helpers ──────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60, flush=True)


def subsection(title):
    print(f"\n--- {title} ---", flush=True)


# ── Stage 1: single __getitem__ latency ──────────────────────────────────────
def bench_getitem(n_warmup=2, n_measure=20):
    """Measure raw __getitem__ cost: first call (scene miss) vs subsequent calls (scene hit)."""
    subsection("Stage 1: __getitem__ latency (single-item, no DataLoader)")
    from preprocessing.tile_dataset import OilSpillTileDataset
    ds = OilSpillTileDataset(TILE_INDEX, split="train", scene_cache_size=2)

    # find two scenes that share tiles
    scene_map: dict[str, list[int]] = {}
    for i, r in enumerate(ds.rows):
        scene_map.setdefault(r["image_id"], []).append(i)
    first_scene_tiles = next(iter(scene_map.values()))[:8]
    idx_first = first_scene_tiles[0]
    idx_same  = first_scene_tiles[1] if len(first_scene_tiles) > 1 else first_scene_tiles[0]

    # cold miss
    ds._cache.clear()
    t0 = time.perf_counter()
    _ = ds[idx_first]
    cold_ms = (time.perf_counter() - t0) * 1000

    # warm hit (same scene, different tile)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        _ = ds[idx_same]
    warm_ms = (time.perf_counter() - t0) * 1000 / n_measure

    print(f"  Cold (scene miss) : {cold_ms:.1f} ms  (rasterio read + lee + norm + stats_skip)")
    print(f"  Warm (scene hit)  : {warm_ms:.2f} ms  (cache lookup + tile slice only)")
    ratio = cold_ms / warm_ms if warm_ms > 0 else float('inf')
    print(f"  Ratio             : {ratio:.0f}x faster on cache hit")
    print(f"  -> With 16 tiles/scene and scene_major, cold fires 1/{16} = {100/16:.1f}% of calls")
    return {"cold_ms": cold_ms, "warm_ms": warm_ms, "speedup_ratio": ratio}


# ── Stage 2: DataLoader batch throughput (num_workers comparison) ─────────────
def bench_dataloader(num_batches=100, workers_list=(0,)):
    """Measure DataLoader throughput for different num_workers values."""
    subsection(f"Stage 2: DataLoader throughput ({num_batches} batches, no GPU)")
    from preprocessing.tile_dataset import OilSpillTileDataset
    from training.dataloader import DataLoaderConfig, make_loader

    results = {}
    for nw in workers_list:
        torch.manual_seed(42)
        cfg = DataLoaderConfig(batch_size=2, num_workers=nw, pin_memory=False)
        loader = make_loader("train", TILE_INDEX, cfg, batch_sampler="scene_major")
        n = 0
        t0 = time.perf_counter()
        for images, masks in loader:
            n += 1
            if n >= num_batches:
                break
        elapsed = time.perf_counter() - t0
        tiles_sec = (n * 2) / elapsed if elapsed > 0 else 0
        batch_ms = elapsed / n * 1000 if n > 0 else 0
        ram_gb = _mem_python_gb()
        print(f"  workers={nw:2d}: {n} batches  {elapsed:.1f}s  "
              f"{batch_ms:.0f}ms/batch  {tiles_sec:.1f} tiles/s"
              + (f"  RAM={ram_gb:.2f}GB" if ram_gb is not None else ""))
        results[f"workers_{nw}"] = {
            "batches": n, "elapsed_s": round(elapsed, 2),
            "ms_per_batch": round(batch_ms, 1),
            "tiles_per_sec": round(tiles_sec, 1)
        }
    return results


# ── Stage 3: GPU pipeline timing ─────────────────────────────────────────────
def bench_gpu_pipeline(num_batches=100, num_workers=0, amp=True):
    """Measure full GPU training loop: H2D, forward, backward, optimizer."""
    subsection(f"Stage 3: Full GPU pipeline ({num_batches} batches, AMP={amp})")
    from models import UNetResNet34, UNetResNet34Config
    from training.dataloader import DataLoaderConfig, make_loader
    from training.losses import BCEDiceLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  WARNING: CUDA not available, skipping GPU pipeline benchmark")
        return {}

    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device)
    model.train()
    crit = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    cfg = DataLoaderConfig(batch_size=2, num_workers=num_workers, pin_memory=True)
    loader = make_loader("train", TILE_INDEX, cfg, batch_sampler="scene_major")

    # timing accumulators
    t_data = t_h2d = t_fwd = t_bwd = t_opt = 0.0
    n = 0
    torch.cuda.synchronize()
    t_epoch_start = time.perf_counter()

    t_data_start = time.perf_counter()
    for images, masks in loader:
        t_load_done = time.perf_counter()
        t_data += t_load_done - t_data_start  # includes data load from last iteration

        # H2D
        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_h2d += time.perf_counter() - t0

        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss   = crit(logits, masks)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        # Backward
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

        # Optimizer
        t0 = time.perf_counter()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        torch.cuda.synchronize()
        t_opt += time.perf_counter() - t0

        n += 1
        if n == 1:
            # Report first-batch breakdown
            print(f"\n  [First batch breakdown]")
            print(f"    data (CPU preprocessing + queue)  : {t_data * 1000:.1f} ms")
            print(f"    H2D (CPU->GPU transfer)           : {t_h2d * 1000:.1f} ms")
            print(f"    forward pass                      : {t_fwd * 1000:.1f} ms")
            print(f"    backward pass                     : {t_bwd * 1000:.1f} ms")
            print(f"    optimizer step                    : {t_opt * 1000:.1f} ms")
            print(f"    GPU alloc: {gpu_allocated_mb()} MB  reserved: {gpu_reserved_mb()} MB",
                  flush=True)

        if n >= num_batches:
            break

        t_data_start = time.perf_counter()

    elapsed = time.perf_counter() - t_epoch_start
    tiles_s = (n * 2) / elapsed if elapsed > 0 else 0

    print(f"\n  Total {n} batches : {elapsed:.1f}s  ({hms(elapsed)})")
    print(f"  avg/batch        : {elapsed/n*1000:.0f} ms")
    print(f"  tiles/sec        : {tiles_s:.1f}")
    print(f"\n  Time breakdown (avg per batch):")
    print(f"    data (CPU)  : {t_data/n*1000:.1f} ms  ({100*t_data/elapsed:.1f}%)")
    print(f"    H2D         : {t_h2d/n*1000:.1f} ms  ({100*t_h2d/elapsed:.1f}%)")
    print(f"    forward     : {t_fwd/n*1000:.1f} ms  ({100*t_fwd/elapsed:.1f}%)")
    print(f"    backward    : {t_bwd/n*1000:.1f} ms  ({100*t_bwd/elapsed:.1f}%)")
    print(f"    optimizer   : {t_opt/n*1000:.1f} ms  ({100*t_opt/elapsed:.1f}%)")
    alloc = gpu_allocated_mb()
    resrv = gpu_reserved_mb()
    ram   = _mem_python_gb()
    print(f"\n  GPU memory: allocated={alloc} MB  reserved={resrv} MB")
    if ram is not None:
        print(f"  System RAM (RSS): {ram:.2f} GB", flush=True)

    # ETA projection
    n_train_batches = 16430
    projected_epoch_s = elapsed / n * n_train_batches
    print(f"\n  Projected epoch time  : {hms(projected_epoch_s)} "
          f"  ({n_train_batches} batches × {elapsed/n:.2f}s)")

    return {
        "batches": n,
        "elapsed_s": round(elapsed, 2),
        "ms_per_batch": round(elapsed / n * 1000, 1),
        "tiles_per_sec": round(tiles_s, 1),
        "ms_data": round(t_data / n * 1000, 1),
        "ms_h2d": round(t_h2d / n * 1000, 1),
        "ms_forward": round(t_fwd / n * 1000, 1),
        "ms_backward": round(t_bwd / n * 1000, 1),
        "ms_optimizer": round(t_opt / n * 1000, 1),
        "projected_epoch_s": round(projected_epoch_s, 0),
        "gpu_alloc_mb": alloc,
        "gpu_reserved_mb": resrv,
    }


# ── Stage 4: Scene cache effectiveness ───────────────────────────────────────
def bench_cache(num_scenes=20):
    """Report cache hit/miss rate for scene_major vs shuffled sampler on first N scenes."""
    subsection(f"Stage 4: Scene cache analysis ({num_scenes} scenes)")
    from preprocessing.tile_dataset import OilSpillTileDataset

    ds = OilSpillTileDataset(TILE_INDEX, split="train", scene_cache_size=2)

    # Build per-scene tile index
    scene_map: dict[str, list[int]] = {}
    for i, r in enumerate(ds.rows):
        scene_map.setdefault(r["image_id"], []).append(i)
    scene_keys = sorted(scene_map.keys())[:num_scenes]

    # Simulate scene_major access
    hits_sm = misses_sm = 0
    ds._cache.clear()
    for sk in scene_keys:
        for idx in scene_map[sk]:
            key = ds.rows[idx]["image_id"]
            if key in ds._cache:
                hits_sm += 1
            else:
                misses_sm += 1
            # simulate the cache update
            if key not in ds._cache:
                ds._cache[key] = True
            else:
                ds._cache.move_to_end(key)
            while len(ds._cache) > ds.scene_cache_size:
                ds._cache.popitem(last=False)

    # Simulate shuffled access (interleaved tiles from different scenes)
    import random
    rng_s = random.Random(42)
    shuffled_tiles = []
    for sk in scene_keys:
        shuffled_tiles.extend(scene_map[sk])
    rng_s.shuffle(shuffled_tiles)

    hits_sh = misses_sh = 0
    ds._cache.clear()
    for idx in shuffled_tiles:
        key = ds.rows[idx]["image_id"]
        if key in ds._cache:
            hits_sh += 1
        else:
            misses_sh += 1
        if key not in ds._cache:
            ds._cache[key] = True
        else:
            ds._cache.move_to_end(key)
        while len(ds._cache) > ds.scene_cache_size:
            ds._cache.popitem(last=False)

    total = hits_sm + misses_sm
    print(f"  scene_major (cache_size=2) : {misses_sm}/{total} misses "
          f"({100*misses_sm/total:.1f}%)  hits={hits_sm} ({100*hits_sm/total:.1f}%)")
    total2 = hits_sh + misses_sh
    print(f"  shuffled    (cache_size=2) : {misses_sh}/{total2} misses "
          f"({100*misses_sh/total2:.1f}%)  hits={hits_sh} ({100*hits_sh/total2:.1f}%)")
    tiles_per_scene = total / num_scenes
    print(f"\n  Tiles/scene: {tiles_per_scene:.1f}")
    print(f"  With scene_major: 1 process_scene() per scene = {num_scenes} calls for {num_scenes} scenes")
    print(f"  With shuffled   : ~{misses_sh} calls for {num_scenes} scenes")
    return {
        "scene_major_misses_pct": round(100 * misses_sm / total, 1),
        "shuffled_misses_pct": round(100 * misses_sh / total2, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark SAR training pipeline")
    ap.add_argument("--num-batches", type=int, default=100,
                    help="Batches to measure in GPU pipeline stage (default 100)")
    ap.add_argument("--workers", type=int, nargs="+", default=[0],
                    help="num_workers values to benchmark (default: 0)")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="Skip GPU pipeline benchmark (just DataLoader stages)")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable AMP in GPU pipeline benchmark")
    return ap.parse_args()


def main():
    args = parse_args()
    section("SAR Training Pipeline Benchmark")
    print(f"Project root : {ROOT}")
    print(f"Tile index   : {TILE_INDEX}")
    print(f"CUDA         : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU          : {p.name}  VRAM={p.total_memory // 1024 ** 2} MB", flush=True)

    all_results: dict = {}

    try:
        all_results["getitem"] = bench_getitem()
    except Exception as e:
        print(f"  getitem bench FAILED: {e}")
        traceback.print_exc()

    try:
        all_results["dataloader"] = bench_dataloader(
            num_batches=args.num_batches,
            workers_list=args.workers)
    except Exception as e:
        print(f"  dataloader bench FAILED: {e}")
        traceback.print_exc()

    try:
        all_results["cache"] = bench_cache(num_scenes=50)
    except Exception as e:
        print(f"  cache bench FAILED: {e}")
        traceback.print_exc()

    if not args.skip_gpu and torch.cuda.is_available():
        best_workers = args.workers[0]
        try:
            all_results["gpu_pipeline"] = bench_gpu_pipeline(
                num_batches=args.num_batches,
                num_workers=best_workers,
                amp=not args.no_amp)
        except Exception as e:
            print(f"  GPU pipeline bench FAILED: {e}")
            traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────────────────
    section("SUMMARY")
    gp = all_results.get("gpu_pipeline", {})
    if gp:
        n   = gp.get("batches", args.num_batches)
        ms  = gp.get("ms_per_batch", "?")
        ts  = gp.get("tiles_per_sec", "?")
        eta = gp.get("projected_epoch_s")
        print(f"  Batches measured   : {n}")
        print(f"  Avg time/batch     : {ms} ms")
        print(f"  Tiles/sec          : {ts}")
        if eta:
            print(f"  Projected epoch 1  : {hms(eta)} ({eta:.0f}s)")
            proj50 = eta * 50
            print(f"  Projected 50 epochs: {hms(proj50)}")
        print(f"  Data (CPU) %       : {gp.get('ms_data','?')} ms/batch "
              f"({100*gp.get('ms_data',0)/ms:.1f}% of total)" if isinstance(ms, (int, float)) and ms > 0 else "")
        print(f"  Forward+backward   : {gp.get('ms_forward',0) + gp.get('ms_backward',0):.1f} ms/batch")
    gi = all_results.get("getitem", {})
    if gi:
        print(f"\n  Cold __getitem__   : {gi.get('cold_ms','?'):.1f} ms")
        print(f"  Warm __getitem__   : {gi.get('warm_ms','?'):.2f} ms")
        print(f"  Cache speedup      : {gi.get('speedup_ratio','?'):.0f}x")

    print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
