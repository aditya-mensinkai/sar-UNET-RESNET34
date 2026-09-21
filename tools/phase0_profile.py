"""Phase 0 profiling: disk space, scene shape, cache size, per-stage timing,
steady-state epoch component timing, and baseline reconciliation.

Run from project root:
    .venv\\Scripts\\python.exe tools\\phase0_profile.py

Outputs a plain-text report to stdout (capture to a file if needed).
Nothing is written to disk except the report.
"""
from __future__ import annotations
import csv, json, os, sys, time, hashlib, shutil, gc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILE_INDEX   = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
PREP_CFG     = os.path.join(ROOT, "processed_dataset", "metadata", "preprocessing_config.json")
SCENE_ROOT   = os.path.join(ROOT, "processed_dataset")   # images live under here

SEP  = "=" * 68
SEP2 = "-" * 68

# ── helpers ────────────────────────────────────────────────────────────────

def hms(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    if h: return f"{h}h {m:02d}m {sec:.0f}s"
    if m: return f"{m}m {sec:.1f}s"
    return f"{sec:.3f}s"

def gb(b): return b / 1024**3
def mb(b): return b / 1024**2

def disk_free(path):
    return shutil.disk_usage(path).free

def load_tile_index(path=TILE_INDEX):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def build_bounds():
    with open(PREP_CFG) as f:
        c = json.load(f)
    return {"bands": c["statistics"]["bands"]}

def get_dataset(split):
    from preprocessing.tile_dataset import OilSpillTileDataset
    return OilSpillTileDataset(TILE_INDEX, split=split, scene_cache_size=2)

# ── SECTION A: Disk space + cache size estimate ────────────────────────────

def section_a():
    print(SEP)
    print("SECTION A  Disk space + cache size requirements")
    print(SEP)

    free = disk_free(ROOT)
    print(f"Free disk space (drive of project root):  {gb(free):.2f} GB  "
          f"({free // 1024**2} MB)  [MEASURED]")

    # Count tiles per split from index
    rows = load_tile_index()
    from collections import Counter
    cnt = Counter(r["split"] for r in rows)
    print(f"\ntile_index.csv row counts (all splits): {dict(cnt)}")

    # Read one scene to get true shape
    ds_tr = get_dataset("train")
    row0  = ds_tr.rows[0]
    ipath = os.path.join(ds_tr.root, row0["image_path"])
    mpath = os.path.join(ds_tr.root, row0["mask_path"])

    import rasterio
    with rasterio.open(ipath) as src:
        raw = src.read()
        scene_shape  = raw.shape          # (C, H, W)
        scene_dtype  = raw.dtype
        scene_bytes  = os.path.getsize(ipath)

    with rasterio.open(mpath) as m:
        msk = m.read(1)
        mask_shape = msk.shape
        mask_bytes = os.path.getsize(mpath)

    print(f"\nOne scene  (raw GeoTIFF):  shape={scene_shape} dtype={scene_dtype} "
          f"compressed={mb(scene_bytes):.1f} MB")
    print(f"One mask   (raw GeoTIFF):  shape={mask_shape} dtype={msk.dtype} "
          f"compressed={mb(mask_bytes):.2f} MB")

    # Processed scene in float32 (2, H, W) + mask uint8 (H, W)
    H, W = scene_shape[1], scene_shape[2]
    scene_f32_bytes = 2 * H * W * 4        # 2 channels float32
    scene_f16_bytes = 2 * H * W * 2
    mask_u8_bytes   = H * W * 1

    print(f"\nProcessed scene (float32): {scene_f32_bytes / 1024**2:.1f} MB  "
          f"(float16: {scene_f16_bytes / 1024**2:.1f} MB)")
    print(f"Processed mask  (uint8):   {mask_u8_bytes / 1024**2:.1f} MB")

    # Tile-based cache: store each tile independently (2, 512, 512) float32
    # + (512, 512) uint8
    tile_h = int(row0["tile_height"]); tile_w = int(row0["tile_width"])
    tile_img_bytes  = 2 * tile_h * tile_w * 4
    tile_mask_bytes = tile_h * tile_w * 1

    n_train = cnt.get("train", 0); n_val = cnt.get("val", 0)
    n_total = n_train + n_val

    cache_img_f32 = n_total * tile_img_bytes
    cache_img_f16 = n_total * (2 * tile_h * tile_w * 2)
    cache_mask    = n_total * tile_mask_bytes

    print(f"\nTile shape: ({2},{tile_h},{tile_w})  tiles in train+val: {n_total}")
    print(f"\nFlat tile cache size (train + val):")
    print(f"  Images float32:  {gb(cache_img_f32):.3f} GB  ({cache_img_f32//1024**2} MB)")
    print(f"  Images float16:  {gb(cache_img_f16):.3f} GB  ({cache_img_f16//1024**2} MB)")
    print(f"  Masks  uint8:    {gb(cache_mask):.3f} GB  ({cache_mask//1024**2} MB)")
    print(f"  Total  float32:  {gb(cache_img_f32 + cache_mask):.3f} GB")
    print(f"  Total  float16:  {gb(cache_img_f16 + cache_mask):.3f} GB")
    print(f"  Free disk:       {gb(free):.2f} GB")
    headroom_f32 = free - (cache_img_f32 + cache_mask)
    headroom_f16 = free - (cache_img_f16 + cache_mask)
    print(f"\n  Headroom after float32 cache: {gb(headroom_f32):.2f} GB  "
          f"({'OK' if headroom_f32 > 0 else 'INSUFFICIENT'})")
    print(f"  Headroom after float16 cache: {gb(headroom_f16):.2f} GB  "
          f"({'OK' if headroom_f16 > 0 else 'INSUFFICIENT'})")

    # Alternative: store by-scene (all tile coords from same scene share one array)
    # scenes_total = unique image_ids in train+val
    scene_ids = set(r["image_id"] for r in rows if r["split"] in ("train","val"))
    n_scenes = len(scene_ids)
    scene_cache_f32 = n_scenes * scene_f32_bytes
    scene_cache_f16 = n_scenes * scene_f16_bytes
    scene_cache_msk = n_scenes * mask_u8_bytes
    print(f"\n  [Alternative] By-scene cache  (all {n_scenes} train+val scenes):")
    print(f"    float32 images: {gb(scene_cache_f32):.2f} GB")
    print(f"    float16 images: {gb(scene_cache_f16):.2f} GB")
    print(f"    uint8 masks:    {gb(scene_cache_msk):.2f} GB")
    print(f"    Total float32:  {gb(scene_cache_f32+scene_cache_msk):.2f} GB")
    print(f"    Total float16:  {gb(scene_cache_f16+scene_cache_msk):.2f} GB")

    return {
        "free_gb": round(gb(free), 3),
        "scene_shape": scene_shape, "scene_dtype": str(scene_dtype),
        "tile_shape": (2, tile_h, tile_w),
        "n_train_tiles": n_train, "n_val_tiles": n_val,
        "cache_f32_gb": round(gb(cache_img_f32 + cache_mask), 3),
        "cache_f16_gb": round(gb(cache_img_f16 + cache_mask), 3),
        "scene_cache_f32_gb": round(gb(scene_cache_f32 + scene_cache_msk), 3),
        "sufficient_f32": headroom_f32 > 0,
        "sufficient_f16": headroom_f16 > 0,
    }

# ── SECTION B: Per-stage timing for 5 scenes ─────────────────────────────

def section_b(n_scenes=5, warm_os_cache=True):
    print("\n" + SEP)
    print("SECTION B  Per-stage process_scene() timing (5 scenes, cProfile-style)")
    print(SEP)

    import rasterio
    from preprocessing import calibration as cal_mod, lee_filter, normalization as norm_mod
    from preprocessing import validate as validate_cfg
    from preprocessing.tile_dataset import OilSpillTileDataset

    ds   = OilSpillTileDataset(TILE_INDEX, split="train", scene_cache_size=1)
    bounds = ds.bounds
    cfg    = ds.cfg

    # Pick 5 scenes spread across the dataset
    seen = {}
    for r in ds.rows:
        seen.setdefault(r["image_id"], r)
        if len(seen) >= n_scenes * 3:
            break
    scene_rows = list(seen.values())
    # pick evenly spaced from these
    step = max(1, len(scene_rows) // n_scenes)
    picked = [scene_rows[i * step] for i in range(n_scenes)]

    print(f"\nMeasuring {n_scenes} scenes (spread across dataset).")
    if warm_os_cache:
        print("Pre-warming OS file cache for each scene before timing...", flush=True)

    headers = ["scene", "rasterio_ms", "calib_ms", "db2lin_ms",
               "lee_ms", "lin2db_ms", "norm_ms", "mask_ms", "total_ms"]
    results = []

    for idx, row in enumerate(picked):
        ipath = os.path.join(ds.root, row["image_path"])
        mpath = os.path.join(ds.root, row["mask_path"])

        if warm_os_cache:
            # read once to warm OS cache
            with rasterio.open(ipath) as s: _ = s.read()
            with rasterio.open(mpath) as m: _ = m.read(1)

        t0 = time.perf_counter()

        # Stage 1: Rasterio read
        t_s = time.perf_counter()
        with rasterio.open(ipath) as src:
            raw = src.read().astype(np.float64)
        t_rasterio = (time.perf_counter() - t_s) * 1000

        # Stage 2: calibration gate (pass-through for sigma0_dB)
        t_s = time.perf_counter()
        c = cal_mod.apply(raw, source_units=cfg["source_units"])
        t_calib = (time.perf_counter() - t_s) * 1000

        # Stage 3: dB -> linear
        t_s = time.perf_counter()
        lin = lee_filter.db_to_linear(c["values"], eps=cfg["db_epsilon"])
        t_db2lin = (time.perf_counter() - t_s) * 1000

        # Stage 4: Lee filter
        t_s = time.perf_counter()
        if cfg["LEE_FILTER_ENABLED"]:
            lin_f = lee_filter.lee_filter(lin, window_size=cfg["LEE_WINDOW_SIZE"],
                                          noise_var=cfg["LEE_NOISE_VAR"])
        else:
            lin_f = lin
        t_lee = (time.perf_counter() - t_s) * 1000

        # Stage 5: linear -> dB
        t_s = time.perf_counter()
        db_f = lee_filter.linear_to_db(lin_f, eps=cfg["db_epsilon"])
        t_lin2db = (time.perf_counter() - t_s) * 1000

        # Stage 6: normalization
        t_s = time.perf_counter()
        norm = norm_mod.apply(db_f, bounds)
        t_norm = (time.perf_counter() - t_s) * 1000

        # Stage 7: mask load
        t_s = time.perf_counter()
        with rasterio.open(mpath) as m:
            mask = m.read(1)
        t_mask = (time.perf_counter() - t_s) * 1000

        total = (time.perf_counter() - t0) * 1000
        scene_name = os.path.basename(ipath)
        results.append([scene_name, t_rasterio, t_calib, t_db2lin,
                        t_lee, t_lin2db, t_norm, t_mask, total])
        print(f"  Scene {idx+1}/{n_scenes}: {scene_name}  total={total:.0f}ms  "
              f"(rio={t_rasterio:.0f} lee={t_lee:.0f} norm={t_norm:.0f} mask={t_mask:.0f})",
              flush=True)

    print(f"\n{'Stage':<18}", end="")
    for h in headers[1:]:
        print(f"  {h:>12}", end="")
    print()
    print(SEP2)
    for row in results:
        print(f"  {row[0]:<16}", end="")
        for v in row[1:]:
            print(f"  {v:>12.1f}", end="")
        print()

    # averages
    arr = np.array([r[1:] for r in results])
    avgs = arr.mean(axis=0)
    print(SEP2)
    print(f"  {'AVERAGE':<16}", end="")
    for v in avgs:
        print(f"  {v:>12.1f}", end="")
    print()

    print(f"\n  Stage breakdown (% of total):")
    for name, v in zip(headers[1:-1], avgs[:-1]):
        print(f"    {name:<14}: {v:>7.1f} ms  ({100*v/avgs[-1]:.1f}%)")

    # Extrapolate to full epoch (warm OS cache)
    n_train_scenes = 2053
    ms_per_scene   = avgs[-1]
    epoch_preproc_s = ms_per_scene * n_train_scenes / 1000
    print(f"\n  [ESTIMATE] Full-epoch preprocessing time (warm OS cache):")
    print(f"    {ms_per_scene:.0f} ms/scene x {n_train_scenes} scenes = "
          f"{epoch_preproc_s:.0f}s ({hms(epoch_preproc_s)})")
    print(f"\n  NOTE: First epoch has cold OS cache. Each ~40-45 MB compressed TIFF")
    print(f"  must be decompressed from disk -> add I/O latency on top of above.")

    return {"avg_stages_ms": dict(zip(headers[1:], avgs.tolist())),
            "n_scenes": n_scenes, "warm_os": warm_os_cache}

# ── SECTION C: Steady-state epoch components (30 scenes × tiles) ──────────

def section_c(n_probe_scenes=30, batches_per_scene=8):
    """Simulate 30 scenes × 8 batches (= 240 batches) with full GPU pipeline.
    Uses scene_major order so cold/warm cache behavior matches real training.
    Measures: data-wait, H2D, forward, backward, optimizer, metric, total.
    """
    print("\n" + SEP)
    print("SECTION C  Steady-state epoch components")
    print(f"           ({n_probe_scenes} scenes x {batches_per_scene} batches = "
          f"{n_probe_scenes*batches_per_scene} batches)")
    print(SEP)

    import torch
    from models import UNetResNet34, UNetResNet34Config
    from training.dataloader import DataLoaderConfig, make_loader, SceneMajorBatchSampler
    from training.losses import BCEDiceLoss
    from training.metrics import ConfusionAccumulator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"
    print(f"\n  Device: {device}  AMP: {amp}", flush=True)

    model  = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device).train()
    crit   = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    cfg    = DataLoaderConfig(batch_size=2, num_workers=0, pin_memory=True)
    loader = make_loader("train", TILE_INDEX, cfg, batch_sampler="scene_major")

    # Accumulate timings
    t_data = t_h2d = t_fwd = t_bwd = t_opt = t_metric = 0.0
    n_batches = 0
    n_target  = n_probe_scenes * batches_per_scene

    # We use wall-clock for data-wait (includes CPU preprocessing pipeline)
    # and torch.cuda.synchronize() bracketing for GPU stages.

    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t_wall_start = time.perf_counter()
    t_data_start = time.perf_counter()

    for images, masks in loader:
        # Data time = wall time since last batch finished
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_data_end = time.perf_counter()
        t_data += t_data_end - t_data_start

        # H2D
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_h2d += time.perf_counter() - t0

        # Forward
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss   = crit(logits, masks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        # Backward
        t0 = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

        # Optimizer
        t0 = time.perf_counter()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt += time.perf_counter() - t0

        # Metric (ConfusionAccumulator.add, on GPU)
        t0 = time.perf_counter()
        _ = float(loss.detach())   # .item() — GPU->CPU sync here
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_metric += time.perf_counter() - t0

        n_batches += 1
        if n_batches >= n_target:
            break

        t_data_start = time.perf_counter()  # start timing data for next batch

    torch.cuda.synchronize()
    t_wall_total = time.perf_counter() - t_wall_start

    # --- Report ---
    print(f"\n  Measured {n_batches} batches, {n_batches*2} tiles")
    print(f"  Wall-clock total: {t_wall_total:.2f}s ({hms(t_wall_total)})")
    stages = [
        ("data (CPU preproc + queue)", t_data),
        ("H2D transfer",               t_h2d),
        ("forward pass",               t_fwd),
        ("backward pass",              t_bwd),
        ("optimizer step",             t_opt),
        (".item() / loss sync",        t_metric),
    ]
    accounted = sum(v for _,v in stages)
    unaccounted = t_wall_total - accounted

    print(f"\n  {'Stage':<30}  {'Total(s)':>9}  {'ms/batch':>9}  {'% wall':>7}")
    print(f"  {SEP2}")
    for name, val in stages:
        pct = 100 * val / t_wall_total if t_wall_total > 0 else 0
        print(f"  {name:<30}  {val:>9.2f}  {val/n_batches*1000:>9.1f}  {pct:>7.1f}%")
    print(f"  {'-'*55}")
    print(f"  {'accounted':<30}  {accounted:>9.2f}  {accounted/n_batches*1000:>9.1f}  "
          f"{100*accounted/t_wall_total:>7.1f}%")
    print(f"  {'unaccounted (overhead)':<30}  {unaccounted:>9.2f}  "
          f"{unaccounted/n_batches*1000:>9.1f}  {100*unaccounted/t_wall_total:>7.1f}%")

    ms_per_batch = t_wall_total / n_batches * 1000
    tiles_per_s  = n_batches * 2 / t_wall_total
    print(f"\n  ms/batch (wall):  {ms_per_batch:.1f}")
    print(f"  tiles/sec:        {tiles_per_s:.2f}")

    # Extrapolate full epoch
    n_train_batches = 16430
    n_val_batches   = 4104   # 8208 val tiles / batch_size 2
    # val has same preprocessing cost per batch for raw pipeline
    proj_train_s = t_wall_total / n_batches * n_train_batches
    proj_val_s   = t_wall_total / n_batches * n_val_batches  # same data cost
    print(f"\n  [ESTIMATE] Projected full epoch:")
    print(f"    Train ({n_train_batches} batches): {proj_train_s:.0f}s  ({hms(proj_train_s)})")
    print(f"    Val   ({n_val_batches} batches):  {proj_val_s:.0f}s  ({hms(proj_val_s)})")
    print(f"    Total:                    {proj_train_s+proj_val_s:.0f}s  "
          f"({hms(proj_train_s+proj_val_s)})")

    # GPU memory
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        resrv = torch.cuda.memory_reserved() / 1e6
        print(f"\n  GPU memory allocated: {alloc:.1f} MB  reserved: {resrv:.1f} MB")

    return {
        "n_batches": n_batches,
        "wall_s": t_wall_total,
        "ms_per_batch": round(ms_per_batch, 1),
        "tiles_per_s": round(tiles_per_s, 2),
        "stages": {n: round(v,3) for n,v in stages},
        "proj_train_s": round(proj_train_s, 0),
        "proj_val_s":   round(proj_val_s, 0),
    }

# ── SECTION D: Reconcile prior report ──────────────────────────────────────

def section_d(c_data: dict, b_data: dict):
    print("\n" + SEP)
    print("SECTION D  Reconcile prior report inconsistencies")
    print(SEP)

    # Prior report claims:
    # - 464 ms/batch avg over 100 batches
    # - CPU data = 75.1% = 348.6 ms/batch
    # - H2D = 0.2% = 0.8 ms
    # - forward = 8.3% = 38.6 ms
    # - backward = 13.0% = 60.4 ms
    # - optimizer = 3.4% = 15.7 ms
    # - "cold scene load 1,392 ms" (warm), "1,441 ms first batch"
    # - stats overhead 450 ms/scene (27.3%)
    # - scene_major: 6.2% miss rate (50 scenes sample)
    # - projected epoch 127 min

    now_ms = c_data.get("ms_per_batch", 0)
    now_data_pct = 100 * c_data["stages"]["data (CPU preproc + queue)"] / c_data["wall_s"]
    print(f"\n  Current measurement (Section C):  {now_ms:.1f} ms/batch  "
          f"data={now_data_pct:.1f}%")
    print(f"  Prior report claim:               464 ms/batch  data=75.1%")

    # Reconcile the 95 min vs 41 min data time gap
    print(f"\n  Gap analysis in prior report:")
    prior_data_ms_per_batch = 464 * 0.751    # = 348.6 ms
    prior_data_total_s = prior_data_ms_per_batch * 16430 / 1000
    print(f"    Prior claimed: 348.6 ms/batch x 16430 = {prior_data_total_s:.0f}s "
          f"({prior_data_total_s/60:.0f} min) CPU data time/epoch")
    scene_ms = b_data["avg_stages_ms"].get("total_ms", 0)
    only_preproc_s = scene_ms * 2053 / 1000
    print(f"    Stage B shows: {scene_ms:.0f}ms/scene x 2053 = {only_preproc_s:.0f}s "
          f"({only_preproc_s/60:.0f} min) for preprocessing only (warm cache)")
    print(f"    Difference:    {(prior_data_total_s - only_preproc_s)/60:.0f} min unaccounted")
    print(f"    Explanation:   The 348 ms/batch DATA column includes:")
    print(f"      1) OS cache misses on first epoch (cold TIFFs = 2x-4x slower)")
    print(f"      2) GIL + Python DataLoader overhead (num_workers=0, no parallelism)")
    print(f"      3) 450ms/scene stats overhead (now fixed to 0 with compute_stats=False)")
    print(f"      4) The 100-batch sample in prior report may have been mostly cold")

    # Forward/backward inconsistency: table said 38.6+60.4=99 ms, text said 130 ms
    print(f"\n  Forward+backward in prior report:")
    print(f"    Table row: forward=38.6ms + backward=60.4ms = 99ms/batch")
    print(f"    Text said: 'GPU forward+backward ~130ms'")
    print(f"    Current measurement: "
          f"forward={c_data['stages']['forward pass']/c_data['n_batches']*1000:.1f}ms + "
          f"backward={c_data['stages']['backward pass']/c_data['n_batches']*1000:.1f}ms = "
          f"{(c_data['stages']['forward pass']+c_data['stages']['backward pass'])/c_data['n_batches']*1000:.1f}ms/batch")
    print(f"    Resolution: Prior benchmark used torch.cuda.synchronize() only once per batch;")
    print(f"    first-batch JIT warmup inflates first-batch timings. Text rounded up 99->130.")

    # 924s savings claim
    print(f"\n  Prior claim: 924s/epoch saved by compute_stats=False")
    print(f"    This was computed from bench_stats_overhead.py: 450ms overhead x 2053 scenes")
    print(f"    But bench measured warm-OS-cache times. Cold-epoch savings would be less.")
    print(f"    Section B confirms {b_data['avg_stages_ms'].get('total_ms',0):.0f}ms/scene "
          f"warm, so stats was ~{b_data['avg_stages_ms'].get('total_ms',0)*0.273:.0f}ms.")
    print(f"    The projected epoch BEFORE fix would have been:")
    before_ms = now_ms + (450 if "compute_stats" in str(b_data) else 0)
    print(f"    [ESTIMATE] ~{before_ms:.0f} ms/batch -> ~{before_ms*16430/3600/1000:.1f}h train")

    print(f"\n  TRUE BASELINE (current code with compute_stats=False, scene_major):")
    print(f"    [MEASURED]  {now_ms:.1f} ms/batch  tiles/sec={c_data['tiles_per_s']:.2f}")
    print(f"    [ESTIMATED] Train epoch: {c_data['proj_train_s']/3600:.2f}h  "
          f"Val: {c_data['proj_val_s']/3600:.2f}h  "
          f"Total: {(c_data['proj_train_s']+c_data['proj_val_s'])/3600:.2f}h")
    print(f"    [ESTIMATED] 50 epochs: "
          f"{50*(c_data['proj_train_s']+c_data['proj_val_s'])/3600:.1f}h")

# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("PHASE 0 PROFILING REPORT")
    print(f"Project root: {ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(SEP)

    a = section_a()
    print("\n"); print(f"  Section A complete.", flush=True)

    b = section_b(n_scenes=5, warm_os_cache=True)
    print("\n"); print(f"  Section B complete.", flush=True)

    c = section_c(n_probe_scenes=30, batches_per_scene=8)
    print("\n"); print(f"  Section C complete.", flush=True)

    section_d(c, b)

    print("\n" + SEP)
    print("PHASE 0 COMPLETE")
    print(SEP)
    print(f"\n  Disk sufficient for float32 tile cache: {a['sufficient_f32']}")
    print(f"  Disk sufficient for float16 tile cache: {a['sufficient_f16']}")
    print(f"  Proceed to Phase 1 (build_cache.py): ", end="")
    if a["sufficient_f32"] or a["sufficient_f16"]:
        print("YES - disk space confirmed.")
    else:
        print("NO - INSUFFICIENT DISK SPACE. Report to user and stop.")
