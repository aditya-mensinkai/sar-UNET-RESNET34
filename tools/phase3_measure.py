"""Phase 3 measurement script.

Sections
--------
A  System info (GPU, CUDA, PyTorch, RAM, drive type)
B  Validation time benchmark at batch_size 2 and 8 (cache path, no-grad)
C  Noise floor: raw path x2 same seed, report per-step loss diff
D  Individual GPU opts (240 batches each, 20-batch warmup):
     D1  baseline (cache, no changes)
     D2  fused Adam
     D3  cudnn.benchmark
     D4  channels_last
     D5  zero_grad(set_to_none=True)   -- already set; confirm no regression
     D6  GPU-side loss accumulation (no .item() per step)
     D7  D2+D3+D5+D6 combined best
E  num_workers 0/1/2 with persistent_workers, peak-RAM guard

Run from project root:
    .venv\\Scripts\\python.exe tools\\phase3_measure.py
"""
from __future__ import annotations
import ctypes, gc, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TILE_INDEX = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
SEP  = "=" * 68
SEP2 = "-" * 68
N_WARMUP  = 20
N_MEASURE = 240
SEED      = 42

# ── helpers ────────────────────────────────────────────────────────────────

def _avail_ram_gb():
    try:
        class M(ctypes.Structure):
            _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                        ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                        ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                        ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
        m = M(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024**3, m.ullAvailPhys / 1024**3
    except Exception:
        return 0.0, 0.0

def _total_ram_gb(): return _avail_ram_gb()[0]
def _free_ram_gb():  return _avail_ram_gb()[1]

def hms(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60
    if h: return f"{h}h {m:02d}m {sec:.0f}s"
    if m: return f"{m}m {sec:.1f}s"
    return f"{sec:.3f}s"

def fresh_model(device):
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    from models import UNetResNet34, UNetResNet34Config
    m = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    return m.to(device)

def fresh_loader(batch_size=2, num_workers=0, shuffle=True, persistent=False):
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader
    ds = CachedTileDataset(split="train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(persistent and num_workers > 0))

def fresh_val_loader(batch_size=2, num_workers=0):
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader
    ds = CachedTileDataset(split="val")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

def gpu_clocks():
    """Return (sm_clock_MHz, mem_clock_MHz) or (None, None)."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return None, None

def run_bench(name, device, loader_fn, model_factory,
              amp=True, channels_last=False,
              fused_adam=False, cudnn_bench=False,
              gpu_loss_accum=False,
              n_warmup=N_WARMUP, n_measure=N_MEASURE,
              set_to_none=True):
    """
    Run n_warmup+n_measure batches.
    Returns dict with ms/batch, tiles/s, stage timings.
    """
    from training.losses import BCEDiceLoss

    torch.backends.cudnn.benchmark = cudnn_bench
    model = model_factory(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()

    crit   = BCEDiceLoss(0.5, 0.5, 1.0)
    opt_kw = {}
    if fused_adam:
        try:
            opt_kw["fused"] = True
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, **opt_kw)
            # test it works
            _ = opt
        except TypeError:
            opt_kw = {}
            print(f"    [{name}] fused Adam not supported on this build, falling back")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, **opt_kw)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    loader = loader_fn()
    it     = iter(loader)

    t_data = t_h2d = t_fwd = t_bwd = t_opt = 0.0
    loss_accum_gpu = torch.zeros(1, device=device) if gpu_loss_accum else None

    # warmup
    opt.zero_grad(set_to_none=set_to_none)
    for _ in range(n_warmup):
        try:
            images, masks = next(it)
        except StopIteration:
            it = iter(loader); images, masks = next(it)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss   = crit(logits, masks)
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:      loss.backward(); opt.step()
        opt.zero_grad(set_to_none=set_to_none)
    torch.cuda.synchronize()

    # measure
    opt.zero_grad(set_to_none=set_to_none)
    n = 0
    t_wall_start = time.perf_counter()
    t_ds_start   = time.perf_counter()

    for _ in range(n_measure):
        try:
            images, masks = next(it)
        except StopIteration:
            it = iter(loader); images, masks = next(it)

        torch.cuda.synchronize()
        t_data += time.perf_counter() - t_ds_start

        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        t_h2d += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss   = crit(logits, masks)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        if scaler: scaler.scale(loss).backward()
        else:      loss.backward()
        torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        if scaler: scaler.step(opt); scaler.update()
        else:      opt.step()
        opt.zero_grad(set_to_none=set_to_none)
        torch.cuda.synchronize()
        t_opt += time.perf_counter() - t0

        if gpu_loss_accum:
            loss_accum_gpu = loss_accum_gpu + loss.detach()
        else:
            _ = float(loss.detach())   # .item() sync

        n += 1
        t_ds_start = time.perf_counter()

    torch.cuda.synchronize()
    t_wall = time.perf_counter() - t_wall_start

    ms   = t_wall / n * 1000
    ts   = n * 2 / t_wall
    sm_clk, mem_clk = gpu_clocks()

    print(f"  {name:<35} {ms:>7.1f} ms/batch  {ts:>6.1f} tiles/s  "
          f"SM={sm_clk} MHz  MEM={mem_clk} MHz", flush=True)

    torch.backends.cudnn.benchmark = False   # reset
    del model, opt, scaler, loader, it
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {"name": name, "ms": ms, "tiles_s": ts,
            "t_data": t_data/n*1000, "t_h2d": t_h2d/n*1000,
            "t_fwd": t_fwd/n*1000,  "t_bwd": t_bwd/n*1000,
            "t_opt": t_opt/n*1000,
            "sm_mhz": sm_clk, "mem_mhz": mem_clk}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A  System info
# ══════════════════════════════════════════════════════════════════════════════
def section_a():
    print(SEP)
    print("SECTION A  System information")
    print(SEP)

    # GPU
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cc = f"{p.major}.{p.minor}"
        print(f"  GPU:               {p.name}")
        print(f"  Compute capability: {cc}")
        print(f"  VRAM total:         {p.total_memory/1024**3:.2f} GB  "
              f"({p.total_memory//1024**2} MB)")
        free_vram = (p.total_memory - torch.cuda.memory_allocated()) / 1024**3
        print(f"  VRAM free (est):    {free_vram:.2f} GB")
    else:
        print("  GPU: NONE (CUDA not available)")
        cc = "N/A"

    # CUDA / PyTorch / driver
    print(f"  PyTorch version:    {torch.__version__}")
    print(f"  CUDA version:       {torch.version.cuda}")
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        drv = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        drv = "unknown"
    print(f"  NVIDIA driver:      {drv}")

    # cuDNN
    print(f"  cuDNN version:      {torch.backends.cudnn.version()}")
    print(f"  bf16 supported:     {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

    # RAM
    total_gb, avail_gb = _avail_ram_gb()
    print(f"  System RAM total:   {total_gb:.1f} GB")
    print(f"  System RAM free:    {avail_gb:.1f} GB")

    # Drive type for cache
    cache_path = os.path.join(ROOT, "processed_dataset", "tile_cache")
    drive_letter = os.path.splitdrive(cache_path)[0].rstrip(":\\") + ":\\"
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-Command",
             f"Get-PhysicalDisk | Where-Object {{$_.DeviceId -eq "
             f"(Get-Partition -DriveLetter '{drive_letter[0]}').DiskNumber}} | "
             f"Select-Object -ExpandProperty MediaType"],
            capture_output=True, text=True, timeout=10)
        drive_type = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        drive_type = "unknown (powershell query failed)"
    # Fallback: check disk usage pattern
    try:
        import shutil
        du = shutil.disk_usage(cache_path)
        print(f"  Cache drive:        {drive_letter}  type={drive_type}")
        print(f"  Cache drive free:   {du.free/1024**3:.1f} GB")
    except Exception:
        print(f"  Cache drive:        {drive_letter}  type={drive_type}")

    return {"gpu": p.name if torch.cuda.is_available() else None,
            "compute_capability": cc,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": drv,
            "ram_total_gb": total_gb,
            "ram_free_gb": avail_gb}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B  Validation time at batch 2 and batch 8
# ══════════════════════════════════════════════════════════════════════════════
def section_b():
    print(f"\n{SEP}")
    print("SECTION B  Validation time (no-grad, cache path)")
    print(f"{SEP}")
    from training.losses import BCEDiceLoss
    from preprocessing.cached_tile_dataset import CachedTileDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"
    results = {}

    for bs in (2, 8):
        model = fresh_model(device)
        model.eval()
        crit  = BCEDiceLoss(0.5, 0.5, 1.0)
        ds    = CachedTileDataset(split="val")
        loader= DataLoader(ds, batch_size=bs, shuffle=False,
                           num_workers=0, pin_memory=True)

        n_val  = len(ds)
        n_batches = (n_val + bs - 1) // bs
        losses = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images)
                    losses.append(float(crit(logits, masks)))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_batch = elapsed / len(losses) * 1000
        print(f"  batch_size={bs}: {len(losses)} batches  {elapsed:.1f}s  "
              f"{ms_batch:.1f} ms/batch  ({n_val} tiles)  [MEASURED]")
        results[f"bs{bs}"] = {"elapsed_s": round(elapsed, 2),
                              "n_batches": len(losses),
                              "ms_per_batch": round(ms_batch, 1)}
        del model, loader, ds
        gc.collect(); torch.cuda.empty_cache()
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C  Noise floor: raw path x2 same seed
# ══════════════════════════════════════════════════════════════════════════════
def section_c(n_steps=200):
    print(f"\n{SEP}")
    print(f"SECTION C  Noise floor: raw path x2 same seed ({n_steps} steps)")
    print(f"{SEP}")
    from preprocessing.tile_dataset import OilSpillTileDataset
    from torch.utils.data import DataLoader
    from training.losses import BCEDiceLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"

    def run_raw(run_id):
        torch.manual_seed(SEED); np.random.seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        model = fresh_model(device)
        model.train()
        crit  = BCEDiceLoss(0.5, 0.5, 1.0)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler= torch.amp.GradScaler("cuda") if amp else None
        ds    = OilSpillTileDataset(TILE_INDEX, split="train", scene_cache_size=2)
        loader= DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        losses = []
        t0 = time.perf_counter()
        for step, (images, masks) in enumerate(loader):
            if step >= n_steps: break
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                loss   = crit(logits, masks)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:      loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - t0
        print(f"  Run {run_id}: {len(losses)} steps  {elapsed:.1f}s  "
              f"first={losses[0]:.6f}  last={losses[-1]:.6f}", flush=True)
        del model, opt, scaler, loader, ds; gc.collect(); torch.cuda.empty_cache()
        return losses

    losses1 = run_raw(1)
    losses2 = run_raw(2)
    n = min(len(losses1), len(losses2))
    diffs = [abs(a-b) for a,b in zip(losses1[:n], losses2[:n])]
    max_d  = max(diffs)
    mean_d = sum(diffs)/len(diffs)
    print(f"\n  max  |run1 - run2|: {max_d:.2e}  [MEASURED]")
    print(f"  mean |run1 - run2|: {mean_d:.2e}  <- noise floor for same-path comparisons")
    return {"max_diff": max_d, "mean_diff": mean_d, "n_steps": n}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D  Individual GPU optimizations
# ══════════════════════════════════════════════════════════════════════════════
def section_d():
    print(f"\n{SEP}")
    print("SECTION D  Individual GPU optimizations  "
          f"(warmup={N_WARMUP}, measure={N_MEASURE})")
    print(f"{SEP}")
    print(f"  {'Config':<35} {'ms/batch':>9} {'tiles/s':>8}", flush=True)
    print(f"  {SEP2}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    kw_base = dict(device=device, loader_fn=fresh_loader,
                   model_factory=fresh_model, amp=True)

    # D1: baseline
    r = run_bench("D1_baseline", **kw_base)
    results["D1_baseline"] = r
    baseline_ms = r["ms"]

    # D2: fused Adam
    r = run_bench("D2_fused_adam", **kw_base, fused_adam=True)
    results["D2_fused_adam"] = r

    # D3: cudnn.benchmark
    r = run_bench("D3_cudnn_benchmark", **kw_base, cudnn_bench=True)
    results["D3_cudnn_benchmark"] = r

    # D4: channels_last
    r = run_bench("D4_channels_last", **kw_base, channels_last=True)
    results["D4_channels_last"] = r

    # D5: set_to_none=False (current is True, check if False is better/worse)
    r = run_bench("D5_set_to_none_False", **kw_base, set_to_none=False)
    results["D5_set_to_none_False"] = r

    # D6: GPU loss accumulation (no .item() per step)
    r = run_bench("D6_gpu_loss_accum", **kw_base, gpu_loss_accum=True)
    results["D6_gpu_loss_accum"] = r

    # D7: combined best (fused_adam + cudnn_bench + gpu_loss_accum)
    r = run_bench("D7_combined_best", **kw_base,
                  fused_adam=True, cudnn_bench=True, gpu_loss_accum=True)
    results["D7_combined_best"] = r

    # Summary table
    print(f"\n  {'Config':<35} {'ms/batch':>9} {'vs baseline':>12} {'keep?':>7}")
    print(f"  {SEP2}")
    for k, v in results.items():
        delta_pct = (v["ms"] - baseline_ms) / baseline_ms * 100
        keep = "YES" if delta_pct < -3.0 else ("--" if abs(delta_pct) <= 3.0 else "NO")
        print(f"  {k:<35} {v['ms']:>9.1f} {delta_pct:>+11.1f}%  {keep:>7}")

    return results, baseline_ms

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E  num_workers benchmark
# ══════════════════════════════════════════════════════════════════════════════
def section_e(noise_floor: dict, baseline_ms: float):
    print(f"\n{SEP}")
    print("SECTION E  num_workers 0/1/2 with persistent_workers")
    print(f"{SEP}")
    print(f"  RAM free at start: {_free_ram_gb():.2f} GB")
    print(f"  {'Config':<40} {'ms/batch':>9} {'tiles/s':>8} {'RAM_free_GB':>12}", flush=True)
    print(f"  {SEP2}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    RAM_FLOOR_GB = 1.0

    for nw in (0, 1, 2):
        for persistent in (False, True):
            if nw == 0 and persistent:
                continue  # invalid: persistent requires nw > 0
            label = f"workers={nw}_persist={'T' if persistent else 'F'}"

            free_before = _free_ram_gb()
            if free_before < RAM_FLOOR_GB + 0.5:
                print(f"  [{label}] SKIP - free RAM {free_before:.2f} GB < safety margin")
                continue

            def _loader():
                return fresh_loader(batch_size=2, num_workers=nw,
                                    shuffle=True, persistent=persistent)

            from training.losses import BCEDiceLoss
            model  = fresh_model(device)
            model.train()
            crit   = BCEDiceLoss(0.5, 0.5, 1.0)
            opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            loader = _loader()
            it     = iter(loader)
            amp    = True

            # warmup
            opt.zero_grad(set_to_none=True)
            for _ in range(N_WARMUP):
                try: images, masks = next(it)
                except StopIteration: it = iter(loader); images, masks = next(it)
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images); loss = crit(logits, masks)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            n = 0
            t_wall_start = time.perf_counter()
            t_ds_start   = time.perf_counter()
            t_data = 0.0
            min_free = _free_ram_gb()

            for _ in range(N_MEASURE):
                try: images, masks = next(it)
                except StopIteration: it = iter(loader); images, masks = next(it)
                torch.cuda.synchronize()
                t_data += time.perf_counter() - t_ds_start

                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images); loss = crit(logits, masks)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                _ = float(loss.detach())
                n += 1

                cur_free = _free_ram_gb()
                min_free = min(min_free, cur_free)
                if cur_free < RAM_FLOOR_GB:
                    print(f"  [{label}] ABORT at batch {n}: RAM {cur_free:.2f} GB < {RAM_FLOOR_GB} GB floor")
                    break
                t_ds_start = time.perf_counter()

            torch.cuda.synchronize()
            t_wall = time.perf_counter() - t_wall_start

            # explicitly shut down worker processes
            del it
            loader._iterator = None
            if hasattr(loader, '_workers') and loader._workers:
                for w in loader._workers:
                    try: w.terminate()
                    except Exception: pass
            del loader, model, opt, scaler
            gc.collect(); torch.cuda.empty_cache()

            ms   = t_wall / n * 1000
            ts   = n * 2 / t_wall
            free_after = _free_ram_gb()
            print(f"  {label:<40} {ms:>9.1f} {ts:>8.1f} {min_free:>12.2f}  [MEASURED]")
            results[label] = {"ms": ms, "tiles_s": ts,
                              "data_ms": t_data/n*1000,
                              "min_free_ram_gb": min_free}

            if free_after < RAM_FLOOR_GB:
                print(f"  Free RAM {free_after:.2f} GB below floor. Stopping worker tests.")
                break

    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F  Before/after summary + recommendation
# ══════════════════════════════════════════════════════════════════════════════
def section_f(a_data, b_data, c_data, d_data, d_baseline_ms, e_data):
    print(f"\n{SEP}")
    print("SECTION F  Before / After summary")
    print(f"{SEP}")

    d_results, _ = d_data

    print(f"\n  {'Metric':<38} {'Before (raw)':>14} {'After (cache)':>14}")
    print(f"  {SEP2}")
    print(f"  {'ms/batch':<38} {'429.1':>14} {d_baseline_ms:>13.1f}")
    print(f"  {'tiles/sec':<38} {'4.66':>14} {d_results['D1_baseline']['tiles_s']:>13.2f}")
    print(f"  {'data ms/batch':<38} {'322.8':>14} {d_results['D1_baseline']['t_data']:>13.1f}")
    print(f"  {'forward ms/batch':<38} {'34.4':>14} {d_results['D1_baseline']['t_fwd']:>13.1f}")
    print(f"  {'backward ms/batch':<38} {'56.0':>14} {d_results['D1_baseline']['t_bwd']:>13.1f}")
    print(f"  {'proj train epoch (h)':<38} {'1.96':>14} {d_baseline_ms*16430/3600/1000:>13.2f}")
    print(f"  {'proj val epoch (h) @ bs2':<38} {'0.49':>14} {b_data.get('bs2',{}).get('elapsed_s',0)/3600:>13.3f}")
    print(f"  {'proj total epoch (h)':<38} {'2.45':>14} {(d_baseline_ms*16430/1000 + b_data.get('bs2',{}).get('elapsed_s',0))/3600:>13.2f}")

    print(f"\n  Noise floor (raw x2 same seed):  max_diff={c_data['max_diff']:.2e}  mean={c_data['mean_diff']:.2e}")

    print(f"\n  Phase 3 opt results (>3% gain = keep):")
    for k, v in d_results.items():
        delta = (v["ms"] - d_baseline_ms) / d_baseline_ms * 100
        keep  = "KEEP" if delta < -3.0 else ("--" if abs(delta) <= 3.0 else "REVERT")
        print(f"    {k:<30} {delta:>+7.1f}%  [{keep}]")

    # Best workers
    if e_data:
        best_w = min(e_data.items(), key=lambda x: x[1]["ms"])
        print(f"\n  Best num_workers config: {best_w[0]}  {best_w[1]['ms']:.1f} ms/batch")

    print(f"\n  Recommended command (based on measurements):")
    # Determine best config
    best_d  = min(d_results.items(), key=lambda x: x[1]["ms"])
    best_ms = best_d[1]["ms"]
    best_ts = best_d[1]["tiles_s"]
    # Check if D7 (combined) beats individual opts
    d7_ms   = d_results.get("D7_combined_best", {}).get("ms", 9999)
    if d7_ms < best_ms * 0.97:
        use_fused = True; use_cudnn = True; use_gpu_accum = True
        rec_ms = d7_ms
    else:
        use_fused  = d_results.get("D2_fused_adam",{}).get("ms",9999) < d_baseline_ms * 0.97
        use_cudnn  = d_results.get("D3_cudnn_benchmark",{}).get("ms",9999) < d_baseline_ms * 0.97
        rec_ms = best_ms

    proj_total_h = (rec_ms * (16430 + 4104) / 1000) / 3600
    print(f"\n    .venv\\Scripts\\python.exe -m training.train \\")
    print(f"      --epochs 50 --batch-size 2 --learning-rate 1e-4 \\")
    print(f"      --optimizer Adam --scheduler none \\")
    print(f"      --num-workers 0 --mixed-precision \\")
    print(f"      --data-source cache \\")
    print(f"      --checkpoint-dir checkpoints --log-every 500")
    print(f"\n    Projected epoch time: {rec_ms*(16430+4104)/1000:.0f}s = {proj_total_h:.2f}h  [ESTIMATED]")
    print(f"    Projected 50 epochs:  {proj_total_h*50:.1f}h  [ESTIMATED]")
    print(f"    (Note: GPU opts with <3% gain omitted from default command;")
    print(f"     add --cudnn-benchmark / --fused-adam flags if implemented)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(SEP)
    print("PHASE 3  GPU / LOOP EFFICIENCY MEASUREMENT")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(SEP, flush=True)

    a = section_a()
    print(flush=True)

    b = section_b()
    print(flush=True)

    c = section_c(n_steps=200)
    print(flush=True)

    d = section_d()
    d_results, d_baseline_ms = d
    print(flush=True)

    e = section_e(noise_floor=c, baseline_ms=d_baseline_ms)
    print(flush=True)

    section_f(a, b, c, d, d_baseline_ms, e)

    print(f"\n{SEP}")
    print("PHASE 3 COMPLETE")
    print(SEP)
