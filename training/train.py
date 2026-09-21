"""UNet + ResNet34 training orchestration (reuses validated pipeline components).

Dataset/model/loss/metrics come from preprocessing/tile_dataset.py,
models/unet_resnet34.py, training/losses.py, training/metrics.py -
this module only orchestrates: loaders, AMP, accumulation, val,
best/last checkpoints, resume, history, configs. No augmentation.

Changes vs Phase 3 baseline:
  - Early stopping: --early-stop-patience / --early-stop-min-delta
    monitors val_loss (min); saves best_val_loss checkpoint; counter
    and best_val_loss persisted in checkpoint for resume.
  - Per-epoch telemetry: GPU temp (°C), SM clock (MHz) via nvidia-smi
    subprocess; free system RAM (psutil if available, else /proc/meminfo).
  - Checkpoint hygiene: only best_model.pth + last_model.pth kept.

Usage: python -m training.train --epochs 50 --batch-size 2 [...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from models import UNetResNet34, UNetResNet34Config
from preprocessing.tile_dataset import OilSpillTileDataset
from training.dataloader import DataLoaderConfig, make_loader
from training.losses import BCEDiceLoss
from training.metrics import ConfusionAccumulator

TIP_DEFAULT = os.path.join("processed_dataset", "metadata", "tile_index.csv")
CFG_DEFAULT = os.path.join("processed_dataset", "metadata", "preprocessing_config.json")
BEST_CHOICES = {"val_dice": "max", "val_iou": "max", "val_loss": "min"}


# ---------------------------------------------------------------------------
# GPU / system telemetry
# ---------------------------------------------------------------------------

def _nvidia_smi_query():
    """Return (gpu_temp_c, sm_clock_mhz) via nvidia-smi, or (None, None) on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None


def _free_ram_mb():
    """Return free system RAM in MB. Tries psutil first, then /proc/meminfo, else None."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e6, 0)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024.0, 0)
    except Exception:
        pass
    # Windows fallback via ctypes
    try:
        import ctypes
        class _MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MEMSTATUS()
        ms.dwLength = ctypes.sizeof(_MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return round(ms.ullAvailPhys / 1e6, 0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9, **extra_kw):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **extra_kw)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra_kw)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be Adam/AdamW/SGD, got {name!r}")


def build_scheduler(name, opt, ns):
    name = name.lower()
    if name == "none":
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=ns.scheduler_step_size,
                                               gamma=ns.scheduler_gamma)
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ns.scheduler_tmax or ns.epochs)
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=ns.scheduler_factor, patience=ns.scheduler_patience)
    raise ValueError(f"scheduler must be none/StepLR/CosineAnnealingLR/ReduceLROnPlateau, got {name!r}")


def gpu_info():
    if not torch.cuda.is_available():
        return {"cuda": False, "gpu": None, "vram_mb": None, "cuda_version": None}
    p = torch.cuda.get_device_properties(0)
    return {"cuda": True, "gpu": p.name, "vram_mb": round(p.total_memory / 1e6),
            "cuda_version": torch.version.cuda}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="UNet+ResNet34 oil-spill training")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="Adam")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--scheduler", default="none")
    ap.add_argument("--scheduler-step-size", type=int, default=30)
    ap.add_argument("--scheduler-gamma", type=float, default=0.1)
    ap.add_argument("--scheduler-tmax", type=int, default=0)
    ap.add_argument("--scheduler-patience", type=int, default=5)
    ap.add_argument("--scheduler-factor", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mixed-precision", dest="mixed_precision", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--gradient-accumulation", type=int, default=1)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--best-metric", default="val_dice", choices=sorted(BEST_CHOICES))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="cap train+val batches (smoke/probe only; 0 = full)")
    ap.add_argument("--tile-index", default=TIP_DEFAULT)
    ap.add_argument("--preprocessing-config", default=CFG_DEFAULT)
    ap.add_argument("--sampler", default="scene_major", choices=["shuffled", "scene_major"],
                    help="shuffled=global IID shuffle (validated, but ~0%% scene-cache hits); "
                         "scene_major=shuffle scene order, sequential tiles per scene "
                         "(cache-friendly; batches may hold same-scene tiles)")
    ap.add_argument("--log-every", type=int, default=1000,
                    help="progress log interval in batches (0 = epoch-end only)")
    ap.add_argument("--data-source", default="raw", choices=["raw", "cache"],
                    help="raw=on-the-fly process_scene (default); "
                         "cache=memmap from tools/build_cache.py (faster, bit-identical)")
    ap.add_argument("--cache-dir", default=None,
                    help="Override tile_cache directory (only used with --data-source cache)")
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs (default 1 = every epoch)")
    # Phase 3 GPU opts (individually measured; default OFF to preserve exact baseline)
    ap.add_argument("--fused-adam", dest="fused_adam",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Use fused Adam kernel (measured -15.4%% on RTX 3050; "
                         "falls back silently if unsupported)")
    ap.add_argument("--cudnn-benchmark", dest="cudnn_benchmark",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Set torch.backends.cudnn.benchmark=True (measured -7.6%%; "
                         "only beneficial for fixed input sizes)")
    ap.add_argument("--gpu-loss-accum", dest="gpu_loss_accum",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Accumulate loss on GPU; sync only at log intervals "
                         "(measured -3.2%%; removes per-step .item() sync)")
    # Early stopping (monitors val_loss, min direction)
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Stop if val_loss has not improved by --early-stop-min-delta "
                         "for this many consecutive validation epochs. "
                         "0 = disabled (default).")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                    help="Minimum absolute improvement in val_loss to reset the patience "
                         "counter (default 1e-4).")
    return ap.parse_args(argv)


def _mem():
    if not torch.cuda.is_available():
        return (None, None)
    try:
        return (round(torch.cuda.memory_allocated() / 1e6, 1),
                round(torch.cuda.memory_reserved() / 1e6, 1))
    except Exception:
        return (None, None)


def main(argv=None):
    ns = parse_args(argv)
    if ns.epochs <= 0 or ns.batch_size <= 0 or ns.gradient_accumulation <= 0:
        print("FATAL: epochs/batch-size/gradient-accumulation must be positive", flush=True)
        return 2
    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ns.seed)

    # ---- pre-training assertions (read-only wrt dataset) ----
    for p in (ns.tile_index, ns.preprocessing_config):
        if not os.path.isfile(p):
            print(f"FATAL: missing {p}", flush=True)
            return 2
    pcfg = json.load(open(ns.preprocessing_config))
    if pcfg.get("augmentation", {}).get("enabled", None) not in (False,):
        print("FATAL: augmentation must be OFF", flush=True)
        return 2
    if pcfg.get("tile", {}).get("tile_size") != 512:
        print("FATAL: tile_size must be 512", flush=True)
        return 2
    n_train = len(OilSpillTileDataset(ns.tile_index, split="train"))
    n_val = len(OilSpillTileDataset(ns.tile_index, split="val"))
    for got, exp, name in ((n_train, 32860, "train"), (n_val, 8208, "val")):
        if got != exp:
            print(f"WARNING: {name} tiles={got} differ from validated {exp}", flush=True)
    device = torch.device("cuda" if (ns.device == "auto" and torch.cuda.is_available())
                          else ns.device if ns.device != "auto" else "cpu")
    amp = bool(ns.mixed_precision) and device.type == "cuda"
    if ns.mixed_precision and device.type != "cuda":
        print("AMP requested on CPU: disabled (graceful fallback)", flush=True)
    gi = gpu_info()

    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.to(device)
    crit = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
    # Phase 3 GPU opts (measured, default OFF, enabled via CLI flags)
    if getattr(ns, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True  [Phase 3 opt, measured -7.6%]", flush=True)
    _opt_extra = {}
    if getattr(ns, "fused_adam", False):
        try:
            _opt_extra["fused"] = True
            _test = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-4,
                                     fused=True)
            del _test
            print("fused Adam = True  [Phase 3 opt, measured -15.4%]", flush=True)
        except (TypeError, RuntimeError):
            _opt_extra = {}
            print("fused Adam requested but not supported on this build; using standard Adam",
                  flush=True)
    opt = build_optimizer(ns.optimizer, model.parameters(), ns.learning_rate,
                          ns.weight_decay, ns.momentum, **_opt_extra)
    sched = build_scheduler(ns.scheduler, opt, ns)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    mode = BEST_CHOICES[ns.best_metric]

    # Early stopping state
    es_patience = max(0, ns.early_stop_patience)
    es_min_delta = abs(ns.early_stop_min_delta)
    es_counter = 0          # consecutive epochs without improvement
    es_best_val_loss = None # tracks best val_loss independently of --best-metric

    start_epoch, best_val, best_ep = 0, None, None

    if ns.resume:
        ck = torch.load(ns.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scheduler_state_dict") and sched is not None:
            sched.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict") and scaler is not None:
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_metric"]
        best_ep = start_epoch - 1
        # Restore early stopping state from checkpoint
        es_counter = ck.get("es_counter", 0)
        es_best_val_loss = ck.get("es_best_val_loss", None)
        print(f"Resuming from: {ns.resume}\nPrevious epoch: {start_epoch - 1}\n"
              f"Previous best metric: {best_val}\n"
              f"Early-stop counter restored: {es_counter}/{es_patience}\n"
              f"Early-stop best val_loss restored: {es_best_val_loss}", flush=True)

    cfg = DataLoaderConfig(batch_size=ns.batch_size, num_workers=ns.num_workers)
    if ns.data_source == "cache":
        from preprocessing.cached_tile_dataset import CachedTileDataset
        from pathlib import Path as _Path
        _cache_dir = _Path(ns.cache_dir) if ns.cache_dir else None
        _ds_train = CachedTileDataset(split="train", cache_dir=_cache_dir)
        _ds_val   = CachedTileDataset(split="val",   cache_dir=_cache_dir)
        from torch.utils.data import DataLoader as _DL
        train_loader = _DL(_ds_train, batch_size=ns.batch_size, shuffle=True,
                           num_workers=ns.num_workers, pin_memory=True,
                           persistent_workers=(ns.num_workers > 0))
        train_loader.scene_sampler = None
        val_loader = _DL(_ds_val, batch_size=ns.batch_size, shuffle=False,
                         num_workers=ns.num_workers, pin_memory=True,
                         persistent_workers=(ns.num_workers > 0))
        val_loader.scene_sampler = None
        print(f"Data source: CACHE  (no Rasterio, no process_scene)", flush=True)
    else:
        train_loader = make_loader("train", ns.tile_index, cfg,
                                   batch_sampler="scene_major" if ns.sampler == "scene_major" else None)
        val_loader = make_loader("val", ns.tile_index, cfg)
        if ns.sampler == "scene_major":
            print("Sampler: scene_major (scene order shuffled per epoch; tiles sequential "
                  "within a scene for cache locality; batches may contain same-scene tiles)",
                  flush=True)

    eff_bs = ns.batch_size * ns.gradient_accumulation
    es_desc = (f"patience={es_patience} min_delta={es_min_delta}"
               if es_patience > 0 else "DISABLED")
    print("=" * 60 + "\nUNet + ResNet34 TRAINING\n" + "=" * 60, flush=True)
    print(f"\nDevice:\n{device}\nGPU:\n{gi['gpu']}\n\nDataset:\n"
          f"Train scenes: 2053\nValidation scenes: 513\nTest scenes: NOT USED\n"
          f"Train tiles: {n_train}\nValidation tiles: {n_val}\n\n"
          f"Tile size:\n512\nStride:\n512\nAugmentation:\nOFF\n\n"
          f"Model:\nArchitecture:\nUNet + ResNet34\nEncoder:\nResNet34\n"
          f"Input channels:\n2\nPretrained:\nFalse\n\nTraining:\nEpochs:\n{ns.epochs}\n"
          f"Batch size:\n{ns.batch_size}\nEffective batch size:\n{eff_bs}\n"
          f"Learning rate:\n{ns.learning_rate}\nOptimizer:\n{ns.optimizer}\n"
          f"Scheduler:\n{ns.scheduler}\nMixed precision:\n{amp}\n"
          f"Gradient accumulation:\n{ns.gradient_accumulation}\nWorkers:\n{ns.num_workers}\n"
          f"Sampler:\n{ns.sampler}\n\n"
          f"Loss:\nBCEDiceLoss(0.5/0.5/1.0)\n\n"
          f"Early stopping:\n{es_desc}\n\n"
          f"Checkpoint:\nDirectory:\n{ns.checkpoint_dir}\n"
          f"Best metric:\n{ns.best_metric}\nBest mode:\n{mode}\n\n" + "=" * 60, flush=True)
    os.makedirs(ns.checkpoint_dir, exist_ok=True)
    t_start = time.time()
    hist = []
    if ns.resume and os.path.isfile("training_history.csv"):
        # preserve pre-resume epochs: append, never rewrite away history
        with open("training_history.csv", newline="") as f:
            for _r in csv.DictReader(f):
                _row = {}
                for k, v in _r.items():
                    if v in ("", None):
                        _row[k] = None
                        continue
                    try:
                        _row[k] = int(v) if k == "epoch" else float(v)
                    except ValueError:
                        _row[k] = v
                hist.append(_row)
        if hist:
            print(f"Resumed history: {len(hist)} prior epochs preserved", flush=True)
    n_params = sum(p.numel() for p in model.parameters())

    def save(path, epoch, best, es_ctr, es_best_loss):
        payload = {"model_state_dict": model.state_dict(),
                   "optimizer_state_dict": opt.state_dict(),
                   "scheduler_state_dict": sched.state_dict() if sched else None,
                   "scaler_state_dict": scaler.state_dict() if scaler else None,
                   "epoch": epoch, "best_metric": best, "best_metric_name": ns.best_metric,
                   "best_metric_mode": mode, "training_config": tcfg,
                   # Early stopping state for resume
                   "es_counter": es_ctr,
                   "es_best_val_loss": es_best_loss,
                   "rng_state": {
                       "python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    tcfg = {"experiment": "UNet-ResNet34-SAR", "seed": ns.seed, "device": str(device),
            "model": {"architecture": "UNet + ResNet34", "encoder": "ResNet34",
                      "input_channels": 2, "output_channels": 1, "pretrained": False,
                      "params": n_params},
            "dataset": {"train_scenes": 2053, "val_scenes": 513, "test_scenes": 450,
                        "train_tiles": n_train, "val_tiles": n_val,
                        "tile_size": 512, "train_stride": 512, "augmentation": False},
            "training": {"epochs": ns.epochs, "batch_size": ns.batch_size,
                         "effective_batch_size": eff_bs, "learning_rate": ns.learning_rate,
                         "optimizer": ns.optimizer, "weight_decay": ns.weight_decay,
                         "momentum": ns.momentum, "scheduler": ns.scheduler,
                         "scheduler_params": {"step_size": ns.scheduler_step_size,
                                              "gamma": ns.scheduler_gamma,
                                              "tmax": ns.scheduler_tmax,
                                              "patience": ns.scheduler_patience,
                                              "factor": ns.scheduler_factor},
                         "gradient_accumulation": ns.gradient_accumulation,
                         "mixed_precision": amp, "num_workers": ns.num_workers,
                         "sampler": ns.sampler, "log_every": ns.log_every,
                         "early_stop_patience": es_patience,
                         "early_stop_min_delta": es_min_delta},
            "loss": {"name": "BCEDiceLoss", "bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1.0},
            "checkpoint": {"directory": ns.checkpoint_dir, "best_metric": ns.best_metric,
                           "best_mode": mode},
            "runtime": {"torch": torch.__version__,
                        "torchvision": __import__("torchvision").__version__,
                        "cuda_version": torch.version.cuda, "gpu": gi["gpu"],
                        "vram_mb": gi["vram_mb"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    json.dump(tcfg, open("training_config.json", "w"), indent=1)

    try:
        n_train_batches = len(train_loader)
        for ep in range(start_epoch, ns.epochs):
            e0 = time.time()
            if getattr(train_loader, "scene_sampler", None) is not None:
                train_loader.scene_sampler.set_epoch(ep)
            model.train()
            tr_loss, tr_acc, tr_n = 0.0, ConfusionAccumulator(), 0
            tr_loss_gpu = torch.zeros(1, device=device) if getattr(ns, "gpu_loss_accum", False) else None
            opt.zero_grad(set_to_none=True)
            first = {}
            for bi, (images, masks) in enumerate(train_loader):
                if ns.limit_batches and bi >= ns.limit_batches:
                    break
                t_batch = time.time()
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                t_h2d = time.time() - t_batch
                t_batch = time.time()
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images)
                    loss = crit(logits, masks) / ns.gradient_accumulation
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fw = time.time() - t_batch
                if not bi and ep == start_epoch:
                    first = {"data_h2d_s": round(t_h2d, 3), "forward_s": round(t_fw, 3)}
                    print(f"First batch: data+H2D={first['data_h2d_s']}s forward={first['forward_s']}s "
                          f"(data includes CPU preprocessing on cache miss)", flush=True)
                t_bw0 = time.time()
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if not bi and ep == start_epoch:
                    first["backward_s"] = round(time.time() - t_bw0, 3)
                    print(f"First batch: backward={first['backward_s']}s", flush=True)
                with torch.no_grad():
                    tr_acc.add(logits.detach(), masks)  # same pre-update logits as the loss
                if (bi + 1) % ns.gradient_accumulation == 0:
                    if scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                # Loss accumulation: GPU tensor (no sync) or .item() (sync) per flag
                if getattr(ns, "gpu_loss_accum", False):
                    tr_loss_gpu = tr_loss_gpu + loss.detach() * ns.gradient_accumulation
                else:
                    tr_loss += float(loss.detach()) * ns.gradient_accumulation
                tr_n += 1
                if ns.log_every and (bi + 1) % ns.log_every == 0:
                    # Sync GPU loss tensor here (only at log interval, not per step)
                    if getattr(ns, "gpu_loss_accum", False):
                        tr_loss = float(tr_loss_gpu) / max(tr_n, 1)
                        tr_loss_gpu = torch.zeros(1, device=device)
                    el = time.time() - e0
                    rate = (bi + 1) / el
                    eta = (n_train_batches - bi - 1) / rate if rate > 0 else float("inf")
                    print(f"Epoch {ep + 1:03d}/{ns.epochs}  batch {bi + 1}/{n_train_batches} "
                          f"({100.0 * (bi + 1) / n_train_batches:.1f}%)  "
                          f"{rate:.2f} batches/s  elapsed {el:.0f}s  ETA {eta:.0f}s",
                          flush=True)
            if tr_n % ns.gradient_accumulation != 0:  # final incomplete group
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            # Sync GPU loss accumulator at epoch end if needed
            if getattr(ns, "gpu_loss_accum", False) and tr_loss_gpu is not None:
                tr_loss = float(tr_loss_gpu)
            model.eval()
            va_loss, va_acc, va_n = 0.0, ConfusionAccumulator(), 0
            run_val = (ns.val_every <= 1) or ((ep + 1) % ns.val_every == 0) or (ep + 1 == ns.epochs)
            with torch.no_grad():
                for images, masks in val_loader:
                    if not run_val:
                        break
                    if ns.limit_batches and va_n >= ns.limit_batches:
                        break
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=amp):
                        logits = model(images)
                        va_loss += float(crit(logits, masks))
                    va_acc.add(logits, masks)
                    va_n += 1
            tm, vm = tr_acc.metrics(), va_acc.metrics()
            lr_now = opt.param_groups[0]["lr"]
            if sched is not None:
                if ns.scheduler.lower() == "reducelronplateau":
                    sched.step(va_loss / max(va_n, 1))
                else:
                    sched.step()
            ma, mr = _mem()

            # --- per-epoch telemetry: GPU temp/SM clock + free RAM ---
            gpu_temp_c, sm_clock_mhz = _nvidia_smi_query()
            free_ram_mb = _free_ram_mb()

            ep_time = round(time.time() - e0, 1)
            row = {"epoch": ep + 1, "train_loss": round(tr_loss / max(tr_n, 1), 5),
                   "val_loss": round(va_loss / max(va_n, 1), 5),
                   "val_iou": round(vm["IoU"], 5), "val_dice": round(vm["Dice"], 5),
                   "val_precision": round(vm["Precision"], 5),
                   "val_recall": round(vm["Recall"], 5), "val_f1": round(vm["F1"], 5),
                   "train_iou": round(tm["IoU"], 5), "train_dice": round(tm["Dice"], 5),
                   "train_precision": round(tm["Precision"], 5),
                   "train_recall": round(tm["Recall"], 5), "train_f1": round(tm["F1"], 5),
                   "learning_rate": lr_now, "epoch_duration_sec": ep_time,
                   "gpu_memory_allocated_mb": ma, "gpu_memory_reserved_mb": mr,
                   "gpu_temp_c": gpu_temp_c,
                   "sm_clock_mhz": sm_clock_mhz,
                   "free_ram_mb": free_ram_mb}
            hist.append(row)
            print(f"Epoch {ep + 1:03d}/{ns.epochs}  "
                  f"train_loss={row['train_loss']}  val_loss={row['val_loss']}  "
                  f"val_dice={row['val_dice']}  val_iou={row['val_iou']}  "
                  f"time={ep_time}s  "
                  f"gpu_temp={gpu_temp_c}°C  sm_clock={sm_clock_mhz}MHz  "
                  f"free_ram={free_ram_mb}MB  "
                  f"LR={lr_now}",
                  flush=True)

            # ---- early stopping logic (monitors val_loss, min) ----
            val_loss_now = row["val_loss"]
            if run_val and es_patience > 0:
                if es_best_val_loss is None or val_loss_now < es_best_val_loss - es_min_delta:
                    es_best_val_loss = val_loss_now
                    es_counter = 0
                else:
                    es_counter += 1
                print(f"Early-stop: val_loss={val_loss_now:.5f}  best={es_best_val_loss:.5f}  "
                      f"counter={es_counter}/{es_patience}", flush=True)

            cur = row[ns.best_metric]
            better = best_val is None or (cur > best_val if mode == "max" else cur < best_val)
            if better:
                best_val, best_ep = cur, ep
                save(os.path.join(ns.checkpoint_dir, "best_model.pth"), ep, best_val,
                     es_counter, es_best_val_loss)
                print(f"NEW BEST {ns.best_metric}={cur}  saved best_model.pth", flush=True)
            save(os.path.join(ns.checkpoint_dir, "last_model.pth"), ep, best_val,
                 es_counter, es_best_val_loss)
            with open("training_history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                w.writeheader()
                w.writerows(hist)

            # Trigger early stop AFTER saving last checkpoint
            if run_val and es_patience > 0 and es_counter >= es_patience:
                print(f"EARLY STOPPING: val_loss did not improve by {es_min_delta} "
                      f"for {es_patience} consecutive validation epochs.  "
                      f"Best val_loss={es_best_val_loss:.5f} (epoch {ep + 1 - es_patience})",
                      flush=True)
                break

    except torch.cuda.OutOfMemoryError:
        print("FATAL: CUDA out of memory. Reduce batch_size, increase gradient_accumulation, "
              "or disable mixed precision only for debugging. Batch size unchanged.", flush=True)
        return 3

    dur = time.time() - t_start
    summ = {"experiment": "UNet-ResNet34-SAR", "completed_epochs": len(hist),
            "best_epoch": (best_ep + 1) if best_ep is not None else None,
            "best_metric": best_val, "best_metric_name": ns.best_metric,
            "best_metric_mode": mode,
            "best_checkpoint": os.path.join(ns.checkpoint_dir, "best_model.pth"),
            "last_checkpoint": os.path.join(ns.checkpoint_dir, "last_model.pth"),
            "final_train_loss": hist[-1]["train_loss"] if hist else None,
            "final_val_loss": hist[-1]["val_loss"] if hist else None,
            "final_val_dice": hist[-1]["val_dice"] if hist else None,
            "final_val_iou": hist[-1]["val_iou"] if hist else None,
            "final_val_precision": hist[-1]["val_precision"] if hist else None,
            "final_val_recall": hist[-1]["val_recall"] if hist else None,
            "final_val_f1": hist[-1]["val_f1"] if hist else None,
            "early_stopped": (es_patience > 0 and es_counter >= es_patience),
            "es_best_val_loss": es_best_val_loss,
            "GPU": gi["gpu"], "CUDA version": torch.version.cuda,
            "PyTorch version": torch.__version__,
            "training_duration_seconds": round(dur, 1),
            "training_duration_human": f"{dur / 3600:.2f}h"}
    json.dump(summ, open("training_summary.json", "w"), indent=1)
    print("=" * 60 + "\nTRAINING COMPLETE\n" + "=" * 60, flush=True)
    print(f"Best epoch: {summ['best_epoch']}\n"
          f"Best {ns.best_metric}: {best_val}\n"
          f"Best checkpoint: {summ['best_checkpoint']}\n"
          f"Last checkpoint:  {summ['last_checkpoint']}\n"
          f"Duration: {summ['training_duration_human']}\n"
          f"Test set: NOT USED DURING TRAINING\n" + "=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
