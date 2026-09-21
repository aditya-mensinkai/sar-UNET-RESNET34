"""Equal-wall-clock pilot: bs=8 lr=2e-4  vs  bs=2 lr=1e-4.

Implements the task-2 step-3 pilot as a standalone driver so the synthetic
batch-size sweep in tools/batch_benchmark.py does not have to be repeated.

Protocol
--------
  Arm A: batch_size=8, lr=2e-4, up to --epochs epochs (default 3),
         hard wall-clock cap --wall-cap seconds (default 3600).
  Arm B: batch_size=2, lr=1e-4, same epoch cap, wall-clock cap set to arm A's
         ACTUAL elapsed time (this is the "same wall-clock, not the same epoch
         count" requirement).
Both arms use the real CachedTileDataset (512x512), fused Adam,
cudnn.benchmark=True, num_workers=1, mixed precision - i.e. the full-run
configuration except batch size / learning rate.

Results are appended to stdout (line-buffered: each epoch prints immediately)
and written to --out as JSON.

Usage:
  .venv\\Scripts\\python.exe -u tools\\pilot_wallclock.py
  .venv\\Scripts\\python.exe -u tools\\pilot_wallclock.py --epochs 3 --wall-cap 3600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from tools.batch_benchmark import run_mini_pilot


def main(argv=None):
    ap = argparse.ArgumentParser(description="Equal-wall-clock pilot: bs8/lr2e-4 vs bs2/lr1e-4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--wall-cap", type=float, default=3600.0,
                    help="wall-clock cap for arm A in seconds (arm B gets arm A's elapsed)")
    ap.add_argument("--bs-a", type=int, default=8)
    ap.add_argument("--lr-a", type=float, default=2e-4)
    ap.add_argument("--bs-b", type=int, default=2)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--out", default=os.path.join("run_logs", "pilot_results.json"))
    ns = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device available", flush=True)
        return 1
    device = torch.device("cuda")

    print("=" * 60, flush=True)
    print(f"EQUAL-WALL-CLOCK PILOT: bs={ns.bs_a} lr={ns.lr_a}  vs  bs={ns.bs_b} lr={ns.lr_b}", flush=True)
    print(f"  real 512x512 cached tiles, fused Adam, cudnn.benchmark, workers=1, AMP", flush=True)
    print(f"  epoch cap={ns.epochs}  arm A wall cap={ns.wall_cap:.0f}s", flush=True)
    print("=" * 60, flush=True)

    print(f"\n[ARM A] bs={ns.bs_a} lr={ns.lr_a}", flush=True)
    t_a = time.time()
    ra = run_mini_pilot(ns.bs_a, ns.lr_a, ns.epochs, device, max_wall_sec=ns.wall_cap)
    elapsed_a = time.time() - t_a
    print(f"  arm A done: val_loss={ra['val_loss']}  val_dice={ra['val_dice']}  "
          f"epochs={ra['epochs_done']}  elapsed={elapsed_a:.0f}s", flush=True)

    print(f"\n[ARM B] bs={ns.bs_b} lr={ns.lr_b}  (wall budget = arm A elapsed = {elapsed_a:.0f}s)",
          flush=True)
    t_b = time.time()
    rb = run_mini_pilot(ns.bs_b, ns.lr_b, ns.epochs, device, max_wall_sec=elapsed_a)
    elapsed_b = time.time() - t_b
    print(f"  arm B done: val_loss={rb['val_loss']}  val_dice={rb['val_dice']}  "
          f"epochs={rb['epochs_done']}  elapsed={elapsed_b:.0f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("TIME-TO-QUALITY COMPARISON (same wall clock)", flush=True)
    print("=" * 60, flush=True)
    print(f"  bs={ns.bs_a} lr={ns.lr_a}: val_loss={ra['val_loss']}  val_dice={ra['val_dice']}  "
          f"epochs={ra['epochs_done']}  wall={elapsed_a:.0f}s", flush=True)
    print(f"  bs={ns.bs_b} lr={ns.lr_b}: val_loss={rb['val_loss']}  val_dice={rb['val_dice']}  "
          f"epochs={rb['epochs_done']}  wall={elapsed_b:.0f}s", flush=True)
    if ra["val_loss"] is not None and rb["val_loss"] is not None:
        print(f"\n  d_val_loss (A-B) = {ra['val_loss'] - rb['val_loss']:+.5f}", flush=True)
        print(f"  d_val_dice (A-B) = {ra['val_dice'] - rb['val_dice']:+.5f}", flush=True)
        who = (f"bs={ns.bs_a} lr={ns.lr_a}" if ra["val_loss"] < rb["val_loss"]
               else f"bs={ns.bs_b} lr={ns.lr_b}")
        print(f"  Lower val_loss at equal wall clock: {who}", flush=True)

    if ns.out:
        os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
        json.dump({"protocol": {"epochs_cap": ns.epochs, "wall_cap_a": ns.wall_cap,
                                "arm_a": {"bs": ns.bs_a, "lr": ns.lr_a},
                                "arm_b": {"bs": ns.bs_b, "lr": ns.lr_b},
                                "data": "CachedTileDataset real 512x512", "workers": 1,
                                "fused_adam": True, "cudnn_benchmark": True, "amp": True},
                   "arm_a": dict(ra, elapsed_sec=round(elapsed_a, 1)),
                   "arm_b": dict(rb, elapsed_sec=round(elapsed_b, 1))},
                  open(ns.out, "w"), indent=2)
        print(f"\nSaved: {ns.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
