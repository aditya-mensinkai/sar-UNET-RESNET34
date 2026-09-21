"""tools/build_cache.py  --  Pre-cache all preprocessed tiles to flat memmaps.

Runs the EXACT same preprocessing chain as OilSpillTileDataset._scene():
    process_scene(augment_override=False, compute_stats=False)
then crops each tile window and stores the result in two flat memmap files:

    tile_cache/tiles_images_{splits}_{dtype}.npy   (N, 2, 512, 512)  float32|float16
    tile_cache/tiles_masks_{splits}.npy             (N, 512, 512)     uint8

A manifest JSON is written LAST after atomic rename, so a partial build is
never silently used.  Resumable via a .partial_done tracker.

Usage (from project root):
    .venv\\Scripts\\python.exe tools\\build_cache.py
    .venv\\Scripts\\python.exe tools\\build_cache.py --dry-run
    .venv\\Scripts\\python.exe tools\\build_cache.py --workers 2
    .venv\\Scripts\\python.exe tools\\build_cache.py --dtype float16

All writes go to  processed_dataset/tile_cache/.
SOURCE FILES ARE NEVER MODIFIED.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TILE_INDEX = ROOT / "processed_dataset" / "metadata" / "tile_index.csv"
PREP_CFG   = ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
CACHE_DIR  = ROOT / "processed_dataset" / "tile_cache"

TILE_H = TILE_W = 512
TILE_C = 2
SPLITS = ("train", "val", "test")


# ── helpers ───────────────────────────────────────────────────────────────

def _hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    if h:  return f"{h}h {m:02d}m {sec:.0f}s"
    if m:  return f"{m}m {sec:.1f}s"
    return f"{sec:.1f}s"


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _config_hash() -> str:
    with open(PREP_CFG) as f:
        c = json.load(f)
    sig = {
        "lee_enabled": c.get("lee_filter", {}).get("enabled"),
        "lee_kernel":  c.get("lee_filter", {}).get("kernel_size"),
        "bands":       c.get("statistics", {}).get("bands"),
        "augmentation_enabled": c.get("augmentation", {}).get("enabled"),
    }
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]


def _avail_ram_gb() -> float:
    try:
        class MEM(ctypes.Structure):
            _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                        ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                        ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                        ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
        m = MEM(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024**3
    except Exception:
        return 4.0


def _worker_cap() -> int:
    return max(1, min(os.cpu_count() - 1, int(_avail_ram_gb() // 2)))


def _ds_root() -> Path:
    """Return the root that tile_index paths are relative to (processed_dataset/)."""
    from preprocessing.tile_dataset import OilSpillTileDataset
    ds = OilSpillTileDataset(str(TILE_INDEX), split="train", scene_cache_size=1)
    return Path(ds.root)


# ── tile index loading ─────────────────────────────────────────────────────

def load_rows(splits: tuple[str, ...]) -> list[dict]:
    rows = []
    with open(TILE_INDEX, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] in splits:
                rows.append(r)
    return rows


def group_by_scene(rows: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["image_id"], []).append(i)
    return groups


# ── worker (module-level, spawn-safe on Windows) ───────────────────────────

def _process_one_scene(args):
    """
    Process one scene and write tiles into pre-allocated memmaps.
    args = (scene_key, tile_indices, rows, img_mm_path, msk_mm_path,
            dtype_str, done_path)
    Returns (scene_key, n_written, elapsed_s, error_or_None)
    """
    scene_key, tile_indices, rows, img_mm_path, msk_mm_path, dtype_str, done_path = args
    t0 = time.perf_counter()
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    try:
        sys.path.insert(0, str(ROOT))

        img_mm = np.lib.format.open_memmap(img_mm_path, mode="r+")
        msk_mm = np.lib.format.open_memmap(msk_mm_path, mode="r+")
        done   = np.lib.format.open_memmap(done_path,   mode="r+")

        from preprocessing.tile_dataset import OilSpillTileDataset as _DS
        from preprocessing.pipeline import process_scene

        row0   = rows[tile_indices[0]]
        _ds    = _DS(str(TILE_INDEX), split=row0["split"], scene_cache_size=1)
        bounds = _ds.bounds
        cfg    = _ds.cfg
        ds_root = Path(_ds.root)
        del _ds

        ipath = ds_root / row0["image_path"]
        mpath = ds_root / row0["mask_path"]

        out = process_scene(
            str(ipath), str(mpath), cfg,
            rng=None, split=row0["split"],
            bounds=bounds, augment_override=False, compute_stats=False)

        img_scene = np.asarray(out["normalized"], dtype=np_dtype)
        msk_scene = np.asarray(out["mask"],       dtype=np.uint8)
        del out

        n_written = 0
        for ti in tile_indices:
            if done[ti]:
                continue
            r = rows[ti]
            x = int(r["x"]); y = int(r["y"])
            h = int(r["tile_height"]); w = int(r["tile_width"])
            img_mm[ti] = img_scene[:, y:y+h, x:x+w]
            msk_mm[ti] = msk_scene[y:y+h, x:x+w]
            done[ti] = 1
            n_written += 1

        img_mm.flush(); msk_mm.flush(); done.flush()
        del img_scene, msk_scene

        return (scene_key, n_written, time.perf_counter() - t0, None)

    except Exception:
        return (scene_key, 0, time.perf_counter() - t0, traceback.format_exc())


# ── pilot ──────────────────────────────────────────────────────────────────

def run_pilot(rows: list[dict], groups: dict[str, list[int]],
              dtype_str: str) -> dict:
    """Process exactly 1 scene to measure per-scene time and RAM delta."""
    print("\n[PILOT] Running 1-scene pilot...", flush=True)
    scene_key = next(iter(groups))
    tile_idxs = groups[scene_key]
    n_scene_tiles = len(tile_idxs)
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    def get_avail_gb():
        try:
            class MEM(ctypes.Structure):
                _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                            ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                            ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                            ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
            m = MEM(); m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1024**3
        except Exception:
            return 0.0

    from preprocessing.tile_dataset import OilSpillTileDataset as _DS
    from preprocessing.pipeline import process_scene

    row0 = rows[tile_idxs[0]]
    _ds  = _DS(str(TILE_INDEX), split=row0["split"], scene_cache_size=1)
    ds_root = Path(_ds.root)

    ram_before = get_avail_gb()
    t0 = time.perf_counter()

    out = process_scene(
        str(ds_root / row0["image_path"]),
        str(ds_root / row0["mask_path"]),
        _ds.cfg, rng=None, split=row0["split"],
        bounds=_ds.bounds, augment_override=False, compute_stats=False)

    img_scene = np.asarray(out["normalized"], dtype=np_dtype)
    msk_scene = np.asarray(out["mask"],       dtype=np.uint8)

    # Crop and store only this scene's tiles (not the full N-tile array)
    img_buf = np.empty((n_scene_tiles, TILE_C, TILE_H, TILE_W), dtype=np_dtype)
    msk_buf = np.empty((n_scene_tiles, TILE_H, TILE_W),          dtype=np.uint8)
    for local_i, ti in enumerate(tile_idxs):
        r = rows[ti]
        x=int(r["x"]); y=int(r["y"]); h=int(r["tile_height"]); w=int(r["tile_width"])
        img_buf[local_i] = img_scene[:, y:y+h, x:x+w]
        msk_buf[local_i] = msk_scene[y:y+h, x:x+w]

    elapsed = time.perf_counter() - t0
    ram_after = get_avail_gb()
    ram_delta = ram_before - ram_after   # GB consumed (positive = more used)

    del img_buf, msk_buf, img_scene, msk_scene, out

    n_scenes_total   = len(groups)
    proj_serial_s    = elapsed * n_scenes_total
    proj_2workers_s  = proj_serial_s / 2

    # Theoretical peak per scene: float64 raw(2ch) + lin + lee + float32 norm
    SH = TILE_H * 4; SW = TILE_W * 4   # 2048x2048
    ram_peak_mb = (2 * SH * SW * 8 * 3 + 2 * SH * SW * 4) / 1024**2

    print(f"  Scene: {scene_key}  tiles in scene: {n_scene_tiles}")
    print(f"  Wall time:          {elapsed:.2f}s  [MEASURED]")
    print(f"  RAM delta:          {ram_delta:+.2f} GB  [MEASURED]")
    print(f"  Theoretical peak:   ~{ram_peak_mb:.0f} MB per worker")
    print(f"  Projected serial:   {_hms(proj_serial_s)} ({proj_serial_s:.0f}s)  [ESTIMATED]")
    print(f"  Projected 2-worker: {_hms(proj_2workers_s)} ({proj_2workers_s:.0f}s)  [ESTIMATED]")
    print(f"  Scenes total:       {n_scenes_total}", flush=True)

    return {"elapsed_s": elapsed, "ram_delta_gb": ram_delta,
            "proj_serial_s": proj_serial_s, "proj_2workers_s": proj_2workers_s,
            "n_scenes": n_scenes_total}


# ── main build ─────────────────────────────────────────────────────────────

def build(splits: tuple[str, ...], dtype_str: str, n_workers: int,
          dry_run: bool, cache_dir: Path, force: bool) -> None:

    np_dtype = np.float32 if dtype_str == "float32" else np.float16
    print(f"\n{'='*68}")
    print(f"build_cache  splits={splits}  dtype={dtype_str}  workers={n_workers}")
    print(f"cache_dir: {cache_dir}")
    print(f"{'='*68}", flush=True)

    rows   = load_rows(splits)
    groups = group_by_scene(rows)
    n_tiles  = len(rows)
    n_scenes = len(groups)
    print(f"\nTiles: {n_tiles}  Scenes: {n_scenes}  splits: {splits}")

    tag        = "_".join(sorted(splits))
    img_path   = cache_dir / f"tiles_images_{tag}_{dtype_str}.npy"
    msk_path   = cache_dir / f"tiles_masks_{tag}.npy"
    done_path  = cache_dir / f".partial_done_{tag}.npy"
    manifest_p = cache_dir / f"manifest_{tag}.json"
    img_tmp    = Path(str(img_path) + ".tmp")
    msk_tmp    = Path(str(msk_path) + ".tmp")

    # Check for complete existing cache
    if manifest_p.exists() and img_path.exists() and msk_path.exists() and not force:
        with open(manifest_p) as f:
            mf = json.load(f)
        if (mf.get("tile_index_md5") == _file_md5(TILE_INDEX) and
                mf.get("config_hash") == _config_hash()):
            print(f"\nValid complete cache found at {manifest_p}")
            print("Pass --force to rebuild.")
            return
        print("\nCache manifest mismatch — rebuilding.")

    # Run pilot
    pilot = run_pilot(rows, groups, dtype_str)
    if dry_run:
        print("\n[DRY RUN] Pilot complete. Remove --dry-run to build.")
        return

    # Allocate memmaps
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_shape = (n_tiles, TILE_C, TILE_H, TILE_W)
    msk_shape = (n_tiles, TILE_H, TILE_W)

    if not img_tmp.exists():
        print(f"\nAllocating image memmap {img_shape} {dtype_str}...", flush=True)
        t0 = time.perf_counter()
        mm = np.lib.format.open_memmap(str(img_tmp), mode="w+", dtype=np_dtype, shape=img_shape)
        mm.flush(); del mm
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print(f"  Resuming: {img_tmp.name} already exists")

    if not msk_tmp.exists():
        print(f"Allocating mask  memmap {msk_shape} uint8...", flush=True)
        t0 = time.perf_counter()
        mm = np.lib.format.open_memmap(str(msk_tmp), mode="w+", dtype=np.uint8, shape=msk_shape)
        mm.flush(); del mm
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print(f"  Resuming: {msk_tmp.name} already exists")

    if not done_path.exists():
        dm = np.lib.format.open_memmap(str(done_path), mode="w+",
                                        dtype=np.uint8, shape=(n_tiles,))
        dm[:] = 0; dm.flush(); del dm

    # Count resume progress
    dm = np.lib.format.open_memmap(str(done_path), mode="r")
    n_done = int(dm.sum()); del dm
    print(f"\nAlready done (resume): {n_done}/{n_tiles}")

    # Build work list: only scenes with at least one undone tile
    dm = np.lib.format.open_memmap(str(done_path), mode="r")
    scene_work = [(sk, tidxs) for sk, tidxs in groups.items()
                  if any(not dm[ti] for ti in tidxs)]
    del dm
    print(f"Scenes remaining: {len(scene_work)}/{n_scenes}", flush=True)

    if scene_work:
        args_list = [
            (sk, tidxs, rows, str(img_tmp), str(msk_tmp), dtype_str, str(done_path))
            for sk, tidxs in scene_work
        ]
        n_todo    = len(scene_work)
        n_written = 0
        errors    = []

        t_start = time.perf_counter()

        if n_workers <= 1:
            print(f"\nBuilding (1 worker / serial)...", flush=True)
            for ai, args in enumerate(args_list):
                sk, nw, el, err = _process_one_scene(args)
                if err:
                    errors.append((sk, err))
                    print(f"  ERROR {sk}: {err[:120]}", flush=True)
                else:
                    n_written += nw
                    done_cnt = ai + 1
                    elapsed  = time.perf_counter() - t_start
                    rate = done_cnt / elapsed
                    eta  = (n_todo - done_cnt) / rate if rate else 0
                    if done_cnt % 100 == 0 or done_cnt <= 3 or done_cnt == n_todo:
                        print(f"  [{done_cnt:4d}/{n_todo}] +{nw} tiles "
                              f"{elapsed:.0f}s  ETA {_hms(eta)}", flush=True)
        else:
            print(f"\nBuilding ({n_workers} workers)...", flush=True)
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for ai, (sk, nw, el, err) in enumerate(
                        pool.imap_unordered(_process_one_scene, args_list)):
                    if err:
                        errors.append((sk, err))
                        print(f"  ERROR {sk}: {err[:120]}", flush=True)
                    else:
                        n_written += nw
                        done_cnt = ai + 1
                        elapsed  = time.perf_counter() - t_start
                        rate = done_cnt / elapsed
                        eta  = (n_todo - done_cnt) / rate if rate else 0
                        if done_cnt % 100 == 0 or done_cnt <= 3 or done_cnt == n_todo:
                            print(f"  [{done_cnt:4d}/{n_todo}] +{nw} tiles "
                                  f"{elapsed:.0f}s  ETA {_hms(eta)}", flush=True)

        t_build = time.perf_counter() - t_start
        print(f"\nBuild done: {t_build:.1f}s ({_hms(t_build)})  "
              f"tiles written: {n_written}  errors: {len(errors)}", flush=True)
        if errors:
            for sk, e in errors[:5]:
                print(f"  ERROR: {sk}: {e[:200]}")

    # Verify completeness
    dm = np.lib.format.open_memmap(str(done_path), mode="r")
    n_final_done = int(dm.sum())
    del dm
    if n_final_done < n_tiles:
        print(f"\nFATAL: {n_tiles - n_final_done} tiles missing. "
              f"Re-run to resume. NOT finalizing.")
        return

    # Atomic finalize: rename tmp files, write manifest last
    print(f"\nAll {n_tiles} tiles written. Renaming...", flush=True)
    img_tmp.rename(img_path)
    msk_tmp.rename(msk_path)

    manifest = {
        "version":        1,
        "splits":         list(splits),
        "dtype":          dtype_str,
        "n_tiles":        n_tiles,
        "n_scenes":       n_scenes,
        "tile_shape_img": [TILE_C, TILE_H, TILE_W],
        "tile_shape_msk": [TILE_H, TILE_W],
        "img_file":       img_path.name,
        "msk_file":       msk_path.name,
        "config_hash":    _config_hash(),
        "tile_index_md5": _file_md5(TILE_INDEX),
        "tile_index_path": str(TILE_INDEX),
        # row_order: tile_id and split for each row index
        "row_order": [{"tile_id": r["tile_id"], "split": r["split"], "row_idx": i}
                      for i, r in enumerate(rows)],
        "build_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    mf_tmp = Path(str(manifest_p) + ".tmp")
    with open(mf_tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    mf_tmp.rename(manifest_p)
    done_path.unlink(missing_ok=True)

    img_gb = img_path.stat().st_size / 1024**3
    msk_gb = msk_path.stat().st_size / 1024**3
    print(f"\nCache complete!")
    print(f"  {img_path.name}  {img_gb:.2f} GB")
    print(f"  {msk_path.name}  {msk_gb:.2f} GB")
    print(f"  {manifest_p.name}")
    print(f"  Total: {img_gb+msk_gb:.2f} GB")


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Pre-cache SAR tiles")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=list(SPLITS))
    ap.add_argument("--dtype",    default="float32", choices=["float32", "float16"])
    ap.add_argument("--workers",  type=int, default=None,
                    help="Worker processes (default: auto)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Pilot only — do not write cache")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--force",    action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n_w = args.workers if args.workers is not None else _worker_cap()
    cap = _worker_cap()
    print(f"Worker auto-cap: min(cpu-1={os.cpu_count()-1}, "
          f"avail_RAM_GB/2={int(_avail_ram_gb()//2)}) = {cap}")
    print(f"Using workers: {n_w}")
    build(
        splits    = tuple(args.splits),
        dtype_str = args.dtype,
        n_workers = n_w,
        dry_run   = args.dry_run,
        cache_dir = Path(args.cache_dir),
        force     = args.force,
    )
