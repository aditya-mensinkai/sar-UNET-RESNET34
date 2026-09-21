"""CachedTileDataset — O(1) memmap-backed tile reader.

Reads from the flat .npy memmaps produced by tools/build_cache.py.
No Rasterio, no process_scene(), no per-scene state.
Preprocessing is already baked in; this is a pure index-and-slice reader.

Interface identical to OilSpillTileDataset:
    __len__()   -> int
    __getitem__(i) -> (image_tensor [2,512,512] float32,
                       mask_tensor  [1,512,512] float32  {0.0, 1.0})

CachedTileDataset is a DROP-IN replacement for OilSpillTileDataset.
No augmentation (augmentation is OFF for this experiment).

The manifest is validated on __init__:
  - tile_index_md5 must match the current tile_index.csv
  - config_hash must match the current preprocessing_config.json
  - dtype, shape, n_tiles must match header of the .npy files

If validation fails a clear error is raised — never silently use a stale cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT      = Path(__file__).resolve().parent.parent
TILE_INDEX = ROOT / "processed_dataset" / "metadata" / "tile_index.csv"
PREP_CFG   = ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
CACHE_DIR  = ROOT / "processed_dataset" / "tile_cache"


# ── manifest helpers (duplicated from build_cache to keep modules independent) ──

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


class CachedTileDataset(Dataset):
    """
    Read preprocessed tiles from flat memmaps built by tools/build_cache.py.

    Parameters
    ----------
    split : "train" | "val" | "test"
    cache_dir : path to the tile_cache directory (default: processed_dataset/tile_cache)
    tile_index_path : override the tile_index.csv path (default: standard location)
    dtype : expected cache dtype ("float32" or "float16")

    Returns (image [2,512,512] float32, mask [1,512,512] float32) per item.
    The image is always returned as float32 even if the cache is float16
    (upcast on read — costs ~0.1 ms, avoids silent precision changes downstream).
    """

    def __init__(self, split: str = "train",
                 cache_dir: Path | str | None = None,
                 tile_index_path: Path | str | None = None,
                 dtype: str = "float32"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.tile_index_path = Path(tile_index_path) if tile_index_path else TILE_INDEX

        # Locate manifest — find file whose "splits" includes this split
        manifest, img_path, msk_path = self._find_manifest(split, dtype)

        # Validate manifest freshness
        idx_md5 = _file_md5(self.tile_index_path)
        if manifest["tile_index_md5"] != idx_md5:
            raise RuntimeError(
                f"Cache manifest tile_index_md5 mismatch.\n"
                f"  manifest: {manifest['tile_index_md5']}\n"
                f"  current:  {idx_md5}\n"
                f"Rebuild the cache with tools/build_cache.py.")
        if manifest["config_hash"] != _config_hash():
            raise RuntimeError(
                f"Cache manifest config_hash mismatch.\n"
                f"  manifest: {manifest['config_hash']}\n"
                f"  current:  {_config_hash()}\n"
                f"Rebuild the cache with tools/build_cache.py.")

        # Load row_order and filter to this split
        all_rows = manifest["row_order"]         # [{tile_id, split, row_idx}, ...]
        self._rows = [r for r in all_rows if r["split"] == split]
        if not self._rows:
            raise ValueError(f"No rows for split={split!r} in manifest {manifest}")
        # local index i -> global memmap row index
        self._global_indices = np.array([r["row_idx"] for r in self._rows], dtype=np.int64)

        # Open memmaps read-only (shared through OS page cache across workers)
        self._img_mm = np.load(str(img_path), mmap_mode="r")
        self._msk_mm = np.load(str(msk_path), mmap_mode="r")

        self._cache_dtype = manifest["dtype"]
        self.n_tiles = len(self._rows)

        # Validate shapes
        exp_img_shape = (manifest["n_tiles"], 2, 512, 512)
        exp_msk_shape = (manifest["n_tiles"], 512, 512)
        if self._img_mm.shape != exp_img_shape:
            raise RuntimeError(f"Image memmap shape {self._img_mm.shape} != {exp_img_shape}")
        if self._msk_mm.shape != exp_msk_shape:
            raise RuntimeError(f"Mask memmap shape {self._msk_mm.shape} != {exp_msk_shape}")

    def _find_manifest(self, split: str, dtype: str):
        """Find a manifest file that covers this split and dtype."""
        # Prefer the train+val combined manifest, fall back to per-split
        candidates = sorted(self.cache_dir.glob("manifest_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No manifest_*.json found in {self.cache_dir}.\n"
                f"Run: .venv\\Scripts\\python.exe tools\\build_cache.py")
        for mf_path in candidates:
            with open(mf_path) as f:
                mf = json.load(f)
            if split in mf.get("splits", []) and mf.get("dtype") == dtype:
                img_path = self.cache_dir / mf["img_file"]
                msk_path = self.cache_dir / mf["msk_file"]
                if img_path.exists() and msk_path.exists():
                    return mf, img_path, msk_path
        raise FileNotFoundError(
            f"No valid cache found for split={split!r} dtype={dtype!r} "
            f"in {self.cache_dir}.\n"
            f"Run: .venv\\Scripts\\python.exe tools\\build_cache.py --splits {split}")

    def __len__(self) -> int:
        return self.n_tiles

    def __getitem__(self, i: int):
        gi = int(self._global_indices[i])
        # memmap slice: O(1) index into the flat array; OS page cache serves subsequent hits
        img_np = np.array(self._img_mm[gi], dtype=np.float32)  # (2,512,512) float32
        msk_np = self._msk_mm[gi]                               # (512,512)   uint8

        # Sanity guards (same as OilSpillTileDataset)
        if img_np.shape != (2, 512, 512):
            raise RuntimeError(f"Unexpected image shape {img_np.shape} at global row {gi}")
        if not np.all(np.isfinite(img_np)):
            raise RuntimeError(f"NaN/Inf in cached image at global row {gi}")
        u = np.unique(msk_np)
        if not set(u.tolist()) <= {0, 1}:
            raise RuntimeError(f"Mask values {u.tolist()} at global row {gi}; expected {{0,1}}")

        image = torch.from_numpy(img_np)
        mask  = torch.from_numpy(msk_np[None].astype(np.float32))
        return image, mask

    def get_tile_id(self, i: int) -> str:
        return self._rows[i]["tile_id"]

    # Allow DataLoader workers to share the same mmap handles
    def __getstate__(self):
        st = self.__dict__.copy()
        # numpy memmaps are file-backed; workers re-open them from paths
        st["_img_mm"] = None
        st["_msk_mm"] = None
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-open memmaps in each worker process
        mf_path_candidates = sorted(self.cache_dir.glob("manifest_*.json"))
        for mf_path in mf_path_candidates:
            with open(mf_path) as f:
                mf = json.load(f)
            if self.split in mf.get("splits", []) and mf.get("dtype") == self._cache_dtype:
                img_path = self.cache_dir / mf["img_file"]
                msk_path = self.cache_dir / mf["msk_file"]
                if img_path.exists() and msk_path.exists():
                    self._img_mm = np.load(str(img_path), mmap_mode="r")
                    self._msk_mm = np.load(str(msk_path), mmap_mode="r")
                    return
        raise RuntimeError("Could not re-open memmaps in worker process")
