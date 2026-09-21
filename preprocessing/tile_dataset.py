"""On-the-fly 512px tile reader for UNet + ResNet34 training (no materialization).

Consumes processed_dataset/metadata/tile_index.csv as the ONLY coordinate
authority. For each tile: process the parent scene once via the existing
preprocessing.pipeline.process_scene (frozen bounds, augmentation OFF),
cache at most `scene_cache_size` processed scenes (LRU, per-process), then
crop the exact CSV (x, y, w, h) window from image AND mask. No augmentation,
no RNG, no file writes, no second coordinate algorithm.
"""
from __future__ import annotations

import csv
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_COLS = ["tile_id", "split", "class", "image_id", "image_path", "mask_path",
                 "x", "y", "tile_width", "tile_height", "original_width", "original_height"]


class OilSpillTileDataset(Dataset):
    """image [2,512,512] float32 + mask [1,512,512] float32 in {0.0, 1.0}."""

    def __init__(self, tile_index_path, split="train", cfg=None, bounds=None,
                 scene_cache_size=2):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if scene_cache_size is not None and scene_cache_size < 1:
            raise ValueError("scene_cache_size must be >= 1 or None (unbounded NOT allowed)")
        with open(tile_index_path) as f:
            rdr = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLS if c not in (rdr.fieldnames or [])]
            if missing:
                raise ValueError(f"tile_index.csv missing columns: {missing}")
            self.rows = [r for r in rdr if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no tile rows for split={split!r}")
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(tile_index_path)))
        self.split = split
        if cfg is None:
            from preprocessing import with_overrides
            cfg = with_overrides()
        else:
            from preprocessing import validate as _validate
            cfg = _validate(dict(cfg))
        self.cfg = cfg
        self.bounds = bounds
        if self.bounds is None:
            import json
            for _rel in ("metadata/preprocessing_config.json",          # authoritative experiment config
                         "reports/preprocessing_config.json"):          # legacy fallback
                _p = os.path.join(self.root, _rel)
                if os.path.isfile(_p):
                    _c = json.load(open(_p))
                    if "NORMALIZATION_BOUNDS" in _c:                     # legacy layout
                        self.bounds = _c["NORMALIZATION_BOUNDS"]
                    elif "bands" in _c.get("statistics", {}):           # experiment layout
                        self.bounds = {"bands": _c["statistics"]["bands"]}
                    break
            if self.bounds is None:
                raise ValueError("no NORMALIZATION_BOUNDS found in metadata/ or reports/ config")
        try:
            _bands = self.bounds["bands"]
            assert len(_bands) == 2 and all(len(_b) == 2 and _b[1] > _b[0] for _b in _bands)
        except (KeyError, TypeError, AssertionError):
            raise ValueError("NORMALIZATION_BOUNDS lacks consumable bands=[[lo,hi],[lo,hi]]")
        self.scene_cache_size = scene_cache_size
        self._cache = OrderedDict()  # image_id -> (normalized float32, mask uint8, W, H)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_cache"] = OrderedDict()  # fresh bounded cache per worker process
        return st

    def __len__(self):
        return len(self.rows)

    def get_metadata(self, i):
        return dict(self.rows[i])

    def _scene(self, row):
        key = row["image_id"]
        hit = self._cache.pop(key, None)
        if hit is not None:
            self._cache[key] = hit
            return hit
        from preprocessing.pipeline import process_scene
        ipath = os.path.join(self.root, row["image_path"])
        mpath = os.path.join(self.root, row["mask_path"])
        if not os.path.isfile(ipath):
            raise RuntimeError(
                f"Failed to read image window.\ntile_id: {row['tile_id']}\n"
                f"image_id: {row['image_id']}\nimage_path: {ipath}\n"
                f"x: {row['x']}\ny: {row['y']}")
        if not os.path.isfile(mpath):
            raise RuntimeError(
                f"Failed to read mask window.\ntile_id: {row['tile_id']}\n"
                f"image_id: {row['image_id']}\nmask_path: {mpath}\n"
                f"x: {row['x']}\ny: {row['y']}")
        try:
            out = process_scene(ipath, mpath, self.cfg or None, rng=None,
                                split=self.split, bounds=self.bounds,
                                augment_override=False, compute_stats=False)
        except TypeError:  # older process_scene without augment_override
            out = process_scene(ipath, mpath, self.cfg or None, rng=None,
                                split="val", bounds=self.bounds)
        img, msk = np.asarray(out["normalized"]), np.asarray(out["mask"])
        H, W = img.shape[-2:]
        if (W, H) != (int(row["original_width"]), int(row["original_height"])):
            raise RuntimeError(
                f"Scene dims {W}x{H} disagree with index "
                f"{row['original_width']}x{row['original_height']} for {row['tile_id']}")
        entry = (img.astype(np.float32, copy=False), msk, W, H)
        self._cache[key] = entry
        while len(self._cache) > self.scene_cache_size:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, i):
        row = self.rows[i]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["tile_width"]), int(row["tile_height"])
        if w != 512 or h != 512:
            raise RuntimeError(f"tile {row['tile_id']} is {w}x{h}, expected 512x512")
        img, msk, W, H = self._scene(row)
        if img.shape[0] != 2:
            raise RuntimeError(f"tile {row['tile_id']}: expected 2 channels, got {img.shape}")
        imw = np.array(img[:, y:y + h, x:x + w], copy=True)   # [y,x] order, never transposed
        maw = np.array(msk[y:y + h, x:x + w], copy=True)
        if imw.shape != (2, h, w) or maw.shape != (h, w):
            raise RuntimeError(f"window shape mismatch in {row['tile_id']}")
        if not (np.all(np.isfinite(imw)) and np.all(np.isfinite(maw))):
            raise RuntimeError(f"NaN/Inf in tile {row['tile_id']}")
        u = np.unique(maw)
        if not set(u.tolist()) <= {0, 1}:
            raise RuntimeError(f"mask values {u.tolist()} in tile {row['tile_id']}; "
                               f"expected subset of {{0,1}}")
        image = torch.from_numpy(imw)
        mask = torch.from_numpy(maw[None].astype(np.float32))
        if not set(torch.unique(mask).tolist()) <= {0.0, 1.0}:
            raise RuntimeError(f"post-conversion mask violation in {row['tile_id']}")
        return image, mask
