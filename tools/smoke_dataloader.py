"""Minimal smoke-test dataloader for materialized 256px tiles. NO augmentation.

Reads tiled/{split}/images|masks/*.tif via tile_index.csv. Returns image
float32 [2,256,256] (stored normalized values, unmodified) and mask float32
[1,256,256] in {0,1}. No resizing, no augmentation, no preprocessing -
tiles are already preprocessed. Raises on any binary violation.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class TileDataset(Dataset):
    def __init__(self, root, split, limit=None):
        self.root = os.path.abspath(root)
        rows = [r for r in csv.DictReader(
            open(os.path.join(self.root, "metadata", "tile_index.csv")))
            if r["split"] == split and r["filtering_status"] == "kept"]
        rows.sort(key=lambda r: r["tile_id"])
        self.rows = rows[:limit] if limit else rows
        if not self.rows:
            raise ValueError(f"no kept tiles for split={split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with rasterio.open(os.path.join(self.root, r["image_path"])) as s:
            img = s.read().astype(np.float32)
        with rasterio.open(os.path.join(self.root, r["mask_path"])) as m:
            msk = m.read(1)
        if img.shape != (2, 256, 256):
            raise ValueError(f"bad image shape {img.shape} in {r['tile_id']}")
        if msk.shape != (256, 256):
            raise ValueError(f"bad mask shape {msk.shape} in {r['tile_id']}")
        if not np.all(np.isfinite(img)):
            raise ValueError(f"non-finite image in {r['tile_id']}")
        u = np.unique(msk)
        if not set(u.tolist()) <= {0, 1}:
            raise ValueError(f"non-binary mask {u.tolist()} in {r['tile_id']}")
        return (torch.from_numpy(img),
                torch.from_numpy(msk[None].astype(np.float32)),
                r["tile_id"])
