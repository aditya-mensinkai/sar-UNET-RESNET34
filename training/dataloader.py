"""DataLoader configuration + factory for the UNet + ResNet34 experiment.

Consumes the existing preprocessing.tile_dataset.OilSpillTileDataset unchanged.
No augmentation, no resizing, no preprocessing here - batching only.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    scene_major_shuffle: bool = False  # group tiles by scene, shuffle scene order
    shuffle_seed: int = 42

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers=True requires num_workers > 0")


class SceneMajorBatchSampler(Sampler):
    """Batches of co-located same-scene tiles; scene order shuffled per epoch.

    Preserves split isolation (built from one split's rows only) and tile
    content; only the PRESENTATION ORDER changes vs global shuffle: tiles of
    one scene are visited consecutively so the dataset's LRU scene cache hits.
    Deterministic given (seed, epoch). Yields lists of dataset indices.
    """

    def __init__(self, rows, batch_size, seed=42, drop_last=False):
        self.scenes = {}
        for i, r in enumerate(rows):
            self.scenes.setdefault(r["image_id"], []).append(i)
        self.scene_keys = sorted(self.scenes)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.scene_keys), generator=g).tolist()
        batch = []
        for k in order:
            for i in self.scenes[self.scene_keys[k]]:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = sum(len(v) for v in self.scenes.values())
        return (n + self.batch_size - 1) // self.batch_size if not self.drop_last \
            else n // self.batch_size


def make_loader(split, tile_index_path, cfg=None, batch_sampler=None, **overrides):
    """Build a DataLoader for split in {train, val, test}.

    Shuffle is derived from split (train=True, else False). Windows-safe:
    default num_workers=0; persistent_workers validated against num_workers.
    If batch_sampler is given it takes precedence (shuffle must be False then).
    """
    from preprocessing.tile_dataset import OilSpillTileDataset

    cfg = cfg or DataLoaderConfig(**overrides) if overrides else (cfg or DataLoaderConfig())
    if not isinstance(cfg, DataLoaderConfig):
        raise TypeError(f"cfg must be DataLoaderConfig, got {type(cfg).__name__}")
    ds = OilSpillTileDataset(tile_index_path, split=split)
    if batch_sampler == "scene_major" and split == "train":
        sampler = SceneMajorBatchSampler(ds.rows, cfg.batch_size, seed=cfg.shuffle_seed)
        loader = DataLoader(ds, batch_sampler=sampler,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            persistent_workers=cfg.persistent_workers)
        loader.scene_sampler = sampler  # for per-epoch set_epoch(); None otherwise
        return loader
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=(split == "train"),
                        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                        persistent_workers=cfg.persistent_workers)
    loader.scene_sampler = None
    return loader
