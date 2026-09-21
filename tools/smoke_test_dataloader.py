"""DataLoader smoke test: one batch per split, contract validation, CUDA transfer.

No training, no epochs, no full-dataset iteration. Exit 0 = PASS, 1 = FAIL.
Usage: python tools/smoke_test_dataloader.py [--batch-size 2]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.dataloader import DataLoaderConfig, make_loader

TIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "processed_dataset", "metadata", "tile_index.csv")


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def check_batch(images, masks, bs, tag):
    if tuple(images.shape) != (bs, 2, 512, 512):
        fail(f"{tag} image shape {tuple(images.shape)}")
    if tuple(masks.shape) != (bs, 1, 512, 512):
        fail(f"{tag} mask shape {tuple(masks.shape)}")
    if images.dtype != torch.float32 or masks.dtype != torch.float32:
        fail(f"{tag} dtypes {images.dtype}/{masks.dtype}")
    if bool(torch.isnan(images).any()) or bool(torch.isnan(masks).any()):
        fail(f"{tag} NaN present")
    if bool(torch.isinf(images).any()) or bool(torch.isinf(masks).any()):
        fail(f"{tag} Inf present")
    u = torch.unique(masks).tolist()
    if not set(u) <= {0.0, 1.0}:
        fail(f"{tag} non-binary mask {u}")
    if images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[2:]:
        fail(f"{tag} image/mask misalignment")
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ns = ap.parse_args()
    cfg = DataLoaderConfig(batch_size=ns.batch_size)
    print("=" * 60, flush=True)
    print("OIL-SPILL DATALOADER SMOKE TEST", flush=True)
    print("=" * 60, flush=True)
    print(f"\nConfiguration\n-------------\nbatch_size: {cfg.batch_size}\n"
          f"num_workers: {cfg.num_workers}\npin_memory: {cfg.pin_memory}\n"
          f"persistent_workers: {cfg.persistent_workers}", flush=True)
    for split, shuf in (("train", True), ("val", False), ("test", False)):
        loader = make_loader(split, TIP, cfg)
        from torch.utils.data import RandomSampler
        actual_shuf = isinstance(loader.sampler, RandomSampler)
        print(f"\n{split}: shuffle={actual_shuf} (expected {shuf})", flush=True)
        if actual_shuf != shuf:
            fail(f"{split} shuffle mismatch")
        images, masks = next(iter(loader))
        u = check_batch(images, masks, ns.batch_size, split)
        if split == "train":
            print(f"\nBatch\n-----\nImage shape: {images.shape}\nMask shape: {masks.shape}",
                  flush=True)
            print(f"\nImage dtype: {images.dtype}\nMask dtype: {masks.dtype}", flush=True)
            print(f"\nImage NaN: False\nMask NaN: False\n\nImage Inf: False\nMask Inf: False",
                  flush=True)
            print(f"\nMask unique values: {u}", flush=True)
            print("\nShape validation: PASS\nImage/mask alignment: PASS\n"
                  "Mask binary validation: PASS", flush=True)
    print("\nCUDA\n----", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        loader = make_loader("train", TIP, cfg)
        images, masks = next(iter(loader))
        ic = images.to("cuda", non_blocking=cfg.pin_memory)
        mc = masks.to("cuda", non_blocking=cfg.pin_memory)
        if ic.device.type != "cuda" or mc.device.type != "cuda":
            fail("CUDA transfer device mismatch")
        print("Image CUDA transfer: PASS\nMask CUDA transfer: PASS", flush=True)
        print(f"GPU allocated MB: {torch.cuda.memory_allocated() / 1e6:.1f}", flush=True)
    else:
        print("CUDA transfer test: SKIPPED", flush=True)
    print("\n" + "=" * 60 + "\nDATALOADER SMOKE TEST: PASS\n" + "=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    main()
