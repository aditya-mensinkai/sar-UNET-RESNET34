"""UNet + ResNet34 smoke test: construct, forward [B,2,512,512], validate logits,
NaN/Inf, backward gradient flow, param counts, CUDA. No training. Exit 0=PASS.
Usage: python tools/smoke_test_unet_resnet34.py [--batch-size 2]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models import UNetResNet34, UNetResNet34Config


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--pretrained", action="store_true")
    ns = ap.parse_args()
    print("=" * 60 + "\nUNET + RESNET34 MODEL SMOKE TEST\n" + "=" * 60, flush=True)

    # channel count from the actual Dataset contract (one real tile, not a magic number)
    from preprocessing.tile_dataset import OilSpillTileDataset
    ds = OilSpillTileDataset("processed_dataset/metadata/tile_index.csv", split="train")
    probe_img, _ = ds[0]
    C = int(probe_img.shape[0])
    print(f"Dataset probe channels: {C}", flush=True)
    if C != 2:
        fail(f"expected 2 SAR channels, got {C}")

    cfg = UNetResNet34Config(in_channels=C, out_channels=1, pretrained=ns.pretrained)
    print(f"\nModel configuration\n-------------------\nArchitecture: UNet + ResNet34\n"
          f"Input channels: {cfg.in_channels}\nOutput channels: {cfg.out_channels}\n"
          f"Pretrained: {cfg.pretrained}\nAutomatic weight download: Disabled", flush=True)
    model = UNetResNet34(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice\n------\nCUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    model.to(device).train()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel\n-----\nTotal parameters: {total}\nTrainable parameters: {trainable}", flush=True)

    x = torch.randn(ns.batch_size, C, 512, 512, dtype=torch.float32, device=device)
    logits = model(x)
    print(f"\nForward\n-------\nInput shape: {tuple(logits.shape) and tuple(x.shape)}\n"
          f"Output shape: {tuple(logits.shape)}\nOutput dtype: {logits.dtype}", flush=True)
    if logits.ndim != 4 or logits.shape[0] != ns.batch_size or logits.shape[1] != 1 \
            or logits.shape[2] != 512 or logits.shape[3] != 512:
        fail(f"bad output shape {tuple(logits.shape)}")
    if torch.isnan(logits).any():
        fail("Output NaN: True")
    print("Output NaN: False", flush=True)
    if torch.isinf(logits).any():
        fail("Output Inf: True")
    print("Output Inf: False", flush=True)
    has_sig = any(isinstance(m, nn.Sigmoid) for m in model.modules())
    print(f"Raw logits: PASS (min={logits.min().item():.3f} max={logits.max().item():.3f}, "
          f"dtype={logits.dtype})", flush=True)
    print(f"Sigmoid in model output: {'YES - FAIL' if has_sig else 'NO'}", flush=True)
    if has_sig:
        fail("sigmoid module present")

    loss = logits.mean()
    loss.backward()
    with_g = sum(1 for p in model.parameters() if p.grad is not None)
    without_g = sum(1 for p in model.parameters() if p.grad is None)
    print(f"\nGradient test\n-------------\nBackward: PASS\n"
          f"Parameters with gradients: {with_g}\n"
          f"Parameters without gradients: {without_g}", flush=True)
    if with_g == 0:
        fail("no gradients produced")
    print("\n" + "=" * 60 + "\nUNET + RESNET34 SMOKE TEST: PASS\n" + "=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    main()
