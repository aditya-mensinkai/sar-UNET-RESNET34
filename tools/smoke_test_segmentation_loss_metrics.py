"""Loss + metrics smoke test: known-answer synthetic tensors only (no test split).

Verifies sigmoid->threshold pipeline, threshold configurability, loss
finite+backward, metric math vs hand computation, Dice==F1, ranges, counts.
CPU + CUDA (if available). Exit 0=PASS. No training.
Usage: python tools/smoke_test_segmentation_loss_metrics.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.losses import BCEDiceLoss
from training.metrics import (ConfusionAccumulator, compute_binary_segmentation_metrics,
                              confusion_counts)


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def main():
    print("=" * 60 + "\nBINARY SEGMENTATION LOSS + METRICS SMOKE TEST\n" + "=" * 60,
          flush=True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    # known-answer fixture: TP=2 FP=1 TN=3 FN=2
    tgt = torch.tensor([[[[0, 0, 1, 1], [0, 1, 1, 0]]]], dtype=torch.float32)
    log = torch.tensor([[[[-2., 2., 2., -2.], [-2., 2., -2., -2.]]]], dtype=torch.float32)
    exp = {"TP": 2, "FP": 1, "TN": 3, "FN": 2, "IoU": 0.4, "Dice": 4 / 7,
           "Precision": 2 / 3, "Recall": 0.5, "F1": 4 / 7}
    for dev in devices:
        print(f"\n--- device: {dev} ---", flush=True)
        t, l = tgt.to(dev), log.to(dev).requires_grad_(True)
        m = compute_binary_segmentation_metrics(l, t, threshold=0.5)
        for k, v in exp.items():
            if not close(m[k], v):
                fail(f"[{dev}] {k}={m[k]} expected {v}")
        print(f"known-answer TP/FP/TN/FN/IoU/Dice/P/R/F1: PASS {m}", flush=True)
        # threshold configurability (0.95 -> all-background prediction)
        m95 = compute_binary_segmentation_metrics(l, t, threshold=0.95)
        if not (m95["TP"] == 0 and m95["FP"] == 0 and m95["TN"] == 4 and m95["FN"] == 4):
            fail(f"[{dev}] threshold 0.95 not applied: {m95}")
        print("threshold 0.5 vs 0.95: PASS (config works, not tuning)", flush=True)
        # empty-ground-truth policy
        me = compute_binary_segmentation_metrics(
            torch.full((1, 1, 2, 2), -5.0, device=dev), torch.zeros(1, 1, 2, 2, device=dev))
        if not (me["IoU"] == 1.0 and me["Dice"] == 1.0):
            fail(f"[{dev}] empty/empty policy: {me}")
        print("empty-GT policy: PASS", flush=True)
        # accumulator == single-shot global
        acc = ConfusionAccumulator()
        acc.add(l[:, :, :1, :], t[:, :, :1, :])
        acc.add(l[:, :, 1:, :], t[:, :, 1:, :])
        am = acc.metrics()
        if not all(close(am[k], exp[k]) for k in exp):
            fail(f"[{dev}] accumulator mismatch: {am}")
        print("accumulator consistency: PASS", flush=True)
        # loss: finite scalar + backward
        crit = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        big = torch.randn(2, 1, 32, 32, device=dev, requires_grad=True)
        bigt = (torch.rand(2, 1, 32, 32, device=dev) > 0.9).float()
        loss = crit(big, bigt)
        if not (loss.ndim == 0 and torch.isfinite(loss)):
            fail(f"[{dev}] loss not finite scalar: {loss}")
        loss.backward()
        if big.grad is None or not torch.isfinite(big.grad).all():
            fail(f"[{dev}] no/invalid grads")
        print(f"Loss: {float(loss.detach()):.5f}\nLoss finite: PASS\nLoss backward: PASS", flush=True)
        # ranges + counts + Dice==F1 on random data
        mr = compute_binary_segmentation_metrics(
            torch.randn(2, 1, 16, 16, device=dev),
            (torch.rand(2, 1, 16, 16, device=dev) > 0.95).float())
        if not all(0.0 <= mr[k] <= 1.0 for k in ("IoU", "Dice", "Precision", "Recall", "F1")):
            fail(f"[{dev}] range violation: {mr}")
        if mr["TP"] + mr["FP"] + mr["TN"] + mr["FN"] != 2 * 16 * 16:
            fail(f"[{dev}] count inconsistency: {mr}")
        if not close(mr["Dice"], mr["F1"]):
            fail(f"[{dev}] Dice != F1: {mr['Dice']} vs {mr['F1']}")
        print(f"TP: {mr['TP']}\nFP: {mr['FP']}\nTN: {mr['TN']}\nFN: {mr['FN']}\n"
              f"IoU: {mr['IoU']:.5f}\nDice: {mr['Dice']:.5f}\nPrecision: {mr['Precision']:.5f}\n"
              f"Recall: {mr['Recall']:.5f}\nF1: {mr['F1']:.5f}\n"
              f"Metric ranges: PASS\nConfusion-count consistency: PASS\n"
              f"Dice/F1 consistency: PASS", flush=True)
    print("\nTest threshold tuning: NOT PERFORMED\n\n" + "=" * 60 +
          "\nLOSS + METRICS SMOKE TEST: PASS\n" + "=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    main()
