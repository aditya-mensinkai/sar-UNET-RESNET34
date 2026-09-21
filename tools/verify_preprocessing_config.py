"""Strict experiment validation for metadata/preprocessing_config.json (read-only).

Enforces: tile 512 / stride 512, augmentation disabled (all stochastic ops),
frozen train-only bounds present/finite/ordered, documented op order, neutral
band order, split behavior frozen-train-only. Exit 0 = valid, 1 = violations.
Usage: python tools/verify_preprocessing_config.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "processed_dataset", "metadata", "preprocessing_config.json")

ERRORS = []


def need(cond, msg):
    if not cond:
        ERRORS.append(msg)
    return cond


def main():
    c = json.load(open(PATH))
    t = c.get("tile", {})
    need(t.get("tile_size") == 512, f"tile.tile_size must be 512, got {t.get('tile_size')}")
    need(t.get("train_stride") == 512, f"tile.train_stride must be 512, got {t.get('train_stride')}")
    aug = c.get("augmentation", {})
    need(aug.get("enabled") is False, "augmentation.enabled must be false")
    for k, v in aug.items():
        if k in ("enabled", "note"):
            continue
        need(v is False, f"augmentation.{k} must be false, got {v!r}")
    ops = c.get("preprocessing", {}).get("operations")
    need(ops == ["calibration_gate", "db_to_linear", "lee_filter", "linear_to_db",
                 "frozen_normalization"], f"unexpected operations order: {ops}")
    lee = c.get("lee_filter", {})
    need(lee.get("enabled") is True and lee.get("kernel_size") == 5 and lee.get("type") == "Lee",
         f"lee_filter must be enabled Lee k=5, got {lee}")
    n = c.get("normalization", {})
    need(n.get("method") == "percentile" and n.get("per_band") is True, f"bad normalization: {n}")
    s = c.get("statistics", {})
    need(s.get("source_split") == "train", "statistics.source_split must be 'train'")
    need(s.get("frozen") is True, "statistics must be frozen")
    need(s.get("validation_uses_training_statistics") is True, "val must reuse train stats")
    need(s.get("test_uses_training_statistics") is True, "test must reuse train stats")
    exp = {"band_0": (-41.22778857146995, 0.0), "band_1": (-34.39919176014351, 0.0)}
    for b, (lo, hi) in exp.items():
        got = (s.get(b, {}).get("lower"), s.get(b, {}).get("upper"))
        need(got == (lo, hi), f"statistics.{b} must be [{lo}, {hi}], got {got}")
    bl = s.get("bands")
    need(bl == [[-41.22778857146995, 0.0], [-34.39919176014351, 0.0]],
         "statistics.bands consumable array mismatch")
    for v in [bl[0][0], bl[0][1], bl[1][0], bl[1][1]]:
        need(isinstance(v, (int, float)), "bounds must be numeric")
    bo = c.get("band_order", {})
    need(bo.get("band_0") == "unresolved" and bo.get("band_1") == "unresolved",
         f"band order must stay unresolved, got {bo}")
    for sp in ("train", "val", "test"):
        b = c.get("split_behavior", {}).get(sp, {})
        need(b.get("augmentation") is False and
             b.get("normalization_statistics") == "frozen_training_statistics",
             f"split_behavior.{sp} must be aug=false + frozen train stats, got {b}")
    if ERRORS:
        print("INVALID:", flush=True)
        for e in ERRORS:
            print(f"  - {e}", flush=True)
        return 1
    print("preprocessing_config.json: VALID (512/512, aug=false, frozen train bounds, "
          "Lee5, order documented, band order unresolved)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
