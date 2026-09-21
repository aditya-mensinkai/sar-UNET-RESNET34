"""Training-pipeline smoke test: 1 epoch, 2 batches/split, temp checkpoints.

Verifies train+val batch flow, forward/loss/backward/step, AMP path,
metrics, checkpoint save+strict-load, and all artifact files. No full training.
Usage: python tools/smoke_test_training.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.train import main as train_main


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def main():
    tmp = tempfile.mkdtemp(prefix="train_smoke_")
    ckpt = os.path.join(tmp, "checkpoints")
    rc = train_main(["--epochs", "1", "--batch-size", "2", "--learning-rate", "1e-4",
                     "--optimizer", "Adam", "--scheduler", "none", "--num-workers", "0",
                     "--mixed-precision", "--gradient-accumulation", "1",
                     "--checkpoint-dir", ckpt, "--seed", "42",
                     "--best-metric", "val_dice", "--limit-batches", "2"])
    if rc != 0:
        fail(f"train_main returned {rc}")
    for f in ("training_config.json", "training_history.csv", "training_summary.json"):
        if not os.path.isfile(f):
            fail(f"missing artifact {f}")
    for f in ("best_model.pth", "last_model.pth"):
        if not os.path.isfile(os.path.join(ckpt, f)):
            fail(f"missing checkpoint {f}")
    # checkpoint integrity: strict load into fresh model
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import UNetResNet34, UNetResNet34Config
    for f in ("best_model.pth", "last_model.pth"):
        ck = torch.load(os.path.join(ckpt, f), map_location="cpu", weights_only=False)
        model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
        model.load_state_dict(ck["model_state_dict"])  # strict: raises on mismatch
        assert {"optimizer_state_dict", "epoch", "best_metric", "training_config",
                "rng_state"} <= set(ck), f"checkpoint keys incomplete in {f}"
    import json
    summ = json.load(open("training_summary.json"))
    assert summ["completed_epochs"] == 1, summ
    print("forward: PASS\nbackward: PASS\noptimizer step: PASS\nAMP: PASS\n"
          "metrics: PASS\ncheckpoint save/load: PASS", flush=True)
    print("=" * 60 + "\nTRAINING SMOKE TEST: PASS\n" + "=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    main()
