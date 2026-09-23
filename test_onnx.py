"""ONNX inference test: run best_model.onnx on one real SAR tile.

Usage (PowerShell)::

    python test_onnx.py --onnx checkpoints\\best_model.onnx --image processed_dataset\\test\\images\\Oil\\00000.tif --y 1024 --x 0 [--save-mask out_mask.tif] [--compare-torch]

What it does:
  1. Loads + checks the ONNX model (onnx.checker).
  2. Preprocesses ONE 512x512 tile with the exact training pipeline
     (dB -> linear -> Lee 5x5 -> dB -> frozen P1/P99 -> float32 NCHW).
  3. Runs ONNX Runtime (CPU) -> raw logits -> sigmoid once -> threshold 0.5.
  4. Prints shapes, probability stats, positive-pixel count.
  5. Optionally saves the binary mask GeoTIFF and/or compares against the
     PyTorch checkpoint (requires --compare-torch + --checkpoint).

Preprocessing is OUTSIDE the ONNX graph; georeferencing stays in the existing
post-processing pipeline (postprocessing/run_postprocess.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TILE = 512
THRESHOLD = 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Test ONNX model on one real SAR tile.")
    ap.add_argument("--onnx", default=str(_REPO_ROOT / "checkpoints" / "best_model.onnx"))
    ap.add_argument("--image", default=str(
        _REPO_ROOT / "processed_dataset" / "test" / "images" / "Oil" / "00000.tif"))
    ap.add_argument("--y", type=int, default=1024)
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--save-mask", default=None, help="Path to save uint8 {0,1} mask GeoTIFF")
    ap.add_argument("--compare-torch", action="store_true")
    ap.add_argument("--checkpoint", default=str(_REPO_ROOT / "checkpoints" / "best_model.pth"))
    args = ap.parse_args(argv)

    import onnx
    import onnxruntime as ort

    if not os.path.isfile(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    m = onnx.load(args.onnx)
    onnx.checker.check_model(m)
    print("[test] onnx.checker: valid")

    print(f"[test] Available providers: {ort.get_available_providers()}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"[test] Selected provider: {sess.get_providers()}")
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"[test] input : name={inp.name} shape={inp.shape} dtype={inp.type}")
    print(f"[test] output: name={outp.name} shape={outp.shape} dtype={outp.type}")

    from preprocessing.pipeline import process_scene
    from preprocessing import with_overrides

    with open(_REPO_ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json") as f:
        bands = json.load(f)["statistics"]["bands"]
    res = process_scene(
        args.image, mask_path=None,
        cfg=with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5),
        bounds={"bands": bands}, split="inference",
        augment_override=False, compute_stats=False)
    img = res["normalized"]
    H, W = img.shape[1], img.shape[2]
    if args.y + TILE > H or args.x + TILE > W:
        print(f"ERROR: crop y={args.y},x={args.x} exceeds scene {H}x{W}", file=sys.stderr)
        return 2
    tile = img[:, args.y:args.y + TILE, args.x:args.x + TILE][np.newaxis].astype(np.float32)
    print(f"[test] scene={args.image} ({H}x{W}) crop y={args.y},x={args.x} tile={tile.shape}")

    logits = sess.run([outp.name], {inp.name: tile})[0]
    prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid exactly once
    mask = (prob >= args.threshold).astype(np.uint8)
    print(f"[test] logits shape={logits.shape} min={logits.min():.4f} max={logits.max():.4f}")
    print(f"[test] prob   shape={prob.shape} min={prob.min():.6f} max={prob.max():.6f} "
          f"mean={prob.mean():.6f}")
    print(f"[test] mask positives: {int(mask.sum())}/{mask.size} (threshold={args.threshold})")

    if args.save_mask:
        import rasterio
        with rasterio.open(args.image) as src:
            crs, transform = src.crs, src.transform
        # Crop transform to the tile origin
        win_transform = rasterio.transform.Affine(
            transform.a, transform.b, transform.c + args.x * transform.a,
            transform.d, transform.e, transform.f + args.y * transform.e)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "height": TILE, "width": TILE, "compress": "deflate"}
        if crs is not None:
            profile.update(crs=crs, transform=win_transform)
        with rasterio.open(args.save_mask, "w", **profile) as dst:
            dst.write(mask[0])
        print(f"[test] mask saved -> {args.save_mask}")

    if args.compare_torch:
        import torch
        from models import UNetResNet34, UNetResNet34Config
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            t_logits = model(torch.from_numpy(tile)).numpy()
        print(f"[test] torch-vs-onnx max abs diff: {float(np.max(np.abs(t_logits - logits))):.3e}")
        print(f"[test] torch-vs-onnx mean abs diff: {float(np.mean(np.abs(t_logits - logits))):.3e}")
        t_mask = (1.0 / (1.0 + np.exp(-t_logits)) >= args.threshold).astype(np.uint8)
        agree = float((t_mask == mask).mean())
        print(f"[test] mask agreement: {agree:.6f} "
              f"(torch_pos={int(t_mask.sum())} onnx_pos={int(mask.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
