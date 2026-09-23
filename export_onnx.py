"""Export the trained SAR oil-spill U-Net/ResNet34 checkpoint to ONNX.

Canonical usage (PowerShell)::

    python export_onnx.py --checkpoint checkpoints\\best_model.pth --output checkpoints\\best_model.onnx

What this script does (and does NOT do):
  * Reconstructs the EXACT architecture from ``models/unet_resnet34.py``:
    ``UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1,
    pretrained=False))`` -- custom torchvision-ResNet34 encoder + UNet concat
    decoder, widths [256, 128, 64, 32], 1x1 conv head -> 1 raw-logit channel.
    NOTE: despite the task brief mentioning ``segmentation_models_pytorch``,
    the repository does NOT use smp anywhere; the model is fully custom.
  * Loads ``checkpoint["model_state_dict"]`` strictly (fails loudly on any
    missing/unexpected key). No weights are fabricated; the .pth is untouched.
  * Exports the RAW LOGITS forward output (no sigmoid/threshold inside ONNX).
    Inference applies ``sigmoid`` exactly once outside the model, then
    thresholds at 0.5 AFTER stitching (see ``inference/predict_scene.py``).
  * Fixed input ``[1, 2, 512, 512]`` float32 (two SAR bands, normalized [0,1]
    via frozen P1/P99 bounds). No dynamic axes: training/inference tiles are
    always 512x512.
  * No quantization / FP16 / pruning -- pure FP32 export.

Exporter choice: ``torch.onnx.export(..., dynamo=False)`` (TorchScript-based)
  is used DELIBERATELY because (a) it supports explicit ``input_names`` /
  ``output_names`` (``input`` / ``segmentation``), (b) it is the most broadly
  compatible path with the installed ONNX Runtime 1.30, and (c) the installed
  torch 2.14 dynamo exporter pulls in an onnxscript-IR pipeline that is less
  mature for this BatchNorm-heavy U-Net. Opset 17 covers every op used
  (Conv/BatchNorm/Relu/MaxPool/Resize for bilinear Upsample/Concat/Sigmoid-free
  graph) and is supported by ORT >= 1.13.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import UNetResNet34, UNetResNet34Config

# ---------------------------------------------------------------------------
# Constants derived from the repository (do NOT change without code change)
# ---------------------------------------------------------------------------
IN_CHANNELS = 2
OUT_CHANNELS = 1
TILE_SIZE = 512
INPUT_NAME = "input"
OUTPUT_NAME = "segmentation"
THRESHOLD = 0.5  # inference/predict_scene.py + postprocessing/config.py
DEFAULT_OPSET = 17


def inspect_checkpoint(ckpt_path: str) -> dict:
    """Load checkpoint, print its structure, return the state_dict to use."""
    print(f"[export] torch version : {torch.__version__}")
    print(f"[export] checkpoint    : {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[export] ckpt type     : {type(ck)}")
    if not isinstance(ck, dict):
        raise TypeError(f"Unexpected checkpoint type {type(ck)}; expected dict.")
    print(f"[export] top-level keys: {sorted(ck.keys())}")
    for k in ("epoch", "best_metric_name", "best_metric", "best_metric_mode"):
        if k in ck:
            print(f"[export]   {k} = {ck[k]!r}")
    if "model_state_dict" in ck:
        sd = ck["model_state_dict"]
        print("[export] using ck['model_state_dict']")
    elif "state_dict" in ck:
        sd = ck["state_dict"]
        print("[export] using ck['state_dict']")
    else:
        raise KeyError(
            "Checkpoint has neither 'model_state_dict' nor 'state_dict'. "
            f"Keys: {sorted(ck.keys())}"
        )
    keys = list(sd.keys())
    print(f"[export] state_dict tensors: {len(keys)}")
    print(f"[export]   first: {keys[0]} {tuple(sd[keys[0]].shape)}")
    print(f"[export]   last : {keys[-1]} {tuple(sd[keys[-1]].shape)}")
    print(f"[export]   stem.0.weight (conv1): {tuple(sd['stem.0.weight'].shape)}")
    return {"payload": ck, "state_dict": sd}


def build_model() -> torch.nn.Module:
    """Reconstruct the EXACT training architecture (train.py line ~248,
    predict_scene.py line ~90)."""
    cfg = UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False)
    print(
        "[export] architecture : UNetResNet34("
        f"in_channels={cfg.in_channels}, out_channels={cfg.out_channels}, "
        f"pretrained={cfg.pretrained}, "
        f"decoder_channels={list(cfg.decoder_channels)})"
    )
    return UNetResNet34(cfg)


def _mask_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(bool)
    b = b.astype(bool)
    tp = int((a & b).sum())
    fp = int((a & ~b).sum())
    fn = int((~a & b).sum())
    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    return {
        "agreement": float((a == b).mean()),
        "iou": float(tp / union) if union else 1.0,
        "dice": float(2 * tp / denom) if denom else 1.0,
        "torch_pos": int(a.sum()),
        "onnx_pos": int(b.sum()),
    }


def compare_outputs(torch_out: np.ndarray, onnx_out: np.ndarray, tag: str) -> dict:
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    mean_diff = float(np.mean(np.abs(torch_out - onnx_out)))
    denom = np.maximum(np.abs(torch_out), 1e-6)
    rel = float(np.max(np.abs(torch_out - onnx_out) / denom))
    print(f"[verify:{tag}] max abs diff : {max_diff:.6e}")
    print(f"[verify:{tag}] mean abs diff: {mean_diff:.6e}")
    print(f"[verify:{tag}] max rel diff : {rel:.6e}")
    return {"max_diff": max_diff, "mean_diff": mean_diff, "rel_diff": rel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export best_model.pth to ONNX (FP32, logits).")
    ap.add_argument("--checkpoint", default=str(_REPO_ROOT / "checkpoints" / "best_model.pth"))
    ap.add_argument("--output", default=str(_REPO_ROOT / "checkpoints" / "best_model.onnx"))
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--device", default="cpu", help="cpu (export always runs on CPU for determinism)")
    ap.add_argument("--real-image", default=None,
                    help="Real SAR GeoTIFF for validation (default: processed_dataset/test/images/Oil/00000.tif)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    # ---- 1-4. inspect checkpoint, rebuild exact model, load strictly ----
    info = inspect_checkpoint(args.checkpoint)
    model = build_model()
    missing, unexpected = model.load_state_dict(info["state_dict"], strict=False), None
    # load_state_dict(strict=False) returns names; re-run strict to enforce
    try:
        model.load_state_dict(info["state_dict"], strict=True)
    except RuntimeError as e:
        print(f"ERROR: state_dict mismatch -- {e}", file=sys.stderr)
        print("STOP: architecture does not match checkpoint; refusing to export.", file=sys.stderr)
        return 3
    print("[export] load_state_dict(strict=True): OK, no missing/unexpected keys")

    # ---- 5. eval mode, no grad ----
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dummy = torch.randn(1, IN_CHANNELS, TILE_SIZE, TILE_SIZE, dtype=torch.float32)
    print(f"[export] dummy input: shape={tuple(dummy.shape)} dtype={dummy.dtype} (NCHW float32)")

    with torch.no_grad():
        ref_logits = model(dummy).numpy()
    print(f"[export] torch forward: logits shape={ref_logits.shape} "
          f"min={ref_logits.min():.4f} max={ref_logits.max():.4f}")

    # ---- export (legacy TorchScript path, deliberate -- see docstring) ----
    print(f"[export] ONNX opset: {args.opset} (exporter: torch.onnx TorchScript, dynamo=False)")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote {args.output}")

    # ---- sizes ----
    pth_mb = os.path.getsize(args.checkpoint) / 1e6
    onnx_mb = os.path.getsize(args.output) / 1e6
    print(f"[export] PyTorch checkpoint size: {pth_mb:.1f} MB")
    print(f"[export] ONNX model size        : {onnx_mb:.1f} MB")

    # ---- onnx checker + graph inspection ----
    import onnx

    m = onnx.load(args.output)
    onnx.checker.check_model(m)
    print("[export] onnx.checker.check_model: ONNX model is valid")
    print(f"[export] IR version={m.ir_version} opset={m.opset_import[0].version} "
          f"producer={m.producer_name} {m.producer_version}")
    for i in m.graph.input:
        d = [(x.dim_value if x.HasField('dim_value') else '?') for x in i.type.tensor_type.shape.dim]
        print(f"[export] Input : name={i.name} shape={d} dtype={i.type.tensor_type.elem_type}")
    for o in m.graph.output:
        d = [(x.dim_value if x.HasField('dim_value') else '?') for x in o.type.tensor_type.shape.dim]
        print(f"[export] Output: name={o.name} shape={d} dtype={o.type.tensor_type.elem_type}")

    # ---- ORT session ----
    import onnxruntime as ort

    print(f"[export] Available providers: {ort.get_available_providers()}")
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.output, providers=providers)
    print(f"[export] Selected provider: {sess.get_providers()}")
    print(f"[export] Session inputs : {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"[export] Session outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

    # ---- dummy comparison (logits) ----
    onnx_logits = sess.run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy()})[0]
    print(f"[verify:dummy] torch {ref_logits.shape} vs onnx {onnx_logits.shape}")
    compare_outputs(ref_logits, onnx_logits, "dummy-logits")

    # probabilities + masks (sigmoid exactly once, threshold 0.5)
    torch_prob = 1.0 / (1.0 + np.exp(-ref_logits))
    onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logits))
    compare_outputs(torch_prob, onnx_prob, "dummy-prob")
    mm = _mask_metrics((torch_prob >= THRESHOLD), (onnx_prob >= THRESHOLD))
    print(f"[verify:dummy-mask] agreement={mm['agreement']:.6f} iou={mm['iou']:.6f} "
          f"dice={mm['dice']:.6f} torch_pos={mm['torch_pos']} onnx_pos={mm['onnx_pos']}")

    # ---- real SAR image validation ----
    real = args.real_image
    if real is None:
        cand = _REPO_ROOT / "processed_dataset" / "test" / "images" / "Oil" / "00000.tif"
        real = str(cand) if cand.exists() else None
    if real is not None and os.path.isfile(real):
        from preprocessing.pipeline import process_scene
        from preprocessing import with_overrides

        with open(_REPO_ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json") as f:
            bands = json.load(f)["statistics"]["bands"]
        out = process_scene(real, mask_path=None,
                            cfg=with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5),
                            bounds={"bands": bands}, split="inference",
                            augment_override=False, compute_stats=False)
        img = out["normalized"]  # (2, H, W) float32 in [0,1]
        print(f"[verify:real] image={real} normalized shape={img.shape} "
              f"dtype={img.dtype} min={img.min():.4f} max={img.max():.4f}")
        # Compare on two 512-crops: top-left (usually background) plus the
        # first crop with positive predictions (stronger mask-equivalence test).
        crops = {"top-left (y=0,x=0)": (0, 0)}
        H, W = img.shape[1], img.shape[2]
        model_cpu = model  # already on CPU, eval, no-grad
        with torch.no_grad():
            for yy in range(0, H - TILE_SIZE + 1, TILE_SIZE):
                for xx in range(0, W - TILE_SIZE + 1, TILE_SIZE):
                    t = torch.from_numpy(
                        img[:, yy:yy + TILE_SIZE, xx:xx + TILE_SIZE][np.newaxis].astype(np.float32))
                    p = torch.sigmoid(model_cpu(t)).numpy()
                    if int((p >= THRESHOLD).sum()) > 0:
                        crops[f"positive (y={yy},x={xx})"] = (yy, xx)
                        break
                if len(crops) > 1:
                    break
        for label, (yy, xx) in crops.items():
            tile = img[:, yy:yy + TILE_SIZE, xx:xx + TILE_SIZE][np.newaxis].astype(np.float32)
            print(f"[verify:real] tile '{label}' shape={tile.shape} (CHW->NCHW, float32)")
            with torch.no_grad():
                t_logits = model(torch.from_numpy(tile)).numpy()
            o_logits = sess.run([OUTPUT_NAME], {INPUT_NAME: tile})[0]
            c1 = compare_outputs(t_logits, o_logits, f"real-logits [{label}]")
            t_prob = 1.0 / (1.0 + np.exp(-t_logits))
            o_prob = 1.0 / (1.0 + np.exp(-o_logits))
            c2 = compare_outputs(t_prob, o_prob, f"real-prob [{label}]")
            mm = _mask_metrics((t_prob >= THRESHOLD), (o_prob >= THRESHOLD))
            print(f"[verify:real:{label}] input shape: {tile.shape}")
            print(f"[verify:real:{label}] PyTorch output shape: {t_logits.shape}")
            print(f"[verify:real:{label}] ONNX output shape   : {o_logits.shape}")
            print(f"[verify:real:{label}] PyTorch positive pixels: {mm['torch_pos']}")
            print(f"[verify:real:{label}] ONNX positive pixels   : {mm['onnx_pos']}")
            print(f"[verify:real:{label}] Mask IoU : {mm['iou']:.6f}")
            print(f"[verify:real:{label}] Mask Dice: {mm['dice']:.6f}")
        print("[verify:real] Post-processing compatibility: ONNX emits raw logits, "
              "identical to PyTorch forward; existing pipeline (sigmoid once -> "
              "overlap-mean stitch -> threshold 0.5 -> morphology/open3/close3 -> "
              "components>=100px -> GeoJSON/SHP) consumes ONNX output unchanged. "
              "Georeferencing (CRS/transform/polygons) stays outside ONNX.")
    else:
        print("[verify:real] skipped (no real image found; pass --real-image)")

    print("[export] DONE. ONNX output = raw LOGITS [1,1,512,512] float32; "
          "apply sigmoid once + threshold 0.5 exactly as inference/predict_scene.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
