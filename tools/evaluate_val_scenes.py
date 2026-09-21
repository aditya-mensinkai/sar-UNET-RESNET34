"""Phase 7 — Validation-only evaluation of the full-scene inference pipeline.

Evaluates on VAL SCENES ONLY.  Test split is refused unless --allow-test is
passed explicitly (do not use that flag during development).

Metrics reported for each (stride, threshold) combination:
  - Dataset-pooled Dice and IoU  (sum TP/FP/FN over all pixels, then divide)
  - Per-scene Dice (mean ± std of per-scene pooled values)
  - The same Dice computed as training's val_dice (pooled over tiles, threshold=0.5)
    so we can see exactly why they differ

Timing per scene: preprocessing, model+stitching, total (at each stride).

Usage::

    python tools/evaluate_val_scenes.py \\
        [--checkpoint PATH]   \\  # default: checkpoints/best_model.pth
        [--n-scenes N]        \\  # default: all val scenes (513)
        [--strides 512 384 256] \\
        [--thresholds ...]    \\  # default: 0.10 0.15 … 0.90 (step 0.05)
        [--batch-size 4]      \\
        [--output-dir results/val_eval]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── constants ──────────────────────────────────────────────────────────────
_DEFAULT_CKPT     = _ROOT / "checkpoints" / "best_model.pth"
_PREP_CFG_PATH    = _ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
_TILE_INDEX_PATH  = _ROOT / "processed_dataset" / "metadata" / "tile_index.csv"
_DEFAULT_OUT_DIR  = _ROOT / "results" / "val_eval"


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers  (pixel-level, integer counts → no per-batch averaging)
# ═══════════════════════════════════════════════════════════════════════════

def _dice_from_counts(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 1.0


def _iou_from_counts(tp: int, fp: int, fn: int) -> float:
    d = tp + fp + fn
    return (tp / d) if d else 1.0


def _threshold_map(prob: np.ndarray, thr: float) -> np.ndarray:
    """Return uint8 {0,1} binary mask from probability map."""
    return (prob >= thr).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Load infrastructure (shared across all stride/threshold combos)
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_and_bounds(ckpt_path: str, device):
    import torch
    from models import UNetResNet34, UNetResNet34Config

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()

    pcfg = json.load(open(_PREP_CFG_PATH))
    bounds = {"bands": pcfg["statistics"]["bands"]}
    return model, bounds


def _load_val_scene_map(n_scenes: Optional[int]) -> Dict[str, List[dict]]:
    """Return {image_id: [tile_rows]} for val split, limited to n_scenes."""
    with open(_TILE_INDEX_PATH) as f:
        all_rows = list(csv.DictReader(f))
    scene_map: Dict[str, List[dict]] = {}
    for r in all_rows:
        if r["split"] == "val":
            scene_map.setdefault(r["image_id"], []).append(r)
    ids = sorted(scene_map.keys())
    if n_scenes is not None:
        ids = ids[:n_scenes]
    return {sid: scene_map[sid] for sid in ids}


# ═══════════════════════════════════════════════════════════════════════════
# Per-scene inference  (returns prob_map + mask_path for ground truth)
# ═══════════════════════════════════════════════════════════════════════════

def _run_scene(
    model,
    bounds: dict,
    scene_rows: List[dict],
    stride: int,
    batch_size: int,
    device,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns
    -------
    prob_map   : float32 (H, W) — full-scene probability map
    gt_mask    : uint8   (H, W) — ground-truth binary mask
    timing     : dict with preprocessing_s, model_s, total_s
    """
    import torch
    from preprocessing.pipeline import process_scene
    from preprocessing import with_overrides
    from inference.windows import tile_grid, TILE_SIZE
    from inference.stitching import ProbabilityStitcher
    import rasterio

    cfg = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5)
    rep = scene_rows[0]
    ipath = str(_ROOT / "processed_dataset" / rep["image_path"])
    mpath = str(_ROOT / "processed_dataset" / rep["mask_path"])

    t0 = time.time()
    out = process_scene(
        ipath, mpath, cfg=cfg, bounds=bounds,
        split="val", augment_override=False, compute_stats=False,
    )
    t_prep = time.time() - t0

    norm = out["normalized"]           # (2, H, W) float32
    H, W = norm.shape[1], norm.shape[2]
    gt = out["mask"]                   # (H, W) uint8 {0,1}

    origins = tile_grid(H, W, tile=TILE_SIZE, stride=stride)
    stitcher = ProbabilityStitcher(H, W, TILE_SIZE)

    t1 = time.time()
    with torch.inference_mode():
        # batch the origins
        buf_origins, buf_patches = [], []
        for (y, x) in origins:
            buf_origins.append((y, x))
            buf_patches.append(norm[:, y : y + TILE_SIZE, x : x + TILE_SIZE])
            if len(buf_patches) == batch_size:
                _flush(model, buf_origins, buf_patches, stitcher, device, amp)
                buf_origins, buf_patches = [], []
        if buf_patches:
            _flush(model, buf_origins, buf_patches, stitcher, device, amp)

    prob_map = stitcher.finalize()     # float32 (H, W) — NO threshold here
    t_model = time.time() - t1

    timing = {
        "preprocessing_s": round(t_prep, 3),
        "model_s":         round(t_model, 3),
        "total_s":         round(t_prep + t_model, 3),
    }
    return prob_map, gt.astype(np.uint8), timing


def _flush(model, origins, patches, stitcher, device, amp):
    import torch
    tensor = torch.from_numpy(
        np.stack(patches).astype(np.float32)
    ).to(device)
    with torch.amp.autocast("cuda", enabled=amp):
        logits = model(tensor)
    probs = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()  # (B,512,512)
    for (y, x), p in zip(origins, probs):
        stitcher.add_tile(y, x, p)


# ═══════════════════════════════════════════════════════════════════════════
# Training-style val_dice  (pooled per-scene tiles at threshold=0.5)
# ═══════════════════════════════════════════════════════════════════════════

def _training_val_dice(
    model,
    bounds: dict,
    scene_map: Dict[str, List[dict]],
    batch_size: int,
    device,
    amp: bool,
) -> float:
    """Reproduce training's val_dice: run model on each 512x512 tile at stride=512,
    accumulate TP/FP/FN globally over all tiles, divide once.
    threshold=0.5  (same as ConfusionAccumulator.add default).
    """
    import torch
    from preprocessing.pipeline import process_scene
    from preprocessing import with_overrides

    cfg = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5)
    total_tp = total_fp = total_fn = 0

    for sid, rows in scene_map.items():
        rep = rows[0]
        ipath = str(_ROOT / "processed_dataset" / rep["image_path"])
        mpath = str(_ROOT / "processed_dataset" / rep["mask_path"])
        out   = process_scene(
            ipath, mpath, cfg=cfg, bounds=bounds,
            split="val", augment_override=False, compute_stats=False,
        )
        norm = out["normalized"]       # (2,H,W)
        gt   = out["mask"].astype(bool)  # (H,W)

        buf_patches, buf_masks = [], []
        for row in rows:
            x, y = int(row["x"]), int(row["y"])
            w, h = int(row["tile_width"]), int(row["tile_height"])
            patch = norm[:, y : y + h, x : x + w]
            buf_patches.append(patch)
            buf_masks.append(gt[y : y + h, x : x + w])
            if len(buf_patches) == batch_size:
                tp, fp, fn = _eval_tile_batch(model, buf_patches, buf_masks, device, amp)
                total_tp += tp; total_fp += fp; total_fn += fn
                buf_patches, buf_masks = [], []
        if buf_patches:
            tp, fp, fn = _eval_tile_batch(model, buf_patches, buf_masks, device, amp)
            total_tp += tp; total_fp += fp; total_fn += fn

    return _dice_from_counts(total_tp, total_fp, total_fn)


def _eval_tile_batch(model, patches, masks, device, amp) -> Tuple[int, int, int]:
    import torch
    tensor = torch.from_numpy(
        np.stack(patches).astype(np.float32)
    ).to(device)
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(tensor)
    probs = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()
    tp = fp = fn = 0
    for p, m in zip(probs, masks):
        pred = p >= 0.5
        tp += int((pred & m).sum())
        fp += int((pred & ~m).sum())
        fn += int((~pred & m).sum())
    return tp, fp, fn


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    ckpt_path: str,
    n_scenes: Optional[int],
    strides: List[int],
    thresholds: List[float],
    batch_size: int,
    output_dir: str,
    allow_test: bool = False,
) -> None:
    import torch

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"
    print(f"[eval] device={device}  amp={amp}")

    model, bounds = _load_model_and_bounds(ckpt_path, device)
    scene_map     = _load_val_scene_map(n_scenes)
    n_eval        = len(scene_map)
    print(f"[eval] val scenes to evaluate: {n_eval}")

    # ── results structure ──────────────────────────────────────────────────
    # stride_results[stride] = {
    #   "timing": [{"preprocessing_s", "model_s", "total_s"}, ...],
    #   "prob_maps": [np.ndarray(H,W), ...],
    #   "gt_masks":  [np.ndarray(H,W), ...],
    # }
    stride_results: Dict[int, dict] = {}

    for stride in strides:
        print(f"\n[eval] ── stride={stride} ──────────────────────────")
        prob_maps, gt_masks, timings = [], [], []

        for i, (sid, rows) in enumerate(scene_map.items()):
            t_scene = time.time()
            prob_map, gt, timing = _run_scene(
                model, bounds, rows, stride, batch_size, device, amp
            )
            prob_maps.append(prob_map)
            gt_masks.append(gt)
            timings.append(timing)

            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == n_eval:
                elapsed = time.time() - t_scene
                print(
                    f"  scene {i+1}/{n_eval}  {sid}  "
                    f"prep={timing['preprocessing_s']:.1f}s  "
                    f"model={timing['model_s']:.1f}s  "
                    f"total={timing['total_s']:.1f}s",
                    flush=True,
                )

        stride_results[stride] = {
            "timing":    timings,
            "prob_maps": prob_maps,
            "gt_masks":  gt_masks,
        }

    # ── training-style val_dice (once, uses stride=512 tiling) ────────────
    print("\n[eval] computing training-style val_dice (stride=512 tile-by-tile)…")
    t_tv = time.time()
    train_val_dice = _training_val_dice(model, bounds, scene_map, batch_size, device, amp)
    print(f"[eval] training-style val_dice = {train_val_dice:.5f}  ({time.time()-t_tv:.1f}s)")

    # ── per-(stride, threshold) metrics ───────────────────────────────────
    print("\n[eval] computing metrics …")
    results = []   # list of dicts, one per (stride, threshold)

    for stride in strides:
        sr = stride_results[stride]
        timings   = sr["timing"]
        prob_maps = sr["prob_maps"]
        gt_masks  = sr["gt_masks"]

        # Timing summary
        mean_prep  = float(np.mean([t["preprocessing_s"] for t in timings]))
        mean_model = float(np.mean([t["model_s"] for t in timings]))
        mean_total = float(np.mean([t["total_s"] for t in timings]))

        for thr in thresholds:
            pool_tp = pool_fp = pool_fn = 0
            per_scene_dice = []

            for prob, gt in zip(prob_maps, gt_masks):
                # threshold AFTER finalize — prob is already the stitched map
                pred = _threshold_map(prob, thr)
                pred_b = pred.astype(bool)
                gt_b   = gt.astype(bool)

                tp = int((pred_b & gt_b).sum())
                fp = int((pred_b & ~gt_b).sum())
                fn = int((~pred_b & gt_b).sum())

                pool_tp += tp
                pool_fp += fp
                pool_fn += fn
                per_scene_dice.append(_dice_from_counts(tp, fp, fn))

            pooled_dice = _dice_from_counts(pool_tp, pool_fp, pool_fn)
            pooled_iou  = _iou_from_counts(pool_tp, pool_fp, pool_fn)
            mean_scene_dice = float(np.mean(per_scene_dice))
            std_scene_dice  = float(np.std(per_scene_dice))

            results.append({
                "stride":           stride,
                "threshold":        round(thr, 4),
                "pooled_dice":      round(pooled_dice, 5),
                "pooled_iou":       round(pooled_iou, 5),
                "mean_scene_dice":  round(mean_scene_dice, 5),
                "std_scene_dice":   round(std_scene_dice, 5),
                "training_val_dice": round(train_val_dice, 5),
                "n_scenes":         n_eval,
                # timing (same for all thresholds within this stride)
                "mean_prep_s":      round(mean_prep, 3),
                "mean_model_s":     round(mean_model, 3),
                "mean_total_s":     round(mean_total, 3),
            })

    # ── find best (stride, threshold) by pooled_dice ──────────────────────
    best = max(results, key=lambda r: r["pooled_dice"])
    print(
        f"\n[eval] Best (stride={best['stride']}, thr={best['threshold']}): "
        f"pooled_dice={best['pooled_dice']:.5f}  "
        f"pooled_iou={best['pooled_iou']:.5f}  "
        f"mean_scene_dice={best['mean_scene_dice']:.5f}"
    )
    print("[eval] NOTE: this is the best on VAL only. The final threshold decision is yours.")

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "val_eval_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"[eval] results -> {csv_path}")

    # ── save JSON summary ─────────────────────────────────────────────────
    summary = {
        "checkpoint":        ckpt_path,
        "n_val_scenes":      n_eval,
        "strides":           strides,
        "thresholds":        thresholds,
        "training_val_dice": round(train_val_dice, 5),
        "best_stride":       best["stride"],
        "best_threshold":    best["threshold"],
        "best_pooled_dice":  best["pooled_dice"],
        "best_pooled_iou":   best["pooled_iou"],
        "results":           results,
        "note": (
            "training_val_dice is computed tile-by-tile at stride=512 "
            "with threshold=0.5, matching training's ConfusionAccumulator. "
            "pooled_dice is computed on the full stitched probability map "
            "at each threshold. They differ because: (a) overlapping tiles "
            "at stride<512 average probabilities before thresholding, and "
            "(b) threshold varies. At stride=512 and threshold=0.5 both "
            "should be nearly identical (any difference = AMP noise)."
        ),
    }
    json_path = os.path.join(output_dir, "val_eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] summary  -> {json_path}")

    # ── print table ───────────────────────────────────────────────────────
    _print_table(results, strides, thresholds, train_val_dice)


def _print_table(results, strides, thresholds, train_val_dice: float) -> None:
    """Print stride × threshold grid of pooled_dice."""
    # Index by (stride, thr)
    idx = {(r["stride"], r["threshold"]): r for r in results}
    thr_vals = sorted(set(r["threshold"] for r in results))

    print("\n" + "=" * 70)
    print("POOLED DICE — stride × threshold")
    print(f"(training-style val_dice at stride=512, thr=0.5 = {train_val_dice:.5f})")
    print("=" * 70)

    header = f"{'thr':>6}" + "".join(f"  S={s:<6}" for s in strides)
    print(header)
    print("-" * len(header))
    for thr in thr_vals:
        row_str = f"{thr:>6.2f}"
        for s in strides:
            key = (s, round(thr, 4))
            r   = idx.get(key)
            val = f"{r['pooled_dice']:.4f}" if r else "  N/A "
            row_str += f"  {val:<8}"
        print(row_str)

    print("\n" + "=" * 70)
    print("MEAN-SCENE DICE — stride × threshold")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for thr in thr_vals:
        row_str = f"{thr:>6.2f}"
        for s in strides:
            key = (s, round(thr, 4))
            r   = idx.get(key)
            val = f"{r['mean_scene_dice']:.4f}" if r else "  N/A "
            row_str += f"  {val:<8}"
        print(row_str)

    print("\n" + "=" * 70)
    print("MEAN TOTAL TIME PER SCENE (s) — stride")
    print("=" * 70)
    timing_by_stride = {}
    for r in results:
        s = r["stride"]
        if s not in timing_by_stride:
            timing_by_stride[s] = r
    for s, r in sorted(timing_by_stride.items()):
        print(
            f"  stride={s}: prep={r['mean_prep_s']:.2f}s  "
            f"model+stitch={r['mean_model_s']:.2f}s  "
            f"total={r['mean_total_s']:.2f}s"
        )
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Val-only evaluation of scene inference")
    ap.add_argument("--checkpoint",  default=str(_DEFAULT_CKPT))
    ap.add_argument("--n-scenes",    type=int,   default=None,
                    help="Number of val scenes (default: all 513)")
    ap.add_argument("--strides",     type=int,   nargs="+", default=[512, 384, 256])
    ap.add_argument("--thresholds",  type=float, nargs="+",
                    default=[round(x * 0.05, 4) for x in range(2, 19)])  # 0.10 .. 0.90
    ap.add_argument("--batch-size",  type=int,   default=4)
    ap.add_argument("--output-dir",  default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--allow-test",  action="store_true",
                    help="Allow test-split scenes (never use for tuning)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    ns = _parse(argv)

    if not ns.allow_test:
        # Refuse if somehow a test scene snuck in (belt-and-suspenders)
        print("[eval] evaluating VAL split only. "
              "Test split refused (pass --allow-test to override).")

    evaluate(
        ckpt_path=ns.checkpoint,
        n_scenes=ns.n_scenes,
        strides=ns.strides,
        thresholds=ns.thresholds,
        batch_size=ns.batch_size,
        output_dir=ns.output_dir,
        allow_test=ns.allow_test,
    )


if __name__ == "__main__":
    main()
