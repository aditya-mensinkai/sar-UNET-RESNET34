"""Val-only diagnosis at the candidate config (stride 384, threshold 0.80),
plus a train-split tile-level gap check from the tile cache.

CONSTRAINTS
-----------
* VAL scenes only for the gallery / false-alarm stats; TRAIN cache tiles only
  for the train-vs-val gap. The test split is never referenced (no --allow-test
  flag exists here). No training: the model runs under torch.inference_mode().
* No protected file is modified and no inference-CLI default is changed:
  inference/, preprocessing/, training/ and models/ are only *imported*.
  Outputs go exclusively to results/val_analysis/gallery/ and
  results/val_analysis/diagnosis.json.
* Every reported number is MEASURED on this run.

Usage:
    python tools/val_diagnosis.py                                   # full run
    python tools/val_diagnosis.py --gallery-limit 1 --max-train-tiles 200 --out-json diagnosis_smoke.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.val_analysis import (
    STRIDES, THRESHOLDS, _dice, connected_component_sizes,
)

_CKPT = _ROOT / "checkpoints" / "best_model.pth"
_TILE_IDX = _ROOT / "processed_dataset" / "metadata" / "tile_index.csv"
_PREP_CFG = _ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
_RECORDS = _ROOT / "results" / "val_analysis" / "records"
_GALLERY = _ROOT / "results" / "val_analysis" / "gallery"
_CACHE = _ROOT / "processed_dataset" / "tile_cache"

STRIDE = 384
THR = 0.80
SI = STRIDES.index(STRIDE)
TI = THRESHOLDS.index(round(THR, 2))

SIZE_BINS = [(1, 24), (25, 99), (100, 399), (400, 1599), (1600, None)]


# ── val records (histograms not needed; tf grid + meta only) ──────────────

def load_val_scenes():
    scenes = []
    for p in sorted(_RECORDS.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        tf = z["tf"]  # (stride, threshold, 3)
        tp, fp, fn = (int(v) for v in tf[SI, TI])
        scenes.append({
            "sid": meta["sid"], "category": meta["category"],
            "gt_total": int(meta["gt_total"]),
            "gt_positive": int(meta["gt_positive"]),
            "tp": tp, "fp": fp, "fn": fn,
            "dice": _dice(tp, fp, fn),
            "pred": tp + fp,
        })
    return scenes


def select(scenes):
    look = sorted([s for s in scenes if s["category"] == "Lookalike"],
                  key=lambda s: -s["fp"])[:12]
    big_oil = sorted([s for s in scenes if s["category"] == "Oil"
                      and s["gt_positive"] > 262144],
                     key=lambda s: s["dice"])[:8]
    small_oil = sorted([s for s in scenes if s["category"] == "Oil"
                        and s["gt_positive"] < 50027],
                       key=lambda s: s["dice"])[:8]
    return look, big_oil, small_oil


def _val_paths():
    paths = {}
    with open(_TILE_IDX) as f:
        for r in csv.DictReader(f):
            if r["split"] == "val" and r["image_id"] not in paths:
                paths[r["image_id"]] = (
                    str(_ROOT / "processed_dataset" / r["image_path"]),
                    str(_ROOT / "processed_dataset" / r["mask_path"]))
    return paths


# ── single-scene inference (val only; maps are transient per scene) ───────

def infer_scene(model, bounds, cfg, ipath, mpath, batch_size, device, amp):
    import torch
    from preprocessing.pipeline import process_scene
    from inference.windows import tile_grid, TILE_SIZE
    from inference.stitching import ProbabilityStitcher
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = process_scene(ipath, mpath, cfg=cfg, bounds=bounds, split="val",
                            augment_override=False, compute_stats=False)
    norm = out["normalized"]
    gt = (out["mask"] > 0)
    db = out["db_filtered"]  # post-Lee dB, (2, H, W) — display bands
    H, W = norm.shape[1], norm.shape[2]
    origins = tile_grid(H, W, tile=TILE_SIZE, stride=STRIDE)
    stitcher = ProbabilityStitcher(H, W, TILE_SIZE)
    with torch.inference_mode():
        buf_o, buf_p = [], []
        for (y, x) in origins:
            buf_o.append((y, x))
            buf_p.append(norm[:, y:y + TILE_SIZE, x:x + TILE_SIZE])
            if len(buf_p) == batch_size:
                t = torch.from_numpy(np.stack(buf_p).astype(np.float32)).to(device)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(t)
                pr = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()
                for (yy, xx), p in zip(buf_o, pr):
                    stitcher.add_tile(yy, xx, p)
                buf_o, buf_p = [], []
        if buf_p:
            t = torch.from_numpy(np.stack(buf_p).astype(np.float32)).to(device)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(t)
            pr = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()
            for (yy, xx), p in zip(buf_o, pr):
                stitcher.add_tile(yy, xx, p)
    prob = stitcher.finalize()
    return db, gt, prob


def _stretch(band: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(band, (1, 99))
    return np.clip((band - lo) / (hi - lo + 1e-12), 0, 1)


def gallery_png(sid, cat, db, gt, prob, dice, fp, gt_pos, out_path):
    b0, b1 = _stretch(db[0]), _stretch(db[1])
    binary = prob >= THR
    rgb = np.stack([b0, b0, b0], axis=-1)
    rgb[gt] = [1.0, 0.15, 0.15]  # GT overlay in red
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))
    ax[0].imshow(b0, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("band 0 (P1-P99 stretch)")
    ax[1].imshow(b1, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("band 1 (P1-P99 stretch)")
    ax[2].imshow(rgb)
    ax[2].set_title(f"ground truth overlay (GT px={gt_pos:,})")
    im = ax[3].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    if binary.any():
        ax[3].contour(binary, levels=[0.5], colors="white", linewidths=0.8)
    ax[3].set_title(f"prob + mask@{THR:.2f} contour (FP px={fp:,})")
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="probability")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{sid}  [{cat}]  stride={STRIDE} thr={THR:.2f}  "
                 f"Dice={dice:.4f} (MEASURED)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def fa_stats(prob, gt):
    binary = prob >= THR
    fa = binary & ~gt
    fa_px = int(fa.sum())
    sizes = connected_component_sizes(fa, connectivity=8)
    hist = {}
    for lo, hi in SIZE_BINS:
        key = f"{lo}-{hi if hi else 'inf'}"
        hist[key] = int(((sizes >= lo) & (sizes <= hi if hi else sizes)).sum()) \
            if sizes.size else 0
    return {
        "fa_px": fa_px,
        "mean_prob_in_fa": round(float(prob[fa].mean()), 5) if fa_px else None,
        "n_components": int(sizes.size),
        "component_size_hist": hist,
        "largest_component_px": int(sizes.max()) if sizes.size else 0,
    }


# ── train cache, tile level ───────────────────────────────────────────────

def train_cache_rows():
    man = json.load(open(_CACHE / "manifest_train_val.json"))
    order = np.asarray(man["row_order"])
    assert len(order) == man["n_tiles"], "row_order length mismatch"
    rows = list(csv.DictReader(open(_TILE_IDX)))
    # row_order indexes into the train+val subset of tile_index, in file order
    tv = [r for r in rows if r["split"] in ("train", "val")]
    assert len(tv) == man["n_tiles"], \
        f"train+val rows {len(tv)} != cache n_tiles {man['n_tiles']}"
    if isinstance(order[0], dict):
        # entries like {"tile_id": ..., "split": ..., "row_idx": ...}:
        # row_idx indexes the train+val subset of tile_index in file order
        mapped = []
        for e in order:
            r = tv[int(e["row_idx"])]
            assert r["tile_id"] == e["tile_id"] and r["split"] == e["split"], \
                f"cache row mismatch: {e} vs {r['tile_id']}/{r['split']}"
            mapped.append(r)
    elif order.dtype.kind in "iu":
        mapped = [tv[int(i)] for i in order]
    else:  # tile_id strings
        by_id = {r["tile_id"]: r for r in tv}
        mapped = [by_id[str(i)] for i in order]
    assert mapped[0]["split"] in ("train", "val")
    return man, mapped


def run_train_tiles(model, device, amp, batch_size, max_tiles, split="train"):
    import torch
    man, mapped = train_cache_rows()
    img = np.load(str(_CACHE / man["img_file"]), mmap_mode="r")
    msk = np.load(str(_CACHE / man["msk_file"]), mmap_mode="r")
    assert img.shape[1:] == (2, 512, 512), img.shape
    train_idx = [i for i, r in enumerate(mapped) if r["split"] == split]
    total_split = len(train_idx)
    if max_tiles:
        train_idx = train_idx[:max_tiles]
    agg = {}  # class -> {n, alarm, tp, fp, fn, empty}
    t0 = time.time()
    with torch.inference_mode():
        for k, i in enumerate(train_idx):
            r = mapped[i]
            g = msk[i] > 0
            t = torch.from_numpy(np.asarray(img[i]).astype(np.float32)[None]).to(device)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(t)
            p = torch.sigmoid(logits).float().cpu().numpy()[0, 0] >= THR
            tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
            a = agg.setdefault(r["class"], {"n": 0, "alarm": 0, "tp": 0, "fp": 0,
                                            "fn": 0, "empty": 0})
            a["n"] += 1; a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
            if tp + fp >= 1:
                a["alarm"] += 1
            if not g.any():
                a["empty"] += 1
            if (k + 1) % 500 == 0 or (k + 1) == len(train_idx):
                el = time.time() - t0
                print(f"  {split} tiles {k + 1}/{len(train_idx)}  "
                      f"{(k + 1) / el:.1f} tiles/s  ETA {el / (k + 1) * (len(train_idx) - k - 1) / 60:.1f}min",
                      flush=True)
    out = {}
    for c, a in agg.items():
        out[c] = {
            "n_tiles": a["n"], "tiles_with_pred_ge1": a["alarm"],
            "alarm_rate": round(a["alarm"] / a["n"], 4),
            "pooled_recall": round(a["tp"] / (a["tp"] + a["fn"]), 5) if (a["tp"] + a["fn"]) else None,
            "pooled_precision": round(a["tp"] / (a["tp"] + a["fp"]), 5) if (a["tp"] + a["fp"]) else None,
            "empty_mask_tiles": a["empty"],
            "empty_mask_fraction": round(a["empty"] / a["n"], 4),
            "tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
        }
    return out, len(train_idx)


# ── main ──────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Val-only diagnosis at 384/0.80 + train tile gap.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gallery-limit", type=int, default=None)
    ap.add_argument("--max-train-tiles", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-gallery", action="store_true")
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="cache split for the tile-level gap check")
    ap.add_argument("--out-json", default="diagnosis.json")
    ns = ap.parse_args(argv)

    import torch
    from models import UNetResNet34, UNetResNet34Config
    from preprocessing import with_overrides

    device = torch.device(ns.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    ck = torch.load(str(_CKPT), map_location=device, weights_only=False)
    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device); model.eval()
    pcfg = json.load(open(_PREP_CFG))
    bounds = {"bands": pcfg["statistics"]["bands"]}
    cfg = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5)

    scenes = load_val_scenes()
    assert all(s["sid"] for s in scenes) and len(scenes) == 513, len(scenes)
    look12, big8, small8 = select(scenes)
    print(f"[diag] selected: {len(look12)} Lookalike (top FP), {len(big8)} big-Oil "
          f"(lowest Dice), {len(small8)} small-Oil (lowest Dice) @384/0.80")

    _GALLERY.mkdir(parents=True, exist_ok=True)
    paths = _val_paths()
    gallery = []
    fa12 = []
    if ns.skip_gallery:
        print("[diag] gallery skipped (--skip-gallery)")
    gallery_targets = [] if ns.skip_gallery else [("lookalike", look12),
                                                   ("big_oil", big8),
                                                   ("small_oil", small8)]
    for group, members in gallery_targets:
        for s in members[:ns.gallery_limit] if ns.gallery_limit else members:
            ipath, mpath = paths[s["sid"]]
            db, gt, prob = infer_scene(model, bounds, cfg, ipath, mpath,
                                       ns.batch_size, device, amp)
            tp = int(((prob >= THR) & gt).sum())
            fp = int(((prob >= THR) & ~gt).sum())
            fn = int(s["gt_positive"] - tp)
            dice = _dice(tp, fp, fn)
            assert (tp, fp, fn) == (s["tp"], s["fp"], s["fn"]), \
                f"stitch mismatch on {s['sid']}"
            name = f"{group}_{s['sid'].replace('/', '__')}.png"
            gallery_png(s["sid"], s["category"], db, gt, prob, dice, fp,
                        s["gt_positive"], _GALLERY / name)
            entry = {"sid": s["sid"], "group": group, "png": f"gallery/{name}",
                     "gt_positive": s["gt_positive"], "dice_384_080": round(dice, 4),
                     "fp_px_384_080": fp, "measured": True}
            gallery.append(entry)
            if group == "lookalike":
                st = fa_stats(prob, gt)
                fa12.append({"sid": s["sid"], **st, "measured": True})
                print(f"  {s['sid']}: FP={fp:,} meanFAprob={st['mean_prob_in_fa']} "
                      f"comps={st['n_components']} largest={st['largest_component_px']:,}")
            del db, gt, prob

    result = {"config": {"stride": STRIDE, "threshold": THR}, "measured": True,
              "gallery": gallery, "fa_regions_lookalike12": fa12}

    if not ns.skip_train:
        print(f"[diag] {ns.split} cache tiles @0.80 …")
        train_metrics, n_done = run_train_tiles(model, device, amp,
                                                ns.train_batch_size, ns.max_train_tiles,
                                                ns.split)
        result[f"{ns.split}_tile_metrics"] = train_metrics
        result["tile_metrics_split"] = ns.split
        result["train_tiles_done"] = n_done
        for c, m in train_metrics.items():
            print(f"  {c}: n={m['n_tiles']} alarm={m['alarm_rate']} "
                  f"recall={m['pooled_recall']} prec={m['pooled_precision']} "
                  f"empty={m['empty_mask_fraction']}")

    with open(_ROOT / "results" / "val_analysis" / ns.out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[diag] gallery -> {_GALLERY} ({len(gallery)} PNGs)")
    print(f"[diag] json -> results/val_analysis/{ns.out_json}")


if __name__ == "__main__":
    main()
