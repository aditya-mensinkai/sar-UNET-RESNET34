"""Tiny deterministic overfit test for UNet + ResNet34 (diagnostic only, no full training).

Subset: 16 train tiles (8 oil-containing + 8 background), seed-42 shuffled scan
of the tile index with windowed mask reads (no preprocessing for selection).
Uses OilSpillTileDataset + process_scene pipeline + BCEDiceLoss + Adam 1e-3
(diagnostic LR, NOT claimed optimal) + FP32, 10 epochs. No validation, no test.
Writes processed_dataset/reports/tiny_overfit_report.json (+ history csv).
Exit 0 = PASS, 1 = FAIL (STOP: do not start full training).
Usage: python tools/tiny_overfit_test.py [--epochs 10] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SEED = 42
N_OIL_TILES = 8
N_BG_TILES = 8
LR = 3e-4  # diagnostic value: 1e-3 learned noisily under batch-size-2 BatchNorm
            # noise; 3e-4 gives a cleaner curve. NOT a final training hyperparameter.


def fail(report_path, report, msg):
    report["result"] = "FAIL"
    report["failure_stage"] = msg
    json.dump(report, open(report_path, "w"), indent=1)
    print(f"TINY OVERFIT TEST: FAIL ({msg})", flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batchnorm-mode", choices=["train", "frozen_stats"], default="train")
    ns = ap.parse_args()

    import rasterio
    from rasterio.windows import Window

    from preprocessing.tile_dataset import OilSpillTileDataset
    from models import UNetResNet34, UNetResNet34Config
    from training.losses import BCEDiceLoss
    from training.metrics import compute_binary_segmentation_metrics

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TIP = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")
    REP = os.path.join(ROOT, "processed_dataset", "reports", "tiny_overfit_report.json")
    report = {"experiment": "tiny_overfit_unet_resnet34", "seed": ns.seed,
              "subset": {}, "model": {}, "loss": {}, "optimizer": {},
              "device": {}, "history": [], "diagnostics": {}, "result": "FAIL"}

    torch.manual_seed(ns.seed)
    np.random.seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        fail(REP, report, "CUDA")
    print(f"CUDA available: True\nCUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    ds = OilSpillTileDataset(TIP, split="train")
    # deterministic scan for 8 oil + 8 background tiles (windowed mask reads only)
    order = list(range(len(ds)))
    random.Random(ns.seed).shuffle(order)
    oil_idx, bg_idx = [], []
    for i in order:
        if len(oil_idx) >= N_OIL_TILES and len(bg_idx) >= N_BG_TILES:
            break
        r = ds.rows[i]
        with rasterio.open(os.path.join(ds.root, r["mask_path"])) as m:
            w = m.read(1, window=Window(int(r["x"]), int(r["y"]),
                                        int(r["tile_width"]), int(r["tile_height"])))
        if (w == 1).any():
            if len(oil_idx) < N_OIL_TILES:
                oil_idx.append(i)
        elif len(bg_idx) < N_BG_TILES:
            bg_idx.append(i)
    if len(oil_idx) < N_OIL_TILES or len(bg_idx) < N_BG_TILES:
        fail(REP, report, "DATASET")
    sub_idx = oil_idx + bg_idx
    sub = Subset(ds, sub_idx)
    # mask content audit on selected tiles
    oil_fracs = []
    for i in sub_idx:
        _, m = ds[i]
        oil_fracs.append(float((m.numpy() == 1).mean()))
    oil_fracs = np.array(oil_fracs)
    report["subset"] = {
        "seed": ns.seed, "split": "train", "n_tiles": len(sub_idx),
        "tile_ids": [ds.rows[i]["tile_id"] for i in sub_idx],
        "scene_ids": sorted(set(ds.rows[i]["image_id"] for i in sub_idx)),
        "positive_tiles": int((oil_fracs > 0).sum()),
        "empty_tiles": int((oil_fracs == 0).sum()),
        "oil_fraction_min": round(float(oil_fracs.min()), 5),
        "oil_fraction_max": round(float(oil_fracs.max()), 5),
        "oil_fraction_mean": round(float(oil_fracs.mean()), 5)}
    print(f"Subset: {len(sub_idx)} tiles ({len(oil_idx)} oil + {len(bg_idx)} bg), "
          f"oil_frac mean={oil_fracs.mean():.4f} "
          f"min={oil_fracs.min():.5f} max={oil_fracs.max():.4f}", flush=True)
    print("tile_id | image_id | class | oil_pixels | oil_fraction", flush=True)
    _cls = [ds.rows[i]["class"] for i in sub_idx]
    for i, f in zip(sub_idx, oil_fracs):
        r = ds.rows[i]
        print(f"{r['tile_id']} | {r['image_id']} | {r['class']} | "
              f"{int(f * 512 * 512)} | {f:.6f}", flush=True)
    print(f"positive={(oil_fracs > 0).sum()} zero={(oil_fracs == 0).sum()} "
          f"median={float(np.median(oil_fracs)):.6f}", flush=True)
    for _c in ("Oil", "No_oil", "Lookalike"):
        _sel = [f for f, cc in zip(oil_fracs, _cls) if cc == _c]
        if _sel:
            print(f"{_c}: n={len(_sel)} positive={sum(1 for f in _sel if f > 0)} "
                  f"zero={sum(1 for f in _sel if f == 0)}", flush=True)

    mcfg = UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False)
    model = UNetResNet34(mcfg).to(device)
    report["model"] = {"arch": "UNet+ResNet34", "in_channels": 2, "out_channels": 1,
                       "pretrained": False,
                       "params": sum(p.numel() for p in model.parameters())}
    crit = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
    report["loss"] = {"name": "BCEDiceLoss", "bce_weight": 0.5, "dice_weight": 0.5,
                      "smooth": 1.0, "threshold": 0.5}
    opt = torch.optim.Adam(model.parameters(), lr=ns.lr)
    report["optimizer"] = {"name": "Adam", "lr": ns.lr, "note": "diagnostic LR only"}
    report["device"] = {"cuda": True, "name": torch.cuda.get_device_name(0)}
    import torch.nn as _nn

    B = ns.batch_size

    def _apply_bn_mode(_model, _mode):
        # model.train() resets every module to training; re-freeze BN only.
        # BN.eval() => forward uses FIXED running stats, no running-stat updates;
        # affine weight/bias keep requires_grad and still update via optimizer.
        n = 0
        for _m in _model.modules():
            if isinstance(_m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)):
                if _mode == "frozen_stats":
                    _m.eval()
                n += 1
        return n

    print(f"BatchNorm diagnostic\n--------------------\nbatch_size: {B}\nseed: {ns.seed}\n"
          f"learning_rate: {ns.lr}\nepochs: {ns.epochs}\nbatchnorm_mode: {ns.batchnorm_mode}",
          flush=True)
    model.train()
    _nbn = _apply_bn_mode(model, ns.batchnorm_mode)
    _states = [(n, m.training, m.track_running_stats, m.affine,
                bool(m.weight.requires_grad), bool(m.bias.requires_grad))
               for n, m in model.named_modules()
               if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d))]
    _nonbn_train = all(m.training for m in model.modules()
                       if not isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)))
    print(f"number_of_BN_layers: {_nbn}\nBN training-state unique: {sorted(set(s[1] for s in _states))}\n"
          f"non-BN modules all in training mode: {_nonbn_train}", flush=True)
    print(f"BN sample states (name,training,track,affine,wgrad,bgrad): {_states[:3]}", flush=True)
    report["batchnorm"] = {"mode": ns.batchnorm_mode, "n_layers": _nbn,
                           "bn_training": sorted(set(s[1] for s in _states)),
                           "affine_trainable": all(s[4] and s[5] for s in _states)}
    loader = DataLoader(sub, batch_size=B, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(ns.seed), num_workers=0)
    print(f"batch_size={B} batches/epoch={len(loader)}", flush=True)
    print(f"Experiment: Learning-rate diagnostic\nbatch_size: {B}\nseed: {ns.seed}\n"
          f"learning_rate: {ns.lr}\nepochs: {ns.epochs}\nbatchnorm_mode: {ns.batchnorm_mode}\n"
          f"augmentation: OFF\noptimizer: Adam\nloss: BCEDiceLoss(0.5/0.5/1.0)", flush=True)
    print(f"GT positive fraction (subset): {round(float(oil_fracs.mean()), 6)}", flush=True)
    before = torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).clone()

    history = []
    for ep in range(ns.epochs):
        model.train()
        _apply_bn_mode(model, ns.batchnorm_mode)  # model.train() resets BN; re-freeze
        tot, acc = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}, 0.0
        nb = 0
        gmax, gmean, ngrad = 0.0, 0.0, 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if images.shape != (B, 2, 512, 512) or masks.shape != (B, 1, 512, 512):
                fail(REP, report, "TENSOR_SHAPE")
            opt.zero_grad()
            logits = model(images)
            if tuple(logits.shape) != (B, 1, 512, 512):
                fail(REP, report, "TENSOR_SHAPE")
            loss = crit(logits, masks)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                fail(REP, report, "LOSS")
            loss.backward()
            gs = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            if not gs:
                fail(REP, report, "GRADIENT")
            g = torch.cat(gs)
            if not torch.isfinite(g).all() or not (g.abs().sum() > 0):
                fail(REP, report, "GRADIENT")
            gmax = max(gmax, float(g.abs().max()))
            gmean += float(g.abs().mean())
            ngrad += 1
            opt.step()
            m = compute_binary_segmentation_metrics(logits.detach(), masks, threshold=0.5)
            for k in ("tp", "fp", "tn", "fn"):
                tot[k] += m[{"tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN"}[k]]
            acc += float(loss.detach())
            nb += 1
        n = tot["tp"] + tot["fp"] + tot["tn"] + tot["fn"]
        iou = tot["tp"] / (tot["tp"] + tot["fp"] + tot["fn"]) if n else 0.0
        dice = 2 * tot["tp"] / (2 * tot["tp"] + tot["fp"] + tot["fn"]) if n else 0.0
        # epoch-global predicted counts (accumulated over all batches, not last-batch only)
        ep_pos = tot["tp"] + tot["fp"]
        ep_neg = tot["tn"] + tot["fn"]
        with torch.no_grad():
            probs = torch.sigmoid(logits.detach())
            last_pos = int((probs >= 0.5).sum())
            last_neg = int((probs < 0.5).sum())
        row = {"epoch": ep + 1, "train_loss": round(acc / nb, 5),
               "train_dice": round(dice, 5), "train_iou": round(iou, 5), "lr": ns.lr,
               "logit_min": round(float(logits.detach().min()), 3),
            "logit_max": round(float(logits.detach().max()), 3),
               "pred_pos": ep_pos, "pred_neg": ep_neg,
               "last_batch_pos": last_pos, "last_batch_neg": last_neg}
        history.append(row)
        print(f"Epoch {ep + 1}/{ns.epochs}  loss={row['train_loss']} dice={row['train_dice']} "
              f"iou={row['train_iou']} lr={ns.lr}", flush=True)
    report["history"] = history
    _d = [r["train_dice"] for r in history]
    _best_i = int(np.argmax(_d))
    _best_j = int(np.argmax([r["train_iou"] for r in history]))
    report["diagnostics"] = {
        "best_dice": _d[_best_i], "best_dice_epoch": _best_i + 1,
        "best_iou": history[_best_j]["train_iou"], "best_iou_epoch": _best_j + 1,
        "mean_abs_dice_change": round(float(np.mean(np.abs(np.diff(_d)))), 5),
        "peak_alloc_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 1e6, 1),
        "nan_inf": "none observed", "grad_max": round(gmax, 6),
        "grad_mean": round(gmean / max(ngrad, 1), 8),
        "params_changed": None, "predictions_nonconstant": None,
        "loss_ratio": None}
    after = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    changed = bool((before != after).any())
    h0, h1 = history[0], history[-1]
    # PASS bar (diagnostic, documented): meaningful loss drop + Dice above 0.5
    # + non-constant predictions. No universal threshold claimed.
    learned = (h1["train_loss"] < 0.75 * h0["train_loss"] and h1["train_dice"] > 0.5
               and h1["pred_pos"] > 0 and h1["pred_neg"] > 0)
    report["diagnostics"].update({
        "params_changed": changed,
        "predictions_nonconstant": h1["pred_pos"] > 0 and h1["pred_neg"] > 0,
        "loss_ratio": round(h1["train_loss"] / h0["train_loss"], 4)})
    report["result"] = "PASS" if (learned and changed) else "FAIL"
    if report["result"] == "FAIL":
        report["failure_stage"] = "LOSS"
    json.dump(report, open(REP, "w"), indent=1)
    with open(REP.replace(".json", "_history.csv"), "w") as f:
        f.write("epoch,train_loss,train_dice,train_iou,lr\n")
        for r in history:
            f.write(f"{r['epoch']},{r['train_loss']},{r['train_dice']},{r['train_iou']},{r['lr']}\n")
    print(f"TINY OVERFIT TEST: {report['result']}", flush=True)
    sys.exit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
