"""Materialize 256px tile pairs from the authoritative scene split (hard links stay sources).

Reads split_index.csv (image-level, seed-42, frozen) + NORMALIZATION_BOUNDS from
preprocessing_config.json. Writes tiled/{split}/images|masks/*.tif (new derived
GeoTIFFs), tile_index.csv, quarantine/tile_quarantine.csv, reports/tiles_balance_report.txt.
Re-runnable: existing tiles are verified, never rewritten. No model code touched.
Usage: python tools/build_tiles.py [--stride 256] [--limit-scenes N] [--splits train val test]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import with_overrides
from preprocessing.make_tiles import TILE_FIELDS, build_tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="processed_dataset")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--classes", nargs="+", default=["Oil", "No_oil", "Lookalike"])
    a = ap.parse_args()
    root = os.path.abspath(a.dataset)
    cfg = with_overrides(**({"TILE_STRIDE": a.stride} if a.stride else {}))
    cfgj = json.load(open(os.path.join(root, "reports", "preprocessing_config.json")))
    fb = cfgj.get("NORMALIZATION_BOUNDS")
    if not fb:
        print("FATAL: preprocessing_config.json lacks NORMALIZATION_BOUNDS; STOPPING")
        return 1
    if cfg["TILE_SIZE"] != 256:
        print(f"FATAL: TILE_SIZE must be 256, got {cfg['TILE_SIZE']}")
        return 1
    # disk guard: ~600KB per tile pair worst case
    import csv as _csv
    all_rows = [r for r in _csv.DictReader(
        open(os.path.join(root, "metadata", "split_index.csv")))
        if r["split"] in a.splits and r["excluded"] == "NO"]
    if a.limit_scenes is not None:
        from collections import defaultdict as _dd
        per = _dd(list)
        for r in all_rows:
            per[(r["split"], r["class"])].append(r)
        n_scenes = sum(min(len(v), a.limit_scenes) for v in per.values())
    else:
        n_scenes = len(all_rows)
    per_scene = ((2048 // cfg["TILE_SIZE"]) + 1) ** 2  # stride-256 upper bound incl. edge row
    have = 0
    for _sp in a.splits:
        _d = os.path.join(root, "tiled", _sp, "images")
        if os.path.isdir(_d):
            have += sum(1 for _f in os.listdir(_d) if _f.endswith(".tif"))
    need = max(0, n_scenes * per_scene - have) * 600 * 1024
    free = shutil.disk_usage(root).free
    print(f"[disk] scenes={n_scenes} est_tiles~{n_scenes * per_scene} "
          f"need~{need / 1e9:.1f}GB free={free / 1e9:.1f}GB", flush=True)
    if need > free:
        print("FATAL: insufficient disk for requested stride; reduce scope or free space. STOPPING.")
        return 1
    rows, quar, stats = build_tiles(os.path.join(root, "metadata", "split_index.csv"),
                                    root, cfg, fb, splits=tuple(a.splits),
                                    limit_scenes=a.limit_scenes, classes=tuple(a.classes))
    tip = os.path.join(root, "metadata", "tile_index.csv")
    if os.path.exists(tip) and os.path.getsize(tip) > 0:
        merged = {r["tile_id"]: r for r in csv.DictReader(open(tip))}
        merged.update({r["tile_id"]: r for r in rows})
        rows = [merged[k] for k in sorted(merged)]
    with open(tip, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TILE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tqp = os.path.join(root, "metadata", "quarantine", "tile_quarantine.csv")
    if os.path.exists(tqp) and os.path.getsize(tqp) > 0:
        seen = {(r["tile_id"], r["reason"]) for r in csv.DictReader(open(tqp))}
        quar = [q for q in quar if (q["tile_id"], q["reason"]) not in seen] + \
               [r for r in csv.DictReader(open(tqp))]
    with open(tqp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_id", "scene_id", "split", "reason", "source_image"])
        w.writeheader()
        w.writerows(quar)
    rep = (f"TILE BALANCE REPORT (size={cfg['TILE_SIZE']} stride={cfg['TILE_STRIDE']} "
           f"overlap={cfg['TILE_SIZE'] - cfg['TILE_STRIDE']})\n"
           f"scenes={stats['scenes']} kept_tiles={stats['kept']} oil_tiles={stats['oil_tiles']} "
           f"bg_tiles={stats['bg_tiles']} filtered_bg={stats['filtered_bg']} "
           f"rejected={stats['rejected']} oil_px_total={stats['oil_px']}\n"
           f"filtering: MIN_OIL_FRACTION={cfg['MIN_OIL_FRACTION']} "
           f"BG_KEEP_FRACTION={cfg['BG_KEEP_FRACTION']} BG_SEED={cfg.get('BG_SEED')}\n"
           f"augmentation: dynamic train-only (flips/rot90 exact, see preprocessing_config.json); "
           f"materialized tiles are UNAUGMENTED.\n")
    open(os.path.join(root, "reports", "tiles_balance_report.txt"), "w").write(rep)
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
