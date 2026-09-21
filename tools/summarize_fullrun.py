"""Batch summary + validation for the full test-set pipeline run.

Reads ONLY outputs under results/full_run/ plus source scenes for geo
comparison. Never touches ground-truth masks except via Get-FileHash outside.
Writes results/full_run/batch_summary.json.

Usage:
    python tools/summarize_fullrun.py
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

_ROOT = Path(__file__).resolve().parent.parent
INF = _ROOT / "results" / "full_run" / "inference"
PP = _ROOT / "results" / "full_run" / "postprocessing"
SRC = _ROOT / "processed_dataset" / "test" / "images"
CLASSES = ["Oil", "Lookalike", "No_oil"]


def shp_shape_type(shp_path: Path):
    with open(shp_path, "rb") as f:
        hdr = f.read(100)
    if len(hdr) < 100:
        return None
    code, _, _, _, _, _, _ = struct.unpack(">7i", hdr[:28])
    (stype,) = struct.unpack("<i", hdr[32:36])
    return (code, stype)  # expect (9994, 5=polygon)


def main():
    rows = []
    t0 = time.time()
    for cls in CLASSES:
        for img in sorted((SRC / cls).glob("*.tif")):
            stem = img.stem
            sid = f"{cls}_{stem}"
            prob_p = INF / cls / f"{stem}_prob.tif"
            meta_p = INF / cls / f"{stem}_inference.json"
            d = PP / sid
            row = {"scene_id": sid, "class": cls, "status": "ok",
                   "threshold": 0.5, "issues": []}
            try:
                src = rasterio.open(img)
                pr = rasterio.open(prob_p)
                a = pr.read(1)
                row["prob_shape"] = [pr.width, pr.height]
                if (pr.width, pr.height) != (src.width, src.height):
                    row["issues"].append("prob dims != source")
                if pr.crs != src.crs:
                    row["issues"].append(f"crs {pr.crs} != {src.crs}")
                if pr.transform != src.transform:
                    row["issues"].append("transform != source")
                if not bool(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()):
                    row["issues"].append("prob values outside [0,1] or non-finite")
                row["prob_min"] = round(float(a.min()), 6)
                row["prob_max"] = round(float(a.max()), 6)
                meta = json.load(open(meta_p))
                row["inference_time_s"] = meta["timing"]["total_s"]
                row["stride"] = meta["stride"]

                for kind in ("binary_mask", "cleaned_mask"):
                    mp = d / f"{stem}_{kind}.tif"
                    m = rasterio.open(mp)
                    if m.crs != src.crs or m.transform != src.transform:
                        row["issues"].append(f"{kind} geo != source")

                g = json.load(open(d / f"{stem}_predictions.geojson"))
                feats = g.get("features", [])
                shp = shp_shape_type(d / f"{stem}_predictions.shp")
                for ext in ("shp", "shx", "dbf", "prj"):
                    p = d / f"{stem}_predictions.{ext}"
                    if not p.exists() or p.stat().st_size == 0:
                        row["issues"].append(f"missing/empty .{ext}")
                if shp != (9994, 5) and feats:
                    row["issues"].append(f"shp header {shp} != (9994, polygon)")
                b = src.bounds
                for f in feats:
                    for ring in f["geometry"]["coordinates"]:
                        c = np.asarray(ring)
                        if not (b.left <= c[:, 0].min() and c[:, 0].max() <= b.right
                                and b.bottom <= c[:, 1].min() and c[:, 1].max() <= b.top):
                            row["issues"].append("polygon outside source bounds")
                            break

                rep = json.load(open(d / f"{stem}_postprocessing_report.json"))
                row["number_of_regions"] = rep["number_of_predicted_regions"]
                row["total_predicted_area_km2"] = rep["total_predicted_area_km2"]
                row["postprocess_time_s"] = rep["processing_time_seconds"]
                n, ar = rep["number_of_predicted_regions"], rep["total_predicted_area_km2"]
                if (n == 0) != (ar == 0.0) or (n > 0 and not ar > 0):
                    row["issues"].append(f"area/region mismatch n={n} area={ar}")
                if len(feats) != n:
                    row["issues"].append(f"geojson {len(feats)} != report {n}")
                row["probability_path"] = str(prob_p.relative_to(_ROOT)).replace("\\", "/")
                row["binary_mask_path"] = str((d / f"{stem}_binary_mask.tif").relative_to(_ROOT)).replace("\\", "/")
                row["cleaned_mask_path"] = str((d / f"{stem}_cleaned_mask.tif").relative_to(_ROOT)).replace("\\", "/")
                row["geojson_path"] = str((d / f"{stem}_predictions.geojson").relative_to(_ROOT)).replace("\\", "/")
                row["shp_path"] = str((d / f"{stem}_predictions.shp").relative_to(_ROOT)).replace("\\", "/")
                row["report_path"] = str((d / f"{stem}_postprocessing_report.json").relative_to(_ROOT)).replace("\\", "/")
                src.close(); pr.close()
            except Exception as e:  # noqa: BLE001 — per-scene isolation, recorded
                row["status"] = f"FAILED: {type(e).__name__}: {e}"
                row["issues"].append("exception during validation")
            rows.append(row)

    ok = [r for r in rows if r["status"] == "ok" and not r["issues"]]
    bad = [r for r in rows if r["status"] != "ok" or r["issues"]]
    tot_reg = sum(r.get("number_of_regions", 0) for r in rows if r.get("number_of_regions") is not None)
    tot_area = sum(r.get("total_predicted_area_km2", 0) or 0 for r in rows)
    it = [r["inference_time_s"] for r in rows if "inference_time_s" in r]
    pt = [r["postprocess_time_s"] for r in rows if "postprocess_time_s" in r]
    summary = {
        "total_scenes": len(rows),
        "successful_scenes": len(ok),
        "failed_scenes": len(bad),
        "total_predicted_regions": tot_reg,
        "total_predicted_area_km2": round(tot_area, 6),
        "inference_time_s": {"mean": round(float(np.mean(it)), 3),
                             "min": round(float(np.min(it)), 3),
                             "max": round(float(np.max(it)), 3)},
        "postprocess_time_s": {"mean": round(float(np.mean(pt)), 3),
                               "min": round(float(np.min(pt)), 3),
                               "max": round(float(np.max(pt)), 3)},
        "validation_seconds": round(time.time() - t0, 1),
        "scenes": rows,
    }
    with open(_ROOT / "results" / "full_run" / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"scenes={len(rows)} ok={len(ok)} bad={len(bad)} "
          f"regions={tot_reg} area={tot_area:.4f}km2")
    for r in bad[:20]:
        print(f"  BAD {r['scene_id']}: {r['status']} {r['issues']}")


if __name__ == "__main__":
    main()
