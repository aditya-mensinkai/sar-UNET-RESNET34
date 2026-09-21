"""Regression test: empty prediction must report total_predicted_area_km2 == 0.0.

Locks in the contract that an empty prediction (no regions) yields the float
0.0 for total_predicted_area_km2 — never None — so downstream consumers can
rely on a numeric total unconditionally.

Usage:
    python tools/test_postprocessing_report.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from postprocessing.report import generate_report


def check(n, name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {n}: {name}" + (f" — {detail}" if detail and not cond else ""))
    return 0 if cond else 1


def main():
    failures = 0
    tmp = tempfile.mkdtemp(prefix="test_pp_report_")
    report_path = os.path.join(tmp, "empty_postprocessing_report.json")

    generate_report(
        scene_id="Lookalike/00001.tif",
        prob_path=os.path.join(tmp, "prob.tif"),
        binary_mask_path=os.path.join(tmp, "mask.tif"),
        cleaned_mask_path=os.path.join(tmp, "cleaned.tif"),
        geojson_path=None,
        shp_path=None,
        report_path=report_path,
        threshold=0.80,
        morphology_enabled=False,
        opening_kernel_size=0,
        closing_kernel_size=0,
        min_region_area_pixels=0,
        regions=[],  # empty prediction: no regions whatsoever
        prob_stats={"min": 0.0, "max": 0.01, "mean": 0.001},
        processing_time_seconds=0.5,
        crs_str=None,
        has_georeferencing=False,
    )
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)

    failures += check(1, "report written with zero regions",
                      rep["number_of_predicted_regions"] == 0,
                      f"got {rep['number_of_predicted_regions']}")
    failures += check(2, "total_predicted_pixels == 0",
                      rep["total_predicted_pixels"] == 0,
                      f"got {rep['total_predicted_pixels']!r}")
    failures += check(3, "total_predicted_area_km2 == 0.0 (never None)",
                      rep["total_predicted_area_km2"] == 0.0
                      and rep["total_predicted_area_km2"] is not None
                      and isinstance(rep["total_predicted_area_km2"], float),
                      f"got {rep['total_predicted_area_km2']!r}")

    print(f"[test_postprocessing_report] {'PASS' if failures == 0 else 'FAIL'} "
          f"({3 - failures}/3 checks)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
