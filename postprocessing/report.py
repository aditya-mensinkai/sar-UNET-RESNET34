"""Post-processing report generation.

Generates a machine-readable JSON report for every processed scene.
All values are calculated from actual processing — no fabricated values.

Report schema::

    {
        "scene_id": str,
        "prob_path": str,
        "threshold": float,
        "morphology": {
            "enabled": bool,
            "opening_kernel_size": int,
            "closing_kernel_size": int
        },
        "min_region_area_pixels": int,
        "number_of_predicted_regions": int,
        "total_predicted_area_km2": float,
        "total_predicted_pixels": int,
        "processing_time_seconds": float,
        "crs": str | null,
        "has_georeferencing": bool,
        "probability_map": {
            "min": float, "max": float, "mean": float
        },
        "outputs": {
            "probability_map": str,
            "binary_mask": str,
            "cleaned_mask": str,
            "geojson": str | null,
            "shapefile": str | null
        },
        "candidates": [
            {
                "candidate_id": str,
                "pixel_area": int,
                "geographic_area_km2": float | null,
                "centroid_pixel": [row, col],
                "centroid_geo": [lon, lat] | null
            },
            ...
        ]
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_report(
    scene_id: str,
    prob_path: str,
    binary_mask_path: str,
    cleaned_mask_path: str,
    geojson_path: Optional[str],
    shp_path: Optional[str],
    report_path: str,
    threshold: float,
    morphology_enabled: bool,
    opening_kernel_size: int,
    closing_kernel_size: int,
    min_region_area_pixels: int,
    regions: List[Dict],
    prob_stats: Dict[str, float],
    processing_time_seconds: float,
    crs_str: Optional[str],
    has_georeferencing: bool,
) -> str:
    """Write a JSON post-processing report and return its path.

    All numeric values in the report are derived from actual processing.
    """
    total_predicted_pixels = sum(r["pixel_area"] for r in regions)
    area_km2_list = [
        r["geographic_area_km2"]
        for r in regions
        if r.get("geographic_area_km2") is not None
    ]
    total_predicted_area_km2 = (
        round(sum(area_km2_list), 6) if area_km2_list else 0.0
    )

    candidates = [
        {
            "candidate_id": r["candidate_id"],
            "pixel_area": r["pixel_area"],
            "geographic_area_km2": r.get("geographic_area_km2"),
            "centroid_pixel": r.get("centroid_pixel"),
            "centroid_geo": r.get("centroid_geo"),
        }
        for r in regions
    ]

    report: Dict[str, Any] = {
        "scene_id": scene_id,
        "prob_path": str(prob_path),
        "threshold": threshold,
        "morphology": {
            "enabled": morphology_enabled,
            "opening_kernel_size": opening_kernel_size,
            "closing_kernel_size": closing_kernel_size,
        },
        "min_region_area_pixels": min_region_area_pixels,
        "number_of_predicted_regions": len(regions),
        "total_predicted_area_km2": total_predicted_area_km2,
        "total_predicted_pixels": total_predicted_pixels,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "crs": crs_str,
        "has_georeferencing": has_georeferencing,
        "probability_map": {
            "min": round(float(prob_stats.get("min", 0.0)), 6),
            "max": round(float(prob_stats.get("max", 0.0)), 6),
            "mean": round(float(prob_stats.get("mean", 0.0)), 6),
        },
        "outputs": {
            "probability_map": str(prob_path),
            "binary_mask": str(binary_mask_path),
            "cleaned_mask": str(cleaned_mask_path),
            "geojson": str(geojson_path) if geojson_path else None,
            "shapefile": str(shp_path) if shp_path else None,
        },
        "candidates": candidates,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path
