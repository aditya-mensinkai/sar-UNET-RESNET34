"""Post-processing configuration (single source of truth).

All post-processing parameters live here.  Call ``validate(cfg)`` before use
to check that values are internally consistent.  Use ``with_overrides(**kw)``
to create a validated copy with individual keys changed.

Parameter names follow the same snake_case convention used by
preprocessing/__init__.py (DEFAULTS dict + validate + with_overrides).
"""
from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    # ---- Thresholding -------------------------------------------------------
    # Applied to the model's full-scene probability map (float32 in [0, 1]).
    # Never applied to ground-truth masks.
    "threshold": 0.5,

    # ---- Morphological cleanup ----------------------------------------------
    # Applied AFTER thresholding, BEFORE connected-component extraction.
    # Uses pure-NumPy binary operations (no scipy dependency).
    "morphology_enabled": True,
    # opening_kernel_size: structuring element side (pixels, must be odd >= 1)
    # Opening = erosion followed by dilation; removes small bright noise.
    "opening_kernel_size": 3,
    # closing_kernel_size: structuring element side (pixels, must be odd >= 1)
    # Closing = dilation followed by erosion; fills small dark holes.
    "closing_kernel_size": 3,

    # ---- Region filtering ---------------------------------------------------
    # Minimum connected-component area in pixels.  Regions smaller than this
    # are removed AFTER morphology and before polygon extraction.
    # Value 0 means no filtering.
    "min_region_area_pixels": 100,

    # ---- Output formats -----------------------------------------------------
    "output_geojson": True,
    "output_shp": True,

    # ---- Naming convention --------------------------------------------------
    # Matches the {stem}_prob.tif / {stem}_mask.tif convention used by
    # predict_scene.py.
    "prob_suffix": "_prob.tif",
    "binary_mask_suffix": "_binary_mask.tif",
    "cleaned_mask_suffix": "_cleaned_mask.tif",
    "geojson_suffix": "_predictions.geojson",
    "shp_suffix": "_predictions.shp",
    "report_suffix": "_postprocessing_report.json",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dict. Returns a copy; raises ValueError on error."""
    cfg = dict(cfg)
    errors = []

    thr = cfg.get("threshold")
    if not isinstance(thr, (int, float)) or not (0.0 < thr < 1.0):
        errors.append(
            f"threshold must be a float in (0, 1), got {thr!r}"
        )

    for k in ("opening_kernel_size", "closing_kernel_size"):
        v = cfg.get(k)
        if not isinstance(v, int) or v < 1 or v % 2 == 0:
            errors.append(
                f"{k} must be a positive odd integer, got {v!r}"
            )

    min_area = cfg.get("min_region_area_pixels")
    if not isinstance(min_area, int) or min_area < 0:
        errors.append(
            f"min_region_area_pixels must be a non-negative integer, got {min_area!r}"
        )

    if errors:
        raise ValueError(
            "postprocessing config invalid: " + "; ".join(errors)
        )
    return cfg


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def with_overrides(**kw: Any) -> Dict[str, Any]:
    """Return a validated configuration dict with the specified keys overridden."""
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    return validate(cfg)
