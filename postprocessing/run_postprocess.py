"""Main post-processing pipeline: probability map -> polygons -> GeoJSON/SHP/report.

Integration point:
  predict_scene.py already saves {stem}_prob.tif (float32, georeferenced) and
  {stem}_mask.tif (uint8, binary) after full-scene inference.  This module
  reads the probability map (NOT the binary mask, NOT the ground truth) and
  applies the full post-processing chain from scratch.

Pipeline:
  1. Load {stem}_prob.tif — extracts CRS + transform from the file itself
     (original scene georeferencing is already embedded by predict_scene)
  2. Validate probability values (must be float32 in [0, 1])
  3. Threshold -> binary mask (saved as {stem}_binary_mask.tif)
  4. Morphological cleanup (opening + closing, pure-NumPy)
  5. Save cleaned mask (saved as {stem}_cleaned_mask.tif)
  6. Extract connected-component regions using rasterio.features.shapes
  7. Filter by minimum area
  8. Write GeoJSON ({stem}_predictions.geojson)
  9. Write SHP ({stem}_predictions.shp + .shx + .dbf + .prj)
  10. Write post-processing report ({stem}_postprocessing_report.json)

Ground-truth masking never occurs here.  This pipeline operates exclusively
on model output (probability map).

CLI::

    python -m postprocessing.run_postprocess \\
        --prob  results/my_inference/00092_prob.tif \\
        --output-dir results/postprocessed/00092 \\
        --threshold 0.5 \\
        --opening-kernel 3 \\
        --closing-kernel 3 \\
        --min-region-area 100

All parameters default to the values in postprocessing/config.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from affine import Affine

# Repository root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from postprocessing.config import DEFAULTS, validate, with_overrides
from postprocessing.morphology import morphological_cleanup
from postprocessing.vectorize import extract_regions, write_geojson, write_shp
from postprocessing.report import generate_report
from inference.predict_scene import _write_geotiff_uint8  # reuse existing writer

logger = logging.getLogger(__name__)

_PREFIX = "[POSTPROCESS]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess_scene(
    prob_path: str,
    output_dir: str,
    threshold: Optional[float] = None,
    morphology_enabled: Optional[bool] = None,
    opening_kernel_size: Optional[int] = None,
    closing_kernel_size: Optional[int] = None,
    min_region_area_pixels: Optional[int] = None,
    output_geojson: Optional[bool] = None,
    output_shp: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full post-processing pipeline on one scene probability map.

    Parameters
    ----------
    prob_path              : path to the float32 GeoTIFF probability map
                             ({stem}_prob.tif from predict_scene.py)
    output_dir             : directory to write all output artefacts
    threshold              : probability threshold (0 < t < 1); default 0.5
    morphology_enabled     : enable morphological cleanup; default True
    opening_kernel_size    : opening structuring element side; default 3
    closing_kernel_size    : closing structuring element side; default 3
    min_region_area_pixels : minimum region area (pixels); default 100
    output_geojson         : write GeoJSON; default True
    output_shp             : write SHP; default True
    verbose                : print progress messages

    Returns
    -------
    dict with keys matching the post-processing report schema
    """
    t_start = time.time()

    # ---- Merge caller args with defaults ----
    overrides: Dict[str, Any] = {}
    if threshold is not None:
        overrides["threshold"] = threshold
    if morphology_enabled is not None:
        overrides["morphology_enabled"] = morphology_enabled
    if opening_kernel_size is not None:
        overrides["opening_kernel_size"] = opening_kernel_size
    if closing_kernel_size is not None:
        overrides["closing_kernel_size"] = closing_kernel_size
    if min_region_area_pixels is not None:
        overrides["min_region_area_pixels"] = min_region_area_pixels
    if output_geojson is not None:
        overrides["output_geojson"] = output_geojson
    if output_shp is not None:
        overrides["output_shp"] = output_shp

    cfg = with_overrides(**overrides)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(prob_path).stem
    # Strip _prob suffix to get the base scene stem for output naming
    if stem.endswith("_prob"):
        scene_stem = stem[:-5]
    else:
        scene_stem = stem

    scene_id = scene_stem

    _log(verbose, f"Loading prediction   : {prob_path}")

    # ---- Step 1: Load probability map ----
    _validate_prob_file_exists(prob_path)
    prob_map, src_crs, src_transform = _load_prob_map(prob_path)
    H, W = prob_map.shape
    has_geo = src_crs is not None and src_transform is not None

    _log(verbose, f"Probability shape    : {H}x{W}")
    _log(verbose, f"CRS                  : {src_crs}")
    _log(verbose, f"Has georeferencing   : {has_geo}")
    _log(verbose, f"Threshold            : {cfg['threshold']:.2f}")

    # ---- Step 2: Validate probability values ----
    _validate_prob_values(prob_map)

    prob_stats = {
        "min": float(prob_map.min()),
        "max": float(prob_map.max()),
        "mean": float(prob_map.mean()),
    }
    predicted_px = int((prob_map >= cfg["threshold"]).sum())
    _log(
        verbose,
        f"Predicted pixels (>= threshold): {predicted_px}"
        f"  ({100.0 * predicted_px / (H * W):.2f}%)"
    )

    # ---- Step 3: Threshold ----
    binary_mask = (prob_map >= cfg["threshold"]).astype(np.uint8)
    binary_mask_path = os.path.join(output_dir, f"{scene_stem}_binary_mask.tif")
    _write_geotiff_uint8(binary_mask_path, binary_mask, src_crs, src_transform, has_geo)
    _log(verbose, f"Binary mask          -> {binary_mask_path}")

    # ---- Step 4 + 5: Morphological cleanup ----
    _log(
        verbose,
        f"Morphological cleanup: enabled={cfg['morphology_enabled']}"
        f"  open={cfg['opening_kernel_size']}  close={cfg['closing_kernel_size']}"
    )
    cleaned_mask = morphological_cleanup(
        binary_mask,
        opening_kernel_size=cfg["opening_kernel_size"],
        closing_kernel_size=cfg["closing_kernel_size"],
        enabled=cfg["morphology_enabled"],
    )
    cleaned_mask_path = os.path.join(output_dir, f"{scene_stem}_cleaned_mask.tif")
    _write_geotiff_uint8(cleaned_mask_path, cleaned_mask, src_crs, src_transform, has_geo)
    _log(verbose, f"Cleaned mask         -> {cleaned_mask_path}")

    # ---- Step 6 + 7: Connected components / candidate extraction ----
    _log(
        verbose,
        f"Extracting regions   : min_area={cfg['min_region_area_pixels']} px"
    )
    if not has_geo:
        _warn(
            verbose,
            "Source raster has no CRS/transform; polygon coordinates will be "
            "in pixel space (not geographic). Georeferencing verification will "
            "not be possible."
        )
        # Fall back to identity transform so shapes() returns pixel coordinates
        _transform = Affine.identity() if src_transform is None else src_transform
    else:
        _transform = src_transform

    regions = extract_regions(
        cleaned_mask,
        transform=_transform,
        crs=src_crs,
        min_region_area_pixels=cfg["min_region_area_pixels"],
    )
    n_regions = len(regions)

    total_geo_area = 0.0
    if regions and regions[0].get("geographic_area_km2") is not None:
        total_geo_area = round(sum(r["geographic_area_km2"] for r in regions), 6)

    _log(verbose, f"Connected components : {n_regions}")
    _log(verbose, f"Total predicted area : {total_geo_area:.4f} km2")

    # ---- Step 8: GeoJSON ----
    geojson_path = None
    if cfg["output_geojson"]:
        geojson_path = os.path.join(output_dir, f"{scene_stem}_predictions.geojson")
        write_geojson(regions, geojson_path, scene_id, cfg["threshold"], src_crs)
        _log(verbose, f"GeoJSON              -> {geojson_path}")

    # ---- Step 9: SHP ----
    shp_path = None
    if cfg["output_shp"]:
        shp_path = os.path.join(output_dir, f"{scene_stem}_predictions.shp")
        write_shp(regions, shp_path, scene_id, cfg["threshold"], src_crs)
        _log(verbose, f"SHP                  -> {shp_path}")

    processing_time = time.time() - t_start
    _log(verbose, f"Processing time      : {processing_time:.3f} s")

    # ---- Step 10: Report ----
    report_path = os.path.join(output_dir, f"{scene_stem}_postprocessing_report.json")
    crs_str = str(src_crs) if src_crs else None

    generate_report(
        scene_id=scene_id,
        prob_path=prob_path,
        binary_mask_path=binary_mask_path,
        cleaned_mask_path=cleaned_mask_path,
        geojson_path=geojson_path,
        shp_path=shp_path,
        report_path=report_path,
        threshold=cfg["threshold"],
        morphology_enabled=cfg["morphology_enabled"],
        opening_kernel_size=cfg["opening_kernel_size"],
        closing_kernel_size=cfg["closing_kernel_size"],
        min_region_area_pixels=cfg["min_region_area_pixels"],
        regions=regions,
        prob_stats=prob_stats,
        processing_time_seconds=processing_time,
        crs_str=crs_str,
        has_georeferencing=has_geo,
    )
    _log(verbose, f"Report               -> {report_path}")

    # Return the full report as a dict for programmatic use
    with open(report_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(verbose: bool, msg: str) -> None:
    logger.info(msg)
    if verbose:
        print(f"{_PREFIX} {msg}", flush=True)


def _warn(verbose: bool, msg: str) -> None:
    logger.warning(msg)
    if verbose:
        print(f"{_PREFIX} WARNING: {msg}", flush=True)


def _validate_prob_file_exists(prob_path: str) -> None:
    """Raise a clear error if the probability file does not exist."""
    if not os.path.isfile(prob_path):
        raise FileNotFoundError(
            f"Prediction file not found: {prob_path}"
        )


def _load_prob_map(prob_path: str):
    """Load probability map from a GeoTIFF, extracting CRS and transform.

    Returns
    -------
    prob_map  : float32 ndarray (H, W)
    crs       : rasterio.CRS or None
    transform : affine.Affine or None
    """
    with rasterio.open(prob_path) as src:
        if src.count != 1:
            raise ValueError(
                f"Probability map must be single-band, got {src.count} bands: {prob_path}"
            )
        prob = src.read(1).astype(np.float32)       # (H, W)
        crs = src.crs
        transform = src.transform

        # Detect identity (no georeferencing)
        _has_crs = crs is not None
        _has_transform = (transform is not None) and (transform != rasterio.transform.IDENTITY)

        # Check if transform is the default identity (rasterio default when no georef)
        if _has_transform:
            # rasterio.transform.IDENTITY may not be directly comparable
            try:
                identity_t = rasterio.transform.IDENTITY
                if transform == identity_t:
                    _has_transform = False
            except AttributeError:
                pass

    if not _has_crs or not _has_transform:
        crs = crs if _has_crs else None
        transform = transform if _has_transform else None

    return prob, crs, transform


def _validate_prob_values(prob_map: np.ndarray) -> None:
    """Validate that probability map contains values in [0, 1]."""
    if not np.all(np.isfinite(prob_map)):
        raise ValueError(
            "Prediction contains invalid probability values (NaN/Inf found). "
            "Ensure sigmoid was applied exactly once during inference."
        )
    pmin, pmax = float(prob_map.min()), float(prob_map.max())
    if pmin < 0.0 or pmax > 1.0:
        raise ValueError(
            f"Prediction contains invalid probability values outside [0, 1]: "
            f"min={pmin:.6f}, max={pmax:.6f}. "
            "Ensure the model output was passed through sigmoid before saving."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "SAR oil-spill post-processing: probability map -> "
            "binary mask -> morphology -> connected components -> "
            "GeoJSON / SHP -> report"
        )
    )
    ap.add_argument(
        "--prob",
        required=True,
        metavar="PATH",
        help="Path to the float32 probability-map GeoTIFF ({stem}_prob.tif "
             "from predict_scene.py)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write all output artefacts",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULTS["threshold"],
        metavar="FLOAT",
        help=f"Probability threshold (default: {DEFAULTS['threshold']})",
    )
    ap.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable morphological cleanup",
    )
    ap.add_argument(
        "--opening-kernel",
        type=int,
        default=DEFAULTS["opening_kernel_size"],
        metavar="INT",
        help=f"Opening structuring element side, pixels (default: {DEFAULTS['opening_kernel_size']})",
    )
    ap.add_argument(
        "--closing-kernel",
        type=int,
        default=DEFAULTS["closing_kernel_size"],
        metavar="INT",
        help=f"Closing structuring element side, pixels (default: {DEFAULTS['closing_kernel_size']})",
    )
    ap.add_argument(
        "--min-region-area",
        type=int,
        default=DEFAULTS["min_region_area_pixels"],
        metavar="INT",
        help=f"Minimum region area in pixels (default: {DEFAULTS['min_region_area_pixels']})",
    )
    ap.add_argument(
        "--no-geojson",
        action="store_true",
        help="Skip GeoJSON output",
    )
    ap.add_argument(
        "--no-shp",
        action="store_true",
        help="Skip SHP output",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return ap.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ns = _parse_args(argv)

    report = postprocess_scene(
        prob_path=ns.prob,
        output_dir=ns.output_dir,
        threshold=ns.threshold,
        morphology_enabled=not ns.no_morphology,
        opening_kernel_size=ns.opening_kernel,
        closing_kernel_size=ns.closing_kernel,
        min_region_area_pixels=ns.min_region_area,
        output_geojson=not ns.no_geojson,
        output_shp=not ns.no_shp,
        verbose=not ns.quiet,
    )

    print(f"\n{_PREFIX} === SUMMARY ===")
    print(f"{_PREFIX}   Scene ID            : {report['scene_id']}")
    print(f"{_PREFIX}   Threshold           : {report['threshold']}")
    print(f"{_PREFIX}   Predicted regions   : {report['number_of_predicted_regions']}")
    if report.get("total_predicted_area_km2") is not None:
        print(f"{_PREFIX}   Total area          : {report['total_predicted_area_km2']:.4f} km²")
    print(f"{_PREFIX}   Processing time     : {report['processing_time_seconds']:.3f} s")
    print(f"{_PREFIX}   Report              -> {report['outputs']['cleaned_mask']}")


if __name__ == "__main__":
    main()
