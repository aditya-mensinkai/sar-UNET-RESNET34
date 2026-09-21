"""Post-processing package for SAR oil-spill predictions.

Converts full-scene probability maps from U-Net + ResNet34 inference into:
  - Thresholded binary mask
  - Morphologically cleaned binary mask
  - Connected-component candidate regions
  - Georeferenced polygons (GeoJSON and SHP)
  - Post-processing report (JSON)

Typical usage::

    from postprocessing.run_postprocess import postprocess_scene
    report = postprocess_scene(
        prob_path="results/my_inference/00092_prob.tif",
        output_dir="results/postprocessed/00092",
    )

Or via CLI::

    python -m postprocessing.run_postprocess \\
        --prob  results/my_inference/00092_prob.tif \\
        --output-dir results/postprocessed/00092 \\
        --threshold 0.5 \\
        --min-region-area 100

Design decisions:
  - NO adapter required: predict_scene.py already saves a clean float32
    single-band GeoTIFF (prob.tif) with the original scene CRS + transform
    embedded. Post-processing reads that file directly.
  - Morphology uses pure-NumPy (no scipy) because scipy is not in the
    project requirements.txt.
  - Connected components are extracted via rasterio.features.shapes, which
    operates at scene level (never per-tile) so regions crossing tile
    boundaries remain continuous.
  - SHP output is written without fiona/geopandas (neither is available)
    using a minimal struct-based .shp / .shx / .dbf / .prj writer.
  - Ground truth is never touched; pipeline operates exclusively on model
    probability output.
"""
from __future__ import annotations

__version__ = "1.0.0"
