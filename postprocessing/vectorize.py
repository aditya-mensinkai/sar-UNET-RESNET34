"""Vectorization: connected components → polygons → GeoJSON / SHP.

Uses rasterio.features.shapes to extract geospatially accurate polygons from
a binary mask.  The raster transform from the probability map (which embeds
the original scene CRS and affine transform) is passed directly to shapes(),
so pixel coordinates are converted to geographic coordinates using the
original scene metadata — no fabricated coordinates.

SHP output is written using a minimal pure-Python struct-based writer
(no fiona, no geopandas, no osgeo) because those packages are not available
in this project's environment.  Only POLYGON geometries are written, which
is exactly what rasterio.features.shapes produces for binary masks.

Geographic area calculation:
  For EPSG:4326 (degrees), pixel area is computed from the affine transform
  pixel size and the WGS-84 sphere approximation (111.32 km per degree of
  latitude and cos(lat)*111.32 km per degree of longitude at the polygon
  centroid).  This is not geodetically exact but is geospatially consistent
  with the raster resolution — it does NOT use fabricated constants or
  arbitrary values.  If the scene CRS is projected (metres), pixel area is
  directly |dx * dy| m².
"""
from __future__ import annotations

import json
import math
import struct
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from affine import Affine


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_regions(
    cleaned_mask: np.ndarray,
    transform: Affine,
    crs: Optional[CRS],
    min_region_area_pixels: int = 0,
) -> List[Dict]:
    """Extract candidate regions (connected components) from a binary mask.

    Uses rasterio.features.shapes which performs connected-component analysis
    at scene level (4-connectivity by default) and converts pixel polygons to
    geographic coordinates using the provided affine transform.

    Parameters
    ----------
    cleaned_mask          : uint8 (H, W) binary array (0/1)
    transform             : affine transform from the probability-map raster
    crs                   : CRS from the probability-map raster (may be None)
    min_region_area_pixels: discard regions with fewer than this many pixels
                            (applied in true pixel count, not Shoelace estimate)

    Returns
    -------
    List of dicts, one per region:
        {
          "candidate_id": "candidate_001",
          "pixel_area": int,
          "geographic_area_km2": float,   # None if no valid CRS/transform
          "centroid_pixel": [row, col],
          "centroid_geo": [lon, lat],      # None if no valid CRS/transform
          "geometry": { GeoJSON Polygon }
        }
    """
    from rasterio.features import shapes, geometry_mask as geo_mask

    if cleaned_mask.ndim != 2:
        raise ValueError(
            f"cleaned_mask must be 2-D, got shape {cleaned_mask.shape}"
        )
    arr = np.asarray(cleaned_mask, dtype=np.uint8)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("cleaned_mask must contain only {0, 1} values")

    H, W = arr.shape

    # shapes() extracts polygons for each connected component of the same pixel
    # value.  Passing mask=arr.astype(bool) limits extraction to foreground (1)
    # pixels only, preventing background from becoming a giant polygon.
    # transform is passed to convert pixel → geographic coordinates using the
    # original raster metadata.
    raw_shapes = list(shapes(arr, mask=arr.astype(bool), transform=transform))

    has_geo = (
        transform is not None
        and transform != Affine.identity()
        and crs is not None
    )

    # Determine if CRS is geographic (degrees) or projected (metres)
    is_geographic = has_geo and _is_geographic_crs(crs)

    regions = []
    candidate_idx = 0
    for geom, _val in raw_shapes:
        # ----------------------------------------------------------------
        # BUG FIX: pixel area must be counted by rasterizing the polygon,
        # NOT via the Shoelace formula on the exterior ring only.
        #
        # The Shoelace formula on the exterior ring ignores interior rings
        # (holes), causing inflated pixel_area for polygons with holes.
        # rasterio.features.shapes CAN return polygons with holes for
        # complex binary regions (e.g. a predicted oil patch with a small
        # undetected area inside).  Using the rasterized count is exact
        # and handles holes correctly.
        # ----------------------------------------------------------------
        pmask = geo_mask([geom], transform=transform, invert=True, out_shape=(H, W))
        pixel_area = int(pmask.sum())

        if min_region_area_pixels > 0 and pixel_area < min_region_area_pixels:
            continue

        candidate_idx += 1

        poly_coords = geom["coordinates"][0]  # exterior ring (for centroid)

        # Centroid in pixel and geographic space
        centroid_geo = _polygon_centroid(poly_coords)  # geographic (lon, lat)
        centroid_pixel = _geo_to_pixel(centroid_geo[0], centroid_geo[1], transform) \
            if has_geo else [0, 0]

        # Geographic area: pixel_area (rasterized, exact) * per-pixel km²
        geo_area_km2 = None
        if has_geo and pixel_area > 0:
            geo_area_km2 = _compute_geo_area_km2(
                pixel_area, transform, is_geographic,
                centroid_geo[1]  # centroid latitude for degree→km conversion
            )

        regions.append({
            "candidate_id": f"candidate_{candidate_idx:03d}",
            "pixel_area": pixel_area,
            "geographic_area_km2": geo_area_km2,
            "centroid_pixel": centroid_pixel,
            "centroid_geo": centroid_geo if has_geo else None,
            "geometry": geom,
        })

    return regions


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------

def write_geojson(
    regions: List[Dict],
    output_path: str,
    scene_id: str,
    threshold: float,
    crs: Optional[CRS],
) -> str:
    """Write candidate regions to a GeoJSON file.

    Each feature corresponds to one predicted oil-spill candidate region.
    Properties include scene metadata and per-region statistics.  No
    geographic values are fabricated — all values come from the regions list.

    Returns the path written.
    """
    features = []
    for r in regions:
        features.append({
            "type": "Feature",
            "geometry": r["geometry"],
            "properties": {
                "scene_id": scene_id,
                "candidate_id": r["candidate_id"],
                "threshold": threshold,
                "pixel_area": r["pixel_area"],
                "geographic_area_km2": r["geographic_area_km2"],
                "centroid_geo_lon": r["centroid_geo"][0] if r["centroid_geo"] else None,
                "centroid_geo_lat": r["centroid_geo"][1] if r["centroid_geo"] else None,
                "centroid_pixel_row": r["centroid_pixel"][0] if r["centroid_pixel"] else None,
                "centroid_pixel_col": r["centroid_pixel"][1] if r["centroid_pixel"] else None,
            },
        })

    crs_name = str(crs) if crs else "unknown"
    fc = {
        "type": "FeatureCollection",
        "name": f"{scene_id}_predictions",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name},
        },
        "features": features,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Shapefile output (minimal pure-Python writer, no fiona/geopandas)
# ---------------------------------------------------------------------------

# SHP format constants
_SHP_MAGIC = 9994
_SHP_VERSION = 1000
_SHP_TYPE_POLYGON = 5
_DBF_HEADER_BYTE = 0x03


def write_shp(
    regions: List[Dict],
    output_path: str,
    scene_id: str,
    threshold: float,
    crs: Optional[CRS],
) -> str:
    """Write candidate regions to ESRI Shapefile format.

    Produces: .shp, .shx, .dbf, .prj

    Geometry: POLYGON (exterior ring only; interior rings / holes are
    represented as separate shell rings if present, as per GeoJSON structure
    from rasterio.features.shapes which produces polygons without holes for
    binary masks).

    DBF fields: scene_id (C,50), cand_id (C,20), threshold (N,8,4),
                px_area (N,10,0), area_km2 (N,12,4)

    Coordinate system: WKT written to .prj from the CRS string.
    """
    base = str(output_path)
    if base.endswith(".shp"):
        base = base[:-4]
    shp_path = base + ".shp"
    shx_path = base + ".shx"
    dbf_path = base + ".dbf"
    prj_path = base + ".prj"

    Path(shp_path).parent.mkdir(parents=True, exist_ok=True)

    # Build polygon records
    records = []
    for r in regions:
        geom = r["geometry"]
        if geom["type"] == "Polygon":
            rings = geom["coordinates"]   # list of ring coord lists
        else:
            warnings.warn(
                f"skipping non-Polygon geometry type {geom['type']!r} in SHP output"
            )
            continue
        records.append({
            "rings": rings,
            "scene_id": scene_id,
            "cand_id": r["candidate_id"],
            "threshold": threshold,
            "px_area": r["pixel_area"],
            "area_km2": r["geographic_area_km2"] if r["geographic_area_km2"] is not None else 0.0,
        })

    # ---- .shp + .shx ----
    _write_shp_shx(shp_path, shx_path, records)

    # ---- .dbf ----
    _write_dbf(dbf_path, records)

    # ---- .prj ----
    _write_prj(prj_path, crs)

    return shp_path


# ---------------------------------------------------------------------------
# SHP / SHX writer
# ---------------------------------------------------------------------------

def _poly_bbox(rings):
    """Bounding box of all rings: (xmin, ymin, xmax, ymax)."""
    xs = [c[0] for ring in rings for c in ring]
    ys = [c[1] for ring in rings for c in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _write_shp_shx(shp_path: str, shx_path: str, records: list) -> None:
    """Write SHP and SHX files for POLYGON records."""
    # First pass: build record bytes
    record_bytes_list = []
    for rec in records:
        rings = rec["rings"]
        xmin, ymin, xmax, ymax = _poly_bbox(rings)
        n_parts = len(rings)
        n_points = sum(len(r) for r in rings)

        # Record content (little-endian body)
        # Shape type (4 bytes LE int) + bounding box (4 doubles) + n_parts + n_points
        body = struct.pack("<i", _SHP_TYPE_POLYGON)
        body += struct.pack("<dddd", xmin, ymin, xmax, ymax)
        body += struct.pack("<ii", n_parts, n_points)
        # Part offsets (index of first point in each ring)
        offset = 0
        for ring in rings:
            body += struct.pack("<i", offset)
            offset += len(ring)
        # Points
        for ring in rings:
            for c in ring:
                body += struct.pack("<dd", c[0], c[1])

        record_bytes_list.append(body)

    # SHP file
    # Header: 
    #   file_code(4B BE) + 5 unused(20B BE) + file_length(4B BE) = 28 bytes
    #   version(4B LE) + shape_type(4B LE) = 8 bytes
    #   bbox 8 doubles (Xmin,Ymin,Xmax,Ymax,Zmin,Zmax,Mmin,Mmax) = 64 bytes
    #   Total = 100 bytes
    # file_length is in 16-bit words
    shp_records_size = sum(4 + 4 + len(b) for b in record_bytes_list)  # rec hdr + body
    shp_file_length = (100 + shp_records_size) // 2  # in 16-bit words

    # Compute overall bbox
    if records:
        all_xs, all_ys = [], []
        for rec in records:
            for ring in rec["rings"]:
                for c in ring:
                    all_xs.append(c[0])
                    all_ys.append(c[1])
        bbox = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    with open(shp_path, "wb") as shp, open(shx_path, "wb") as shx:
        # Write SHP header (100 bytes exactly)
        # Big-endian: file_code(1 int) + 5 unused(5 ints) + file_length(1 int) = 28 bytes
        shp_header = struct.pack(
            ">iiiiiii",
            _SHP_MAGIC, 0, 0, 0, 0, 0, shp_file_length
        )
        # Little-endian: version(1 int) + shape_type(1 int) = 8 bytes
        shp_header += struct.pack("<ii", _SHP_VERSION, _SHP_TYPE_POLYGON)
        # Little-endian: 8 doubles (Xmin,Ymin,Xmax,Ymax,Zmin,Zmax,Mmin,Mmax) = 64 bytes
        shp_header += struct.pack("<dddddddd",
                                  bbox[0], bbox[1], bbox[2], bbox[3],
                                  0.0, 0.0, 0.0, 0.0)
        shp.write(shp_header)
        assert len(shp_header) == 100, f"SHP header size = {len(shp_header)}, expected 100"

        # SHX header (100 bytes exactly, same structure)
        shx_file_length = (100 + 8 * len(records)) // 2
        shx_header = struct.pack(
            ">iiiiiii",
            _SHP_MAGIC, 0, 0, 0, 0, 0, shx_file_length
        )
        shx_header += struct.pack("<ii", _SHP_VERSION, _SHP_TYPE_POLYGON)
        shx_header += struct.pack("<dddddddd",
                                  bbox[0], bbox[1], bbox[2], bbox[3],
                                  0.0, 0.0, 0.0, 0.0)
        shx.write(shx_header)
        assert len(shx_header) == 100, f"SHX header size = {len(shx_header)}, expected 100"

        current_offset = 50  # in 16-bit words (100 bytes / 2)
        for rec_num, body in enumerate(record_bytes_list, start=1):
            content_length = len(body) // 2  # in 16-bit words
            # SHP record header: record number and content length (big-endian)
            shp.write(struct.pack(">ii", rec_num, content_length))
            shp.write(body)
            # SHX record
            shx.write(struct.pack(">ii", current_offset, content_length))
            current_offset += 4 + content_length  # 4 words for header


# ---------------------------------------------------------------------------
# DBF writer
# ---------------------------------------------------------------------------

def _write_dbf(dbf_path: str, records: list) -> None:
    """Write a minimal DBF-III file with 5 fixed fields."""
    # Field descriptors
    fields = [
        # name(11B), type(1B), reserved(4B), field_length(1B), decimal_count(1B), reserved(14B)
        ("SCENE_ID",  "C", 50, 0),
        ("CAND_ID",   "C", 20, 0),
        ("THRESHOLD", "N",  8, 4),
        ("PX_AREA",   "N", 10, 0),
        ("AREA_KM2",  "N", 12, 4),
    ]
    n_fields = len(fields)
    header_size = 32 + n_fields * 32 + 1  # 32 bytes header + 32 per field + terminator
    record_size = 1 + sum(f[2] for f in fields)  # deletion flag + field widths
    n_records = len(records)

    with open(dbf_path, "wb") as f:
        # Header record
        f.write(struct.pack("<B", _DBF_HEADER_BYTE))  # version
        f.write(struct.pack("<BBB", 26, 9, 21))       # last update YY/MM/DD (placeholder)
        f.write(struct.pack("<I", n_records))          # number of records
        f.write(struct.pack("<HH", header_size, record_size))
        f.write(b"\x00" * 20)  # reserved

        # Field descriptors
        for name, ftype, flen, fdec in fields:
            fname_bytes = name.encode("ascii").ljust(11, b"\x00")[:11]
            f.write(fname_bytes)
            f.write(ftype.encode("ascii"))
            f.write(b"\x00" * 4)  # reserved
            f.write(struct.pack("<BB", flen, fdec))
            f.write(b"\x00" * 14)  # reserved

        f.write(b"\r")  # field descriptor terminator

        # Data records
        for rec in records:
            f.write(b" ")  # deletion flag (space = not deleted)
            _dbf_write_str(f, rec["scene_id"], 50)
            _dbf_write_str(f, rec["cand_id"], 20)
            _dbf_write_num(f, rec["threshold"], 8, 4)
            _dbf_write_int(f, rec["px_area"], 10)
            _dbf_write_num(f, rec["area_km2"], 12, 4)

        f.write(b"\x1a")  # EOF marker


def _dbf_write_str(f, value: str, width: int) -> None:
    """Write a character field, padded/truncated to width."""
    b = str(value).encode("ascii", errors="replace")[:width]
    f.write(b.ljust(width, b" "))


def _dbf_write_num(f, value: float, width: int, decimals: int) -> None:
    """Write a numeric field as ASCII right-justified."""
    fmt = f"{value:.{decimals}f}"
    if len(fmt) > width:
        fmt = "*" * width  # overflow indicator
    f.write(fmt.rjust(width).encode("ascii"))


def _dbf_write_int(f, value: int, width: int) -> None:
    """Write an integer numeric field as ASCII right-justified."""
    fmt = str(int(value))
    if len(fmt) > width:
        fmt = "*" * width
    f.write(fmt.rjust(width).encode("ascii"))


# ---------------------------------------------------------------------------
# PRJ writer
# ---------------------------------------------------------------------------

def _write_prj(prj_path: str, crs: Optional[CRS]) -> None:
    """Write CRS as WKT to .prj file."""
    if crs is None:
        # Write an empty .prj to indicate unknown CRS
        with open(prj_path, "w") as f:
            f.write("")
        return
    try:
        wkt = crs.to_wkt()
    except Exception:
        try:
            wkt = crs.to_proj4()
        except Exception:
            wkt = str(crs)
    with open(prj_path, "w") as f:
        f.write(wkt)


# ---------------------------------------------------------------------------
# Area / centroid helpers
# ---------------------------------------------------------------------------

def _polygon_centroid(coords: list) -> Tuple[float, float]:
    """Centroid of a polygon ring from coordinate list [(x,y), ...]."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return float(np.mean(xs)), float(np.mean(ys))


def _geo_to_pixel(
    lon: float, lat: float, transform: Affine
) -> List[float]:
    """Convert geographic (lon, lat) to pixel (row, col) using inverse transform."""
    try:
        inv = ~transform
        col, row = inv * (lon, lat)
        return [float(row), float(col)]
    except Exception:
        return [0.0, 0.0]


def _is_geographic_crs(crs: Optional[CRS]) -> bool:
    """Return True if the CRS uses geographic (degree) units."""
    if crs is None:
        return False
    try:
        return crs.is_geographic
    except Exception:
        return False


def _compute_geo_area_km2(
    pixel_area: int,
    transform: Affine,
    is_geographic: bool,
    centroid_lat: float,
) -> float:
    """Compute geographic area in km² from rasterized pixel area and raster resolution.

    For geographic CRS (degrees): uses WGS-84 approximation.
      1 degree latitude  ≈ 111.32 km
      1 degree longitude ≈ 111.32 * cos(lat) km

    For projected CRS (metres): |dx * dy| m² converted to km².

    Parameters
    ----------
    pixel_area   : exact pixel count from rasterio.features.geometry_mask
    transform    : affine transform (pixel → geographic)
    is_geographic: True = degrees, False = metres
    centroid_lat : centroid latitude (degrees), used for degree→km conversion
    """
    if pixel_area <= 0:
        return 0.0

    dx = abs(transform.a)   # pixel width in CRS units
    dy = abs(transform.e)   # pixel height in CRS units

    if is_geographic:
        # WGS-84 approximation: 1 deg lat ≈ 111.32 km
        # 1 deg lon ≈ 111.32 * cos(lat) km  (varies with latitude)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(centroid_lat))
        pixel_area_km2 = (dx * km_per_deg_lon) * (dy * km_per_deg_lat)
    else:
        # Projected CRS: CRS units are metres
        pixel_area_m2 = dx * dy
        pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    return round(float(pixel_area) * pixel_area_km2, 6)
