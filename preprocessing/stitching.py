"""Stitching entry point (re-exported from tiling for the documented layout).

Probability averaging of overlaps, full-size reconstruction, post-reconstruction
thresholding, CRS/transform preservation. See tiling.py.
"""
from __future__ import annotations

from .tiling import stitch

__all__ = ["stitch"]
