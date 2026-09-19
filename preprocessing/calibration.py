"""Radiometric calibration gate.

B1 + radiometry finding: local TIFFs are ALREADY author-calibrated Sigma0 in dB
(Zenodo 8346860/8253899/13761290, HIGH confidence). Calibrating again would
double-calibrate. This module therefore inspects and PASSES THROUGH, refusing
to invent calibration constants.
"""
from __future__ import annotations

import numpy as np

VALID_SOURCE_UNITS = {"sigma0_dB"}


def inspect(values, bands=2, dtype_expected=np.float32):
    """Report-only inspection of a loaded (bands, H, W) array. Never modifies."""
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    return {"shape": tuple(a.shape), "dtype": str(a.dtype),
            "bands": int(a.shape[0]) if a.ndim == 3 else None,
            "nan": int(np.isnan(a).sum()), "posinf": int(np.isposinf(a).sum()),
            "neginf": int(np.isneginf(a).sum()),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "calibration_state": ("already-Sigma0-dB (author-stated); "
                                  "calibration_applied=false")}


def apply(values, source_units="sigma0_dB"):
    """Pass-through gate. Raises instead of inventing calibration."""
    if source_units not in VALID_SOURCE_UNITS:
        raise ValueError(
            f"source_units={source_units!r} is not calibrated Sigma0 dB; "
            "raw DN calibration is impossible here (no LUTs/constants in this "
            "dataset) - refusing to invent calibration constants.")
    a = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite values in calibration input (NaN/Inf rejected)")
    return {"values": a, "calibration_applied": False,
            "calibration_type": "none (input already Sigma0 dB)",
            "source_units": source_units, "output_units": "sigma0_dB"}
