"""SAR radiometry supporting numerical analysis (READ-ONLY) + report writer.

Reads a deterministic sample of dataset TIFFs (no modification) and writes:
  processed_dataset/reports/sar_radiometry_report.txt
  processed_dataset/reports/sar_radiometry_report.json
  processed_dataset/reports/sar_sources.txt
Authoritative conclusions come from Zenodo records + paper metadata (see SOURCES
in this script); numerical results are supporting evidence only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SOURCES = [
    {"SOURCE": "Zenodo record 8346860 (official dataset archive, Part I, Oil)",
     "TITLE": "Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part I.",
     "URL / DOI": "https://zenodo.org/records/8346860 / 10.5281/zenodo.8346860",
     "RELEVANT INFORMATION": ("'The dataset comprises Sentinel-1 SAR images in Sigma0, in decibels (db)'; "
                               "'1200 Sentinel-1 SAR Sigma0 images, in db'; images 2048x2048x2 TIFF; "
                               "masks 2048x2048, foreground=1/background=0, same number as image. Notes: "
                               "'only the Sentinel-1 Sigma0 images in decibels (db) with two polarizations "
                               "(VV, VH) and dimensions of 2048x2048x2 are georeferenced. The masks ... are not "
                               "georeferenced because they were treated as matrices'."),
     "WHAT IT PROVES": ("Stored values are calibrated Sigma0 backscatter in dB (linear->dB done by authors); "
                        "polarization set is {VV, VH}; image/mask layout matches local dataset (1200 Oil)."),
     "CONFIDENCE": "HIGH"},
    {"SOURCE": "Zenodo record 8253899 (official dataset archive, Part II, No Oil + Lookalike)",
     "TITLE": "Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part II.",
     "URL / DOI": "https://zenodo.org/records/8253899 / 10.5281/zenodo.8253899",
     "RELEVANT INFORMATION": ("'685 images of oil-free Sentinel-1 SAR Sigma0 images'; '685 Sentinel-1 SAR Sigma0 "
                               "images ... of look-alikes'; No_oil/Lookalike ground truths are all 0 "
                               "('the value of the ground truth ... is only 0'). Same Sigma0-dB / (VV,VH) notes."),
     "WHAT IT PROVES": "Same representation for No_oil/Lookalike; all-zero masks are by design.",
     "CONFIDENCE": "HIGH"},
    {"SOURCE": "Zenodo record 13761290 (official dataset archive, Part III, test)",
     "TITLE": "Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part III",
     "URL / DOI": "https://zenodo.org/records/13761290 / 10.5281/zenodo.13761290",
     "RELEVANT INFORMATION": ("'This part contains the test images'; 150 Oil + 150 No oil + 150 Lookalike "
                               "Sigma0-dB images plus *_segmentation ground truths. Same (VV,VH) notes."),
     "WHAT IT PROVES": ("Local Test_Images/ (150/class, no masks shipped locally) and local nested "
                        "Mask/<class>/*_segmentation.tif (150/class) both originate from Part III."),
     "CONFIDENCE": "HIGH"},
    {"SOURCE": "Crossref metadata for the dataset paper",
     "TITLE": "Marine oil spill detection and segmentation in SAR data with two steps Deep Learning framework",
     "URL / DOI": "https://doi.org/10.1016/j.marpolbul.2024.116549 (Mar. Pollut. Bull. 204, 116549)",
     "RELEVANT INFORMATION": ("Peer-reviewed paper by the dataset authors (Trujillo-Acatitla et al., IPICYT, "
                               "Mexico); Zenodo records I-III declare this DOI under 'documents'. Full text is "
                               "paywalled (Elsevier 429 on TDM fetch); abstract-level metadata confirms "
                               "authorship, venue, year. References SNAP (ESA Sentinel Application Platform)."),
     "WHAT IT PROVES": "Dataset is peer-reviewed and documented; preprocessing pipeline details live in the paper.",
     "CONFIDENCE": "MEDIUM (metadata inspected; full methods text not accessible)"},
    {"SOURCE": "Kaggle third-party pipeline for this exact dataset (NON-authoritative)",
     "TITLE": "Oil Spill Segmentation (Sentinel-1 SAR, dual-pol VV+VH) — oil-spill-code",
     "URL / DOI": "https://www.kaggle.com/datasets/shuddhabrotabanerjee/oil-spill-code (no DOI)",
     "RELEVANT INFORMATION": ("Claims local TIFF layout '2048x2048x2 float32, Sigma0 dB (band0=VV, band1=VH)'. "
                               "Author is unrelated to the dataset creators."),
     "WHAT IT PROVES": "Nothing authoritative; records a band-order claim (band0=VV) that CONFLICTS with the "
                        "physical-consistency check below. Listed only to document the conflict.",
     "CONFIDENCE": "LOW"},
    {"SOURCE": "Terrascope S1 ATBD + STEP forum / Copernicus openEO docs (sensor background only)",
     "TITLE": "Terrascope Sentinel-1 ATBD S1 SIGMA0; forum.step.esa.int; dataspace copernicus OilSpillMapping",
     "URL / DOI": "https://docs.terrascope.be/.../S1%20SIGMA0%20V131.pdf (no DOI)",
     "RELEVANT INFORMATION": ("Standard S1 IW GRD dual-pol (VV,VH) products; sigma0 dB = 10*log10(sigma0_linear); "
                               "VH ocean backscatter is weaker than VV; SNAP is the standard calibration tool."),
     "WHAT IT PROVES": ("General SAR background ONLY - used solely to interpret (not to decide) the local "
                        "band statistics. Never applied to this dataset's files."),
     "CONFIDENCE": "LOW (background context only)"},
]

SAMPLE = {"Oil": ["00000.tif", "00007.tif", "00356.tif"],
          "No_oil": ["00000.tif", "00027.tif"],
          "Lookalike": ["00000.tif", "00087.tif"]}


def band_stats(a):
    f = a[np.isfinite(a)]
    out = {"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()),
           "median": float(np.median(f)), "std": float(f.std())}
    for q, v in zip((1, 5, 25, 50, 75, 95, 99), np.percentile(f, (1, 5, 25, 50, 75, 95, 99))):
        out[f"P{int(q)}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="processed_dataset")
    a = ap.parse_args()
    ds = os.path.abspath(a.dataset)
    rep = os.path.join(os.path.abspath(a.out), "reports")
    os.makedirs(rep, exist_ok=True)

    num, corrs = {}, []
    for cls, fns in SAMPLE.items():
        for fn in fns:
            with rasterio.open(os.path.join(ds, cls, fn)) as src:
                arr = src.read().astype(np.float64)
            b1, b2 = arr[0].ravel(), arr[1].ravel()
            m = np.isfinite(b1) & np.isfinite(b2)
            corrs.append(float(np.corrcoef(b1[m][::97], b2[m][::97])[0, 1]))
            num[f"{cls}/{fn}"] = {"band1": band_stats(arr[0]), "band2": band_stats(arr[1])}
    num["band1_vs_band2_pearson_sampled"] = {"n_files": len(corrs),
                                             "mean": float(np.mean(corrs)),
                                             "min": float(np.min(corrs)),
                                             "max": float(np.max(corrs))}
    print(json.dumps(num["band1_vs_band2_pearson_sampled"], indent=1), flush=True)

    txt = """========================================
SAR RADIOMETRY INVESTIGATION
========================================
(Read-only audit. No file in dataset/ was modified, rescaled, converted or rewritten.)

Dataset:
Local folders Oil/ (1200), No_oil/ (685), Lookalike/ (685) + flat masks + nested
Mask/<class>/*_segmentation.tif (150/class) + Test_Images/<class>/ (150/class).
Identified (Part 1, file counts + layout match) as the Trujillo-Acatitla et al.
IPICYT Sentinel-1 SAR oil-spill dataset, Zenodo Parts I-III.

Original source:
Zenodo official archives 10.5281/zenodo.8346860 (Part I, Oil),
10.5281/zenodo.8253899 (Part II, No Oil + Lookalike),
10.5281/zenodo.13761290 (Part III, test images + ground truth).

Paper:
Trujillo-Acatitla, Tuxpan-Vargas, Ovando-Vazquez, Monterrubio-Martinez (2024),
"Marine oil spill detection and segmentation in SAR data with two steps Deep
Learning framework", Marine Pollution Bulletin 204, 116549.

DOI:
Dataset: 10.5281/zenodo.8346860 / 10.5281/zenodo.8253899 / 10.5281/zenodo.13761290
Paper: 10.1016/j.marpolbul.2024.116549

Official dataset source:
https://zenodo.org/records/8346860, https://zenodo.org/records/8253899,
https://zenodo.org/records/13761290 (CC BY 4.0, authors = paper authors).

----------------------------------------
BAND INFORMATION
----------------------------------------

Number of bands:
2 (float32 + float32 in every train and test image; verified in B1 audit)

Band 1:
First channel of the 2048x2048x2 Sigma0-dB stack. Polarization assignment UNKNOWN
(see POLARIZATION conflict report below).

Band 2:
Second channel of the 2048x2048x2 Sigma0-dB stack. Polarization assignment UNKNOWN.

----------------------------------------
POLARIZATION
----------------------------------------

Band 1 polarization:
UNKNOWN (member of {VV, VH}; order unresolved)

Band 2 polarization:
UNKNOWN (member of {VV, VH}; order unresolved)

Confidence:
HIGH that the polarization set is {VV, VH} (stated verbatim by the dataset authors
in the notes of all three Zenodo records). UNKNOWN for the per-band order.

Evidence:
FOR set {VV, VH}: author notes on all three records: "Sigma0 images in decibels (db)
with two polarizations (VV, VH)". AGAINST any other set (HH/HV/other): Sentinel-1 IW
dual-pol over ocean is VV+VH and authors state VV,VH explicitly. FOR band1=VV/band2=VH:
one unrelated third-party Kaggle pipeline claims "band0=VV, band1=VH" (LOW, explicitly
non-authoritative). AGAINST trusting that claim: physical-consistency check on local
files shows band 2 systematically STRONGER than band 1 (e.g. Oil file-means approx
-33.2 vs -19.9 dB; sample medians band1 ~-33.6 / band2 ~-20.5), whereas cross-pol VH
ocean backscatter is canonically WEAKER than co-pol VV - i.e. the observed order is
more consistent with [VH, VV] stacking (e.g. alphabetical export) than with [VV, VH].
This contradiction is documented, NOT resolved: numerical behavior must not determine
polarization, so the order stays UNKNOWN until the authors' paper/code states it.

----------------------------------------
RADIOMETRIC REPRESENTATION
----------------------------------------

Stored representation:
sigma0 (calibrated normalized radar cross-section) in decibels (dB)

Physical quantity:
sigma0 (sigma-nought)

Unit:
dB (10*log10 of linear sigma0)

Scale:
logarithmic

Confidence:
HIGH (stated verbatim by the dataset authors: "Sentinel-1 SAR images in Sigma0,
in decibels (db)" on all three records; consistent 2048x2048x2 float32 TIFFs).

Evidence:
Authoritative: three Zenodo descriptions + notes quoted above. Supporting (only):
strongly negative float32 values with a positive bright-target tail (Oil band1
min -112.3 / max +22.8 / mean -33.2; band2 min -114.7 / max +29.9 / mean -19.9),
no NaN/Inf/NoData - compatible with, but not proof of, log-scaled backscatter.
Band1/band2 Pearson on the read-only sample: MEAN_CORR (positive, oil and sea
structure shared across channels; supporting only).

----------------------------------------
DATA GENERATION PIPELINE
----------------------------------------

Original product:
Sentinel-1 SAR (authors do not state mode/level in the Zenodo text; IW GRD
dual-pol is the standard source for this kind of product but that is unconfirmed
for THESE files -> UNKNOWN specifics).

Radiometric calibration:
YES (HIGH) - "Sigma0" stated by authors; sigma0 exists only after calibration.

Terrain correction:
UNKNOWN - images are georeferenced (EPSG:4326, verified) so SOME geocoding was
applied, but the authors do not state Range-Doppler terrain vs ellipsoid correction.

Speckle filtering:
UNKNOWN - no statement, no detectable signature asserted (no analysis performed
that could decide this; must not be guessed).

Linear -> dB:
YES (HIGH) - authors state the stored Sigma0 IS in decibels.

dB -> linear:
NO (HIGH) - stored values are already dB; no linear data present.

Other transformations:
Tiling/chipping to 2048x2048x2 windows + TIFF export (HIGH, evident); resampling
details, orbit files, border-noise removal: UNKNOWN (paper full text paywalled).

----------------------------------------
CONCLUSION
----------------------------------------

SAR representation:
sigma0 in dB (HIGH confidence, author-stated).

Polarization:
set {VV, VH} (HIGH); per-band order UNKNOWN (conflict documented, not guessed).

Confidence:
HIGH for representation + polarization set; UNKNOWN for band order, terrain
correction method, speckle filtering, exact source product level.

Remaining uncertainty:
1. Band order (VV/VH vs VH/VV) - needs author paper/code statement.
2. Terrain-correction method, speckle filtering, exact S1 product/mode.
3. Origin of extreme minima (down to -128) and max==0.0 files (fill vs real).
Paper full text (Elsevier, paywalled, TDM 429) should be consulted for 1-2.
"""
    mean_corr = num["band1_vs_band2_pearson_sampled"]["mean"]
    txt = txt.replace("MEAN_CORR", f"{mean_corr:.4f}")
    with open(os.path.join(rep, "sar_radiometry_report.txt"), "w") as f:
        f.write(txt)

    ev_set = [f"{s['SOURCE']}: {s['RELEVANT INFORMATION']}" for s in SOURCES[:3]]
    report = {
        "dataset": ("Trujillo-Acatitla et al. Sentinel-1 SAR oil-spill dataset, Zenodo Parts I-III "
                    "(10.5281/zenodo.8346860, 10.5281/zenodo.8253899, 10.5281/zenodo.13761290); "
                    "paper 10.1016/j.marpolbul.2024.116549"),
        "source": ["https://zenodo.org/records/8346860", "https://zenodo.org/records/8253899",
                   "https://zenodo.org/records/13761290", "https://doi.org/10.1016/j.marpolbul.2024.116549"],
        "bands": {
            "band_1": {"polarization": "UNKNOWN (member of {VV, VH})",
                       "physical_quantity": "sigma0", "representation": "decibels (dB)",
                       "unit": "dB", "confidence": "HIGH (representation); UNKNOWN (polarization order)",
                       "evidence": ev_set + [
                           "Third-party claim 'band0=VV' (LOW, non-authoritative, conflicts with observed band strength order).",
                           f"Sample stats band1 (supporting only): {json.dumps(num)}"[:600]]},
            "band_2": {"polarization": "UNKNOWN (member of {VV, VH})",
                       "physical_quantity": "sigma0", "representation": "decibels (dB)",
                       "unit": "dB", "confidence": "HIGH (representation); UNKNOWN (polarization order)",
                       "evidence": ev_set + [
                           "Band2 systematically stronger than band1 (e.g. Oil means -19.9 vs -33.2 dB); "
                           "consistent-with-[VH,VV]-order hypothesis offered as supporting evidence ONLY."]}},
        "radiometric_calibration": {"status": "YES",
                                    "evidence": ["Authors state stored values are Sigma0 (exists only post-calibration)."]},
        "terrain_correction": {"status": "UNKNOWN",
                               "evidence": ["Georeferenced EPSG:4326 verified; method (Range-Doppler vs ellipsoid) unstated."]},
        "speckle_filtering": {"status": "UNKNOWN", "evidence": ["No author statement found."]},
        "linear_to_db": {"status": "YES",
                         "evidence": ["Authors state stored Sigma0 is in decibels."]},
        "overall_status": "PARTIALLY_CONFIRMED",
        "overall_status_reason": ("Representation sigma0-dB CONFIRMED (HIGH); polarization set {VV,VH} CONFIRMED "
                                 "(HIGH); per-band polarization order UNCONFIRMED (UNKNOWN) due to documented "
                                 "conflict; pipeline specifics partly UNKNOWN."),
        "numerical_support_only": num,
    }
    with open(os.path.join(rep, "sar_radiometry_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(os.path.join(rep, "sar_sources.txt"), "w") as f:
        for s in SOURCES:
            f.write(f"SOURCE: {s['SOURCE']}\nTITLE: {s['TITLE']}\nURL / DOI: {s['URL / DOI']}\n"
                    f"RELEVANT INFORMATION: {s['RELEVANT INFORMATION']}\n"
                    f"WHAT IT PROVES: {s['WHAT IT PROVES']}\nCONFIDENCE: {s['CONFIDENCE']}\n\n")
    print("reports written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
