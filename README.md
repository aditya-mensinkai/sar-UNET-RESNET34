# Sentinel-1 SAR Oil-Spill Segmentation — Dataset & SAR Validation

Validation/audit record for the Sentinel-1 SAR oil-spill segmentation dataset
(Trujillo-Acatitla et al., IPICYT). All validation is **read-only**: no script in
`tools/` ever modifies, rewrites, or deletes files under `dataset/`.

## Dataset identity (established from authoritative sources)

| Item | Value |
|---|---|
| Source | Zenodo official archives, CC BY 4.0 |
| Part I (Oil, 1200) | `10.5281/zenodo.8346860` |
| Part II (No_oil + Lookalike, 685 + 685) | `10.5281/zenodo.8253899` |
| Part III (test, 150/class + ground truth) | `10.5281/zenodo.13761290` |
| Paper | Trujillo-Acatitla et al. (2024), *Mar. Pollut. Bull.* 204, 116549 — `10.1016/j.marpolbul.2024.116549` |
| Stored values | **Sigma0 backscatter in decibels (dB)**, 2-band `float32` TIFFs (HIGH confidence, author-stated) |
| Polarization set | **{VV, VH}** (HIGH confidence, author-stated) |
| Per-band order | **UNKNOWN** — band1=VV/band2=VH is *not* established (see conflict note below) |

> **Band-order conflict (unresolved, not guessed):** an unrelated third-party
> pipeline claims band0=VV/band1=VH, but local statistics show band 2
> systematically *stronger* than band 1 (Oil means ≈ −19.9 vs −33.2 dB), which is
> opposite to the canonical VV > VH ocean ordering. Numerical behavior must not
> decide polarization, so the order stays UNKNOWN pending an author statement.

## Validation stages completed

1. **B1 audit** (`tools/validate_b1_dataset.py` → `processed_dataset/reports/b1_dataset_audit.json/.txt`):
   full read-only scan of all 6040 files — structure, headers, per-band stats,
   masks, pairing (2570/2570 flat pairs PASS), CRS (EPSG:4326), duplicates,
   test independence, 9 visual-QC figures.
2. **Issue resolution** (`tools/resolve_b1_issues.py` → `b1_resolution_report.json/.txt`,
   `duplicate_resolution.csv`, `mask_set_comparison.csv`, `mask_authority_report.txt`,
   `sar_representation_report.txt`, `train_test_independence.txt`).
3. **SAR radiometry** (`tools/investigate_sar_radiometry.py` → `sar_radiometry_report.txt/.json`,
   `sar_sources.txt`): Sigma0-dB CONFIRMED, {VV,VH} CONFIRMED, band order + speckle/
   terrain-correction details UNKNOWN. Overall: **PARTIALLY CONFIRMED**.
4. **Split construction** (`tools/create_dataset_split.py`, seed 42, NTFS **hard links**,
   0 physical copies) → `processed_dataset/{train,val,test}/images|masks/<class>/`,
   `metadata/split_index.csv`, `metadata/quarantine/quarantine.csv`,
   `reports/dataset_split_report.txt`.
5. **Preprocessing** (`preprocessing/`, numpy-only): inspect → calibration-gate
   (SKIPPED — input already Sigma0 dB, `calibration_applied=false`) → dB→linear →
   Lee 5×5 (configurable 3/5/7, linear domain only) → linear→dB → percentile
   (P1–P99) normalization → train-only exact flips/rot90 (masks stay binary) →
   deterministic 256/192 tiling → overlap-average stitching, threshold after
   reconstruction. Outputs: `reports/preprocessing_config.json`,
   `reports/preprocessing_qc/` (8-panel QC figures + stats).
6. **Preprocessing audit** (read-only, no repo writes): full verification of the
   `preprocessing/` implementation — Lee math, windows, conversions, tiling,
   stitching, augmentation, split hygiene all PASS with measured evidence.
   Two FAILs found and fixed (see 7). Verdict: YES, WITH SPECIFIC FIXES.
7. **Preprocessing fixes** (minimal, Lee/tiling/augmentation untouched):
   - Global frozen normalization bounds from the 2053-scene train split
     (deterministic seeded reservoir over post-Lee dB; one scene in RAM):
     **band 0 P1 = −41.2278, P99 = 0.0; band 1 P1 = −34.3992, P99 = 0.0**,
     saved as `NORMALIZATION_BOUNDS` in `preprocessing_config.json`.
   - `process_scene(..., bounds=None)` now **requires** frozen bounds for every
     split (train/val/test/inference) — missing bounds raise `ValueError`;
     per-scene refitting is impossible through the pipeline.
   - `db_to_linear` is now exactly `10**(dB/10)` (additive eps removed);
     zero-protection lives only in `linear_to_db` (`max(lin, eps)`, 0 → −100 dB).
8. **Zero-value investigations** (read-only, local + source-level, config untouched):
   ~2.6% of train pixels are exact 0.0, forming single edge-anchored rectangles
   with 100% inter-band coincidence (e.g. `Oil/01034` 60.8% zeros, rows 663–2047;
   two 100%-blank test scenes). No tag/doc declares NoData; author code paywalled
   or absent; SNAP terrain-correction sources confirm no-coverage areas are written
   as 0. P99 = 0 is mathematically forced and verified. Classification:
   **LIKELY FILL / NODATA — NOT PROVEN**; recommendation PATH D (confirm with
   authors before any masking/normalization change).

## Key findings

- **G1 label conflict (MANUAL REVIEW REQUIRED):** `Oil/00007 == Oil/01339` pixel-identical,
  masks differ (fg 70772 vs 74906). Both quarantined from train/val. Mtime forensics:
  identical image timestamps (packaging duplication), masks ~2 days apart.
- **Exact duplicates:** `Oil/00356 == 00357`, `No_oil/00542 == 00543` (second copies excluded);
  intra-test `Test_Images/No oil/00005 == 00087` (official test set kept intact, flagged for eval).
- **Mask authority:** flat `Mask_oil/` etc. are authoritative for train/val. Nested `Mask/`
  is Part III test ground truth (`*_segmentation.tif`, IDs match `Test_Images` 1:1);
  nested Oil masks provably belong to a different sample (median same-ID IoU 0.003) — never merged.
- **Odd dimensions:** 4 Lookalike files (00140/00166/00288/00390) are valid, aligned, kept as-is.
- **Split:** train 2053 (958/547/548) · val 513 (239/137/137) · test 450 (150×3); 0 hash
  mismatches, 0 cross-split overlaps. No NaN/Inf/NoData anywhere.
- **Normalization (frozen):** train-only global P1/P99 — band 0 [−41.2278, 0.0],
  band 1 [−34.3992, 0.0]; 2.66%/2.70% of pixels saturate at 1.0 (zeros + rare
  bright scatterers). Val/test/inference reuse these bounds; refitting raises.
- **Zero pixels:** ~2.6% exact zeros (edge-anchored fill-like blocks, both bands
  identical); test mean 7.8%. Treated as ordinary pixels until authors confirm
  fill semantics — no masking, no P99 change.

## DATASET_STATUS: NOT_READY

Blockers: G1 contradictory ground truth; per-band polarization order unconfirmed;
zero fill-semantics unconfirmed by authors (PATH D pending).
(Missing mask CRS is accepted — pixel alignment verified via dimensions.)
No model code exists in this repo yet: any future LinkNet+ResNet34 training must
use the `preprocessing/` chain exactly as recorded in `preprocessing_config.json`
(train≡inference); Lee-window ablations (A: off, B: 3, C: 5, D: 7) require retraining.

## Reproduce

```powershell
.venv\Scripts\python.exe tools\validate_b1_dataset.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\resolve_b1_issues.py  --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\investigate_sar_radiometry.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\create_dataset_split.py --dataset dataset --out processed_dataset --seed 42
$env:PYTHONPATH = "<root>"
.venv\Scripts\python.exe -m preprocessing.visualize --image <sar.tif> --mask <mask.tif> --out <qc.png> [--lee-window 5] [--no-lee]
```

`dataset/` is the immutable source (newest write May 2023). Do not delete it until
`processed_dataset/` has been manually verified.
