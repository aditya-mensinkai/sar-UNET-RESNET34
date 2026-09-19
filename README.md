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

## DATASET_STATUS: NOT_READY

Blockers: G1 contradictory ground truth; per-band polarization order unconfirmed.
(Missing mask CRS is accepted — pixel alignment verified via dimensions.)

## Reproduce

```powershell
.venv\Scripts\python.exe tools\validate_b1_dataset.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\resolve_b1_issues.py  --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\investigate_sar_radiometry.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\create_dataset_split.py --dataset dataset --out processed_dataset --seed 42
```

`dataset/` is the immutable source (newest write May 2023). Do not delete it until
`processed_dataset/` has been manually verified.
