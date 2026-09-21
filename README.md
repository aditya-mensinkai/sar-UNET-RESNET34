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
   Lee 5×5 (configurable 3/5/7, linear domain only) → linear→dB → frozen global
   percentile normalization → train-only exact flips/rot90 (masks stay binary) →
   deterministic tiling (256/256 materialized; 256/192 overlap available for
   inference stitching) → overlap-average stitching, threshold after
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
9. **Tiling + augmentation materialization** (`preprocessing/make_tiles.py`,
   `tools/build_tiles.py`, `tools/test_augmentation_tiling.py`):
   scene-level split → frozen-bounds preprocessing → paired 256×256 tiling
   (materialized **stride 256** — stride-192 materialization needs ~216 GB vs
   ~117 GB free; recorded per-tile, overlap stitching path unchanged) →
   validation → keep-every-valid-tile filtering (MIN_OIL_FRACTION=0.0,
   BG_KEEP_FRACTION=1.0; reject only corrupt) → `tiled/{train,val,test}/`,
   `metadata/tile_index.csv` (**193,048 tiles**: train 131,416 / val 32,832 /
   test 28,800; 18,578 positive), `tiles_balance_report.txt`. Augmentation is
   dynamic train-only (single sampled param-set, exact ops, masks binary);
   `augmented/` intentionally empty. TESTs 1–14: **14/14 pass**.
10. **Pre-training smoke test** (`tools/smoke_dataloader.py`, no aug, no resize):
    batches [4,2,256,256]/[4,1,256,256], float32, finite, binary masks on all
    splits; U-Net + ResNet34 (`segmentation-models-pytorch`, `in_channels=2`,
    `classes=1`, first conv [64,2,7,7], 24.43M params) forward → [4,1,256,256],
   BCEWithLogits loss computed, weights untouched. RTX 3050 6 GB: peak
   254 MB, no OOM (AMP verified). Status: **READY FOR TRAINING**.
11. **Training** (`training/`, `training_config.json` → `checkpoints/`):
    U-Net + ResNet34 (2ch→1ch, 24.43M params, random init), BCEDiceLoss
    0.5/0.5, Adam lr 1e-4, batch 2, 512×512 tiles, AMP on, early-stop
    patience 8 on val loss. Ran 24/50 epochs (8.65 h, RTX 3050); best epoch 20,
    tile-level val_dice 0.55826 @0.5 → `checkpoints/best_model.pth`
    (`last_model.pth` kept). Curve: `training_history.csv`, summary:
    `training_summary.json`.
12. **Full-scene inference** (`inference/`, CLI defaults stride 384 /
    threshold 0.5 — do not retune here): preprocess whole scene once →
    sliding-window forward (sigmoid exactly once) → overlap-average stitching
    → threshold AFTER reconstruction → `<stem>_prob.tif` / `<stem>_mask.tif` /
    `<stem>_inference.json`. Test split is refused without `--allow-test`.
13. **Val-only analysis** (`tools/val_analysis.py` → `results/val_analysis/`,
    val split only, nothing frozen): each of the 513 val scenes processed
    EXACTLY ONCE (preprocessed array reused for strides 512/384/256 + fp32
    control); only 1000-bin pos/neg probability histograms + integer counts
    retained. Items: stride×threshold grid (0.50–0.95) · per-category FA/
    detection · Oil Dice by spill-area tercile · min-area sweep (analysis
    only) · paired bootstrap (1000×) · stride-256 diagnosis · 10-bin
    calibration · fp32-vs-AMP (exact per-pixel). Verification: 3078
   histogram-vs-direct pairs, 0 mismatches — PASS. Best grid point
   512@0.80 (Dice 0.61114) is a text-only suggestion; the decision is human.
14. **Final test evaluation** (`evaluation/` → `evaluation/test_results.csv`,
    `test_summary.json`, `qualitative/`, test split only, canonical protocol
    for all future model comparisons): `best_model.pth` (val-selected, never
    test-tuned) on the fixed 450 test scenes (150×3) at full-scene level —
    dynamic 512 windows (stride 384), training-identical preprocessing, AMP
    inference, overlap-mean stitching, protocol-fixed threshold 0.5 (NOT tuned
    on test; the val 0.80 observation was never frozen). Per-scene TP/FP/TN/FN
    + IoU/Dice/Precision/Recall/F1/specificity with NaN (never silent 0) for
    undefined positive-class metrics; mean/median/sample-std over valid scenes
    + micro-averaged global metrics; Oil/No-oil/Lookalike groups from the
    `class` manifest column. Integrity: manifest = 450 distinct scenes, no
    train/val ID overlap, checkpoint strict-loaded, GT SHA-256 identical
    450/450 before/after. Result (MEASURED): 450/450 success; scene-mean
    IoU 0.254 / Dice 0.316; global IoU 0.150 / Dice 0.261 / precision 0.776 /
    recall 0.157. Test is much harder than val at 0.5 (35/150 Oil scenes get
    zero predicted pixels) — that is the measurement, not a bug.

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
- **Val inference (MEASURED, val only, `results/val_analysis/`):** best grid
  point 512@0.80 Dice 0.61114, but the 512-vs-384 gap (+0.00137) is inside
  bootstrap noise (95% CI [−0.01485, +0.01615]); 384 beats 256 and 0.80 beats
  0.50. Lookalike scenes contribute ~72% of FP pixels (FA ≈ 61–65% @0.80)
  vs No-oil FA ≈ 2%; small spills score ~0.10 Dice below medium/large.
   Model is OVER-confident (pred−obs +0.31); min-area filtering barely moves
   Dice; AMP ≡ fp32 (Dice diff 1e-05, 3e-06 pixels change label).
- **Final test (MEASURED, test only, `evaluation/`):** 450/450 scenes success,
   0 failures; Oil mean IoU 0.337/Dice 0.420 with FP scenes 112/150;
   No-oil FP rate 0.02; Lookalike FP rate 0.307. Threshold stays 0.5 by
   protocol — test was never used for tuning or checkpoint selection.

## DATASET_STATUS: READY FOR TRAINING (data pipeline)

Training blockers cleared: frozen bounds enforced, 193k verified tiles, leak-free
split, CUDA smoke passed. Standing limitations (not training blockers): G1
quarantine, per-band polarization order, zero fill-semantics (PATH D pending).
(Missing mask CRS is accepted — pixel alignment verified via dimensions.)
Train with the `preprocessing/` chain exactly as recorded in
`preprocessing_config.json` (train≡inference); augmentation NONE for this
experiment; Lee-window ablations require retraining.

## Reproduce

```powershell
.venv\Scripts\python.exe tools\validate_b1_dataset.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\resolve_b1_issues.py  --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\investigate_sar_radiometry.py --dataset dataset --out processed_dataset
.venv\Scripts\python.exe tools\create_dataset_split.py --dataset dataset --out processed_dataset --seed 42
$env:PYTHONPATH = "<root>"
.venv\Scripts\python.exe -m preprocessing.visualize --image <sar.tif> --mask <mask.tif> --out <qc.png> [--lee-window 5] [--no-lee]
# Training (config: training_config.json; ~9 h on RTX 3050 6 GB)
.venv\Scripts\python.exe training\train.py
# Full-scene inference (defaults: stride 384, threshold 0.5)
.venv\Scripts\python.exe -m inference.predict_scene --scene <sar.tif> --checkpoint checkpoints\best_model.pth --output-dir <out-dir> [--stride 384] [--threshold 0.5]
# Val-only analysis, all 8 items (~35 min, val split only, nothing frozen)
.venv\Scripts\python.exe tools\val_analysis.py
# Canonical final test evaluation, 450 scenes (~15 min, test split only)
.venv\Scripts\python.exe evaluation\evaluate_model.py [--n-scenes 3]
```

`dataset/` is the immutable source (newest write May 2023). Do not delete it until
`processed_dataset/` has been manually verified.
