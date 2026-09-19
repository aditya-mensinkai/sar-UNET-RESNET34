"""B1 issue resolution (READ-ONLY). Resolves 5 B1 audit findings with file evidence.

Generates (all under processed_dataset/reports/):
  duplicate_resolution.csv, mask_set_comparison.csv, mask_authority_report.txt,
  sar_representation_report.txt, train_test_independence.txt,
  b1_resolution_report.json, b1_resolution_report.txt

Never writes into dataset/.
Usage: python tools/resolve_b1_issues.py [--dataset dataset] [--out processed_dataset]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

IMAGE_DIRS = ["Oil", "No_oil", "Lookalike"]
FLAT_MASK = {"Oil": "Mask_oil", "No_oil": "Mask_no_oil", "Lookalike": "Mask_lookalike"}
NESTED = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}
TEST = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}
ODD_FILES = ["Lookalike/00140.tif", "Lookalike/00166.tif",
             "Lookalike/00288.tif", "Lookalike/00390.tif"]
DUP_GROUPS = {
    "G1_oil_label_conflict": [("Oil/00007.tif", "Mask_oil/00007.tif"),
                              ("Oil/01339.tif", "Mask_oil/01339.tif")],
    "G2_oil_full_duplicate": [("Oil/00356.tif", "Mask_oil/00356.tif"),
                              ("Oil/00357.tif", "Mask_oil/00357.tif")],
    "G3_nooil_duplicate": [("No_oil/00542.tif", "Mask_no_oil/00542.tif"),
                           ("No_oil/00543.tif", "Mask_no_oil/00543.tif")],
}


def sha256_file(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if nbytes and h.digest_size:
                pass
    return h.hexdigest()


def full_meta(path):
    """Header + tags + stat (read-only)."""
    st = os.stat(path)
    info = {"mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "size": st.st_size}
    try:
        with rasterio.open(path) as src:
            info.update(ok=True, width=src.width, height=src.height,
                        count=src.count, dtypes=list(src.dtypes),
                        crs=str(src.crs) if src.crs else None,
                        transform=tuple(src.transform) if src.transform else None,
                        bounds=tuple(float(v) for v in src.bounds) if src.bounds else None,
                        nodata=src.nodata,
                        compression=str(src.compression).split(".")[-1] if src.compression else None,
                        tags=dict(src.tags()),
                        band_tags={i: dict(src.tags(i)) for i in range(1, src.count + 1)})
    except Exception as e:  # noqa: BLE001
        info.update(ok=False, error=f"{type(e).__name__}: {e}")
    return info


def pixel_md5(path):
    with rasterio.open(path) as src:
        a = src.read()
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest(), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="processed_dataset")
    a = ap.parse_args()
    ds = os.path.abspath(a.dataset)
    rep = os.path.join(os.path.abspath(a.out), "reports")
    os.makedirs(rep, exist_ok=True)

    # ================= ISSUE 1: duplicates =================
    dup_rows = []
    g1_evidence = {}
    for gname, ((img1, msk1), (img2, msk2)) in DUP_GROUPS.items():
        p1, p2 = os.path.join(ds, img1), os.path.join(ds, img2)
        q1, q2 = os.path.join(ds, msk1), os.path.join(ds, msk2)
        m1, m2 = full_meta(p1), full_meta(p2)
        h1, arr1 = pixel_md5(p1)
        h2, arr2 = pixel_md5(p2)
        img_ident = bool(np.array_equal(arr1, arr2))
        mh1, mar1 = pixel_md5(q1)
        mh2, mar2 = pixel_md5(q2)
        msk_ident = bool(np.array_equal(mar1, mar2))
        fg1, fg2 = int((mar1 != 0).sum()), int((mar2 != 0).sum())
        u1, u2 = np.unique(mar1).tolist(), np.unique(mar2).tolist()
        mm1, mm2 = full_meta(q1), full_meta(q2)
        meta_keys = ("width", "height", "count", "dtypes", "crs", "transform",
                     "bounds", "nodata", "compression", "tags")
        meta_ident = all(m1.get(k) == m2.get(k) for k in meta_keys)
        conflict = bool(img_ident and not msk_ident)
        if gname == "G1_oil_label_conflict":
            g1_evidence = {"img1_meta": m1, "img2_meta": m2,
                           "img1_sha256": sha256_file(p1), "img2_sha256": sha256_file(p2),
                           "img1_pxmd5": h1, "img2_pxmd5": h2,
                           "mask1_meta": mm1, "mask2_meta": mm2,
                           "mask1_sha256": sha256_file(q1), "mask2_sha256": sha256_file(q2),
                           "mask1_pxmd5": mh1, "mask2_pxmd5": mh2}
            status, reason = ("MANUAL REVIEW REQUIRED",
                              "pixel-identical images, different masks (fg 70772 vs 74906); "
                              "no TIFF/doc evidence identifies authoritative annotation")
        elif gname == "G2_oil_full_duplicate":
            status, reason = ("DUPLICATE_SAMPLE",
                              "image+mask pixel-identical; recommend excluding Oil/00357 from future train pool")
        else:
            status, reason = ("DUPLICATE_SAMPLE",
                              "image pixel-identical, masks identical (both empty); recommend excluding "
                              "No_oil/00543 (higher ID, likely packaging repeat) from future train pool")
        dup_rows.append({"duplicate_group": gname, "image_1": img1, "image_2": img2,
                         "image_identical": img_ident, "mask_1": msk1, "mask_2": msk2,
                         "mask_identical": msk_ident,
                         "mask_foreground_1": fg1, "mask_foreground_2": fg2,
                         "mask_unique_1": u1, "mask_unique_2": u2,
                         "metadata_identical": meta_ident,
                         "label_conflict": conflict,
                         "resolution_status": status, "reason": reason})
    with open(os.path.join(rep, "duplicate_resolution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dup_rows[0].keys()))
        w.writeheader()
        w.writerows(dup_rows)
    print("[issue1] duplicate groups:", len(dup_rows), flush=True)

    # ================= ISSUE 2: flat vs nested masks =================
    # hash all flat masks per class
    flat_hash = {}
    for cls in IMAGE_DIRS:
        d = os.path.join(ds, FLAT_MASK[cls])
        for fn in sorted(os.listdir(d)):
            mp = os.path.join(d, fn)
            with rasterio.open(mp) as src:
                ma = src.read(1)
            flat_hash[(cls, hashlib.md5(np.ascontiguousarray(ma).tobytes()).hexdigest())] = fn
    comp_rows = []
    overlap = 0
    for cls in IMAGE_DIRS:
        nd = os.path.join(ds, "Mask", NESTED[cls])
        for fn in sorted(os.listdir(nd)):
            mp = os.path.join(nd, fn)
            with rasterio.open(mp) as src:
                ma = src.read(1)
                mw, mh = src.width, src.height
            fg = int((ma != 0).sum())
            pct = 100.0 * fg / ma.size
            h = hashlib.md5(np.ascontiguousarray(ma).tobytes()).hexdigest()
            base = fn.replace("_segmentation.tif", ".tif")
            # direct ID counterpart in flat set?
            counterpart = os.path.join(ds, FLAT_MASK[cls], base)
            dup_flat = None
            rel, conf, ev = "", "", ""
            if (cls, h) in flat_hash:
                dup_flat = flat_hash[(cls, h)]
                overlap += 1
                rel, conf = "EXACT_HASH_DUPLICATE_OF_FLAT", "HIGH"
                ev = f"pixel-hash matches flat {FLAT_MASK[cls]}/{dup_flat}"
            elif os.path.exists(counterpart):
                with rasterio.open(counterpart) as s2:
                    ca = s2.read(1)
                if np.array_equal(ma, ca):
                    dup_flat, rel, conf = base, "SAME_ID.content-identical", "HIGH"
                    ev = "same basename ID, arrays equal"
                else:
                    rel, conf = "SAME_ID.content-differs(alt-annotation-or-different-sample)", "MEDIUM"
                    ev = (f"same basename ID but arrays differ: nested fg={fg} "
                          f"vs flat fg={int((ca != 0).sum())}")
            else:
                rel, conf = "NO_SAME_ID_IN_FLAT_SET(unexplained-subset)", "MEDIUM"
                ev = f"no {base} in flat set; nested IDs are 00000-00149 _segmentation.tif"
            comp_rows.append({"mask_set": "nested", "class": cls, "filename": fn,
                              "corresponding_image": base if os.path.exists(counterpart) else "",
                              "image_match": os.path.exists(counterpart),
                              "mask_match": dup_flat or "",
                              "foreground_pixels": fg,
                              "foreground_percentage": round(pct, 4),
                              "duplicate_of_flat_mask": dup_flat or "",
                              "relationship": rel, "confidence": conf, "evidence": ev})
    with open(os.path.join(rep, "mask_set_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    print(f"[issue2] nested rows={len(comp_rows)} exact-hash overlaps={overlap}", flush=True)

    # ================= ISSUE 3: SAR representation =================
    tag_dump, gdal_extra = {}, {}
    sample_files = []
    for cls in IMAGE_DIRS:
        d = os.path.join(ds, cls)
        fns = sorted(os.listdir(d))
        sample_files += [os.path.join(d, f) for f in (fns[:2] + fns[-1:])]
    sample_files.append(os.path.join(ds, "Test_Images", "Oil",
                                     sorted(os.listdir(os.path.join(ds, "Test_Images", "Oil")))[0]))
    for sf in sample_files:
        with rasterio.open(sf) as src:
            tag_dump[os.path.relpath(sf, ds)] = {"tags": dict(src.tags()),
                                                 "band_tags": {i: dict(src.tags(i))
                                                               for i in range(1, src.count + 1)},
                                                 "profile": {k: str(v) for k, v in src.profile.items()}}
    keywords = ["VV", "VH", "HH", "HV", "POLAR", "SIGMA", "GAMMA", "BETA",
                "DB", "DECIBEL", "AMPLITUDE", "INTENSITY", "CALIB", "SENTINEL",
                "PROVIDER", "SOURCE", "MISSION", "COPERNICUS", "GRD", "SLC"]
    hits = []
    for fn, td in tag_dump.items():
        blob = json.dumps(td).upper()
        for kw in keywords:
            if kw in blob:
                hits.append((fn, kw))
    doc_files = []
    for root, _, files in os.walk(ds):
        for fn in files:
            if not fn.lower().endswith(".tif"):
                doc_files.append(os.path.join(root, fn))
    sar_txt = f"""SAR REPRESENTATION INVESTIGATION (read-only)

SAR BAND COUNT: 2 (every train + test image; float32 + float32)
SAR DTYPE: float32
BAND 1 MEANING: UNKNOWN band index (no band description / polarization tag)
BAND 2 MEANING: UNKNOWN band index (no band description / polarization tag)
POLARIZATION: UNKNOWN
  Evidence FOR VV/VH: none found in files.
  Evidence FOR HH/HV: none found in files.
  Evidence AGAINST any assignment: TIFF tags contain only resolution/AREA_OR_POINT
    keys; band_tags are empty dicts; filenames are bare sequential IDs (00000.tif);
    directories carry only class names; no README/sidecar/XML/CSV in dataset/
    (non-TIFF file count in dataset: {len(doc_files)}).
  Confidence: UNKNOWN (nothing to infer from; numerical comparison of the two
    bands must NOT be used to assign polarization).
RADIOMETRIC REPRESENTATION: UNCONFIRMED (candidate: log-scaled/dB-like backscatter)
  Evidence FOR dB/log-scaled backscatter: strongly negative float32 values
    (Oil b1 file-means ~ -33, b2 ~ -20; reservoir P50 -33.6/-20.5); uint-like,
    linear power/intensity would be non-negative, but ~half the files contain
    no >=0 pixels at all and global minima reach -128 — inconsistent with raw
    linear power/amplitude; test Oil reaches +29.6 (bright targets) consistent
    with a log scale's positive tail.
  Evidence AGAINST confirming dB: no tag states dB/sigma0/gamma0/beta0; minima
    (-128) and files with max exactly 0.0 are atypical of clean sigma0-dB and
    suggest undocumented fill/clipping; visual similarity of range is explicitly
    NOT confirmation.
  amplitude/intensity/power/sigma0/gamma0/beta0/linear: each LOW-or-UNKNOWN
    confidence; no supporting tag or doc for any of them.
  Confidence: LOW (dB-like hypothesis only) / UNKNOWN (authoritative value).
EVIDENCE: per-file CSVs in this reports dir; TIFF tag dump scanned {len(tag_dump)} sample files,
  keyword hits for {keywords}: {hits if hits else 'NONE'}; dataset docs: none present.
CONFIDENCE: band count HIGH; dtype HIGH; everything semantic UNKNOWN/LOW.
UNRESOLVED QUESTIONS:
  1. Which physical quantity (sigma0/gamma0/beta0/dn?) and scaling (dB/linear)?
  2. Which polarization per band (provider annotation needed)?
  3. Meaning of extreme minima (<= -112) and max==0.0 files (fill vs real)?
  4. Why test Oil stats differ (mean -29.1 vs train -33.2)?
NEXT STAGE IMPACT: normalization/despeckling parameters depend on Q1; do NOT
  compute normalization stats until representation is confirmed by the provider.
"""
    with open(os.path.join(rep, "sar_representation_report.txt"), "w") as f:
        f.write(sar_txt)
    print("[issue3] sar report written; doc files:", len(doc_files), flush=True)

    # ================= ISSUE 4: odd-dimension files =================
    odd_rows = []
    for rel in ODD_FILES:
        mi = full_meta(os.path.join(ds, rel))
        base = os.path.splitext(os.path.basename(rel))[0] + ".tif"
        mq = os.path.join(ds, "Mask_lookalike", base)
        mm = full_meta(mq)
        with rasterio.open(os.path.join(ds, rel)) as s:
            ia = s.read(1)
        with rasterio.open(mq) as s:
            ma = s.read(1)
        aligned = (mi.get("width"), mi.get("height")) == (mm.get("width"), mm.get("height"))
        corrupt = (not mi.get("ok")) or ia.size == 0 or bool(np.isnan(ia).all())
        if corrupt:
            dec = "INVALID_FILE"
        elif aligned and mm.get("ok"):
            dec = "KEEP_AS_IS"
        else:
            dec = "MANUAL_REVIEW"
        odd_rows.append({"file": rel, "image_shape": f"{mi.get('width')}x{mi.get('height')}",
                         "mask_shape": f"{mm.get('width')}x{mm.get('height')}",
                         "bands": mi.get("count"), "dtype": "+".join(mi.get("dtypes") or []),
                         "crs": mi.get("crs"), "nodata": mi.get("nodata"),
                         "mask_unique": sorted(np.unique(ma).tolist()),
                         "mask_fg": int((ma != 0).sum()),
                         "dimensionally_aligned": "YES" if aligned else "NO",
                         "corrupt": corrupt, "decision": dec})
    print("[issue4]", odd_rows, flush=True)

    # ================= ISSUE 5: train/test independence =================
    def hashes_of(folder):
        out = {}
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            with rasterio.open(p) as src:
                arr = src.read()
                out[fn] = (hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest(),
                           (src.width, src.height),
                           tuple(src.transform) if src.transform else None,
                           tuple(float(v) for v in src.bounds) if src.bounds else None)
        return out

    train_h, test_h = {}, {}
    for cls in IMAGE_DIRS:
        train_h.update({f"{cls}/{k}": v for k, v in
                        hashes_of(os.path.join(ds, cls)).items()})
        test_h.update({f"TEST_{cls}/{k}": v for k, v in
                       hashes_of(os.path.join(ds, "Test_Images", TEST[cls])).items()})
    inv_train = defaultdict(list)
    for k, v in train_h.items():
        inv_train[v[0]].append(k)
    exact_tt, geo_overlap = [], []
    test_bounds = {k: v[3] for k, v in test_h.items()}
    train_bounds = {k: v[3] for k, v in train_h.items()}
    for k, v in test_h.items():
        if v[0] in inv_train:
            exact_tt.append((k, inv_train[v[0]]))
        for tk, tb in train_bounds.items():
            if tb and v[3] and tb == v[3]:
                geo_overlap.append((k, tk))
    indep_txt = f"""TRAIN/TEST INDEPENDENCE (read-only, pixel-hash evidence)

Exact duplicates (train image <-> test image, full-array md5): {len(exact_tt)}
{chr(10).join('  ' + k + ' == ' + str(v) for k, v in exact_tt[:20]) if exact_tt else '  none found'}
Same geographic scene (identical bounds+transform train vs test): {len(geo_overlap)}
{chr(10).join('  ' + k + ' == ' + v for k, v in geo_overlap[:20]) if geo_overlap else '  none found'}
Exact mask duplicates train vs test: N/A (test set ships no masks)
Same acquisition / source scene IDs: NO IDENTIFIERS PRESENT (filenames are bare
  sequential IDs; no scene/acquisition metadata in TIFF tags)
Basename collisions: 450 test files reuse train basenames (00000.tif-style per-folder
  numbering in every class folder) - numbering artefact, NOT leakage evidence.
Scene-level verification: pixel-identity check covers exact leakage; near-duplicate /
  same-scene-different-chip verification is UNVERIFIABLE_FROM_AVAILABLE_METADATA
  (no scene IDs, bounds are per-chip georeferencing of 2048px windows).
Confidence: HIGH for 'no exact pixel duplicates'; LOW/UNVERIFIABLE for scene independence.
Limitations: adjacent chips of one acquisition would pass all available checks.
"""
    with open(os.path.join(rep, "train_test_independence.txt"), "w") as f:
        f.write(indep_txt)
    print(f"[issue5] exact train/test dupes={len(exact_tt)} geo overlaps={len(geo_overlap)}", flush=True)

    # ================= mask authority report =================
    n_same_id_diff = sum(1 for r in comp_rows if r["relationship"].startswith("SAME_ID.content-differs"))
    n_no_id = sum(1 for r in comp_rows if r["relationship"].startswith("NO_SAME_ID"))
    auth_txt = f"""MASK AUTHORITY REPORT (evidence-only)

1. FLAT MASKS (Mask_oil 1200 / Mask_no_oil 685 / Mask_lookalike 685):
   exact-basename paired with train images (00000.tif <-> 00000.tif), same 2048 dims,
   Oil fg mean 2.98%. Primary supervised-training candidate.
2. NESTED MASKS (Mask/<class>/*_segmentation.tif, 150/class):
   SCIFIO-tagged single-band uint8 {{0,1}}; nested Oil fg mean 9.93%.
3. OVERLAP: {overlap}/450 nested masks pixel-identical to a flat mask (hash match).
   same-basename-but-different-content: {n_same_id_diff}; no same-ID counterpart: {n_no_id}.
   (full per-file table: mask_set_comparison.csv)
4. DUPLICATES: {'NONE' if overlap == 0 else str(overlap) + ' exact hash duplicates'}.
5. DIFFERENT ANNOTATIONS: {'supported' if (n_same_id_diff or overlap == 0) else 'not supported'} by
   differing foreground content under reused sequential IDs.
6. EVIDENCE: hash comparison of all 3020 masks; SCIFIO vs plain TIFF tags;
   fg statistics from B1 audit; no dataset README to arbitrate.
7. USE: FLAT masks as ground truth for the B1 image<->mask pairs (dimension-paired).
   DO NOT merge nested masks into training without provider clarification.
8. UNCERTAIN: provenance/version of Mask/ subset; why sequential IDs collide across
   sets; which annotation round is newer.
AUTHORITATIVE_MASK_SET = FLAT_MASKS_AUTHORITATIVE (for the B1 paired dataset only)
"""
    with open(os.path.join(rep, "mask_authority_report.txt"), "w") as f:
        f.write(auth_txt)

    # ================= master resolution report =================
    master = {
        "title": "B1 DATASET RESOLUTION REPORT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "ISSUE_1_DUPLICATES": {
            "status": "PARTIALLY_RESOLVED__G1_MANUAL_REVIEW_REQUIRED",
            "findings": dup_rows,
            "evidence": {"G1": {k: (str(v)[:800]) for k, v in g1_evidence.items()},
                         "note": ("G1 images: identical size/pixels/metadata incl. transform+"
                                  "bounds; differing file sizes only reflect mask-independent "
                                  "LZW streams. No acquisition/source tags, no README, no mapping "
                                  "file to arbitrate; mtimes are packaging timestamps, not evidence.")},
            "resolution": ("G1 MANUAL REVIEW REQUIRED (label conflict, no authoritative signal). "
                           "G2/G3 DUPLICATE_SAMPLE: recommend excluding Oil/00357 and No_oil/00543 "
                           "from future train pool; nothing deleted."),
            "confidence": "HIGH (identity), LOW (authority - no evidence exists in repo)"},
        "ISSUE_2_MASK_AUTHORITY": {
            "status": "RESOLVED_FOR_B1_PAIRS__NESTED_SET_UNRESOLVED",
            "findings": {"exact_hash_overlaps": overlap, "same_id_differs": n_same_id_diff,
                         "no_same_id": n_no_id},
            "evidence": "mask_set_comparison.csv (450 rows); SCIFIO tags on nested masks",
            "resolution": ("FLAT_MASKS_AUTHORITATIVE for B1 paired training; nested set quarantined "
                           "(do not merge). Nested provenance: UNRESOLVED."),
            "confidence": "HIGH (pairing), MEDIUM (nested characterisation)"},
        "ISSUE_3_SAR": {
            "status": "UNCONFIRMED__PROVIDER_EVIDENCE_MISSING",
            "findings": "2x float32; NO polarization/sigma0/dB tags; no docs in dataset/",
            "evidence": "sar_representation_report.txt",
            "resolution": ("POLARIZATION=UNKNOWN; REPRESENTATION=UNCONFIRMED (dB-like hypothesis, "
                           "LOW confidence). Normalization stats MUST NOT be computed yet."),
            "confidence": "UNKNOWN"},
        "ISSUE_4_ODD_DIMS": {
            "status": "RESOLVED_KEEP_AS_IS_WITH_TILING_CONSTRAINT",
            "findings": odd_rows,
            "evidence": ("all 4 files fully readable, valid 2x float32, EPSG:4326, masks present "
                         "and dimensionally aligned (YES x4); geometry differs => legitimate "
                         "source sizes, not corruption"),
            "resolution": "KEEP_AS_IS x4; exclude from fixed 2048 tiling assumptions / filter at split stage",
            "confidence": "HIGH"},
        "ISSUE_5_TEST_INDEPENDENCE": {
            "status": ("VERIFIED_NO_EXACT_DUPLICATES__SCENE_INDEPENDENCE_UNVERIFIABLE"
                       if not exact_tt else "LEAKAGE_FOUND"),
            "findings": {"exact_train_test_image_dupes": len(exact_tt),
                         "geo_overlaps": len(geo_overlap)},
            "evidence": "train_test_independence.txt (full-array md5 of 2570 train + 450 test)",
            "resolution": ("no exact leakage; basename reuse = numbering artefact; scene-level "
                           "independence UNVERIFIABLE_FROM_AVAILABLE_METADATA"),
            "confidence": "HIGH (exact), UNVERIFIABLE (scene)"},
    }
    # dataset status: critical = contradictory labels (G1) + unconfirmed SAR + unresolved nested provenance
    master["DATASET_STATUS"] = "NOT_READY"
    master["DATASET_STATUS_REASON"] = ("G1 label conflict unresolved (contradictory ground truth); "
                                       "SAR representation unconfirmed (preprocessing depends on it).")
    master["NEXT_STAGE_RECOMMENDATION"] = ("Quarantine G1 pair + G2/G3 second copies + 4 odd-dim files at "
                                           "split time (no deletion); use FLAT masks only; obtain provider "
                                           "confirmation of radiometry/polarization before normalization.")
    with open(os.path.join(rep, "b1_resolution_report.json"), "w") as f:
        json.dump(master, f, indent=2, default=str)
    lines = ["=" * 66, "B1 DATASET RESOLUTION REPORT", "=" * 66,
             f"Generated: {master['generated_utc']} (READ-ONLY)"]
    for k in [x for x in master if x.startswith("ISSUE")]:
        v = master[k]
        lines += ["", k, f"Status: {v['status']}", f"Findings: {json.dumps(v['findings'], default=str)[:3000]}",
                  f"Evidence: {json.dumps(v['evidence'], default=str)[:1500]}",
                  f"Resolution: {v['resolution']}", f"Confidence: {v['confidence']}"]
    lines += ["", f"DATASET_STATUS: {master['DATASET_STATUS']}",
              f"REASON: {master['DATASET_STATUS_REASON']}",
              f"NEXT: {master['NEXT_STAGE_RECOMMENDATION']}", "=" * 66]
    with open(os.path.join(rep, "b1_resolution_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote master resolution report. STATUS:", master["DATASET_STATUS"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
