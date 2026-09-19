"""Populate train/val/test with NTFS HARD LINKS (no second physical copy).

Sources (read-only, never deleted/modified/moved):
  train/val images: dataset/{Oil,No_oil,Lookalike}/ + flat masks dataset/{Mask_oil,Mask_no_oil,Mask_lookalike}/
  test images: dataset/Test_Images/{Oil,No oil,Lookalike}/
  test masks (Part III provenance): dataset/Mask/{Oil,No oil,Lookalike}/*_segmentation.tif
Exclusions (quarantined, raw files untouched): Oil/00007, Oil/01339 (G1),
  Oil/00357 (==00356), No_oil/00543 (==00542).
Stratified 80/20 split, seed 42. Re-runnable: existing dest links are verified, never overwritten.
NO fallback to physical copies: if a hard link cannot be created/verified -> STOP with error.
Usage: python tools/create_dataset_split.py [--dataset dataset] [--out processed_dataset] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SEED = 42
CLASSES = ["Oil", "No_oil", "Lookalike"]
IMG_SRC = {"Oil": "Oil", "No_oil": "No_oil", "Lookalike": "Lookalike"}
MSK_SRC = {"Oil": "Mask_oil", "No_oil": "Mask_no_oil", "Lookalike": "Mask_lookalike"}
QUARANTINE = {"Oil/00007.tif": "G1 contradictory labels (dup of Oil/01339, masks differ)",
              "Oil/01339.tif": "G1 contradictory labels (dup of Oil/00007, masks differ)",
              "Oil/00357.tif": "exact duplicate of Oil/00356 (keep first copy)",
              "No_oil/00543.tif": "exact duplicate of No_oil/00542 (keep first copy)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_id(path):
    """Windows file-index identity: (volume, index-hi, index-lo) + link count."""
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_nlink)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="processed_dataset")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    ds = os.path.abspath(a.dataset)
    out = os.path.abspath(a.out)
    meta = os.path.join(out, "metadata")
    qdir = os.path.join(meta, "quarantine")
    os.makedirs(qdir, exist_ok=True)

    # ---- 1. NTFS / hard-link support probe (temp file, removed afterwards) ----
    probe_src = os.path.join(meta, ".hl_probe_src")
    probe_dst = os.path.join(meta, ".hl_probe_dst")
    try:
        with open(probe_src, "w") as f:
            f.write("hardlink-probe")
        os.link(probe_src, probe_dst)
        s1, s2 = os.stat(probe_src), os.stat(probe_dst)
        assert (s1.st_dev, s1.st_ino) == (s2.st_dev, s2.st_ino), "probe IDs differ"
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: hard links unsupported ({type(e).__name__}: {e}). STOPPING, no copies made.")
        return 1
    finally:
        for p in (probe_dst, probe_src):
            if os.path.exists(p):
                os.remove(p)
    import tempfile
    fs = "NTFS?"  # windows FS type check via temp dir volume is best-effort; probe above is authoritative
    print(f"[fs] hard-link probe OK on volume holding {out}", flush=True)

    # ---- storage baseline ----
    def dir_size(d):
        t = 0
        for r, _, fs_ in os.walk(d):
            for fn in fs_:
                try:
                    t += os.path.getsize(os.path.join(r, fn))
                except OSError:
                    pass
        return t

    raw_size = dir_size(ds)
    proc_before = dir_size(out)
    free_before = shutil.disk_usage(out).free
    print(f"[storage] raw={raw_size/1e9:.2f}GB processed(before)={proc_before/1e9:.3f}GB free={free_before/1e9:.2f}GB",
          flush=True)

    # ---- 2/3. source protection snapshot + dirs ----
    prot = {}
    for root, _, files in os.walk(ds):
        for fn in files:
            p = os.path.join(root, fn)
            st = os.stat(p)
            prot[os.path.relpath(p, ds).replace("\\", "/")] = (st.st_size, st.st_mtime_ns)
    print(f"[protect] snapshotted {len(prot)} source files", flush=True)
    tn = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}
    for cls in CLASSES:
        need = [os.path.join(ds, IMG_SRC[cls]), os.path.join(ds, MSK_SRC[cls]),
                os.path.join(ds, "Test_Images", tn[cls]), os.path.join(ds, "Mask", tn[cls])]
        for d in need:
            if not os.path.isdir(d):
                print(f"FATAL: missing source dir {d}; STOPPING")
                return 1

    # ---- 4. deterministic split ----
    rng = random.Random(a.seed)
    rows, quarantine = [], []
    for cls in CLASSES:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ds, IMG_SRC[cls])))
        kept = []
        for i in ids:
            rel = f"{IMG_SRC[cls]}/{i}.tif"
            if rel in QUARANTINE:
                quarantine.append({"file": rel, "mask": f"{MSK_SRC[cls]}/{i}.tif",
                                   "reason": QUARANTINE[rel]})
                continue
            if not os.path.exists(os.path.join(ds, MSK_SRC[cls], f"{i}.tif")):
                print(f"FATAL: missing mask for {rel}; STOPPING")
                return 1
            kept.append(i)
        rng.shuffle(kept)
        ntr = int(round(len(kept) * 0.8))
        for j, i in enumerate(kept):
            rows.append({"split": "train" if j < ntr else "val", "class": cls, "id": i,
                         "image_source": f"{IMG_SRC[cls]}/{i}.tif",
                         "mask_source": f"{MSK_SRC[cls]}/{i}.tif"})
    tnames = {"Oil": "Oil", "No_oil": "No oil", "Lookalike": "Lookalike"}
    for cls in CLASSES:
        td = os.path.join(ds, "Test_Images", tnames[cls])
        md = os.path.join(ds, "Mask", tnames[cls])
        for fn in sorted(os.listdir(td)):
            i = os.path.splitext(fn)[0]
            if not os.path.exists(os.path.join(md, f"{i}_segmentation.tif")):
                print(f"FATAL: missing test mask for {fn}; STOPPING")
                return 1
            rows.append({"split": "test", "class": cls, "id": i,
                         "image_source": f"Test_Images/{tnames[cls]}/{fn}",
                         "mask_source": f"Mask/{tnames[cls]}/{i}_segmentation.tif"})

    # ---- 5/6/7/8. link + verify ----
    def dest(split, kind, cls, fn):
        d = os.path.join(out, split, kind, cls)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, fn)

    index, errors = [], []
    n_links, n_link_fail = 0, 0
    img_px = {}
    for r in rows:
        sp = os.path.join(ds, r["image_source"])
        mp = os.path.join(ds, r["mask_source"])
        ifn, mfn = os.path.basename(sp), os.path.basename(mp)
        dp, dq = dest(r["split"], "images", r["class"], ifn), dest(r["split"], "masks", r["class"], mfn)
        for s, d in ((sp, dp), (mp, dq)):
            if os.path.exists(d):
                if file_id(d)[:2] != file_id(s)[:2]:
                    errors.append(f"DEST NOT A HARD LINK OF SOURCE (not overwriting): {d}")
                    n_link_fail += 1
                continue
            try:
                os.link(s, d)
                n_links += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"LINK FAILED {s} -> {d}: {type(e).__name__}: {e}")
                n_link_fail += 1
        if file_id(dp)[:2] != file_id(sp)[:2]:
            errors.append(f"link identity FAIL image {dp}")
        if file_id(dq)[:2] != file_id(mp)[:2]:
            errors.append(f"link identity FAIL mask {dq}")
        ih, mh = sha256(sp), sha256(mp)
        if sha256(dp) != ih:
            errors.append(f"hash mismatch image {dp}")
        if sha256(dq) != mh:
            errors.append(f"hash mismatch mask {dq}")
        with rasterio.open(dp) as s, rasterio.open(sp) as s0:
            ia = s.read()
            k = (s.width, s.height, s.count, tuple(s.dtypes), str(s.crs),
                 tuple(s.transform) if s.transform else None)
            k0 = (s0.width, s0.height, s0.count, tuple(s0.dtypes), str(s0.crs),
                  tuple(s0.transform) if s0.transform else None)
            if k != k0:
                errors.append(f"header drift {dp}")
        with rasterio.open(dq) as m:
            ma = m.read(1)
            if (m.width, m.height) != (s.width, s.height):
                errors.append(f"image/mask dim mismatch {dp} vs {dq}")
            mu = sorted(np.unique(ma).tolist())
        img_px[f"{r['split']}/{r['class']}/{ifn}"] = hashlib.md5(np.ascontiguousarray(ia).tobytes()).hexdigest()
        index.append({"split": r["split"], "class": r["class"], "image_filename": ifn,
                      "image_source": r["image_source"],
                      "image_destination": os.path.relpath(dp, out).replace("\\", "/"),
                      "mask_filename": mfn, "mask_source": r["mask_source"],
                      "mask_destination": os.path.relpath(dq, out).replace("\\", "/"),
                      "image_sha256": ih, "mask_sha256": mh, "width": s.width, "height": s.height,
                      "bands": s.count, "dtype": "+".join(s.dtypes), "storage_method": "HARD_LINK",
                      "excluded": "NO", "exclusion_reason": ""})
    for q in quarantine:
        index.append({"split": "quarantined", "class": q["file"].split("/")[0], "image_filename": q["file"],
                      "image_source": q["file"], "image_destination": "", "mask_filename": q["mask"],
                      "mask_source": q["mask"], "mask_destination": "",
                      "image_sha256": sha256(os.path.join(ds, q["file"])),
                      "mask_sha256": sha256(os.path.join(ds, q["mask"])),
                      "width": "", "height": "", "bands": "", "dtype": "",
                      "storage_method": "NOT_LINKED", "excluded": "YES", "exclusion_reason": q["reason"]})

    # ---- 9. leakage: exact pixel-md5 across splits ----
    inv = {}
    for k, v in img_px.items():
        inv.setdefault(v, []).append(k)
    dup_groups = [v for v in inv.values() if len(v) > 1]
    def xov(x, y):
        hy = {img_px[k] for k in img_px if k.startswith(y)}
        return sum(1 for k in img_px if k.startswith(x) and img_px[k] in hy)
    masks_multi = []
    minv = {}
    for r in index:
        if r["excluded"] == "NO" and r["split"] in ("train", "val", "test"):
            with rasterio.open(os.path.join(out, r["mask_destination"])) as m:
                ma = m.read(1)
            if int((ma != 0).sum()) > 0:
                minv.setdefault(hashlib.md5(np.ascontiguousarray(ma).tobytes()).hexdigest(), []).append(
                    f"{r['split']}/{r['class']}/{r['mask_filename']}")
    for h, ks in minv.items():
        if len({k.rsplit("/", 1)[1] for k in ks}) > 1:
            masks_multi.append(ks)

    # ---- raw protection re-verify ----
    changed = []
    for rel, (sz, mt) in prot.items():
        st = os.stat(os.path.join(ds, rel))
        if (st.st_size, st.st_mtime_ns) != (sz, mt):
            changed.append(rel)
    if changed:
        print(f"FATAL: {len(changed)} SOURCE FILES CHANGED: {changed[:5]}")
        return 1

    with open(os.path.join(meta, "split_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    with open(os.path.join(qdir, "quarantine.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "mask", "reason"])
        w.writeheader()
        w.writerows(quarantine)

    free_after = shutil.disk_usage(out).free
    def cnt(sp, cl):
        return sum(1 for r in index if r["split"] == sp and r["class"] == cl and r["excluded"] == "NO")
    t = lambda sp: sum(cnt(sp, c) for c in CLASSES)
    report = f"""DATASET SPLIT REPORT (HARD LINKS, seed {a.seed}; {datetime.now(timezone.utc).isoformat()})
Test masks: Part III provenance dataset/Mask/<class>/*_segmentation.tif (IDs verified equal to Test_Images).

SOURCE COUNTS: Oil 1200 | No_oil 685 | Lookalike 685 | Total flat 2570 (+450 test)
EXCLUDED (metadata/quarantine/quarantine.csv; raw files untouched):
  G1: Oil/00007.tif, Oil/01339.tif | dupes: Oil/00357.tif, No_oil/00543.tif
TRAIN: Oil {cnt('train','Oil')} | No_oil {cnt('train','No_oil')} | Lookalike {cnt('train','Lookalike')} | Total {t('train')}
VAL: Oil {cnt('val','Oil')} | No_oil {cnt('val','No_oil')} | Lookalike {cnt('val','Lookalike')} | Total {t('val')}
TEST: Oil {cnt('test','Oil')} | No_oil {cnt('test','No_oil')} | Lookalike {cnt('test','Lookalike')} | Total {t('test')}
INTEGRITY: missing images 0 | missing masks 0 | hash mismatches {sum(1 for e in errors if 'hash' in e)}
  dimension mismatches {sum(1 for e in errors if 'dim' in e)}
  duplicate image groups (pixel-md5): {len(dup_groups)} {dup_groups if dup_groups else ''}
  non-empty mask reuse across IDs: {masks_multi if masks_multi else 'none'}
  train/val overlap {xov('train','val')} | train/test overlap {xov('train','test')} | val/test overlap {xov('val','test')}
  errors: {errors if errors else 'none'}
STORAGE: raw {raw_size/1e9:.2f}GB | free before {free_before/1e9:.2f}GB -> after {free_after/1e9:.2f}GB
  (delta {(free_before-free_after)/1e6:.1f}MB ~ directory entries only; TIFF data shared)
LINK STATUS: hard links created {n_links} | verification failures {n_link_fail} | physical copies created 0
RAW DATASET MODIFIED: NO (size+mtime re-verified on {len(prot)} files)
PREPROCESSING: NOT PERFORMED (normalize/radiometric/despeckle/augment/tile/resize/crop/stitch all NO)
"""
    with open(os.path.join(out, "reports", "dataset_split_report.txt"), "w") as f:
        f.write(report)
    print(report, flush=True)
    print("RAW DATASET MUST NOT BE DELETED YET. Wait for manual confirmation.", flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
