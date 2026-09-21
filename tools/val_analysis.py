"""Val-only deep analysis of the full-scene inference pipeline (items 1-8).

CONSTRAINTS HONOURED (v2 — see run_logs/val_analysis_v1_backup.py for v1)
------------------------------------------------------------------------
* VAL split only.  The test split is never referenced; there is no --allow-test
  flag here at all.  No training, no fitting, no threshold/"best config" freeze.
* No protected file and no inference-CLI default is modified: inference/,
  preprocessing/, training/ and models/ are only *imported*.  This script writes
  exclusively to results/val_analysis/ (plus an optional --output-dir).
* Each val scene is processed EXACTLY ONCE: `process_scene` runs once per scene
  and the preprocessed (2, H, W) array is reused for stride 512, 384 and 256 and
  for the autocast-OFF (fp32) control runs of that same scene.  v1 re-ran the
  whole scene a second time for item 4; v2 does not.
* Full-scene probability maps are NEVER stored between scenes.  Per scene we keep
  only (a) positive/negative probability histograms (1000 bins, in [0, 1]) and
  (b) small integer on-the-fly counts (TP/FP/FN, connected-component sizes,
  AMP-vs-fp32 label-change counts).  The (H, W) float32 map of the scene being
  processed is transient working memory released before the next scene starts,
  and the min-area filter is applied row-by-row so that no filtered full-scene
  map is ever materialised.
* Every reported number is MEASURED on this val pass.  Nothing in the outputs is
  extrapolated, estimated or copied from a previous run.

Binarisation convention (identical for every item):
    predicted positive  <=>  stitched probability >= threshold
  computed either directly (`prob_map >= thr`) or, for the histogram-based items,
  from the 1000-bin histograms as "pixels in bins whose left edge >= thr".
  `verification.json` proves the two paths agree exactly on every scene.

Outputs (default results/val_analysis/):
    item1_metrics.csv          precision/recall/pooled Dice/pooled IoU per stride x threshold
    item2_categories.json      per-category pooled Dice, false alarms, FP shares, detection
    item3_terciles.json        Oil-only pooled Dice by GT spill-area tercile
    item4_minarea.csv          min-area filter sweep (ANALYSIS ONLY, not added to the CLI)
    item5_bootstrap.json       paired bootstrap 95% CI for Dice differences
    item6_stride256.json       stride-256 vs stride-512 diagnosis at threshold 0.80
    item7_calibration.json     10-bin reliability table + over/under-confidence verdict
    item8_fp32_vs_amp.json     autocast ON vs OFF: pooled Dice + EXACT label-change fraction
    recommended_config.json    best grid point by pooled Dice (verdict text only)
    verification.json          internal cross-checks for this run
    pass_summary.json          run manifest (counts, timings, checkpoint hash, config)
    records/<scene>.npz        per-scene aggregate cache (resumable; histograms + counts only)

Usage:
    python tools/val_analysis.py                     # full val pass, all 8 items
    python tools/val_analysis.py --n-scenes 4        # smoke subset
    python tools/val_analysis.py --selftest-cc       # connected-component self-test only
    python tools/val_analysis.py --no-resume         # ignore the per-scene cache
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PREP_CFG = _ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
_TILE_IDX = _ROOT / "processed_dataset" / "metadata" / "tile_index.csv"
_CKPT     = _ROOT / "checkpoints" / "best_model.pth"

STRIDES    = [512, 384, 256]
THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]      # 0.50 .. 0.95
N_BINS     = 1000
BIN_EDGES  = np.linspace(0.0, 1.0, N_BINS + 1)

# Item 4 — min-area sweep.  Strides 512 and 384 are both swept because the grid
# maximum (512) and the inference-CLI default (384) disagree by less than the
# bootstrap noise in item 5; 256 is available via --item4-strides.
ITEM4_STRIDES    = [512, 384]
ITEM4_THRESHOLDS = [0.70, 0.80, 0.85]
ITEM4_NVALUES    = [0, 25, 100, 400, 1600]
ITEM4_CONN       = 8          # 8-connectivity (documented; see connected_components)

# Item 8 — "best configuration(s)" run with autocast OFF.  Both candidates are
# run so the answer does not depend on which one the reader calls best:
#   512@0.80 = highest pooled Dice in the item-1 grid (previous run: 0.61114)
#   384@0.80 = the inference CLI default stride
ITEM8_CFGS = [(512, 0.80), (384, 0.80)]

A_VALUES = [1, 50, 200, 1000]  # item 2 area/level thresholds
CATEGORIES = ["Oil", "Lookalike", "No_oil"]

SCHEMA = "val_analysis/2"      # bump when record semantics change (cache guard)


# ══════════════════════════════════════════════════════════════════════════
# Metric helpers (integer counts -> no per-batch averaging anywhere)
# ══════════════════════════════════════════════════════════════════════════

def _dice(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0


def _iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else 1.0


def _precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else 1.0


def _recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 1.0


def _thr_bin(thr: float) -> int:
    """First histogram bin index whose left edge is >= thr.

    Bin b covers [BIN_EDGES[b], BIN_EDGES[b+1]) except the last bin, which is
    closed.  np.histogram puts values exactly equal to an internal edge in the
    right-hand bin, so ``pixels in bins >= _thr_bin(thr)`` is exactly
    ``prob >= thr`` up to float64 representation of the edges.
    """
    return min(int(np.searchsorted(BIN_EDGES, thr, side="left")), N_BINS)


def _counts_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray, thr: float):
    """TP/FP/FN for one threshold from (1000-bin) positive/negative histograms."""
    tb = _thr_bin(thr)
    return (int(pos_hist[tb:].sum()), int(neg_hist[tb:].sum()), int(pos_hist[:tb].sum()))


def _histogram(prob: np.ndarray, gt: np.ndarray):
    """(pos_hist, neg_hist) with 1000 bins on [0, 1]; exact pixel accounting."""
    flat = prob.ravel()
    g = gt.ravel()
    pos = flat[g]
    neg = flat[~g]
    ph = np.histogram(pos, bins=BIN_EDGES)[0].astype(np.int64)
    nh = np.histogram(neg, bins=BIN_EDGES)[0].astype(np.int64)
    return ph, nh


def _tercile_labels(areas: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Rank-based terciles (equal-count groups, no boundary ambiguity).

    Returns (labels with 0=small/1=medium/2=large, [max_small, min_large]).
    """
    order = np.argsort(areas, kind="stable")
    n = areas.size
    labels = np.empty(n, dtype=np.int64)
    k1 = n // 3
    k2 = n - n // 3
    labels[order[:k1]] = 0
    labels[order[k1:k2]] = 1
    labels[order[k2:]] = 2
    max_small = int(areas[order[k1 - 1]]) if k1 else 0
    min_large = int(areas[order[k2]]) if k2 < n else 0
    return labels, [max_small, min_large]


def _pct(num: int, den: int) -> float:
    """num / den as a 4-dp fraction (0.0 when the denominator is empty)."""
    return round(num / den, 4) if den else 0.0



# ══════════════════════════════════════════════════════════════════════════
# Connected components — pure numpy (no scipy in this environment)
# ══════════════════════════════════════════════════════════════════════════
#
# Run-length + union-find, no Python loop over pixels:
#   1. every row is decomposed into horizontal runs of True;
#   2. runs of consecutive rows that touch (overlap, or are 1 column apart when
#      connectivity == 8) are unioned;
#   3. union-find roots are resolved by repeated vectorised pointer jumping
#      (no per-element Python loop), so sizes are run-length sums per root;
#   4. `min_area_counts` walks the image row by row a second time and counts
#      TP/FP/FN directly from the surviving runs, so no filtered full-scene map
#      is ever materialised.
#
# Equivalence with scipy.ndimage.label (4-conn) and with 8-connected flood fill
# is proven by `--selftest-cc` against a brute-force reference implementation.

def _row_runs(mask_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inclusive (start, end) column indices of the True runs of one row."""
    m = np.empty(mask_row.size + 2, dtype=bool)
    m[0] = False
    m[-1] = False
    m[1:-1] = mask_row
    d = m[1:].astype(np.int8) - m[:-1].astype(np.int8)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def _expand(counts: np.ndarray) -> np.ndarray:
    """np.arange(total) - repeat(run offsets): ragged per-group index generator."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - offs


def _run_graph(binary: np.ndarray, connectivity: int):
    """Row runs plus the union-find edges between runs of consecutive rows."""
    H = binary.shape[0]
    tol = 1 if connectivity == 8 else 0
    row_starts: List[np.ndarray] = []
    row_ends: List[np.ndarray] = []
    for y in range(H):
        s, e = _row_runs(binary[y])
        row_starts.append(s)
        row_ends.append(e)
    run_counts = np.array([s.size for s in row_starts], dtype=np.int64)
    run_off = np.concatenate(([0], np.cumsum(run_counts)))
    n_runs = int(run_off[-1])
    pair_a: List[np.ndarray] = []
    pair_b: List[np.ndarray] = []
    for y in range(H - 1):
        cs, ce = row_starts[y], row_ends[y]
        ns, ne = row_starts[y + 1], row_ends[y + 1]
        if cs.size == 0 or ns.size == 0:
            continue
        # per-run candidate slice of the row above (both arrays are sorted, so the
        # overlapping runs form one contiguous slice)
        lo = np.searchsorted(ce, ns - tol, side="left")
        hi = np.searchsorted(cs, ne + tol, side="right")
        cnt = np.maximum(hi - lo, 0)
        if not np.any(cnt > 0):
            continue
        pair_a.append(run_off[y] + np.repeat(lo, cnt) + _expand(cnt))
        pair_b.append(run_off[y + 1] + np.repeat(np.arange(ns.size), cnt))
    return row_starts, row_ends, run_counts, run_off, n_runs, pair_a, pair_b


def _resolve_parents(n_runs: int, pair_a: List[np.ndarray], pair_b: List[np.ndarray]):
    parent = np.arange(n_runs, dtype=np.int64)
    if not pair_a:
        return parent
    pa = np.concatenate(pair_a).tolist()
    pb = np.concatenate(pair_b).tolist()
    pl = parent.tolist()
    for a, b in zip(pa, pb):          # one bound-refinement per graph edge
        ra = a
        while pl[ra] != ra:
            pl[ra] = pl[pl[ra]]
            ra = pl[ra]
        rb = b
        while pl[rb] != rb:
            pl[rb] = pl[pl[rb]]
            rb = pl[rb]
        if ra != rb:
            if ra < rb:
                pl[rb] = ra
            else:
                pl[ra] = rb
    parent = np.asarray(pl, dtype=np.int64)
    for _ in range(64):               # vectorised root resolution to a fixed point
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    else:                             # pragma: no cover - defensive
        raise RuntimeError("union-find did not converge")
    return parent



def _roots_and_weights(binary: np.ndarray, gt: np.ndarray, connectivity: int):
    """Union-find roots + per-root run length and per-root GT-positive count.

    Returns (roots_indexed_by_root_id, lengths_per_root, gtpos_per_root).
    Entries for ids that are not roots are 0 (callers index by root id only).
    """
    (rs, re_, _, _, n_runs, pa, pb) = _run_graph(binary, connectivity)
    if n_runs == 0:
        z = np.zeros(1, dtype=np.float64)
        return z.astype(np.int64), z, z
    parent = _resolve_parents(n_runs, pa, pb)
    rows = [y for y in range(binary.shape[0]) if rs[y].size]
    S = np.concatenate([rs[y] for y in rows])
    E = np.concatenate([re_[y] for y in rows])
    R = np.repeat(np.asarray(rows, dtype=np.int64), [rs[y].size for y in rows])
    cs = np.cumsum(gt, axis=1, dtype=np.int64)          # per-row GT prefix sums
    left = np.where(S > 0, cs[R, np.maximum(S - 1, 0)], 0)
    gtpos = (cs[R, E] - left).astype(np.float64)
    lengths = (E - S + 1).astype(np.float64)
    n_ids = int(parent.max()) + 1
    return (parent,
            np.bincount(parent, weights=lengths, minlength=n_ids),
            np.bincount(parent, weights=gtpos, minlength=n_ids))


def connected_component_sizes(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Size of EVERY connected component (4- or 8-connectivity).

    Returns a 1-D int64 array of component sizes; isolated pixels are size-1
    components.  Order of the sizes is unspecified (callers sort if needed).
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    binary = np.ascontiguousarray(binary, dtype=bool)
    gt = np.zeros_like(binary)
    _, sizes, _ = _roots_and_weights(binary, gt, connectivity)
    if not binary.any():
        return np.zeros(0, dtype=np.int64)
    return sizes[sizes > 0].astype(np.int64)             # drop non-root ids


def min_area_sweep(binary: np.ndarray, gt: np.ndarray,
                   n_values: Sequence[int],
                   connectivity: int = 8) -> Dict[int, Tuple[int, int, int, int]]:
    """One connected-component pass for ALL n_values at once.

    For every n_min in `n_values`: remove connected components smaller than
    n_min pixels, then return (TP, FP, FN, predicted-positive) against `gt`.
    Components are counted through per-root run lengths and per-root GT-positive
    sums, so neither the label map nor the filtered map is ever materialised.
    """
    binary = np.ascontiguousarray(binary, dtype=bool)
    gt = np.ascontiguousarray(gt, dtype=bool)
    gt_total = int(np.count_nonzero(gt))
    out: Dict[int, Tuple[int, int, int, int]] = {}

    small = [n for n in n_values if n <= 1]
    if small:
        tp = int(np.count_nonzero(binary & gt))
        pred = int(np.count_nonzero(binary))
        for n in small:
            out[n] = (tp, pred - tp, gt_total - tp, pred)
    rest = [n for n in n_values if n > 1]
    if not rest:
        return out

    _, sizes, gtpos = _roots_and_weights(binary, gt, connectivity)
    for n in rest:
        keep = sizes >= n                                # indexed by root id
        pred = int(sizes[keep].sum())
        tp = int(gtpos[keep].sum())
        out[n] = (tp, pred - tp, gt_total - tp, pred)
    return out


def min_area_counts(binary: np.ndarray, gt: np.ndarray, n_min: int,
                    connectivity: int = 8) -> Tuple[int, int, int, int]:
    """Single-n_min convenience wrapper around `min_area_sweep`."""
    return min_area_sweep(binary, gt, [n_min], connectivity=connectivity)[n_min]



# ── brute-force reference for the self-test (small images only) ────────────

def _brute_force_components(binary: np.ndarray, connectivity: int) -> List[List[Tuple[int, int]]]:
    H, W = binary.shape
    nbrs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    seen = np.zeros((H, W), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for y0 in range(H):
        for x0 in range(W):
            if not binary[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in nbrs:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and binary[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            comps.append(comp)
    return comps


def _brute_force_filter(binary: np.ndarray, n_min: int, connectivity: int) -> np.ndarray:
    keep = np.zeros_like(binary, dtype=bool)
    if n_min <= 1:
        return binary.copy()
    for comp in _brute_force_components(binary, connectivity):
        if len(comp) >= n_min:
            for (y, x) in comp:
                keep[y, x] = True
    return keep


def selftest_cc(verbose: bool = True) -> dict:
    """Prove the numpy connected components match the brute-force reference."""
    rng = np.random.default_rng(0)
    size_cases = 0
    for shape in [(1, 1), (1, 17), (17, 1), (9, 13), (24, 31)]:
        for density in (0.15, 0.5, 0.85):
            for _ in range(4):
                b = rng.random(shape) < density
                for conn in (4, 8):
                    got = np.sort(connected_component_sizes(b, connectivity=conn))
                    ref = np.sort(np.asarray(
                        [len(c) for c in _brute_force_components(b, conn)], dtype=np.int64))
                    assert np.array_equal(got, ref), (
                        f"CC mismatch shape={shape} conn={conn} density={density}\n"
                        f"got={got}\nref={ref}")
                    size_cases += 1

    # diagonal-only and touching-only shapes (the 4/8 distinction must bite)
    shapes = {
        "diag": np.array([[1, 0], [0, 1]], dtype=bool),
        "edge": np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        "one":  np.array([[1]], dtype=bool),
        "empty": np.zeros((3, 3), dtype=bool),
    }
    for name, b in shapes.items():
        for conn in (4, 8):
            got = np.sort(connected_component_sizes(b, connectivity=conn))
            ref = np.sort(np.asarray(
                [len(c) for c in _brute_force_components(b, conn)], dtype=np.int64))
            assert np.array_equal(got, ref), f"{name} conn={conn}: {got} != {ref}"
            size_cases += 1
    assert np.array_equal(np.sort(connected_component_sizes(shapes["diag"], 4)), [1, 1])
    assert np.array_equal(np.sort(connected_component_sizes(shapes["diag"], 8)), [2])

    count_cases = 0
    for _ in range(6):
        b = rng.random((21, 23)) < 0.45
        gt = rng.random((21, 23)) < 0.4
        for conn in (4, 8):
            for n_min in (0, 1, 2, 5, 10, 40):
                keep = _brute_force_filter(b, n_min, conn)
                tp, fp, fn, pred = min_area_counts(b, gt, n_min, connectivity=conn)
                exp = (int(np.count_nonzero(keep & gt)),
                       int(np.count_nonzero(keep & ~gt)),
                       int(np.count_nonzero(gt & ~keep)),
                       int(np.count_nonzero(keep)))
                assert (tp, fp, fn, pred) == exp, (
                    f"min_area_counts mismatch conn={conn} n_min={n_min}: "
                    f"{(tp, fp, fn, pred)} != {exp}")
                count_cases += 1

    out = {"status": "PASS", "component_size_cases": size_cases,
           "min_area_count_cases": count_cases,
           "reference": "brute-force stack flood fill (independent implementation)"}
    if verbose:
        print(f"[selftest-cc] PASS  {size_cases} component-size cases + "
              f"{count_cases} min-area count cases match the brute-force reference")
    return out



# ══════════════════════════════════════════════════════════════════════════
# Scene inventory (VAL split only) + per-scene single-pass inference
# ══════════════════════════════════════════════════════════════════════════

def _load_scene_map(n_scenes: Optional[int]) -> "Dict[str, List[dict]]":
    """{image_id: [tile rows]} for split == 'val' only, sorted by image_id."""
    with open(_TILE_IDX) as f:
        rows = list(csv.DictReader(f))
    m: Dict[str, List[dict]] = {}
    for r in rows:
        if r["split"] == "val":
            m.setdefault(r["image_id"], []).append(r)
    ids = sorted(m.keys())
    if n_scenes:
        ids = ids[:n_scenes]
    return {sid: m[sid] for sid in ids}


def _forward(model, origins, patches, stitcher, device, amp: bool, force_fp32: bool):
    """One batched forward pass; sigmoid applied exactly once; no thresholding."""
    import torch
    tensor = torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
    with torch.amp.autocast("cuda", enabled=(amp and not force_fp32)):
        logits = model(tensor)
    probs = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()
    for (y, x), p in zip(origins, probs):
        stitcher.add_tile(y, x, p)


def _stitch_stride(model, norm, stride, batch_size, device, amp, force_fp32=False):
    """Sliding-window stitch of one stride for the already-preprocessed scene."""
    import torch
    from inference.windows import tile_grid, TILE_SIZE
    from inference.stitching import ProbabilityStitcher

    H, W = norm.shape[1], norm.shape[2]
    origins = tile_grid(H, W, tile=TILE_SIZE, stride=stride)
    stitcher = ProbabilityStitcher(H, W, TILE_SIZE)
    with torch.inference_mode():
        buf_o, buf_p = [], []
        for (y, x) in origins:
            buf_o.append((y, x))
            buf_p.append(norm[:, y:y + TILE_SIZE, x:x + TILE_SIZE])
            if len(buf_p) == batch_size:
                _forward(model, buf_o, buf_p, stitcher, device, amp, force_fp32)
                buf_o, buf_p = [], []
        if buf_p:
            _forward(model, buf_o, buf_p, stitcher, device, amp, force_fp32)
    return stitcher.finalize()



def process_scene_once(model, bounds, cfg, rep_row, batch_size, device, amp,
                       strides=STRIDES, item4_strides=ITEM4_STRIDES,
                       item4_thresholds=ITEM4_THRESHOLDS, item4_n=ITEM4_NVALUES,
                       item8_cfgs=ITEM8_CFGS, item4_conn=ITEM4_CONN):
    """THE single pass over one val scene.

    Preprocesses the scene once, then reuses that array for every stride, for the
    min-area sweep and for the autocast-OFF control runs.  Returns aggregates only
    (histograms + integer counts); the probability maps are dropped before return.
    """
    from preprocessing.pipeline import process_scene

    ipath = str(_ROOT / "processed_dataset" / rep_row["image_path"])
    mpath = str(_ROOT / "processed_dataset" / rep_row["mask_path"])

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = process_scene(ipath, mpath, cfg=cfg, bounds=bounds, split="val",
                            augment_override=False, compute_stats=False)
    t_prep = time.time() - t0

    norm = out["normalized"]                       # (2, H, W) float32
    gt = np.ascontiguousarray(out["mask"] > 0)     # (H, W) bool
    gt_total = int(gt.size)
    gt_positive = int(np.count_nonzero(gt))

    hists: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    tf = np.zeros((len(strides), len(THRESHOLDS), 3), dtype=np.int64)
    i4_cfgs = [(s, t) for s in item4_strides for t in item4_thresholds]
    i4 = np.zeros((len(i4_cfgs), len(item4_n), 4), dtype=np.int64)  # tp, fp, fn, pred
    # item 8 columns: amp tp/fp/fn, fp32 tp/fp/fn, exact label changes, intersection
    i8 = np.zeros((len(item8_cfgs), 8), dtype=np.int64)
    i8f = np.zeros((len(item8_cfgs), 2), dtype=np.float64)
    i4_idx = {c: k for k, c in enumerate(i4_cfgs)}
    i8_idx = {c: k for k, c in enumerate(item8_cfgs)}
    checks = {"hist_vs_direct_pairs": 0, "hist_vs_direct_mismatch": 0,
              "hist_mismatch_detail": []}

    t_model = 0.0
    for si, stride in enumerate(strides):
        t1 = time.time()
        prob = _stitch_stride(model, norm, stride, batch_size, device, amp)
        t_model += time.time() - t1

        ph, nh = _histogram(prob, gt)
        assert int(ph.sum()) + int(nh.sum()) == gt_total, "histogram pixel accounting"
        hists[stride] = (ph, nh)
        for ti, thr in enumerate(THRESHOLDS):
            tf[si, ti] = _counts_from_hist(ph, nh, thr)

        for (s4, t4) in i4_cfgs:
            if s4 != stride:
                continue
            binary = prob >= t4                       # pipeline binarisation
            res = min_area_sweep(binary, gt, item4_n, connectivity=item4_conn)
            row = i4_idx[(s4, t4)]
            for ni, n in enumerate(item4_n):
                i4[row, ni] = res[n]
            if round(t4, 2) in THRESHOLDS:            # N=0 must match the histogram
                ti = THRESHOLDS.index(round(t4, 2))
                htp, hfp, hfn = _counts_from_hist(ph, nh, t4)
                dtp, dfp, dfn, _ = res[0]
                checks["hist_vs_direct_pairs"] += 1
                if (htp, hfp, hfn) != (dtp, dfp, dfn):
                    checks["hist_vs_direct_mismatch"] += 1
                    checks["hist_mismatch_detail"].append(
                        {"stride": s4, "threshold": t4,
                         "hist": [htp, hfp, hfn], "direct": [dtp, dfp, dfn]})
            del binary

        for (s8, t8) in item8_cfgs:
            if s8 != stride:
                continue
            prob_fp32 = _stitch_stride(model, norm, stride, batch_size, device,
                                       amp, force_fp32=True)
            k = i8_idx[(s8, t8)]
            ba = prob >= t8
            bf = prob_fp32 >= t8
            i8[k, 0] = int(np.count_nonzero(ba & gt))
            i8[k, 1] = int(np.count_nonzero(ba & ~gt))
            i8[k, 2] = gt_positive - i8[k, 0]
            i8[k, 3] = int(np.count_nonzero(bf & gt))
            i8[k, 4] = int(np.count_nonzero(bf & ~gt))
            i8[k, 5] = gt_positive - i8[k, 3]
            i8[k, 6] = int(np.count_nonzero(ba != bf))     # exact label changes
            i8[k, 7] = int(np.count_nonzero(ba & bf))      # AMP ∩ fp32
            diff = np.abs(prob - prob_fp32)
            i8f[k, 0] = float(diff.mean())
            i8f[k, 1] = float(diff.max())
            del prob_fp32, ba, bf, diff

        del prob
    del norm

    timing = {"prep_s": round(t_prep, 3), "model_s": round(t_model, 3),
              "total_s": round(t_prep + t_model, 3)}
    return {"gt_total": gt_total, "gt_positive": gt_positive, "hists": hists,
            "tf": tf, "item4": i4, "item8": i8, "item8_f": i8f,
            "timing": timing, "checks": checks}



# ══════════════════════════════════════════════════════════════════════════
# Per-scene aggregate record + resumable cache (histograms and counts only)
# ══════════════════════════════════════════════════════════════════════════

class SceneRecord:
    """Everything the eight items need, per scene, with no maps stored."""

    __slots__ = ("sid", "category", "gt_total", "gt_positive", "hists",
                 "tf", "item4", "item8", "item8_f", "timing", "checks")

    def __init__(self, sid, category, agg):
        self.sid = sid
        self.category = category
        self.gt_total = int(agg["gt_total"])
        self.gt_positive = int(agg["gt_positive"])
        self.hists: Dict[int, Tuple[np.ndarray, np.ndarray]] = agg["hists"]
        self.tf: np.ndarray = np.asarray(agg["tf"], dtype=np.int64)
        self.item4: np.ndarray = np.asarray(agg["item4"], dtype=np.int64)
        self.item8: np.ndarray = np.asarray(agg["item8"], dtype=np.int64)
        self.item8_f: np.ndarray = np.asarray(agg["item8_f"], dtype=np.float64)
        self.timing: dict = agg.get("timing", {})
        self.checks: dict = agg.get("checks", {})

    # ── query helpers ─────────────────────────────────────────────────────
    def tf_at(self, stride: int, thr: float) -> Tuple[int, int, int]:
        """TP/FP/FN at (stride, threshold) from the stored histograms."""
        ph, nh = self.hists[stride]
        return _counts_from_hist(ph, nh, thr)

    def pred_at(self, stride: int, thr: float) -> int:
        tp, fp, _ = self.tf_at(stride, thr)
        return tp + fp

    def tf_grid(self, stride: int, thr: float) -> Tuple[int, int, int]:
        """TP/FP/FN from the precomputed 10-threshold grid (must be on-grid)."""
        si = STRIDES.index(stride)
        ti = THRESHOLDS.index(round(thr, 2))
        a, b, c = self.tf[si, ti]
        return int(a), int(b), int(c)


def _record_path(cache_dir: Path, sid: str) -> Path:
    return cache_dir / (sid.replace("/", "__").replace("\\\\", "__") + ".npz")


def save_record(path: Path, rec: SceneRecord, meta: dict) -> None:
    arrs = {"tf": rec.tf, "item4": rec.item4,
            "item8": rec.item8, "item8_f": rec.item8_f,
            "meta": np.array(json.dumps({**meta, "sid": rec.sid,
                                         "category": rec.category,
                                         "gt_total": rec.gt_total,
                                         "gt_positive": rec.gt_positive,
                                         "timing": rec.timing,
                                         "checks": rec.checks}))}
    for stride, (ph, nh) in rec.hists.items():
        arrs[f"pos_{stride}"] = ph
        arrs[f"neg_{stride}"] = nh
    np.savez_compressed(path, **arrs)


def load_record(path: Path, expect: dict) -> Optional[SceneRecord]:
    """Load a cached record, refusing anything that does not match `expect`."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    meta = json.loads(str(z["meta"]))
    for k, v in expect.items():
        if meta.get(k) != v:
            return None
    hists = {s: (z[f"pos_{s}"].astype(np.int64), z[f"neg_{s}"].astype(np.int64))
             for s in STRIDES if f"pos_{s}" in z.files}
    agg = {"gt_total": meta["gt_total"], "gt_positive": meta["gt_positive"],
           "hists": hists, "tf": z["tf"], "item4": z["item4"],
           "item8": z["item8"], "item8_f": z["item8_f"],
           "timing": meta.get("timing", {}), "checks": meta.get("checks", {})}
    return SceneRecord(meta["sid"], meta["category"], agg)



# ══════════════════════════════════════════════════════════════════════════
# MAIN — one val pass, then all eight items from the per-scene records
# ══════════════════════════════════════════════════════════════════════════

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_pass(ns):
    import torch
    from models import UNetResNet34, UNetResNet34Config
    from preprocessing import with_overrides

    out_dir = Path(ns.output_dir)
    cache_dir = out_dir / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(ns.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"

    ckpt_sha = _sha256(_CKPT)
    cache_expect = {"schema": SCHEMA, "strides": STRIDES,
                    "thresholds": THRESHOLDS, "item4_thresholds": ns.item4_thresholds,
                    "item4_n": list(ITEM4_NVALUES), "item4_strides": list(ns.item4_strides),
                    "conn": ITEM4_CONN, "item8_cfgs": [list(c) for c in ITEM8_CFGS],
                    "ckpt_sha256": ckpt_sha}

    print(f"[analysis] device={device}  amp={amp}  batch={ns.batch_size}")
    print(f"[analysis] checkpoint sha256={ckpt_sha[:16]}…")
    ck = torch.load(str(_CKPT), map_location=device, weights_only=False)
    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1,
                                            pretrained=False))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()

    pcfg = json.load(open(_PREP_CFG))
    bounds = {"bands": pcfg["statistics"]["bands"]}
    cfg = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5)

    scene_map = _load_scene_map(ns.n_scenes)
    n = len(scene_map)
    print(f"[analysis] val scenes: {n}   strides: {STRIDES}   item4 strides: "
          f"{list(ns.item4_strides)}   item8 cfgs: {ITEM8_CFGS}")

    records: List[SceneRecord] = []
    checks = {"hist_vs_direct_pairs": 0, "hist_vs_direct_mismatch": 0,
              "hist_mismatch_detail": [], "cache_hits": 0}
    t0 = time.time()
    tsum_prep = tsum_model = 0.0

    for i, (sid, rows) in enumerate(scene_map.items()):
        rep = rows[0]
        cat = rep["class"]                       # Oil / Lookalike / No_oil
        meta = {k: v for k, v in cache_expect.items()}
        rec = None
        if not ns.no_resume:
            rec = load_record(_record_path(cache_dir, sid), cache_expect)
            if rec is not None:
                checks["cache_hits"] += 1
        if rec is None:
            agg = process_scene_once(
                model, bounds, cfg, rep, ns.batch_size, device, amp,
                strides=STRIDES, item4_strides=list(ns.item4_strides),
                item4_thresholds=list(ns.item4_thresholds),
                item4_n=list(ITEM4_NVALUES), item8_cfgs=ITEM8_CFGS)
            rec = SceneRecord(sid, cat, agg)
            save_record(_record_path(cache_dir, sid), rec, meta)

        c = rec.checks or {}
        checks["hist_vs_direct_pairs"] += int(c.get("hist_vs_direct_pairs", 0))
        checks["hist_vs_direct_mismatch"] += int(c.get("hist_vs_direct_mismatch", 0))
        checks["hist_mismatch_detail"] += list(c.get("hist_mismatch_detail", []))
        tsum_prep += float(rec.timing.get("prep_s", 0.0))
        tsum_model += float(rec.timing.get("model_s", 0.0))
        records.append(rec)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == n:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n}  {sid}  prep={rec.timing.get('prep_s', 0):.1f}s "
                  f"model={rec.timing.get('model_s', 0):.1f}s  "
                  f"elapsed={el/60:.1f}min  ETA={eta/60:.1f}min", flush=True)
    pass_s = time.time() - t0
    print(f"[analysis] pass complete in {pass_s/60:.1f} min "
          f"(cache hits {checks['cache_hits']})")

    summary = {
        "schema": SCHEMA,
        "measured_on": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": "val",
        "val_scenes": n,
        "cache_hits": checks["cache_hits"],
        "pass_seconds": round(pass_s, 1),
        "sum_prep_seconds": round(tsum_prep, 1),
        "sum_model_seconds": round(tsum_model, 1),
        "strides": STRIDES,
        "thresholds": THRESHOLDS,
        "tile_size": 512,
        "samples_per_pixel": "uniform-average stitching (inference/stitching.py)",
        "checkpoint": {"path": str(_CKPT), "sha256": ckpt_sha,
                       "epoch": int(ck.get("epoch", -1)),
                       "best_metric_name": ck.get("best_metric_name"),
                       "best_metric": float(ck.get("best_metric", float("nan")))},
        "device": str(device), "amp": amp, "batch_size": ns.batch_size,
        "preprocessing": {"pipeline": "preprocessing.pipeline.process_scene",
                          "lee_window": 5, "compute_stats": False,
                          "bounds": bounds["bands"], "augmentation": False},
        "item4": {"strides": list(ns.item4_strides),
                  "thresholds": list(ns.item4_thresholds),
                  "min_areas_px": list(ITEM4_NVALUES),
                  "connectivity": ITEM4_CONN,
                  "note": "ANALYSIS ONLY — the inference CLI is unchanged"},
        "item8": {"configs": [list(c) for c in ITEM8_CFGS],
                  "note": "autocast OFF (fp32) vs autocast ON (AMP), same stitched "
                          "window geometry; label changes counted exactly per pixel"},
    }
    return records, checks, summary, out_dir



# ══════════════════════════════════════════════════════════════════════════
# Item 1 — precision / recall / pooled Dice / pooled IoU per stride x threshold
# ══════════════════════════════════════════════════════════════════════════

def compute_item1(records: List[SceneRecord]) -> List[dict]:
    rows = []
    for stride in STRIDES:
        for thr in THRESHOLDS:
            tp = fp = fn = 0
            for rec in records:
                a, b, c = rec.tf_grid(stride, thr)
                tp += a; fp += b; fn += c
            rows.append(dict(
                stride=stride, threshold=thr,
                pooled_dice=round(_dice(tp, fp, fn), 5),
                pooled_iou=round(_iou(tp, fp, fn), 5),
                precision=round(_precision(tp, fp), 5),
                recall=round(_recall(tp, fn), 5),
                tp=tp, fp=fp, fn=fn))
    return rows


def print_item1(rows):
    print(f"  {'stride':>6} {'thr':>5} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Rec':>7}")
    print("  " + "-" * 45)
    for r in rows:
        print(f"  {r['stride']:>6} {r['threshold']:>5.2f} {r['pooled_dice']:>7.4f} "
              f"{r['pooled_iou']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Item 2 — per-category analysis
#    category := the `class` column of processed_dataset/metadata/tile_index.csv
#    ("Oil", "Lookalike", "No_oil"), which is the split manifest built from the
#    dataset directory names.  No category is inferred from predictions.
# ══════════════════════════════════════════════════════════════════════════

def compute_item2(records: List[SceneRecord], stride: int) -> dict:
    oil = [r for r in records if r.category == "Oil"]
    look = [r for r in records if r.category == "Lookalike"]
    noil = [r for r in records if r.category == "No_oil"]
    out = {
        "stride": stride,
        "category_source": "tile_index.csv column 'class' (directory name of the "
                           "val scene); no category is derived from predictions",
        "category_counts": {"Oil": len(oil), "Lookalike": len(look),
                            "No_oil": len(noil)},
        "per_threshold": [],
    }
    for thr in THRESHOLDS:
        tp_o = fp_o = fn_o = 0
        for rec in oil:
            a, b, c = rec.tf_grid(stride, thr)
            tp_o += a; fp_o += b; fn_o += c
        fp_cat = {c: 0 for c in CATEGORIES}
        for rec in records:
            _, fp, _ = rec.tf_grid(stride, thr)
            fp_cat[rec.category] += fp
        fp_all = sum(fp_cat.values())

        def fa(recs):
            d = {}
            for A in A_VALUES:
                n_ge = sum(1 for r in recs if r.pred_at(stride, thr) >= A)
                d[str(A)] = {"scenes_ge_A": n_ge, "fraction": _pct(n_ge, len(recs))}
            return d

        def detection(recs):
            d = {}
            for A in A_VALUES:
                ge = [r for r in recs if r.pred_at(stride, thr) >= A]
                ov = [r for r in ge if r.tf_grid(stride, thr)[0] > 0]
                d[str(A)] = {
                    "scenes_with_pred_area_ge_A": len(ge),
                    "scenes_with_overlap": len(ov),
                    "fraction_of_those_overlapping": _pct(len(ov), len(ge)),
                    "fraction_of_all_oil_scenes": _pct(len(ov), len(recs)),
                }
            return d

        out["per_threshold"].append({
            "stride": stride,
            "threshold": thr,
            "oil_pooled_dice": round(_dice(tp_o, fp_o, fn_o), 5),
            "oil_pooled_iou": round(_iou(tp_o, fp_o, fn_o), 5),
            "oil_precision": round(_precision(tp_o, fp_o), 5),
            "oil_recall": round(_recall(tp_o, fn_o), 5),
            "lookalike_false_alarm": fa(look),
            "no_oil_false_alarm": fa(noil),
            "non_oil_false_alarm": fa(look + noil),
            "fp_pixels_by_category": fp_cat,
            "fp_share_by_category": {c: _pct(fp_cat[c], fp_all) for c in CATEGORIES},
            "oil_detection": detection(oil),
        })
    return out


def print_item2(item2):
    print(f"  stride={item2['stride']}  categories: {item2['category_counts']}  "
          f"(from {item2['category_source'].split('(')[0].strip()})")
    print(f"  {'thr':>5} {'OilDice':>8} {'FA_Look>=1':>11} {'FA_NoOil>=1':>12} "
          f"{'FA_NonOil>=1':>13} {'shareOil':>9} {'shareLook':>10} {'detect>=200':>11}")
    print("  " + "-" * 84)
    for r in item2["per_threshold"]:
        print(f"  {r['threshold']:>5.2f} {r['oil_pooled_dice']:>8.4f} "
              f"{r['lookalike_false_alarm']['1']['fraction']:>11.4f} "
              f"{r['no_oil_false_alarm']['1']['fraction']:>12.4f} "
              f"{r['non_oil_false_alarm']['1']['fraction']:>13.4f} "
              f"{r['fp_share_by_category']['Oil']:>9.4f} "
              f"{r['fp_share_by_category']['Lookalike']:>10.4f} "
              f"{r['oil_detection']['200']['fraction_of_all_oil_scenes']:>11.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Item 3 — Oil-only pooled Dice by GT spill-area tercile (rank-based,
# equal-count groups via _tercile_labels; stride 512, all thresholds).
# ══════════════════════════════════════════════════════════════════════════

ITEM3_STRIDE = 512


def compute_item3(records: List[SceneRecord], stride: int = ITEM3_STRIDE) -> dict:
    oil = [r for r in records if r.category == "Oil"]
    if not oil:
        return {"stride": stride, "error": "no Oil scenes in this subset",
                "per_threshold": []}
    areas = np.array([r.gt_positive for r in oil], dtype=np.int64)
    labels, bounds = _tercile_labels(areas)
    groups = {0: [r for r, lb in zip(oil, labels) if lb == 0],
              1: [r for r, lb in zip(oil, labels) if lb == 1],
              2: [r for r, lb in zip(oil, labels) if lb == 2]}
    out = {
        "stride": stride,
        "measured": True,
        "tercile_rule": "rank-based equal-count terciles over Oil GT-positive "
                        "pixel areas (no boundary ambiguity)",
        "tercile_boundaries_px": bounds,   # [max_small, min_large]
        "n": {"small": len(groups[0]), "medium": len(groups[1]),
              "large": len(groups[2])},
        "per_threshold": [],
    }
    for thr in THRESHOLDS:
        row: dict = {"threshold": thr}
        for key, gid in (("small", 0), ("medium", 1), ("large", 2)):
            tp = fp = fn = 0
            for rec in groups[gid]:
                a, b, c = rec.tf_grid(stride, thr)
                tp += a; fp += b; fn += c
            row[f"dice_{key}"] = round(_dice(tp, fp, fn), 5)
            row[f"n_{key}"] = len(groups[gid])
        out["per_threshold"].append(row)
    return out


def print_item3(item3):
    if "error" in item3:
        print(f"  ({item3['error']})")
        return
    print(f"  stride={item3['stride']}  {item3['tercile_rule']}")
    print(f"  boundaries [max_small, min_large] = {item3['tercile_boundaries_px']}  "
          f"n = {item3['n']}  (all MEASURED)")
    print(f"  {'thr':>5} {'small':>8} {'medium':>8} {'large':>8}")
    print("  " + "-" * 34)
    for r in item3["per_threshold"]:
        print(f"  {r['threshold']:>5.2f} {r['dice_small']:>8.4f} "
              f"{r['dice_medium']:>8.4f} {r['dice_large']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Item 4 — min-area filter sweep from the single-pass per-scene counts.
# No re-inference: rec.item4[row(cfg), col(N)] = (tp, fp, fn, pred) measured
# during process_scene_once.  ANALYSIS ONLY — the inference CLI is unchanged.
# False-alarm rate := fraction of non-oil (Lookalike + No_oil) scenes with
# predicted-positive area >= 1 after filtering.  Dice/precision/recall are
# pooled over ALL val scenes.  (false_alarm_rate_all uses the v1-compatible
# denominator of all scenes for comparability.)
# ══════════════════════════════════════════════════════════════════════════

def compute_item4(records: List[SceneRecord],
                  item4_strides, item4_thresholds, item4_n) -> List[dict]:
    i4_cfgs = [(s, t) for s in item4_strides for t in item4_thresholds]
    i4_idx = {c: k for k, c in enumerate(i4_cfgs)}
    n_idx = {n: j for j, n in enumerate(item4_n)}
    nonoil = [r for r in records if r.category != "Oil"]
    rows = []
    for (s, t) in i4_cfgs:
        k = i4_idx[(s, t)]
        for n in item4_n:
            j = n_idx[n]
            tp = fp = fn = 0
            fa_nonoil = 0
            for rec in records:
                a, b, c, p = (int(v) for v in rec.item4[k, j])
                tp += a; fp += b; fn += c
            for rec in nonoil:
                p = int(rec.item4[k, j][3])
                if p >= 1:
                    fa_nonoil += 1
            rows.append(dict(
                stride=s, threshold=round(float(t), 2), min_area_px=n,
                dice=round(_dice(tp, fp, fn), 5),
                precision=round(_precision(tp, fp), 5),
                recall=round(_recall(tp, fn), 5),
                false_alarm_rate=round(fa_nonoil / len(nonoil), 4) if nonoil else 0.0,
                false_alarm_rate_all=round(fa_nonoil / len(records), 4) if records else 0.0,
                n_nonoil=len(nonoil), n_all=len(records),
                tp=tp, fp=fp, fn=fn))
    return rows


def print_item4(rows):
    if not rows:
        print("  (skipped)")
        return
    print(f"  {'stride':>6} {'thr':>5} {'min_area':>9} {'Dice':>7} "
          f"{'Prec':>7} {'Rec':>7} {'FA_nonoil':>10}  (all MEASURED)")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['stride']:>6} {r['threshold']:>5.2f} {r['min_area_px']:>9} "
              f"{r['dice']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['false_alarm_rate']:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Item 5 — paired bootstrap over val scenes (1000 resamples, seed 42):
# recompute pooled Dice from per-scene TP/FP/FN for each resample.
# ══════════════════════════════════════════════════════════════════════════

ITEM5_PAIRS = [
    ("(512,0.80) vs (384,0.80)", (512, 0.80), (384, 0.80)),
    ("(384,0.80) vs (256,0.80)", (384, 0.80), (256, 0.80)),
    ("(384,0.80) vs (384,0.50)", (384, 0.80), (384, 0.50)),
]


def compute_item5(records: List[SceneRecord], n_boot: int = 1000,
                  seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    n = len(records)
    needed = {k for _, a, b in ITEM5_PAIRS for k in (a, b)}
    scene_counts = {k: np.array(
        [[int(v) for v in rec.tf_grid(*k)] for rec in records],
        dtype=np.int64) for k in needed}

    def pooled_dice(counts: np.ndarray) -> float:
        return _dice(int(counts[:, 0].sum()),
                     int(counts[:, 1].sum()),
                     int(counts[:, 2].sum()))

    out = []
    for label, keyA, keyB in ITEM5_PAIRS:
        cA, cB = scene_counts[keyA], scene_counts[keyB]
        obs = pooled_dice(cA) - pooled_dice(cB)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            diffs[b] = pooled_dice(cA[idx]) - pooled_dice(cB[idx])
        out.append({
            "comparison": label,
            "observed_diff": round(float(obs), 5),
            "ci_95_lo": round(float(np.percentile(diffs, 2.5)), 5),
            "ci_95_hi": round(float(np.percentile(diffs, 97.5)), 5),
            "n_resamples": n_boot,
            "seed": seed,
            "n_scenes": n,
            "measured": True,
        })
    return out


def print_item5(rows):
    print(f"  {'Comparison':<30} {'Obs diff':>9} {'95% CI':<20}  (all MEASURED)")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['comparison']:<30} {r['observed_diff']:>+9.5f}  "
              f"[{r['ci_95_lo']:+.4f}, {r['ci_95_hi']:+.4f}]")


# ══════════════════════════════════════════════════════════════════════════
# Item 6 — stride-256 diagnosis at threshold 0.80: per-scene Dice change
# (256 minus 512), 10 largest drops (category + spill size), mean
# predicted-positive area per stride (512/384/256).
# ══════════════════════════════════════════════════════════════════════════

ITEM6_THRESHOLD = 0.80


def compute_item6(records: List[SceneRecord],
                  thr: float = ITEM6_THRESHOLD) -> dict:
    diffs = []
    for rec in records:
        a256, b256, c256 = rec.tf_grid(256, thr)
        a512, b512, c512 = rec.tf_grid(512, thr)
        d256 = _dice(a256, b256, c256)
        d512 = _dice(a512, b512, c512)
        diffs.append({
            "sid": rec.sid, "category": rec.category,
            "gt_positive": rec.gt_positive,
            "dice_256": round(d256, 4), "dice_512": round(d512, 4),
            "diff_256_minus_512": round(d256 - d512, 4),
            "pred_positive_256": a256 + b256,
            "pred_positive_512": a512 + b512,
        })
    diffs.sort(key=lambda x: x["diff_256_minus_512"])
    vals = np.array([d["diff_256_minus_512"] for d in diffs])
    mean_pred = {}
    for s in STRIDES:
        tot = sum(rec.pred_at(s, thr) for rec in records)
        mean_pred[f"stride_{s}"] = round(tot / len(records), 1) if records else 0.0
    return {
        "threshold": thr,
        "n_scenes": len(diffs),
        "measured": True,
        "mean_diff_256_minus_512": round(float(vals.mean()), 5),
        "std_diff": round(float(vals.std()), 5),
        "pct_scenes_256_worse": round(float((vals < 0).mean()), 4),
        "mean_predicted_positive_px": mean_pred,
        "worst10_scenes": diffs[:10],
    }


def print_item6(item6):
    print(f"  At threshold={item6['threshold']}:  (all MEASURED)")
    print(f"  Mean Dice(256) - Dice(512) = {item6['mean_diff_256_minus_512']:+.5f}  "
          f"std={item6['std_diff']:.5f}")
    print(f"  Fraction of scenes where 256 is worse = {item6['pct_scenes_256_worse']:.4f}")
    mp = item6["mean_predicted_positive_px"]
    print("  Mean predicted-positive px: " +
          "  ".join(f"stride-{s}={mp[f'stride_{s}']:.0f}" for s in STRIDES))
    print()
    print("  10 scenes with largest Dice drop (256 minus 512):")
    print(f"  {'sid':<30} {'cat':>10} {'spill_px':>9} {'dice_256':>9} {'dice_512':>9} {'diff':>7}")
    for d in item6["worst10_scenes"]:
        print(f"  {d['sid']:<30} {d['category']:>10} {d['gt_positive']:>9} "
              f"{d['dice_256']:>9.4f} {d['dice_512']:>9.4f} "
              f"{d['diff_256_minus_512']:>+7.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Item 7 — calibration: 10-bin reliability table at stride 512 over all val
# pixels (100 fine bins per calibration bin), plus a plain over/under verdict.
# ══════════════════════════════════════════════════════════════════════════

ITEM7_STRIDE = 512
ITEM7_NBINS = 10


def compute_item7(records: List[SceneRecord], stride: int = ITEM7_STRIDE,
                  n_cal: int = ITEM7_NBINS) -> dict:
    cal_edges = np.linspace(0.0, 1.0, n_cal + 1)
    pos_count = np.zeros(n_cal, dtype=np.int64)
    neg_count = np.zeros(n_cal, dtype=np.int64)
    k = N_BINS // n_cal
    assert N_BINS % n_cal == 0, "1000 fine bins must divide evenly into cal bins"
    for rec in records:
        ph, nh = rec.hists[stride]
        for c in range(n_cal):
            pos_count[c] += int(ph[c * k:(c + 1) * k].sum())
            neg_count[c] += int(nh[c * k:(c + 1) * k].sum())
    rows = []
    for c in range(n_cal):
        total = int(pos_count[c] + neg_count[c])
        obs = float(pos_count[c] / total) if total else float("nan")
        rows.append({
            "bin_lo": round(float(cal_edges[c]), 1),
            "bin_hi": round(float(cal_edges[c + 1]), 1),
            "mean_pred": round(float((cal_edges[c] + cal_edges[c + 1]) / 2), 2),
            "obs_freq": round(obs, 5),
            "n_pixels": total,
        })
    diffs = [r["mean_pred"] - r["obs_freq"] for r in rows
             if r["n_pixels"] > 1000 and np.isfinite(r["obs_freq"])]
    mean_d = float(np.mean(diffs)) if diffs else float("nan")
    if np.isfinite(mean_d) and mean_d > 0.02:
        verdict = ("OVER-confident: predicted probabilities are systematically "
                   "HIGHER than the observed positive frequency")
    elif np.isfinite(mean_d) and mean_d < -0.02:
        verdict = ("UNDER-confident: predicted probabilities are systematically "
                   "LOWER than the observed positive frequency")
    else:
        verdict = "approximately calibrated"
    return {"stride": stride, "measured": True, "bins": rows,
            "mean_pred_minus_obs": round(mean_d, 4), "verdict": verdict}


def print_item7(item7):
    print(f"  stride={item7['stride']}  (all MEASURED)")
    print(f"  {'bin':>12} {'mean_pred':>10} {'obs_freq':>10} {'n_pixels':>12}")
    print("  " + "-" * 48)
    for r in item7["bins"]:
        gap = r["mean_pred"] - r["obs_freq"]
        flag = " <<OVER" if gap > 0.05 else (" <<UNDER" if gap < -0.05 else "")
        print(f"  [{r['bin_lo']:.1f},{r['bin_hi']:.1f})   "
              f"{r['mean_pred']:>10.2f} {r['obs_freq']:>10.5f} "
              f"{r['n_pixels']:>12,}{flag}")
    print(f"\n  Verdict: {item7['verdict']}")
    print(f"  Mean (predicted - observed) over populated bins: "
          f"{item7['mean_pred_minus_obs']:+.4f}  (MEASURED)")


# ══════════════════════════════════════════════════════════════════════════
# Item 8 — fp32 (autocast OFF) vs AMP (autocast ON) from the single-pass
# EXACT per-pixel counts in rec.item8: cols are
# [amp_tp, amp_fp, amp_fn, fp32_tp, fp32_fp, fp32_fn, label_changes, inter].
# rec.item8_f holds [mean_abs_prob_diff, max_abs_prob_diff] diagnostics.
# ══════════════════════════════════════════════════════════════════════════

def compute_item8(records: List[SceneRecord]) -> dict:
    cfgs = [tuple(c) for c in ITEM8_CFGS]
    total_px = sum(r.gt_total for r in records)
    per_cfg = []
    for k, (s, t) in enumerate(cfgs):
        agg = np.zeros(8, dtype=np.int64)
        mean_num = 0.0
        max_d = 0.0
        for rec in records:
            agg += rec.item8[k]
            mean_num += float(rec.item8_f[k][0]) * rec.gt_total
            max_d = max(max_d, float(rec.item8_f[k][1]))
        dice_amp = _dice(int(agg[0]), int(agg[1]), int(agg[2]))
        dice_fp32 = _dice(int(agg[3]), int(agg[4]), int(agg[5]))
        per_cfg.append({
            "config": f"stride={s}, threshold={t:.2f}",
            "stride": s, "threshold": t,
            "measured": True,
            "dice_amp": round(dice_amp, 5),
            "dice_fp32": round(dice_fp32, 5),
            "dice_diff_fp32_minus_amp": round(dice_fp32 - dice_amp, 5),
            "label_change_fraction": round(int(agg[6]) / total_px, 6) if total_px else 0.0,
            "label_change_pixels": int(agg[6]),
            "total_pixels": total_px,
            "mean_abs_prob_diff": round(mean_num / total_px, 8) if total_px else 0.0,
            "max_abs_prob_diff": round(max_d, 6),
        })
    return {"per_config": per_cfg,
            "note": "autocast OFF (fp32) vs autocast ON (AMP) on identical stitched "
                    "window geometry; label changes counted EXACTLY per pixel "
                    "(no histogram estimate)"}


def print_item8(item8):
    for c in item8["per_config"]:
        print(f"  Config: {c['config']}  (all MEASURED)")
        print(f"  Dice AMP:   {c['dice_amp']:.5f}")
        print(f"  Dice fp32:  {c['dice_fp32']:.5f}")
        print(f"  Difference (fp32 - AMP): {c['dice_diff_fp32_minus_amp']:+.5f}")
        print(f"  Fraction of pixels whose binary label changes: "
              f"{c['label_change_fraction']:.6f} "
              f"({c['label_change_pixels']:,}/{c['total_pixels']:,} px)")
        print(f"  Mean |prob diff|: {c['mean_abs_prob_diff']:.2e}   "
              f"Max |prob diff|: {c['max_abs_prob_diff']:.6f}")


# ══════════════════════════════════════════════════════════════════════════
# Writers + verification + main
# ══════════════════════════════════════════════════════════════════════════

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _save_csv(rows: List[dict], path) -> None:
    path = str(path)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        print(f"  saved -> {path} (empty)")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved -> {path}")


def _save_json(obj, path) -> None:
    with open(str(path), "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"  saved -> {path}")


def build_verification(records: List[SceneRecord], checks: dict,
                       summary: dict) -> dict:
    """Cross-checks proving histogram path == direct path (must show 0 mismatch).

    Every number here is MEASURED on this val pass.
    """
    acct_checked = 0
    acct_fail = 0
    for rec in records:
        for stride in STRIDES:
            ph, nh = rec.hists[stride]
            if int(ph.sum()) + int(nh.sum()) != rec.gt_total:
                acct_fail += 1
            acct_checked += 1
    ok = (checks.get("hist_vs_direct_mismatch", 0) == 0 and acct_fail == 0)
    return {
        "measured": True,
        "threshold_convention": "predicted positive <=> stitched probability >= "
                                "threshold; histogram path = pixels in bins whose "
                                "left edge >= threshold",
        "hist_vs_direct_pairs": checks.get("hist_vs_direct_pairs", 0),
        "hist_vs_direct_mismatch": checks.get("hist_vs_direct_mismatch", 0),
        "hist_mismatch_detail": checks.get("hist_mismatch_detail", []),
        "histogram_pixel_accounting": {"checked": acct_checked,
                                       "failures": acct_fail},
        "n_records": len(records),
        "schema": SCHEMA,
        "checkpoint_sha256": summary["checkpoint"]["sha256"],
        "status": "PASS" if ok else "FAIL",
    }


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Val-only analysis (items 1-8). Val split only; "
                    "writes exclusively to the output dir.")
    ap.add_argument("--n-scenes", type=int, default=None,
                    help="Number of val scenes (default: all)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-dir", default=str(_ROOT / "results" / "val_analysis"))
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore the per-scene records/ cache")
    ap.add_argument("--selftest-cc", action="store_true",
                    help="Run the connected-component self-test only")
    ap.add_argument("--item4-strides", type=int, nargs="+", default=list(ITEM4_STRIDES))
    ap.add_argument("--item4-thresholds", type=float, nargs="+",
                    default=list(ITEM4_THRESHOLDS))
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap resamples for item 5")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    ns = _parse_args(argv)

    if ns.selftest_cc:
        selftest_cc(verbose=True)
        return

    records, checks, summary, out_dir = run_pass(ns)

    # ── Item 1 ──
    print("\n== Item 1: precision / recall / pooled Dice / IoU (MEASURED) ==")
    item1 = compute_item1(records)
    print_item1(item1)
    _save_csv(item1, out_dir / "item1_metrics.csv")

    # ── Item 2 (every stride; 384 = the inference-CLI default) ──
    print("\n== Item 2: per-category analysis (MEASURED) ==")
    item2 = {
        "measured": True,
        "category_source": "tile_index.csv column 'class' (directory name of the "
                           "val scene); no category is derived from predictions",
        "category_counts": {
            "Oil": sum(1 for r in records if r.category == "Oil"),
            "Lookalike": sum(1 for r in records if r.category == "Lookalike"),
            "No_oil": sum(1 for r in records if r.category == "No_oil"),
        },
        "by_stride": {str(s): compute_item2(records, s) for s in STRIDES},
    }
    for s in STRIDES:
        print(f"--- stride {s} ---")
        print_item2(item2["by_stride"][str(s)])
    _save_json(item2, out_dir / "item2_categories.json")

    # ── Item 3 ──
    print("\n== Item 3: Oil Dice by spill-area tercile (MEASURED) ==")
    item3 = compute_item3(records)
    print_item3(item3)
    _save_json(item3, out_dir / "item3_terciles.json")

    # ── Item 4 (ANALYSIS ONLY — not added to the CLI) ──
    print("\n== Item 4: min-area filter sweep, ANALYSIS ONLY (MEASURED) ==")
    item4 = compute_item4(records, list(ns.item4_strides),
                          list(ns.item4_thresholds), list(ITEM4_NVALUES))
    print_item4(item4)
    _save_csv(item4, out_dir / "item4_minarea.csv")

    # ── Item 5 ──
    print("\n== Item 5: paired bootstrap 95% CI (MEASURED) ==")
    item5 = compute_item5(records, n_boot=ns.n_boot)
    print_item5(item5)
    _save_json(item5, out_dir / "item5_bootstrap.json")

    # ── Item 6 ──
    print("\n== Item 6: stride-256 diagnosis @0.80 (MEASURED) ==")
    item6 = compute_item6(records)
    print_item6(item6)
    _save_json(item6, out_dir / "item6_stride256.json")

    # ── Item 7 ──
    print("\n== Item 7: calibration (MEASURED) ==")
    item7 = compute_item7(records)
    print_item7(item7)
    _save_json(item7, out_dir / "item7_calibration.json")

    # ── Item 8 (EXACT per-pixel counts from the single pass) ──
    print("\n== Item 8: fp32 vs AMP (MEASURED, EXACT) ==")
    item8 = compute_item8(records)
    print_item8(item8)
    _save_json(item8, out_dir / "item8_fp32_vs_amp.json")

    # ── verification + manifest (no freezing: recommendation is text only) ──
    verification = build_verification(records, checks, summary)
    print(f"\n[verification] {verification['status']}  "
          f"hist_vs_direct={verification['hist_vs_direct_pairs']} pairs, "
          f"{verification['hist_vs_direct_mismatch']} mismatches; "
          f"accounting {verification['histogram_pixel_accounting']['checked']} "
          f"checked, {verification['histogram_pixel_accounting']['failures']} failures")
    _save_json(verification, out_dir / "verification.json")

    best = max(item1, key=lambda r: r["pooled_dice"])
    recommendation = {
        "measured": True,
        "best_grid_point_by_pooled_dice": {
            "stride": best["stride"], "threshold": best["threshold"],
            "pooled_dice": best["pooled_dice"],
            "pooled_iou": best["pooled_iou"],
            "precision": best["precision"], "recall": best["recall"],
        },
        "note": "Verdict text only — NOT FROZEN. The human decides; no CLI "
                "default or protected file was changed. Compare the best grid "
                "point against item 5 (bootstrap noise) and item 2 "
                "(false alarms) before deciding.",
        "status": "NOT_FROZEN_HUMAN_DECIDES",
    }
    _save_json(recommendation, out_dir / "recommended_config.json")
    print(f"\n[recommendation] best grid point: stride={best['stride']} "
          f"thr={best['threshold']:.2f} Dice={best['pooled_dice']:.5f} "
          f"(NOT FROZEN — you decide)")

    pass_summary = {**summary, "verification": verification["status"],
                    "files": ["item1_metrics.csv", "item2_categories.json",
                              "item3_terciles.json", "item4_minarea.csv",
                              "item5_bootstrap.json", "item6_stride256.json",
                              "item7_calibration.json", "item8_fp32_vs_amp.json",
                              "recommended_config.json", "verification.json",
                              "pass_summary.json"]}
    _save_json(pass_summary, out_dir / "pass_summary.json")
    print(f"\n[analysis] outputs -> {out_dir}")


if __name__ == "__main__":
    main()

