"""Full-scene inference: preprocess -> tile -> model -> stitch -> save.

CLI::

    python -m inference.predict_scene \\
        --scene PATH \\
        --checkpoint PATH \\
        --output-dir DIR \\
        [--stride 384] \\
        [--threshold 0.5] \\
        [--batch-size 4] \\
        [--prep-config PATH]   # default: processed_dataset/metadata/preprocessing_config.json

Outputs (all in output-dir):
  <stem>_prob.tif          float32 single-band GeoTIFF (deflate, tiled 256x256)
  <stem>_mask.tif          uint8 {0,1} GeoTIFF  (threshold applied AFTER finalize)
  <stem>_inference.json    metadata and timing

Design decisions documented in PHASE 0 findings:
  - process_scene called ONCE on the whole scene; windows sliced from the result.
  - Normalization bounds loaded from preprocessing_config.json; never hardcoded.
  - Sigmoid applied EXACTLY ONCE, after the model forward pass.
  - No threshold applied to individual tiles; only the final stitched map.
  - AMP autocast replicates training validation (mixed_precision=True on CUDA).
  - CRS/transform copied from source only if the source has them.
  - Test split is REFUSED unless --allow-test is passed (never use that flag here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

# Repository root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import UNetResNet34, UNetResNet34Config
from inference.windows import tile_grid, TILE_SIZE
from inference.stitching import ProbabilityStitcher

logger = logging.getLogger(__name__)

_DEFAULT_PREP_CFG = _REPO_ROOT / "processed_dataset" / "metadata" / "preprocessing_config.json"
_TILE_INDEX_PATH  = _REPO_ROOT / "processed_dataset" / "metadata" / "tile_index.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    """SHA-256 of a file, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_bounds(prep_cfg_path: str) -> dict:
    """Load frozen normalization bounds from preprocessing_config.json."""
    with open(prep_cfg_path) as f:
        pcfg = json.load(f)
    bands = pcfg.get("statistics", {}).get("bands")
    if not bands or len(bands) != 2:
        raise ValueError(
            f"preprocessing_config.json at {prep_cfg_path} must contain "
            "statistics.bands = [[lo0,hi0],[lo1,hi1]]"
        )
    return {"bands": bands}


def _load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Build UNetResNet34 and load checkpoint weights. Returns (model, ckpt_meta)."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetResNet34(UNetResNet34Config(in_channels=2, out_channels=1, pretrained=False))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "epoch": int(ck.get("epoch", -1)),
        "best_metric_name": ck.get("best_metric_name", "unknown"),
        "best_metric_value": float(ck.get("best_metric", float("nan"))),
    }
    return model, meta


def _preprocess_scene(
    scene_path: str,
    bounds: dict,
    prep_cfg_path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Run training preprocessing on the whole scene.

    Returns
    -------
    img   : (2, H, W) float32, normalised to [0, 1]
    valid : (H, W) bool array where True = finite source pixel;
            None if source has no nodata (all pixels valid)
    geo   : dict with 'crs', 'transform', 'height', 'width', 'nodata_count'
    """
    from preprocessing.pipeline import process_scene
    from preprocessing import with_overrides

    cfg = with_overrides(LEE_FILTER_ENABLED=True, LEE_WINDOW_SIZE=5)

    # Record source nodata mask BEFORE process_scene (which raises on NaN)
    with rasterio.open(scene_path) as src:
        raw = src.read()                          # (2, H, W) float32
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        H, W = src.height, src.width

    # Build valid-data mask from source pixels
    valid_mask: Optional[np.ndarray] = None
    nodata_count = 0
    if src_nodata is not None:
        invalid = np.any(raw == src_nodata, axis=0)
        if invalid.any():
            valid_mask = ~invalid
            nodata_count = int(invalid.sum())
            logger.info("Scene %s: %d nodata pixels detected.", scene_path, nodata_count)

    # Also check for NaN/Inf (process_scene raises on them; catch them here)
    nan_mask = ~np.all(np.isfinite(raw), axis=0)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        logger.warning("Scene %s: %d NaN/Inf pixels detected.", scene_path, n_nan)
        nodata_count += n_nan
        if valid_mask is None:
            valid_mask = ~nan_mask
        else:
            valid_mask &= ~nan_mask

    # Call process_scene (raises on non-finite input, so we need clean data)
    # If there are NaN/Inf pixels we cannot call process_scene — stop and report.
    if nan_mask.any():
        raise RuntimeError(
            f"Scene {scene_path} contains {int(nan_mask.sum())} NaN/Inf pixels. "
            "process_scene() raises on non-finite input. Replace nodata pixels "
            "before inference or pre-fill them."
        )

    out = process_scene(
        scene_path,
        mask_path=None,
        cfg=cfg,
        bounds=bounds,
        split="inference",
        augment_override=False,
        compute_stats=False,
    )

    img = out["normalized"]                       # (2, H, W) float32

    # Determine whether source has georeferencing
    has_crs = src_crs is not None
    has_transform = (src_transform is not None and src_transform != rasterio.transform.IDENTITY)
    # rasterio returns identity when no geotransform is set; detect that
    try:
        _id = rasterio.transform.IDENTITY
    except AttributeError:
        _id = None

    geo = {
        "crs": src_crs if has_crs else None,
        "transform": src_transform if (has_crs and has_transform) else None,
        "height": H,
        "width": W,
        "nodata_count": nodata_count,
    }

    return img, valid_mask, geo


def _iter_batches(
    img: np.ndarray,
    origins: list,
    tile: int,
    batch_size: int,
    device: torch.device,
    padded_H: int,
    padded_W: int,
) -> Generator[Tuple[list, torch.Tensor], None, None]:
    """Yield (origin_batch, image_batch_tensor) without disk writes."""
    buf = []
    for y, x in origins:
        patch = img[:, y : y + tile, x : x + tile]          # (2, tile, tile)
        buf.append(((y, x), patch))
        if len(buf) == batch_size:
            origins_b, patches_b = zip(*buf)
            tensor = torch.from_numpy(
                np.stack(patches_b).astype(np.float32)
            ).to(device)
            yield list(origins_b), tensor
            buf = []
    if buf:
        origins_b, patches_b = zip(*buf)
        tensor = torch.from_numpy(
            np.stack(patches_b).astype(np.float32)
        ).to(device)
        yield list(origins_b), tensor


def predict_scene(
    scene_path: str,
    checkpoint_path: str,
    output_dir: str,
    stride: int = 384,
    threshold: float = 0.5,
    batch_size: int = 4,
    prep_cfg_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Run full-scene inference. Returns paths to outputs.

    Parameters
    ----------
    scene_path      : path to input GeoTIFF (2-channel SAR, Sigma0-dB)
    checkpoint_path : path to best_model.pth
    output_dir      : directory to write outputs
    stride          : sliding-window stride (pixels); default 384
    threshold       : probability threshold for binary mask; default 0.5
    batch_size      : tiles per forward pass; default 4
    prep_cfg_path   : preprocessing_config.json; defaults to repo metadata copy
    device          : 'cuda', 'cpu', or None (auto-detect)
    verbose         : print progress with ETA

    Returns
    -------
    dict with keys 'prob_path', 'mask_path', 'json_path'
    """
    t_total_start = time.time()

    if prep_cfg_path is None:
        prep_cfg_path = str(_DEFAULT_PREP_CFG)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(scene_path).stem

    # ---- device ----
    if device is None:
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _dev = torch.device(device)
    amp = _dev.type == "cuda"

    if verbose:
        print(f"[predict_scene] device={_dev}  amp={amp}")

    # ---- checkpoint sha256 ----
    ck_sha = _sha256(checkpoint_path)

    # ---- preprocessing ----
    if verbose:
        print(f"[predict_scene] preprocessing {scene_path} ...", flush=True)
    t_prep_start = time.time()
    bounds = _load_bounds(prep_cfg_path)
    img, valid_mask, geo = _preprocess_scene(scene_path, bounds, prep_cfg_path)
    t_prep = time.time() - t_prep_start
    H, W = geo["height"], geo["width"]

    if verbose:
        print(
            f"[predict_scene] scene {H}x{W}  nodata={geo['nodata_count']}  "
            f"prep={t_prep:.1f}s",
            flush=True,
        )

    # ---- pad if scene < tile ----
    pad_h = max(0, TILE_SIZE - H)
    pad_w = max(0, TILE_SIZE - W)
    if pad_h > 0 or pad_w > 0:
        warnings.warn(
            f"Scene {H}x{W} is smaller than tile {TILE_SIZE}x{TILE_SIZE}; "
            f"padding with reflect ({pad_h} rows, {pad_w} cols)."
        )
        img = np.pad(
            img,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    pH, pW = img.shape[1], img.shape[2]

    # ---- window origins ----
    origins = tile_grid(pH, pW, tile=TILE_SIZE, stride=stride)
    n_windows = len(origins)

    if verbose:
        print(
            f"[predict_scene] stride={stride}  windows={n_windows}  "
            f"batch_size={batch_size}",
            flush=True,
        )

    # ---- model ----
    model, ck_meta = _load_model(checkpoint_path, _dev)

    # ---- inference ----
    stitcher = ProbabilityStitcher(pH, pW, TILE_SIZE)
    t_model_start = time.time()
    n_done = 0

    with torch.inference_mode():
        for origins_b, tensor_b in _iter_batches(
            img, origins, TILE_SIZE, batch_size, _dev, pH, pW
        ):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(tensor_b)          # (B, 1, 512, 512) raw logits
            # sigmoid exactly once — converts logits to probabilities
            probs = torch.sigmoid(logits).float() # float32 (B, 1, 512, 512)
            probs_np = probs.squeeze(1).cpu().numpy()  # (B, 512, 512)

            for (y, x), prob_tile in zip(origins_b, probs_np):
                stitcher.add_tile(y, x, prob_tile)

            n_done += len(origins_b)
            if verbose and n_done % max(1, n_windows // 20) == 0:
                elapsed = time.time() - t_model_start
                rate = n_done / elapsed if elapsed > 0 else float("inf")
                eta = (n_windows - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"\r[predict_scene] {n_done}/{n_windows} windows  "
                    f"{rate:.1f} win/s  ETA {eta:.0f}s   ",
                    end="",
                    flush=True,
                )

    if verbose:
        print(flush=True)  # newline after \r progress

    t_model = time.time() - t_model_start

    # ---- stitch (finalize, no threshold here) ----
    t_stitch_start = time.time()
    prob_map = stitcher.finalize()               # float32 (pH, pW)
    t_stitch = time.time() - t_stitch_start

    # Crop back to original size if padding was applied
    if pad_h > 0 or pad_w > 0:
        prob_map = prob_map[:H, :W]

    assert prob_map.shape == (H, W), (
        f"probability map shape {prob_map.shape} != scene shape ({H}, {W})"
    )

    # Zero-out nodata pixels in the BINARY mask (prob map keeps model output)
    nodata_zeroed = 0
    if valid_mask is not None:
        nodata_zeroed = int(np.sum(~valid_mask))
        if verbose:
            print(
                f"[predict_scene] zeroing {nodata_zeroed} nodata pixels in binary mask",
                flush=True,
            )

    # ---- threshold AFTER finalize (only here, never per tile) ----
    binary_map = (prob_map >= threshold).astype(np.uint8)
    if valid_mask is not None:
        binary_map[~valid_mask] = 0

    # ---- write outputs ----
    crs   = geo["crs"]
    transform = geo["transform"]
    has_geo = crs is not None and transform is not None

    prob_path = os.path.join(output_dir, f"{stem}_prob.tif")
    mask_path = os.path.join(output_dir, f"{stem}_mask.tif")

    _write_geotiff_float32(prob_path, prob_map, crs, transform, has_geo)
    _write_geotiff_uint8(mask_path, binary_map, crs, transform, has_geo)

    if not has_geo:
        logger.info(
            "Scene %s has no CRS/transform; outputs written without georeferencing.",
            scene_path,
        )
        if verbose:
            print(
                "[predict_scene] WARNING: source has no CRS/transform; "
                "outputs written without georeferencing.",
                flush=True,
            )

    t_total = time.time() - t_total_start

    # ---- inference metadata JSON ----
    meta_out = {
        "scene_path": str(scene_path),
        "scene_height": H,
        "scene_width": W,
        "tile_size": TILE_SIZE,
        "stride": stride,
        "threshold": threshold,
        "n_windows": n_windows,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ck_meta["epoch"],
        "checkpoint_sha256": ck_sha,
        "best_metric_name": ck_meta["best_metric_name"],
        "best_metric_value": ck_meta["best_metric_value"],
        "nodata_count": geo["nodata_count"],
        "nodata_zeroed_in_mask": nodata_zeroed,
        "has_georeferencing": has_geo,
        "timing": {
            "preprocessing_s": round(t_prep, 3),
            "model_inference_s": round(t_model, 3),
            "stitching_s": round(t_stitch, 3),
            "total_s": round(t_total, 3),
        },
        "device": str(_dev),
        "amp": amp,
    }

    json_path = os.path.join(output_dir, f"{stem}_inference.json")
    with open(json_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    if verbose:
        print(
            f"[predict_scene] done  "
            f"prep={t_prep:.1f}s  model={t_model:.1f}s  stitch={t_stitch:.3f}s  "
            f"total={t_total:.1f}s",
            flush=True,
        )
        print(f"[predict_scene] prob  -> {prob_path}", flush=True)
        print(f"[predict_scene] mask  -> {mask_path}", flush=True)
        print(f"[predict_scene] meta  -> {json_path}", flush=True)

    return {"prob_path": prob_path, "mask_path": mask_path, "json_path": json_path}


# ---------------------------------------------------------------------------
# GeoTIFF writers
# ---------------------------------------------------------------------------

def _write_geotiff_float32(
    path: str,
    arr: np.ndarray,
    crs,
    transform,
    has_geo: bool,
) -> None:
    """Write (H, W) float32 array as a single-band, deflate-compressed, tiled GeoTIFF."""
    H, W = arr.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": H,
        "width": W,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2,
    }
    if has_geo:
        profile["crs"] = crs
        profile["transform"] = transform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # suppress NotGeoreferencedWarning
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr[np.newaxis].astype(np.float32))


def _write_geotiff_uint8(
    path: str,
    arr: np.ndarray,
    crs,
    transform,
    has_geo: bool,
) -> None:
    """Write (H, W) uint8 {0,1} array as a single-band GeoTIFF."""
    H, W = arr.shape
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": H,
        "width": W,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if has_geo:
        profile["crs"] = crs
        profile["transform"] = transform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr[np.newaxis].astype(np.uint8))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Full-scene oil-spill inference (UNet + ResNet34)"
    )
    ap.add_argument("--scene",      required=True,  help="Input GeoTIFF path")
    ap.add_argument("--checkpoint", required=True,  help="Path to best_model.pth")
    ap.add_argument("--output-dir", required=True,  help="Directory for outputs")
    ap.add_argument("--stride",     type=int,   default=384)
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--prep-config", default=None,
                    help="preprocessing_config.json (default: repo metadata copy)")
    ap.add_argument("--device",    default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--allow-test", action="store_true",
                    help="Allow test-split scenes. DO NOT use for tuning.")
    return ap.parse_args(argv)


def _is_test_scene(scene_path: str) -> bool:
    """Return True if the scene is in the test split according to tile_index.csv.

    Matching compares the scene_path's path components against the image_path
    column (e.g. "test/images/Lookalike/00000.tif").  The leading
    processed_dataset/ prefix is stripped from the scene_path if present so
    that both absolute and relative paths work correctly.
    """
    import csv
    if not _TILE_INDEX_PATH.exists():
        return False
    try:
        p = Path(scene_path).resolve() if Path(scene_path).is_absolute() else Path(scene_path)

        # Build the set of test image_paths (forward-slash normalised)
        test_image_paths: set = set()
        with open(_TILE_INDEX_PATH) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if row.get("split") == "test":
                    test_image_paths.add(
                        row.get("image_path", "").replace("\\", "/")
                    )

        # Try matching after stripping common prefixes.
        # image_path values look like "test/images/Lookalike/00000.tif"
        scene_str = str(p).replace("\\", "/")
        for candidate in test_image_paths:
            if scene_str.endswith("/" + candidate) or scene_str == candidate:
                return True
        return False
    except Exception:
        pass
    return False


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ns = _parse_args(argv)

    if _is_test_scene(ns.scene) and not ns.allow_test:
        print(
            "ERROR: The scene appears to be in the TEST split. "
            "Test data must NOT be used for tuning or debugging. "
            "Pass --allow-test only if you are doing a final held-out evaluation "
            "and are aware that results cannot be used to inform any model decision.",
            file=sys.stderr,
        )
        sys.exit(1)

    predict_scene(
        scene_path=ns.scene,
        checkpoint_path=ns.checkpoint,
        output_dir=ns.output_dir,
        stride=ns.stride,
        threshold=ns.threshold,
        batch_size=ns.batch_size,
        prep_cfg_path=ns.prep_config,
        device=ns.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
