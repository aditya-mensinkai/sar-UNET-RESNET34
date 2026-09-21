"""One-shot patch: fix run_pilot in build_cache.py to not allocate 80GB array."""
import re, sys
path = "tools/build_cache.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# Find the run_pilot function and replace it wholesale
old_fn_marker = "def run_pilot(rows: list[dict], groups: dict[str, list[int]],"
new_fn = '''def run_pilot(rows: list[dict], groups: dict[str, list[int]],
              dtype_str: str, root: Path) -> dict:
    """Process one scene, measure wall time and peak RAM delta."""
    print("\\n[PILOT] Running 1-scene pilot to measure time + RAM...", flush=True)
    scene_key = next(iter(groups))
    tile_idxs = groups[scene_key]
    n_scene_tiles = len(tile_idxs)
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    class MEM(ctypes.Structure):
        _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                    ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                    ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                    ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
    def get_avail():
        m = MEM(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024**3

    ram_before = get_avail()
    t0 = time.perf_counter()

    from preprocessing.tile_dataset import OilSpillTileDataset as _DS
    _ds = _DS(str(TILE_INDEX), split=rows[tile_idxs[0]]["split"], scene_cache_size=1)
    from preprocessing.pipeline import process_scene

    row0  = rows[tile_idxs[0]]
    ipath = root / row0["image_path"]
    mpath = root / row0["mask_path"]
    out   = process_scene(str(ipath), str(mpath), _ds.cfg, rng=None,
                          split=row0["split"], bounds=_ds.bounds,
                          augment_override=False, compute_stats=False)
    img_scene = np.asarray(out["normalized"], dtype=np_dtype)
    msk_scene = np.asarray(out["mask"], dtype=np.uint8)

    # Only allocate this one scene's tiles (not all N tiles)
    img_pilot = np.empty((n_scene_tiles, TILE_C, TILE_H, TILE_W), dtype=np_dtype)
    msk_pilot = np.empty((n_scene_tiles, TILE_H, TILE_W), dtype=np.uint8)
    for local_i, ti in enumerate(tile_idxs):
        r = rows[ti]
        x=int(r["x"]); y=int(r["y"]); h=int(r["tile_height"]); w=int(r["tile_width"])
        img_pilot[local_i] = img_scene[:, y:y+h, x:x+w]
        msk_pilot[local_i] = msk_scene[y:y+h, x:x+w]

    elapsed = time.perf_counter() - t0
    ram_after  = get_avail()
    ram_delta_gb = ram_before - ram_after

    n_scenes_total   = len(groups)
    proj_serial_s    = elapsed * n_scenes_total
    proj_2workers_s  = proj_serial_s / 2

    SH = TILE_H * 4; SW = TILE_W * 4  # full scene 2048x2048
    ram_per_scene_mb = (2 * SH * SW * 8 * 3 + 2 * SH * SW * 4) / 1024**2

    print(f"  Scene: {scene_key}  tiles: {n_scene_tiles}")
    print(f"  Wall time:        {elapsed:.2f}s  [MEASURED]")
    print(f"  RAM delta:        {ram_delta_gb:.2f} GB  [MEASURED]")
    print(f"  RAM/scene (est):  ~{ram_per_scene_mb:.0f} MB per worker during processing")
    print(f"  Projected serial: {_hms(proj_serial_s)}  ({proj_serial_s:.0f}s)  [ESTIMATED]")
    print(f"  Projected 2-wkr: {_hms(proj_2workers_s)}  ({proj_2workers_s:.0f}s)  [ESTIMATED]")

    del img_pilot, msk_pilot, img_scene, msk_scene, out

    return {"elapsed_s": elapsed, "ram_delta_gb": ram_delta_gb,
            "proj_serial_s": proj_serial_s, "proj_2workers_s": proj_2workers_s,
            "n_scenes": n_scenes_total}

'''

# Find the old function start and end (next def at module level or EOF)
start = src.index(old_fn_marker)
# Find next module-level def after run_pilot
next_def = src.find("\ndef build(", start)
if next_def == -1:
    print("ERROR: could not find end of run_pilot"); sys.exit(1)
old_fn = src[start:next_def]
src2 = src[:start] + new_fn + src[next_def:]
with open(path, "w", encoding="utf-8") as f:
    f.write(src2)
print(f"Patched {path}: replaced {len(old_fn)} chars with {len(new_fn)} chars")
