"""Quick one-shot: measure stats computation overhead per scene."""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.tile_dataset import OilSpillTileDataset
from preprocessing.pipeline import process_scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_INDEX = os.path.join(ROOT, "processed_dataset", "metadata", "tile_index.csv")

ds = OilSpillTileDataset(TILE_INDEX, split="train", scene_cache_size=1)
row = ds.rows[0]
ipath = os.path.join(ds.root, row["image_path"])
mpath = os.path.join(ds.root, row["mask_path"])

# warm OS file cache
process_scene(ipath, mpath, ds.cfg, rng=None, split="train",
              bounds=ds.bounds, augment_override=False, compute_stats=False)

N = 5
times_no = []
for _ in range(N):
    t0 = time.perf_counter()
    process_scene(ipath, mpath, ds.cfg, rng=None, split="train",
                  bounds=ds.bounds, augment_override=False, compute_stats=False)
    times_no.append(time.perf_counter() - t0)

times_yes = []
for _ in range(N):
    t0 = time.perf_counter()
    process_scene(ipath, mpath, ds.cfg, rng=None, split="train",
                  bounds=ds.bounds, augment_override=False, compute_stats=True)
    times_yes.append(time.perf_counter() - t0)

no_s  = min(times_no)
yes_s = min(times_yes)
print(f"compute_stats=False : {no_s*1000:.1f} ms/scene  (after OS cache warm)")
print(f"compute_stats=True  : {yes_s*1000:.1f} ms/scene")
print(f"Stats overhead      : {(yes_s-no_s)*1000:.1f} ms/scene  ({100*(yes_s-no_s)/yes_s:.1f}% of total)")
print(f"Speedup             : {yes_s/no_s:.2f}x faster without stats")
print(f"")
print(f"With 2053 scenes/epoch (warm OS cache):")
print(f"  compute_stats=True  : {yes_s*2053:.0f}s = {yes_s*2053/60:.1f} min")
print(f"  compute_stats=False : {no_s*2053:.0f}s = {no_s*2053/60:.1f} min")
print(f"  Time saved/epoch    : {(yes_s-no_s)*2053:.0f}s = {(yes_s-no_s)*2053/60:.1f} min")
