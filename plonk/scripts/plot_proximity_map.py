"""
Extract GPS coordinates from all training images, group by neighbor count,
and plot a world map coloured by proximity group.

Outputs a PDF (vector land outline + rasterised scatter) suitable for LaTeX.

Results are cached to proximity_map_cache.npz so re-runs skip the tar scan.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
import shapefile


INDEX_PATH  = "plonk/datasets/osv5m/neighborhood_index/train/index_street_clip_r100.pkl"
TAR_DIR     = "plonk/datasets/osv5m/train"
CACHE_PATH  = "proximity_map_cache.npz"
OUT_PDF     = "diagrams/proximity_map.pdf"

LAND_SHP    = os.path.expanduser(
    "~/.local/share/cartopy/shapefiles/natural_earth/physical/ne_110m_land.shp"
)

MAX_NEIGHBOR_GROUP = 5
MAX_PTS_PER_GROUP  = 150_000   # subsample for fast rendering


# ── palette ───────────────────────────────────────────────────────────────────

GROUPS = {
    0: ("#999999", "0 neighbors"),
    1: ("#4e9af1", "1 neighbor"),
    2: ("#27ae60", "2 neighbors"),
    3: ("#f39c12", "3 neighbors"),
    4: ("#e74c3c", "4 neighbors"),
    5: ("#8e44ad", f"{MAX_NEIGHBOR_GROUP}+ neighbors"),
}


# ── tar scanning ──────────────────────────────────────────────────────────────

def _scan_tar(tar_path):
    records = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".json"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                meta = json.loads(f.read())
                records.append((str(meta["id"]), meta["latitude"], meta["longitude"]))
    except Exception:
        pass
    return records


def scan_all_tars(tar_dir, workers):
    tar_paths = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    print(f"Scanning {len(tar_paths)} tars with {workers} workers...")
    with mp.Pool(workers) as pool:
        results = []
        for i, batch in enumerate(pool.imap_unordered(_scan_tar, tar_paths, chunksize=4)):
            results.extend(batch)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tar_paths)} tars done, {len(results):,} records")
    return results


def build_cache(workers):
    print("Loading neighbor index...")
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    neighbor_index = index["neighbor_index"]

    records = scan_all_tars(TAR_DIR, workers)
    print(f"Scanned {len(records):,} images")

    ids  = np.array([r[0] for r in records])
    lats = np.array([r[1] for r in records], dtype=np.float32)
    lons = np.array([r[2] for r in records], dtype=np.float32)
    nbrs = np.array(
        [min(len(neighbor_index.get(iid, [])), MAX_NEIGHBOR_GROUP) for iid in ids],
        dtype=np.uint8,
    )

    np.savez_compressed(CACHE_PATH, ids=ids, lats=lats, lons=lons, nbrs=nbrs)
    print(f"Cache saved to {CACHE_PATH}")
    return lats, lons, nbrs


def load_cache():
    data = np.load(CACHE_PATH, allow_pickle=True)
    return data["lats"], data["lons"], data["nbrs"]


# ── map drawing ───────────────────────────────────────────────────────────────

def add_land(ax, shp_path):
    """Draw Natural Earth land polygons as vector patches."""
    sf = shapefile.Reader(shp_path)
    for shape in sf.shapes():
        parts = list(shape.parts) + [len(shape.points)]
        for start, end in zip(parts[:-1], parts[1:]):
            pts = shape.points[start:end]
            if len(pts) < 3:
                continue
            xs, ys = zip(*pts)
            ax.fill(xs, ys, color="#e8e8e8", linewidth=0)
        for start, end in zip(parts[:-1], parts[1:]):
            pts = shape.points[start:end]
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#aaaaaa", linewidth=0.3)


# ── main ──────────────────────────────────────────────────────────────────────

def plot(lats, lons, nbrs):
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#d6eaf8")
    fig.patch.set_facecolor("white")

    print("Drawing land polygons...")
    add_land(ax, LAND_SHP)

    # plot groups from least to most neighbours so denser/rarer groups are on top
    print("Plotting scatter groups...")
    for n in sorted(GROUPS.keys()):
        color, label = GROUPS[n]
        mask = nbrs == n
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        if len(idx) > MAX_PTS_PER_GROUP:
            idx = rng.choice(idx, MAX_PTS_PER_GROUP, replace=False)
        ax.scatter(
            lons[idx], lats[idx],
            s=0.3, color=color, alpha=0.4, linewidths=0,
            rasterized=True,
            label=f"{label}  ({mask.sum():,})",
            zorder=2,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(
        "OSV5M training images — grouped by number of neighbours within 100 m",
        fontsize=11, pad=8,
    )
    ax.tick_params(labelsize=7)

    legend = ax.legend(
        title="Neighbour count",
        title_fontsize=8,
        fontsize=7,
        markerscale=8,
        framealpha=0.9,
        loc="lower left",
        handlelength=1,
    )

    plt.tight_layout()
    fig.savefig(OUT_PDF, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_PDF}")

    print("\nGroup summary:")
    for n, (_, label) in GROUPS.items():
        count = int((nbrs == n).sum())
        if count:
            print(f"  {label}: {count:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore cache and re-scan tars")
    args = parser.parse_args()

    if os.path.exists(CACHE_PATH) and not args.rebuild:
        print(f"Loading cache from {CACHE_PATH}  (--rebuild to re-scan)")
        lats, lons, nbrs = load_cache()
    else:
        lats, lons, nbrs = build_cache(args.workers)

    print(f"Plotting {len(lats):,} points...")
    plot(lats, lons, nbrs)


if __name__ == "__main__":
    main()
