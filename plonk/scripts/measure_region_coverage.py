"""
Measure what percentage of the OSV5M training set falls in Europe, USA, and Japan.

Uses the cached GPS coordinates (proximity_map_cache.npz) from plot_proximity_map.py.
Country polygons come from Natural Earth 110m via cartopy's shapereader (pyshp only,
no broken cartopy crs/pyproj import).

Run from the repo root:
    conda run -n plonk python plonk/scripts/measure_region_coverage.py
"""

import numpy as np
import shapefile
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path

CACHE_PATH = "proximity_map_cache.npz"

# Europe = CONTINENT=='Europe' excluding Russia
EUROPE_EXCLUDE = {"RUS"}

# Western world: Europe (excl. Russia) + USA + Canada + Australia + NZ
WESTERN_WORLD_EXTRA = {"USA", "CAN", "AUS", "NZL"}

REGIONS = {
    "Europe":        lambda r: r["CONTINENT"] == "Europe" and r["ISO_A3"] not in EUROPE_EXCLUDE,
    "USA":           lambda r: r["ISO_A3"] == "USA",
    # "Japan":       lambda r: r["ISO_A3"] == "JPN",
    "Western World": lambda r: (
        (r["CONTINENT"] == "Europe" and r["ISO_A3"] not in EUROPE_EXCLUDE)
        or r["ISO_A3"] in WESTERN_WORLD_EXTRA
    ),
}


def build_region_shapes(sf):
    """Return {region_name: [(bbox, [Path, ...]), ...]} for polygon membership tests."""
    region_shapes = {name: [] for name in REGIONS}
    for shape_rec in sf.shapeRecords():
        rec = shape_rec.record.as_dict()
        shape = shape_rec.shape
        for name, predicate in REGIONS.items():
            if not predicate(rec):
                continue
            parts = list(shape.parts) + [len(shape.points)]
            for start, end in zip(parts[:-1], parts[1:]):
                pts = shape.points[start:end]
                if len(pts) < 3:
                    continue
                xs, ys = zip(*pts)
                bbox = (min(xs), min(ys), max(xs), max(ys))
                region_shapes[name].append((bbox, Path(np.array(pts))))
    return region_shapes


def points_in_region(lons, lats, region_polys):
    """
    Return boolean mask of which points fall inside any polygon of the region.
    Uses bbox pre-filter before the expensive Path.contains_points check.
    """
    pts = np.column_stack([lons, lats])
    result = np.zeros(len(pts), dtype=bool)
    already_in = result  # alias

    for (x0, y0, x1, y1), path in region_polys:
        # skip polygons where all points are already classified
        remaining = ~already_in
        if not remaining.any():
            break
        # bbox pre-filter
        bbox_mask = remaining & (lons >= x0) & (lons <= x1) & (lats >= y0) & (lats <= y1)
        if not bbox_mask.any():
            continue
        # precise polygon check on candidates
        candidates = np.where(bbox_mask)[0]
        inside = path.contains_points(pts[candidates])
        result[candidates[inside]] = True

    return result


def main():
    print(f"Loading GPS cache from {CACHE_PATH}...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    lats = data["lats"].astype(np.float64)
    lons = data["lons"].astype(np.float64)
    total = len(lats)
    print(f"  {total:,} training images\n")

    print("Loading country polygons...")
    shp_path = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    sf = shapefile.Reader(shp_path)
    region_shapes = build_region_shapes(sf)
    for name, polys in region_shapes.items():
        print(f"  {name}: {len(polys)} polygon parts")

    print("\nClassifying points...\n")
    masks = {}
    for name, polys in region_shapes.items():
        print(f"  Testing {name}...", flush=True)
        masks[name] = points_in_region(lons, lats, polys)
        count = masks[name].sum()
        print(f"    {count:,}  ({100 * count / total:.2f}%)")

    combined = np.zeros(total, dtype=bool)
    for mask in masks.values():
        combined |= mask
    combined_count = combined.sum()

    print("\n" + "=" * 45)
    print(f"{'Region':<12}  {'Count':>10}  {'% of train':>10}")
    print("-" * 45)
    for name, mask in masks.items():
        c = int(mask.sum())
        print(f"{name:<12}  {c:>10,}  {100 * c / total:>9.2f}%")
    print("-" * 45)
    print(f"{'Combined':<12}  {combined_count:>10,}  {100 * combined_count / total:>9.2f}%")
    print(f"{'Rest':<12}  {total - combined_count:>10,}  {100 * (total - combined_count) / total:>9.2f}%")
    print(f"{'Total':<12}  {total:>10,}  {'100.00%':>10}")
    print("=" * 45)
    print("\nNote: Europe excludes Russia. Uses Natural Earth 110m polygons.")


if __name__ == "__main__":
    main()
