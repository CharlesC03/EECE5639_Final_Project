"""
List the percentage of OSV5M training images by country, sorted by count.
Also shows top-5 country distribution for each neighbor-count group.

Uses the cached GPS coordinates (proximity_map_cache.npz) from plot_proximity_map.py.

Run from the repo root:
    conda run -n plonk python plonk/scripts/measure_country_coverage.py
"""

import numpy as np
import shapefile
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path

CACHE_PATH = "proximity_map_cache.npz"
TOP_N = 5


def classify_by_country(lons, lats, sf):
    """
    Assign each point to a country using bbox pre-filter + polygon check.
    Returns (assigned array of country indices, country_names list, unassigned count).
    assigned[i] == -1 means point i was not matched to any country.
    """
    pts = np.column_stack([lons, lats])
    assigned = np.full(len(pts), -1, dtype=np.int32)
    country_names = []

    for country_idx, shape_rec in enumerate(sf.shapeRecords()):
        rec = shape_rec.record.as_dict()
        name = rec["NAME"]
        country_names.append(name)
        shape = shape_rec.shape

        parts = list(shape.parts) + [len(shape.points)]
        for start, end in zip(parts[:-1], parts[1:]):
            poly_pts = shape.points[start:end]
            if len(poly_pts) < 3:
                continue
            xs, ys = zip(*poly_pts)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            unassigned = assigned == -1
            bbox_mask = unassigned & (lons >= x0) & (lons <= x1) & (lats >= y0) & (lats <= y1)
            if not bbox_mask.any():
                continue

            candidates = np.where(bbox_mask)[0]
            path = Path(np.array(poly_pts))
            inside = path.contains_points(pts[candidates])
            assigned[candidates[inside]] = country_idx

        if (country_idx + 1) % 25 == 0:
            classified = (assigned >= 0).sum()
            print(f"  {country_idx + 1}/177 countries done  "
                  f"({classified:,} / {len(pts):,} points assigned)", flush=True)

    return assigned, country_names


def top_countries(assigned, country_names, mask, total_in_group, top_n=5):
    """Return top_n (country_name, count, pct_of_group) tuples for points in mask."""
    group_assigned = assigned[mask]
    counts = {}
    for idx, name in enumerate(country_names):
        n = int((group_assigned == idx).sum())
        if n > 0:
            counts[name] = n
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [(name, cnt, 100 * cnt / total_in_group) for name, cnt in sorted_counts[:top_n]]


def print_top_countries_per_group(assigned, country_names, nbr_counts, total):
    groups = [
        ("0 neighbors",  nbr_counts == 0),
        ("1 neighbor",   nbr_counts == 1),
        ("2 neighbors",  nbr_counts == 2),
        ("3 neighbors",  nbr_counts == 3),
        ("4 neighbors",  nbr_counts == 4),
        ("5+ neighbors", nbr_counts >= 5),
    ]

    print(f"\n{'=' * 55}")
    print(f"Top {TOP_N} countries by neighbor-count group")
    print(f"{'=' * 55}")

    for group_name, mask in groups:
        group_size = int(mask.sum())
        if group_size == 0:
            continue
        pct_of_total = 100 * group_size / total
        print(f"\n{group_name}  ({group_size:,} images, {pct_of_total:.2f}% of train)")
        print(f"  {'Country':<25}  {'Count':>8}  {'% of group':>10}")
        print(f"  {'-' * 48}")
        for name, cnt, pct in top_countries(assigned, country_names, mask, group_size, TOP_N):
            print(f"  {name:<25}  {cnt:>8,}  {pct:>9.2f}%")


def main():
    print(f"Loading GPS cache from {CACHE_PATH}...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    lats = data["lats"].astype(np.float64)
    lons = data["lons"].astype(np.float64)
    nbr_counts = data["nbrs"].astype(np.int32)
    total = len(lats)
    print(f"  {total:,} training images\n")

    print("Loading country polygons...")
    shp_path = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    sf = shapefile.Reader(shp_path)
    print(f"  {len(sf.shapeRecords())} countries\n")

    print("Classifying points by country...")
    assigned, country_names = classify_by_country(lons, lats, sf)

    counts = {}
    for idx, name in enumerate(country_names):
        n = int((assigned == idx).sum())
        if n > 0:
            counts[name] = n
    unassigned = int((assigned == -1).sum())

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    col_w = max(len(name) for name, _ in sorted_counts) + 2
    header = f"{'Country':<{col_w}}  {'Count':>10}  {'% of train':>10}"
    sep = "-" * len(header)

    print(f"\n{header}")
    print(sep)
    for name, count in sorted_counts:
        print(f"{name:<{col_w}}  {count:>10,}  {100 * count / total:>9.3f}%")
    print(sep)
    if unassigned:
        print(f"{'(unassigned)':<{col_w}}  {unassigned:>10,}  {100 * unassigned / total:>9.3f}%")
    print(f"{'Total':<{col_w}}  {total:>10,}  {'100.000%':>10}")

    print("\nNote: Uses Natural Earth 110m polygons. Points at sea or in "
          "uncovered territories appear as unassigned.")

    print_top_countries_per_group(assigned, country_names, nbr_counts, total)


if __name__ == "__main__":
    main()
