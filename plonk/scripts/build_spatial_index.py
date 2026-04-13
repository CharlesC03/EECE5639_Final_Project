"""
Pre-compute spatial neighbor index and embedding lookup for neighborhood sampling.

This script scans all webdataset tar files, extracts coordinates and embeddings,
builds a KD-tree spatial index, and groups images by proximity. The output is used
by the NeighborhoodWebdataset wrapper at training/inference time.

Usage:
    conda run -n plonk python -m plonk.scripts.build_spatial_index \
        --data_dir plonk/datasets/osv5m \
        --splits train val test \
        --embedding_name street_clip \
        --radius 500 \
        --max_neighbors 20
"""

import argparse
import json
import os
import pickle
import tarfile
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


def haversine_np(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between arrays of coordinates (in degrees)."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def latlon_to_cartesian(lat_deg, lon_deg):
    """Convert lat/lon in degrees to 3D cartesian (unit sphere)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def meters_to_cartesian_dist(meters):
    """Convert a surface distance in meters to a straight-line (chord) distance
    through the Earth, for use with the cartesian KD-tree."""
    R = 6_371_000
    angle = meters / R  # central angle in radians
    return 2 * np.sin(angle / 2)  # chord length on unit sphere


def scan_tar_file(tar_path, embedding_name):
    """Extract image IDs, coordinates, and embeddings from a single tar file."""
    records = []
    tf = tarfile.open(tar_path)
    members_by_prefix = defaultdict(dict)

    for member in tf.getmembers():
        name = member.name
        # Parse: {id}.json, {id}.{embedding_name}.npy
        if name.endswith(".json"):
            prefix = name[: -len(".json")]
            members_by_prefix[prefix]["json"] = member
        elif name.endswith(f".{embedding_name}.npy"):
            prefix = name[: -len(f".{embedding_name}.npy")]
            members_by_prefix[prefix]["emb"] = member

    for prefix, parts in members_by_prefix.items():
        if "json" not in parts or "emb" not in parts:
            continue
        # Read metadata
        f = tf.extractfile(parts["json"])
        metadata = json.loads(f.read())
        lat = float(metadata["latitude"])
        lon = float(metadata["longitude"])
        image_id = prefix

        # Read embedding
        f = tf.extractfile(parts["emb"])
        emb = np.load(io.BytesIO(f.read()))

        records.append((image_id, lat, lon, emb))

    tf.close()
    return records


def build_index_for_split(data_dir, split, embedding_name, radius_m, max_neighbors):
    """Build spatial index for a single data split."""
    split_dir = os.path.join(data_dir, split)
    tar_files = sorted(
        [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".tar")]
    )

    print(f"\n=== Building index for {split} ({len(tar_files)} tar files) ===")

    # Phase 1: Scan all tar files
    all_ids = []
    all_lats = []
    all_lons = []
    all_embs = []

    for tar_path in tqdm(tar_files, desc=f"Scanning {split} tars"):
        records = scan_tar_file(tar_path, embedding_name)
        for image_id, lat, lon, emb in records:
            all_ids.append(image_id)
            all_lats.append(lat)
            all_lons.append(lon)
            all_embs.append(emb)

    n = len(all_ids)
    print(f"Found {n} samples")

    all_lats = np.array(all_lats, dtype=np.float64)
    all_lons = np.array(all_lons, dtype=np.float64)

    # Phase 2: Build KD-tree on cartesian coordinates (handles wraparound correctly)
    print("Building KD-tree...")
    coords_3d = latlon_to_cartesian(all_lats, all_lons)
    tree = KDTree(coords_3d)

    # Phase 3: Query neighbors for every point
    search_radius = meters_to_cartesian_dist(radius_m)
    print(f"Querying neighbors within {radius_m}m (cartesian dist={search_radius:.6f})...")
    all_neighbor_indices = tree.query_ball_tree(tree, r=search_radius)

    # Phase 4: Build neighbor lookup (excluding self) and cap at max_neighbors
    neighbor_index = {}
    neighbor_counts = []
    for i in tqdm(range(n), desc="Building neighbor groups"):
        # Exclude self
        neighbors = [j for j in all_neighbor_indices[i] if j != i]
        if len(neighbors) > max_neighbors:
            # Keep closest ones
            dists = np.linalg.norm(coords_3d[neighbors] - coords_3d[i], axis=1)
            closest = np.argsort(dists)[:max_neighbors]
            neighbors = [neighbors[c] for c in closest]
        neighbor_index[all_ids[i]] = [all_ids[j] for j in neighbors]
        neighbor_counts.append(len(neighbors))

    neighbor_counts = np.array(neighbor_counts)
    print(f"Neighbor stats: mean={neighbor_counts.mean():.1f}, "
          f"median={np.median(neighbor_counts):.0f}, "
          f"max={neighbor_counts.max()}, "
          f"zero={np.sum(neighbor_counts == 0)} ({np.sum(neighbor_counts == 0)/n*100:.1f}%)")

    # Phase 5: Save embeddings as memory-mapped file
    out_dir = os.path.join(data_dir, "neighborhood_index", split)
    os.makedirs(out_dir, exist_ok=True)

    emb_dim = all_embs[0].shape[0]
    emb_path = os.path.join(out_dir, f"embeddings_{embedding_name}.npy")
    print(f"Saving embeddings memmap ({n} x {emb_dim}) to {emb_path}...")
    emb_mmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float32, shape=(n, emb_dim)
    )
    for i, emb in enumerate(all_embs):
        emb_mmap[i] = emb
    emb_mmap.flush()

    # Phase 6: Save ID-to-index mapping and neighbor index
    id_to_idx = {image_id: i for i, image_id in enumerate(all_ids)}

    meta_path = os.path.join(out_dir, f"index_{embedding_name}_r{radius_m}.pkl")
    print(f"Saving index to {meta_path}...")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "id_to_idx": id_to_idx,
                "neighbor_index": neighbor_index,
                "radius_m": radius_m,
                "max_neighbors": max_neighbors,
                "embedding_name": embedding_name,
                "num_samples": n,
                "emb_dim": emb_dim,
            },
            f,
        )

    print(f"Done with {split}!")
    return n


def main():
    parser = argparse.ArgumentParser(description="Build spatial neighbor index")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (e.g., plonk/datasets/osv5m)")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"], help="Splits to process")
    parser.add_argument("--embedding_name", type=str, default="street_clip", help="Embedding name (e.g., street_clip, dinov2_vitl14_registers)")
    parser.add_argument("--radius", type=float, default=500, help="Neighborhood radius in meters")
    parser.add_argument("--max_neighbors", type=int, default=20, help="Maximum neighbors per image")
    args = parser.parse_args()

    for split in args.splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split}: {split_dir} not found")
            continue
        build_index_for_split(
            args.data_dir, split, args.embedding_name, args.radius, args.max_neighbors
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()
