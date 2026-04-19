"""
Pre-compute spatial neighbor index and embedding lookup for neighborhood sampling.

This script scans all webdataset tar files, extracts coordinates and embeddings,
builds a KD-tree spatial index, and groups images by proximity. The output is used
by the NeighborhoodWebdataset wrapper at training/inference time.

Memory: embeddings are streamed directly into a memory-mapped .npy file as tars
are scanned, so peak RAM is bounded by the metadata + KD-tree + neighbor-query
batch, not by the full embedding tensor.

Usage:
    conda run -n plonk python -m plonk.scripts.build_spatial_index \
        --data_dir plonk/datasets/osv5m \
        --splits train val test \
        --embedding_name street_clip \
        --radius 100 \
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


def _group_members(tf, embedding_name):
    """Walk a tarfile and return {prefix: {'json': TarInfo, 'emb': TarInfo}}."""
    members_by_prefix = defaultdict(dict)
    for member in tf.getmembers():
        name = member.name
        if name.endswith(".json"):
            prefix = name[: -len(".json")]
            members_by_prefix[prefix]["json"] = member
        elif name.endswith(f".{embedding_name}.npy"):
            prefix = name[: -len(f".{embedding_name}.npy")]
            members_by_prefix[prefix]["emb"] = member
    return members_by_prefix


def peek_emb_dim(tar_path, embedding_name):
    """Open one tar and read the first embedding to get its dimensionality."""
    with tarfile.open(tar_path) as tf:
        groups = _group_members(tf, embedding_name)
        for prefix, parts in groups.items():
            if "emb" not in parts:
                continue
            f = tf.extractfile(parts["emb"])
            emb = np.load(io.BytesIO(f.read()))
            return int(emb.shape[0])
    raise RuntimeError(f"No {embedding_name}.npy found in {tar_path}")


def stream_tar_into_memmap(tar_path, embedding_name, emb_mmap, offset,
                           out_ids, out_lats, out_lons):
    """Scan one tar, writing embeddings directly into emb_mmap[offset:offset+k]
    and appending (id, lat, lon) to the lightweight metadata lists.
    Returns the new offset."""
    with tarfile.open(tar_path) as tf:
        groups = _group_members(tf, embedding_name)
        for prefix, parts in groups.items():
            if "json" not in parts or "emb" not in parts:
                continue
            f = tf.extractfile(parts["json"])
            metadata = json.loads(f.read())
            lat = float(metadata["latitude"])
            lon = float(metadata["longitude"])

            f = tf.extractfile(parts["emb"])
            emb = np.load(io.BytesIO(f.read())).astype(np.float32, copy=False)

            emb_mmap[offset] = emb
            out_ids.append(prefix)
            out_lats.append(lat)
            out_lons.append(lon)
            offset += 1
    return offset


def _total_samples_from_sizes(split_dir, tar_files):
    """Use sizes.json if present for an exact count; else count by opening each tar."""
    sizes_path = os.path.join(split_dir, "sizes.json")
    if os.path.exists(sizes_path):
        with open(sizes_path) as f:
            sizes = json.load(f)
        return sum(int(sizes[os.path.basename(p)]) for p in tar_files)
    total = 0
    for p in tqdm(tar_files, desc="Counting samples (no sizes.json)"):
        with tarfile.open(p) as tf:
            total += sum(1 for m in tf.getmembers() if m.name.endswith(".json"))
    return total


def build_index_for_split(data_dir, split, embedding_name, radius_m, max_neighbors,
                          query_batch_size=10_000):
    """Build spatial index for a single data split (memory-streaming version)."""
    split_dir = os.path.join(data_dir, split)
    tar_files = sorted(
        [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".tar")]
    )

    n_expected = _total_samples_from_sizes(split_dir, tar_files)
    print(f"\n=== Building index for {split} ({len(tar_files)} tars, ~{n_expected} samples) ===")

    emb_dim = peek_emb_dim(tar_files[0], embedding_name)
    print(f"Embedding dim: {emb_dim}")

    out_dir = os.path.join(data_dir, "neighborhood_index", split)
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, f"embeddings_{embedding_name}.npy")

    print(f"Allocating memmap ({n_expected} x {emb_dim} float32 "
          f"= {n_expected * emb_dim * 4 / 1e9:.1f} GB on disk) at {emb_path}...")
    emb_mmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float32, shape=(n_expected, emb_dim)
    )

    # Pass: stream embeddings to memmap, keep only lightweight metadata in RAM
    all_ids = []
    all_lats = []
    all_lons = []
    offset = 0
    for tar_path in tqdm(tar_files, desc=f"Scanning {split}"):
        offset = stream_tar_into_memmap(
            tar_path, embedding_name, emb_mmap, offset,
            all_ids, all_lats, all_lons,
        )
    emb_mmap.flush()

    n = offset
    if n != n_expected:
        print(f"Warning: expected {n_expected} samples, got {n}. "
              f"Resizing memmap header (on-disk extra rows will be wasted).")
        # Rewrite the header with the actual size by reopening as a fresh memmap
        # of (n, emb_dim) and copying via a view — easier: just record actual n,
        # the consumer uses the index pickle's num_samples.
    print(f"Wrote {n} embeddings to memmap")

    all_lats = np.array(all_lats, dtype=np.float64)
    all_lons = np.array(all_lons, dtype=np.float64)

    print("Building KD-tree on 3D unit-sphere coords...")
    coords_3d = latlon_to_cartesian(all_lats, all_lons)
    tree = KDTree(coords_3d)

    search_radius = meters_to_cartesian_dist(radius_m)
    print(f"Querying neighbors within {radius_m}m "
          f"(cartesian chord={search_radius:.6f}) in batches of {query_batch_size}...")

    neighbor_index = {}
    neighbor_counts = np.zeros(n, dtype=np.int32)
    for start in tqdm(range(0, n, query_batch_size), desc="Neighbor query"):
        end = min(start + query_batch_size, n)
        batch_neighbors = tree.query_ball_point(coords_3d[start:end], r=search_radius)
        for local_i, nbrs in enumerate(batch_neighbors):
            i = start + local_i
            nbrs = [j for j in nbrs if j != i]
            if len(nbrs) > max_neighbors:
                dists = np.linalg.norm(coords_3d[nbrs] - coords_3d[i], axis=1)
                closest = np.argsort(dists)[:max_neighbors]
                nbrs = [nbrs[c] for c in closest]
            neighbor_index[all_ids[i]] = [all_ids[j] for j in nbrs]
            neighbor_counts[i] = len(nbrs)

    print(f"Neighbor stats: mean={neighbor_counts.mean():.2f}, "
          f"median={np.median(neighbor_counts):.0f}, "
          f"max={neighbor_counts.max()}, "
          f"zero={(neighbor_counts == 0).sum()} "
          f"({(neighbor_counts == 0).sum() / n * 100:.1f}%)")

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
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (e.g., plonk/datasets/osv5m)")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"], help="Splits to process")
    parser.add_argument("--embedding_name", type=str, default="street_clip",
                        help="Embedding name (e.g., street_clip, mobileclip2_s4)")
    parser.add_argument("--radius", type=int, default=500,
                        help="Neighborhood radius in meters (integer)")
    parser.add_argument("--max_neighbors", type=int, default=20,
                        help="Maximum neighbors per image stored in the index")
    parser.add_argument("--query_batch_size", type=int, default=10_000,
                        help="Batch size for KD-tree neighbor queries")
    args = parser.parse_args()

    for split in args.splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split}: {split_dir} not found")
            continue
        build_index_for_split(
            args.data_dir, split, args.embedding_name,
            args.radius, args.max_neighbors,
            query_batch_size=args.query_batch_size,
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()
