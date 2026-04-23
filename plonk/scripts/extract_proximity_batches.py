"""
Extract a few batches of images (anchor + neighbors) that are within 100m of
each other, using the pre-built test split streetclip r100 spatial index.
Saves each batch to proximity_batches/batch_{i}/anchor.jpg, neighbor_{j}.jpg,
and anchor_location.json with the anchor's GPS coordinates and metadata.
"""

import json
import pickle
import shutil
import tarfile
import glob
import os
import numpy as np

INDEX_PATH  = "plonk/datasets/osv5m/neighborhood_index/test/index_street_clip_r100.pkl"
EMB_PATH    = "plonk/datasets/osv5m/neighborhood_index/test/embeddings_street_clip.npy"
TAR_DIR     = "plonk/datasets/osv5m/test"
OUT_DIR     = "proximity_batches"

NEIGHBOR_COUNTS = [1, 2, 3, 4, 5]  # one batch per count, with exactly this many neighbors


def load_index(index_path):
    with open(index_path, "rb") as f:
        return pickle.load(f)


def extract_batches(index, neighbor_counts):
    """Return one batch per entry in neighbor_counts with exactly that many neighbors."""
    neighbor_index = index["neighbor_index"]
    batches = []
    for n in neighbor_counts:
        for anchor_id, nbr_ids in neighbor_index.items():
            if len(nbr_ids) == n:
                batches.append((anchor_id, nbr_ids))
                break
    return batches


def build_id_to_tar(tar_dir, all_ids):
    """Scan tar files once and map each needed image ID to its tar path + stem name."""
    needed = {str(iid) for iid in all_ids}
    found  = {}
    tar_paths = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    for tar_path in tar_paths:
        if not needed:
            break
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                name = member.name
                if name.endswith(".jpg"):
                    img_id = name.split(".")[0]
                    if img_id in needed:
                        found[img_id] = (tar_path, img_id)
                        needed.discard(img_id)
        if found:
            print(f"  scanned {tar_path}  ({len(found)}/{len(found)+len(needed)} found so far)")
    return found


def extract_from_tar(tar_path, stem, extension):
    member_name = f"{stem}.{extension}"
    with tarfile.open(tar_path, "r") as tf:
        f = tf.extractfile(tf.getmember(member_name))
        return f.read()


def save_bytes(data, out_path):
    with open(out_path, "wb") as f:
        f.write(data)


def main():
    print(f"Loading index...")
    index = load_index(INDEX_PATH)
    print(f"  radius_m={index['radius_m']}  num_samples={index['num_samples']}")

    batches = extract_batches(index, NEIGHBOR_COUNTS)

    all_ids = []
    for anchor_id, nbr_ids in batches:
        all_ids.append(anchor_id)
        all_ids.extend(nbr_ids)

    print(f"\nScanning {TAR_DIR} tars for {len(all_ids)} images...")
    id_to_tar = build_id_to_tar(TAR_DIR, all_ids)
    print(f"  found {len(id_to_tar)}/{len(all_ids)} images")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    for i, (anchor_id, nbr_ids) in enumerate(batches):
        batch_dir = os.path.join(OUT_DIR, f"batch_{i}")
        os.makedirs(batch_dir)

        anchor_str = str(anchor_id)
        if anchor_str in id_to_tar:
            tar_path, stem = id_to_tar[anchor_str]
            save_bytes(extract_from_tar(tar_path, stem, "jpg"),
                       os.path.join(batch_dir, "anchor.jpg"))
            meta = json.loads(extract_from_tar(tar_path, stem, "json").decode())
            location = {
                "latitude":  meta["latitude"],
                "longitude": meta["longitude"],
                "city":      meta.get("city"),
                "region":    meta.get("region"),
                "country":   meta.get("country"),
                "google_maps_url": (
                    f"https://www.google.com/maps?q={meta['latitude']},{meta['longitude']}"
                ),
            }
            with open(os.path.join(batch_dir, "anchor_location.json"), "w") as f:
                json.dump(location, f, indent=2)

        for j, nid in enumerate(nbr_ids):
            nid_str = str(nid)
            if nid_str in id_to_tar:
                tar_path, stem = id_to_tar[nid_str]
                save_bytes(extract_from_tar(tar_path, stem, "jpg"),
                           os.path.join(batch_dir, f"neighbor_{j}.jpg"))

        saved = os.listdir(batch_dir)
        print(f"\nbatch_{i}/  ({len(saved)} files): {sorted(saved)}")
        print(f"  anchor:    {anchor_id}  @ {location['latitude']:.5f}, {location['longitude']:.5f}"
              f"  ({location['city']}, {location['country']})")
        print(f"  neighbors: {nbr_ids}")

    print(f"\nDone. Images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
