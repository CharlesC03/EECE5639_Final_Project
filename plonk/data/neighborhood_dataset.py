"""
Neighborhood-aware webdataset that enriches each sample's embedding
by averaging in embeddings from spatially nearby images.

At each training step, for a given anchor image:
1. Look up its pre-computed neighbor group (images within radius R)
2. Randomly sample K neighbors (K is random between min_k and max_k)
3. Average the anchor + neighbor embeddings to produce the conditioning signal

This is a drop-in replacement for GPSWebdataset — it produces samples with
the same keys ("emb", "gps", etc.) so the rest of the pipeline is unchanged.
"""

import json
import logging
import os
import pickle
import random
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import webdataset as wds

from plonk.data.webdataset import (
    GPSWebdataset,
    SharedEpoch,
    detshuffle2,
    filter_dict_keys,
    get_dataset_size,
    get_gps,
    get_attr,
    log_and_continue,
    tarfile_to_samples_nothrow,
)


class NeighborhoodWebdataset(wds.DataPipeline):
    """GPSWebdataset variant that fuses each sample's embedding with neighbors.

    Requires a pre-built spatial index (from build_spatial_index.py).
    """

    def __init__(
        self,
        root,
        index_dir,
        embedding_name="street_clip",
        radius=500,
        min_neighbors=0,
        max_neighbors=5,
        neighbor_weight=1.0,
        fuse_mode="average",
        image_transforms=None,
        distributed=True,
        train=True,
        epoch=0,
        seed=3407,
        return_image=False,
        shard_shuffle_size=2000,
        shard_shuffle_initial=500,
        sample_shuffle_size=5000,
        sample_shuffle_initial=1000,
        metadata_attributes=[],
    ):
        """
        Args:
            root: path to tar files
            index_dir: path to neighborhood_index/{split}/ directory
            embedding_name: which embedding to use (e.g., "street_clip")
            radius: radius in meters used when building the index
            min_neighbors: minimum number of neighbors to sample (0 = sometimes use anchor alone)
            max_neighbors: maximum number of neighbors to sample per anchor
            neighbor_weight: weight of each neighbor relative to anchor (1.0 = equal weight)
            fuse_mode: "average" — weighted-average into emb (original behavior);
                       "attention" — return padded neighbor_embs + neighbor_mask for model-side fusion
            image_transforms: optional image transforms
            Other args: same as GPSWebdataset
        """
        self.image_transforms = image_transforms
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.neighbor_weight = neighbor_weight
        self.fuse_mode = fuse_mode
        self.train = train

        # Load the pre-built spatial index
        index_path = os.path.join(
            index_dir, f"index_{embedding_name}_r{radius}.pkl"
        )
        print(f"Loading neighborhood index from {index_path}...")
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        self.id_to_idx = index_data["id_to_idx"]
        self.neighbor_index = index_data["neighbor_index"]
        print(
            f"  Loaded index: {index_data['num_samples']} samples, "
            f"radius={index_data['radius_m']}m, "
            f"max_neighbors={index_data['max_neighbors']}"
        )

        # Load the embeddings memmap
        emb_path = os.path.join(index_dir, f"embeddings_{embedding_name}.npy")
        print(f"Loading embeddings memmap from {emb_path}...")
        self.embeddings = np.load(emb_path, mmap_mode="r")
        print(f"  Embeddings shape: {self.embeddings.shape}")

        # Build the webdataset pipeline (same as GPSWebdataset but preserving __key__)
        dataset_tar_files = []
        if " " in root:
            root = root.split(" ")
        if isinstance(root, str):
            tar_files = sorted(f for f in os.listdir(root) if f.endswith(".tar"))
            first_tar = tar_files[0].split(".")[0]
            last_tar = tar_files[-1].split(".")[0]
            for tf in tar_files:
                dataset_tar_files.append(f"{root}/{tf}")
            dataset_pattern = f"{root}/{{{first_tar}..{last_tar}}}.tar"
            self.num_samples, _ = get_dataset_size(dataset_pattern)
        elif isinstance(root, list):
            num_samples = 0
            for r in root:
                tar_files = sorted(f for f in os.listdir(r) if f.endswith(".tar"))
                first_tar = tar_files[0].split(".")[0]
                last_tar = tar_files[-1].split(".")[0]
                for tf in tar_files:
                    dataset_tar_files.append(f"{r}/{tf}")
                num_samples += get_dataset_size(
                    f"{r}/{{{first_tar}..{last_tar}}}.tar"
                )[0]
            self.num_samples = num_samples
        else:
            raise ValueError(f"root must be a string or list of strings")

        from lightning_fabric.utilities.rank_zero import _get_rank

        rank = _get_rank()
        self.shared_epoch = SharedEpoch(epoch)
        pipeline = [wds.SimpleShardList(dataset_tar_files)]

        if distributed:
            if train:
                pipeline.extend(
                    [
                        detshuffle2(
                            bufsize=shard_shuffle_size,
                            initial=shard_shuffle_initial,
                            seed=seed,
                            epoch=self.shared_epoch,
                        ),
                        wds.split_by_node,
                        wds.split_by_worker,
                        tarfile_to_samples_nothrow,
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        ),
                    ]
                )
            else:
                pipeline.extend(
                    [wds.split_by_node, wds.split_by_worker, tarfile_to_samples_nothrow]
                )
        else:
            if train:
                pipeline.extend(
                    [
                        wds.shuffle(
                            bufsize=shard_shuffle_size,
                            initial=sample_shuffle_initial,
                        ),
                        wds.split_by_worker,
                        tarfile_to_samples_nothrow,
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        ),
                    ]
                )
            else:
                pipeline.extend([wds.split_by_worker, tarfile_to_samples_nothrow])

        # Rename and transform (same as GPSWebdataset but also extracting __key__)
        outputs_transforms = OrderedDict()
        outputs_rename = OrderedDict()
        if return_image:
            outputs_rename["img.jpg"] = "jpg;png;webp;jpeg"
            outputs_transforms["img.jpg"] = (
                self.image_transforms
                if self.image_transforms is not None
                else lambda x: x
            )
        if embedding_name is not None:
            outputs_rename["emb.npy"] = f"{embedding_name}.npy"
            outputs_transforms["emb.npy"] = lambda x: torch.from_numpy(x)
        if metadata_attributes:
            for attr in metadata_attributes:
                outputs_rename[f"{attr}.json"] = "json"
                outputs_transforms[f"{attr}.json"] = partial(get_attr, attr=attr)
        outputs_rename["gps"] = "json"
        outputs_transforms["gps"] = get_gps

        # Preserve __key__ through filter_dict_keys so _fuse_neighbors can read it.
        # (wds.rename/map_dict/decode all preserve it automatically, but filter_dict_keys
        # drops any key not explicitly listed.)
        outputs_rename["__key__"] = "__key__"

        pipeline.extend(
            [
                wds.rename(**outputs_rename),
                filter_dict_keys(*outputs_rename.keys(), handler=log_and_continue),
            ]
        )
        if return_image:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        else:
            pipeline.append(wds.decode(handler=log_and_continue))
        pipeline.extend(
            [
                wds.map_dict(**outputs_transforms, handler=log_and_continue),
                wds.rename(
                    **{k.split(".")[0]: k for k in outputs_transforms.keys()},
                ),
            ]
        )

        pipeline.append(wds.map(self._fuse_neighbors, handler=log_and_continue))

        super().__init__(*pipeline)

    def _fuse_neighbors(self, sample):
        """Look up neighbors and fuse according to fuse_mode.

        "average": weighted-average neighbor embeddings into sample["emb"] (original behavior).
        "attention": leave sample["emb"] as the raw anchor and add:
            sample["neighbor_embs"]: (max_neighbors, emb_dim) zero-padded
            sample["neighbor_mask"]: (max_neighbors,) bool — True = valid slot
        """
        image_id = sample.get("__key__", None)
        k = 0
        selected_idxs = []

        if image_id is not None and image_id in self.neighbor_index:
            neighbors = self.neighbor_index[image_id]
            if len(neighbors) > 0:
                k = random.randint(
                    min(self.min_neighbors, len(neighbors)),
                    min(self.max_neighbors, len(neighbors)),
                )
                if k > 0:
                    selected = random.sample(neighbors, k)
                    selected_idxs = [self.id_to_idx[nid] for nid in selected]

        if self.fuse_mode == "average":
            if k > 0:
                neighbor_embs = torch.from_numpy(
                    self.embeddings[selected_idxs].copy()
                )
                anchor_emb = sample["emb"]
                total_weight = 1.0 + k * self.neighbor_weight
                sample["emb"] = (
                    anchor_emb + self.neighbor_weight * neighbor_embs.sum(dim=0)
                ) / total_weight

        elif self.fuse_mode == "attention":
            emb_dim = sample["emb"].shape[-1]
            max_k = self.max_neighbors
            padded = torch.zeros(max_k, emb_dim, dtype=sample["emb"].dtype)
            mask = torch.zeros(max_k, dtype=torch.bool)
            if k > 0:
                n_embs = torch.from_numpy(self.embeddings[selected_idxs].copy())
                padded[:k] = n_embs
                mask[:k] = True
            sample["neighbor_embs"] = padded
            sample["neighbor_mask"] = mask

        else:
            raise ValueError(f"Unknown fuse_mode: {self.fuse_mode!r}")

        return sample

    def __len__(self):
        return self.num_samples
