import io
import tarfile
import json
from pathlib import Path
from threading import Thread

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm
from timm.utils import reparameterize_model

# --- Config ---
INPUT_BASE = Path("plonk/datasets/osv5m")
OUTPUT_BASE = Path("plonk/datasets/osv5m/embeddings/mobileclip2_s4")
SPLITS = ["train", "val", "test"]
BATCH_SIZE = 512
NUM_WORKERS = 8
EMBEDDING_NAME = "mobileclip2_s4"

# --- Model setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "MobileCLIP2-S4", pretrained="dfndr2b"
)
model.eval()
model.to(device)
model = reparameterize_model(model)
model = torch.compile(model, mode="max-autotune")


class TarImageDataset(Dataset):
    """Dataset that reads all jpg/json pairs from a single tar file.

    Only returns preprocessed tensors + sample index from __getitem__
    to minimize cross-process data transfer. Raw bytes for tar writing
    are accessed directly from self.samples in the main process.
    """

    def __init__(self, tar_path, transform):
        self.transform = transform
        self.samples = []  # list of (key, jpg_bytes, json_bytes)

        with tarfile.open(tar_path, "r") as tar:
            current_key = None
            jpg_data = None
            json_data = None

            for member in tar:
                f = tar.extractfile(member)
                if f is None:
                    continue
                name = member.name
                key = name.split(".")[0]

                if key != current_key:
                    if current_key is not None and jpg_data is not None:
                        self.samples.append((current_key, jpg_data, json_data))
                    current_key = key
                    jpg_data = None
                    json_data = None

                if name.endswith((".jpg", ".jpeg", ".png")):
                    jpg_data = f.read()
                elif name.endswith(".json"):
                    json_data = f.read()

            if current_key is not None and jpg_data is not None:
                self.samples.append((current_key, jpg_data, json_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, jpg_bytes, _ = self.samples[idx]
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        tensor = self.transform(img)
        # Only return the tensor and index — no bulky bytes across processes
        return tensor, idx


def collate_fn(batch):
    tensors, indices = zip(*batch)
    return torch.stack(tensors), list(indices)


@torch.no_grad()
def process_split(split):
    input_dir = INPUT_BASE / split
    output_dir = OUTPUT_BASE / split
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(input_dir.glob("*.tar"))
    if not tar_files:
        print(f"No tar files found in {input_dir}, skipping {split}")
        return

    sizes = {}
    print(f"\nProcessing {split}: {len(tar_files)} tar files")

    write_thread = None
    prefetch_thread = None
    prefetched_dataset = [None]  # mutable container for thread result

    def write_tar(out_path, samples, embeddings):
        """Write output tar on a background thread."""
        written = 0
        with tarfile.open(out_path, "w") as out_tar:
            for idx, (key, jpg_bytes, json_bytes) in enumerate(samples):
                if embeddings[idx] is None:
                    continue
                if json_bytes is not None:
                    _add_bytes_to_tar(out_tar, f"{key}.json", json_bytes)
                emb_buf = io.BytesIO()
                np.save(emb_buf, embeddings[idx])
                _add_bytes_to_tar(
                    out_tar, f"{key}.{EMBEDDING_NAME}.npy", emb_buf.getvalue()
                )
                written += 1
        sizes[out_path.name] = written

    def prefetch_tar(path):
        """Load next tar dataset on a background thread."""
        prefetched_dataset[0] = TarImageDataset(path, preprocess)

    # Prefetch the first tar
    prefetch_thread = Thread(target=prefetch_tar, args=(tar_files[0],))
    prefetch_thread.start()

    for i, tar_path in enumerate(tqdm(tar_files, desc=f"{split} tars")):
        # Wait for this tar's dataset to be ready
        prefetch_thread.join()
        dataset = prefetched_dataset[0]

        # Immediately start prefetching the next tar
        if i + 1 < len(tar_files):
            prefetch_thread = Thread(target=prefetch_tar, args=(tar_files[i + 1],))
            prefetch_thread.start()

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=collate_fn,
        )

        # Collect all embeddings indexed by sample position
        all_embeddings = [None] * len(dataset)
        for tensor_batch, indices in loader:
            tensor_batch = tensor_batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16):
                embeddings = model.encode_image(tensor_batch)
            embeddings = embeddings.float().cpu().numpy()
            for emb, idx in zip(embeddings, indices):
                all_embeddings[idx] = emb

        # Shut down workers to free file descriptors before next iteration
        del loader

        # Wait for previous tar write to finish before starting the next one
        if write_thread is not None:
            write_thread.join()

        # Write output tar in background while next tar starts processing
        out_tar_path = output_dir / tar_path.name
        write_thread = Thread(
            target=write_tar,
            args=(out_tar_path, dataset.samples, all_embeddings),
        )
        write_thread.start()

    # Wait for final tar write
    if write_thread is not None:
        write_thread.join()

    with open(output_dir / "sizes.json", "w") as f:
        json.dump(sizes, f)
    print(f"{split} done: {sum(sizes.values())} samples across {len(sizes)} shards")


def _add_bytes_to_tar(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)
    print("\nAll done!")
