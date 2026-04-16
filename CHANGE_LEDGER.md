# Change Ledger

Changes made by Claude, in reverse chronological order.

---

### 2026-04-16

**Prompt:** "For the demo I need to install other packages, is there a way to download both the demo and training packages? Yes please"
**Model:** Claude Sonnet 4.6
**Summary:** Updated `setup.py`: added `open-clip-torch>=2.20.0` and `timm` to base `install_requires` (needed for `MobileClipFeatureExtractor` in `pipe.py`); added `joblib` and `peft` to the `train` extras. Both extras can now be installed together with `pip install -e ".[train,demo]"`.

**Prompt:** "For this code would it be possible to see how the randomness evolves over time into a position?"
**Model:** Claude Sonnet 4.6
**Summary:** Added `return_trajectories=False` parameter to `PlonkPipeline.__call__` in `plonk/pipe.py`. When `True`, the sampler's intermediate states are converted to GPS degrees and returned alongside the final output as a list of `(batch_size, 2)` arrays — one per denoising step.

**Prompt:** "I tried running the next cell but got the following error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x1024 and 768x512)"
**Model:** Claude Sonnet 4.6
**Summary:** Added `MobileClipFeatureExtractor` class to `plonk/pipe.py` using `open_clip` with `MobileCLIP2-S4 (pretrained="dfndr2b")` — the same model used during training. Added `"mobile_clip"` branch to `load_prepocessing()`. Updated the `local_models/my_plonk_model` entry in `MODELS` from `emb_name: "street_clip"` to `emb_name: "mobile_clip"` to match the 768-dim embeddings the checkpoint was trained with.

---

### 2026-04-15

**Prompt:** "Why am I getting the following error? TypeError: LinearScheduler.__init__() got an unexpected keyword argument 'tau'"
**Model:** Claude Sonnet 4.6
**Summary:** Removed `tau: 1.0` from `train_noise_scheduler` and `inference_noise_scheduler` sections in `plonk/configs/exp/osv_5m_default_dino.yaml` and `plonk/configs/exp/osv_5m_default_mobile.yaml`. The configs were using the `linear` scheduler but had `tau` overrides copied from a sigmoid-based config — `LinearScheduler` doesn't accept `tau`.

---

### 2026-04-12

**Prompt:** "I tried running it but it broke half way through its validation run, why?"
**Model:** Claude Sonnet 4.6
**Summary:** Fixed typo in `plonk/configs/config.yaml` line 52: `val_check_interval: 250   00` → `val_check_interval: 25000`. Also reverted `num_workers` back to 0 for val/test dataloaders in `plonk/data/datamodule.py`. The previous change to use 12 workers caused `wds.split_by_worker` to split shards across 12 workers — if fewer shards existed than workers, some workers got nothing and validation stopped early (half-way through).

---

### 2026-04-11

**Prompt:** "Training pausing during training — diagnosed as DataLoader round-robin stall + batch size too large for data pipeline"
**Summary:** (Claude Opus 4.6)
- Added `accumulate_grad_batches: 4` to `plonk/configs/exp/osv_5m_default_mobile.yaml` with `full_batch_size: 2048` (effective batch 8192 via gradient accumulation, keeps data pipeline ahead of GPU)
- Added `num_workers` and `pin_memory` to val and test dataloaders in `plonk/data/datamodule.py` (were hardcoded to `num_workers=0`)

**Prompt:** "ValueError: didn't find dinov2_vitl14_registers.npy in mobileclip2_s4 tars"
**Summary:** (Claude Opus 4.6) Fixed `embedding_name` in `plonk/configs/dataset/osv5m_emb_mobile.yaml` from `dinov2_vitl14_registers` to `mobileclip2_s4` to match the actual `.npy` files in the extracted embedding tars.

**Prompt:** "Too many open files error after 59/490 tar files — screen turned off during tmux run"
**Summary:** (Claude Opus 4.6) Fixed file descriptor leak in `mobile_predict.py`:
- Removed `persistent_workers=True` from DataLoader — it provides no benefit here since a new DataLoader with a new dataset is created every iteration, so workers can't be reused
- Added explicit `del loader` after each iteration to promptly shut down worker processes and free file descriptors

**Prompt:** "Change it to not copy the images, just keep the npy files and the json"
**Summary:** (Claude Opus 4.6) Removed image copying from `mobile_predict.py` output tars — `write_tar` no longer writes `.jpg` files, only `.json` and `.npy` embeddings, dramatically reducing output tar size.

**Prompt:** "What was the limitation?" (profiler showed aten::copy_ at 60% CPU time)
**Summary:** (Claude Opus 4.6) Eliminated cross-process data transfer bottleneck in `mobile_predict.py`:
- `__getitem__` now only returns `(tensor, idx)` instead of `(tensor, key, jpg_bytes, json_bytes)`
- Raw bytes for tar writing stay in `dataset.samples`, accessed in main process after inference
- Cuts ~90% of DataLoader IPC overhead — workers no longer serialize tens of KB of image/json bytes per sample

**Prompt:** "GPU utilization jumping 100% to 3% constantly — is there a better way with pytorch?"
**Summary:** (Claude Opus 4.6) Rewrote `plonk/data/extract_embeddings/mobile_predict.py` to use a proper PyTorch `DataLoader` pipeline:
- `TarImageDataset` reads one tar into memory (only jpg+json, skips .npy), applies `preprocess` transform in `__getitem__`
- `DataLoader` with `num_workers=8`, `pin_memory=True`, `persistent_workers=True` handles CPU/GPU overlap
- Workers preprocess the next batch while the GPU computes the current one — eliminates the idle gap
- Kept `torch.compile`, `torch.autocast`, batch size 512

**Prompt:** "Is this code optimal?" (on mobile_predict.py)
**Summary:** (Claude Opus 4.6) Optimized `plonk/data/extract_embeddings/mobile_predict.py`:
- Stream tar files instead of loading all contents into memory at once
- Skip existing `.npy` embeddings during read (only keep `.jpg` and `.json`)
- Process and write in batches as samples are streamed, instead of accumulating all images first
- Added `torch.compile(model, mode="max-autotune")` for faster inference
- Added `torch.autocast` with float16 for the 4090
- Moved `@torch.no_grad()` to function decorator

**Prompt:** "Read the tar files, extract the json, run processing on images, save embeddings as npy in new tars matching the existing structure"
**Summary:** (Claude Opus 4.6) Rewrote `plonk/data/extract_embeddings/mobile_predict.py` to read from existing WebDataset tar files instead of raw image directories:
- Reads source tars from `plonk/datasets/osv5m/{train,val,test}/` (where the downloaded dataset lives)
- For each tar: decodes JPGs, batches through MobileCLIP2-S4, writes new tars with `<id>.jpg`, `<id>.json`, and `<id>.mobileclip2_s4.npy`
- Outputs to `datasets/osv5m/embeddings/mobileclip2_s4/{split}/` with matching shard structure and `sizes.json`
- Can be used directly with `GPSWebdataset` by pointing the dataset config root at the output directory

**Prompt:** "Yes update the file [mobile_predict.py]"
**Summary:** Rewrote `plonk/data/extract_embeddings/mobile_predict.py` to use MobileCLIP2-S4 via `open_clip`:
- Replaced HuggingFace transformers pattern with `open_clip.create_model_and_transforms('MobileCLIP2-S4', pretrained='dfndr2b')`
- Inlined `reparameterize_model` (copied from Apple's repo) to avoid needing `ml-mobileclip` as a dependency
- Passed `preprocess` transform into `ImageWithPathDataset` (following DINOv2 pattern)
- Changed inference loop to `model.encode_image(images)` on stacked tensors
- Updated output path to `mobileclip2_s4`

**Prompt:** "Should I add this to setup.py and or requirements.txt?"
**Summary:** Added `open-clip-torch>=2.20.0` to `requirements.txt` for MobileCLIP2-S4 embedding extraction support.

**Prompt:** "Add persistent workers, also is there no way for me to enable pin memory?"
**Summary:** Enabled `persistent_workers=True` and `pin_memory=True` on the train `WebLoader` in `plonk/data/datamodule.py:84-91` to reduce data-loading overhead and improve GPU utilization.

**Prompt:** "Is there a way to make persistent workers and pin memory a configurable setting?"
**Summary:** Made `persistent_workers` and `pin_memory` configurable via Hydra:
- Added both fields to `plonk/configs/computer/rtx4090.yaml` (set to `True`)
- Wired them through `plonk/configs/config.yaml` datamodule section with `false` defaults for other computer configs
- Added constructor params to `ImageDataModule` in `plonk/data/datamodule.py`, used in train dataloader

**Prompt:** "Can I change [prefetch_factor] in my config? Yes please do"
**Summary:** Added `prefetch_factor` as a configurable setting following the same pattern as `persistent_workers`/`pin_memory`:
- Added `prefetch_factor: 4` to `plonk/configs/computer/rtx4090.yaml`
- Wired through `plonk/configs/config.yaml` with default of 2
- Added to `ImageDataModule` constructor and train `WebLoader` in `plonk/data/datamodule.py`

---

### 2026-04-10

**Prompt:** "Build out the code [for neighborhood sampling]. Make sure not to interfere with the current code, just make it an additional option for sampling from the distribution which I can select in the config"
**Summary:** Added neighborhood-aware embedding fusion as an optional dataset mode:
- Created `plonk/scripts/build_spatial_index.py` — pre-computation script that scans webdataset tar files, builds a KD-tree spatial index from image coordinates, groups images by proximity, and saves neighbor groups + embeddings as a memory-mapped file.
- Created `plonk/data/neighborhood_dataset.py` — `NeighborhoodWebdataset` class (drop-in replacement for `GPSWebdataset`) that, for each sample, looks up its pre-computed neighbor group, randomly samples a variable number of neighbors, and averages their embeddings with the anchor image's embedding before passing to the model.
- Created `plonk/configs/dataset/osv5m_emb_neighborhood.yaml` — Hydra config to enable neighborhood sampling via `dataset: osv5m_emb_neighborhood`. Configurable parameters: `radius`, `min_neighbors`, `max_neighbors`, `neighbor_weight`.

---

### 2026-04-09

**Prompt:** "Shouldn't I use a device? Not specific to cuda?"
**Summary:** Updated `plonk/models/samplers/custom_sampler.py:78` — replaced hardcoded `'cuda'` in `torch.amp.autocast` with `x.device.type` to be device-agnostic (works on CPU, CUDA, MPS, etc.).

**Prompt:** "Make the weights_only=false"
**Summary:** Updated `plonk/train.py:46` — added `weights_only=False` to `load_from_checkpoint` call to fix PyTorch 2.6+ deserialization error with OmegaConf objects in checkpoints.

**Prompt:** "When evaluating, it prints updates which interfere with the progress bar, fix it so that the progress bar does get broken"
**Summary:** Updated `plonk/models/module.py` — replaced `print()` calls with `tqdm.write()` at lines 514, 527, and 549 to prevent breaking the tqdm progress bar during evaluation. Original prints commented out.

**Prompt:** "Is there a way I can log this or print this or add it to the progress bar?" / "Use self.log and see how that works"
**Summary:** Updated `plonk/models/module.py` — replaced remaining `print("ODE solver")` and `print(f"nfe: {nfe[0]}")` with `self.log()` calls to log NFE to W&B without breaking the progress bar.

**Prompt:** "How do I clear this? ... isinstance(treespec, LeafSpec) is deprecated..."
**Summary:** Updated `plonk/train.py` — added `warnings.filterwarnings` to suppress PyTorch Lightning `LeafSpec` deprecation warning.

**Prompt:** "Evaluation not updating W&B website at the end of running"
**Summary:** Added `wandb.finish()` at the end of `main()` in `plonk/train.py` to ensure W&B syncs all logged metrics (including test metrics) before the process exits.

**Prompt:** "What is the best function to be using here? I want them to log and sync with W&B" / "I want it to run immediately"
**Summary:** Replaced commented-out `self.print(f"nfe: {nfe[0]}")` with `self.log("test/nfe", ...)` using `on_step=True, on_epoch=False` in `plonk/models/module.py:547` to log NFE to W&B immediately per batch.
