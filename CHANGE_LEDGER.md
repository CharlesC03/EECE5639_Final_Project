# Change Ledger

Changes made by Claude, in reverse chronological order.

---

### 2026-04-23

**Prompt:** "What is the total portion of data in the western world?"
**Model:** Claude Sonnet 4.6
**Summary:** Added "Western World" region to `plonk/scripts/measure_region_coverage.py`. Defined as Europe (excl. Russia) + USA + Canada + Australia + New Zealand. Result: 65.96% of training data.

---

**Prompt:** "Print out the top 5 countries distribution of data for each number of neighbors"
**Model:** Claude Sonnet 4.6
**Summary:** Extended `plonk/scripts/measure_country_coverage.py` to also show top-5 country breakdown per neighbor-count group (0, 1, 2, 3, 4, 5+). Loads `nbrs` array from the GPS cache, classifies all points once, then filters by group to compute per-group country distributions.

---

**Prompt:** "Make a new script which lists out the percentage of data by country"
**Model:** Claude Sonnet 4.6
**Summary:** Created `plonk/scripts/measure_country_coverage.py`. Iterates all 177 Natural Earth 110m countries, assigns each of the 4.9M training points to a country using bbox pre-filter + matplotlib Path.contains_points, and prints a table sorted by image count descending with country name, count, and percentage.

---

**Prompt:** "Write a script to measure the percentage of training data in Europe, USA, and Japan individually and combined"
**Model:** Claude Sonnet 4.6
**Summary:** Created `plonk/scripts/measure_region_coverage.py`. Loads the GPS cache from `proximity_map_cache.npz`, downloads the Natural Earth 110m countries shapefile via cartopy's shapereader (pyshp only), and classifies all 4.9M points using bounding-box pre-filtering + matplotlib `Path.contains_points`. Europe excludes Russia. Results: Europe 35.46%, USA 24.70%, Japan 3.92%, combined 64.08%.

---

**Prompt:** "Move all of these new python files you made into plonk/scripts"
**Model:** Claude Sonnet 4.6
**Summary:** Moved `extract_proximity_batches.py` and `plot_proximity_map.py` from the repo root into `plonk/scripts/`. Both scripts still run from the repo root.

---

**Prompt:** "Make a script to extract all locations of training images grouped by proximity neighbor count and plot a map coloured by group — output as PDF for LaTeX; rasterize dots at 1000 DPI"
**Model:** Claude Sonnet 4.6
**Summary:** Created `plot_proximity_map.py`. Scans all 490 training tars in parallel (multiprocessing) to extract GPS coordinates, joins with `index_street_clip_r100.pkl` neighbor counts, caches to `proximity_map_cache.npz`, and renders `proximity_map.pdf` — a 14×7 inch matplotlib figure with a vector Natural Earth land outline (pyshp) and 1000-DPI rasterized scatter points (one colour per group: 0=grey, 1=blue, 2=green, 3=orange, 4=red, 5+=purple). PDF is ~6 MB. Re-runs load from cache; use `--rebuild` to re-scan tars.

---

**Prompt:** "Extract a few batches of images in proximity to each other inside the r100 for streetclip — do it for the test dataset / Save them as images into their own folder / Also save the anchor's actual location / Make each batch have its own amount of images (1, 2, 3, 4, 5 neighbors)"
**Model:** Claude Sonnet 4.6
**Summary:** Created `extract_proximity_batches.py`. Loads the pre-built test split spatial index (`index_street_clip_r100.pkl`), selects one anchor per neighbor count in `[1,2,3,4,5]`, scans test tars, and saves each batch to `proximity_batches/batch_{i}/anchor.jpg`, `neighbor_{j}.jpg`, and `anchor_location.json` (lat/lon, city, country, Google Maps URL). Output directory is wiped clean on each run.

---

### 2026-04-23

**Prompt:** "Can you add in the support for multi_image_mode, if there is not in the config for the model, ignore it and don't let you input multiple images into the model"
**Model:** Claude Sonnet 4.6
**Summary:** Modified `PlonkPipeline.__call__` in `plonk/pipe.py` to support grouped image input (`[[img1a, img1b], [img2a, img2b, img2c], ...]`). Added `multi_image_mode` flag to `MODELS` dict entries: `"average"` mean-fuses all images in a group into one embedding before the model; `"attention"` passes the first image as anchor (`batch["emb"]`) and the rest as `batch["neighbor_embs"]`/`batch["neighbor_mask"]` for model-side attention pooling. If `multi_image_mode` is absent from the model's entry and grouped input is provided, a `ValueError` is raised. Flat list input (one prediction per image) is unchanged.

---

### 2026-04-22

**Prompt:** "I don't mind concentrated attention, can you make the mask a variable I can change in the config?"
**Model:** Claude Sonnet 4.6
**Summary:** Added `use_attention_mask: bool = True` parameter to `NeighborhoodAttentionPooler` and `GeoAdaLNMLP` in `plonk/models/networks/mlp.py`. When `false`, passes `key_padding_mask=None` to `nn.MultiheadAttention`, allowing zero-padded slots to dilute the softmax over valid neighbors. Added `use_attention_mask: true` to both `geo_adaln_mlp_neighborhood.yaml` and `geo_adaln_mlp_neighborhood_rand_cond.yaml`.

---

**Prompt:** "Create the config files so I can use them in experiments"
**Model:** Claude Sonnet 4.6
**Summary:** Created three new dataset config files in `plonk/configs/dataset/`: `osv5m_emb_neighborhood_dinov2_r100.yaml` (dinov2_vitl14_registers at 100m), `osv5m_emb_neighborhood_dinov2_r500.yaml` (dinov2_vitl14_registers at 500m), and `osv5m_emb_neighborhood_streetclip_r500.yaml` (street_clip at 500m). All use `NeighborhoodWebdataset` and mirror the structure of the existing `osv5m_emb_neighborhood.yaml`. cond_dim is 1024 for all three (verified from memmap shapes).

---

### 2026-04-21

**Prompt:** "Would it be better to randomize the location of where the images are inputted into the model? ... Add this in as a parameter which i can adjust from the config file"
**Model:** Claude Sonnet 4.6
**Summary:** Added `shuffle_neighbors` parameter to `NeighborhoodAttentionPooler` and `GeoAdaLNMLP` in `plonk/models/networks/mlp.py`. When enabled, randomly permutes the K neighbor slots per batch item during training only (no-op at inference), so the model cannot learn a positional bias from neighbor order. Added `shuffle_neighbors: false` to `plonk/configs/model/network/geo_adaln_mlp_neighborhood.yaml` to expose the flag via config.

---

### 2026-04-20

**Prompt:** "Could you update the neighbor testing to also list the number of images its tested for each of the amount of neighbors testing so I can know the sample size"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Log per-bucket sample counts to W&B in `DiffGeolocalizer.on_test_epoch_end` (`plonk/models/module.py`). Added a `self.log("test/<label>/n_samples", count, reduce_fx="sum")` call alongside the existing metric logging so the number of samples contributing to each `test/0/`, `test/1/`, …, `test/overall/` bucket now appears in the wandb summary/JSON. The printed summary table already included the `n` column — this just mirrors it into the logged run state. Used `reduce_fx="sum"` so counts aggregate correctly across DDP ranks (default is mean).

**Prompt:** "I want to do learned cross-attention in the model"
**Model:** Claude Sonnet 4.6
**Summary:** Added `NeighborhoodAttentionPooler` to `plonk/models/networks/mlp.py` — uses `nn.MultiheadAttention` where the anchor is the query and the neighbor sequence (anchor + padded neighbors) is keys/values; out_proj zero-init so training starts identical to no-pooler baseline. Added `use_neighbor_attention` and `neighbor_attention_heads` args to `GeoAdaLNMLP`; pooler is skipped gracefully at inference when `neighbor_embs` is absent from the batch. Added `fuse_mode` param to `NeighborhoodWebdataset` in `neighborhood_dataset.py`: `"average"` preserves the existing weighted-average behavior; `"attention"` returns padded `neighbor_embs` `(max_neighbors, emb_dim)` and `neighbor_mask` `(max_neighbors,)` bool tensors alongside the raw anchor `emb`. Updated `osv5m_emb_neighborhood.yaml` to thread `fuse_mode` through to all three dataset splits (default `average`, so existing configs are unaffected). Created `plonk/configs/model/network/geo_adaln_mlp_neighborhood.yaml` and `plonk/configs/exp/osv_5m_neighborhood_attention.yaml` for the attention experiment.

**Prompt:** "Update the testing when using multiple images, to print out results for each amount of images used"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Added per-neighbor-count test evaluation in `plonk/models/module.py`. New optional `model.test_neighbor_counts` list (e.g. `[0, 1, 2, 3, 4, 5]`); when set, `DiffGeolocalizer.__init__` instantiates a `ModuleDict` of `test_metrics` copies keyed by k, and `test_step` dispatches to the new `test_step_vary_neighbors` which draws x_N once and samples once per k using the first k valid neighbors from `batch["neighbor_mask"]` (same noise across k for apples-to-apples comparison). `on_test_epoch_end` logs each metric as `test/k=K/<name>` and prints a formatted summary table via `_print_per_k_summary`. Updated `plonk/models/samplers/riemannian_flow_sampler.py` CFG path to also stack `neighbor_embs`/`neighbor_mask` (uncond half gets zeros → no neighbors attended), so the pooler sees neighbor context during CFG sampling instead of silently falling back to anchor-only. Added `test_neighbor_counts: [0, 1, 2, 3, 4, 5]` to `osv_5m_neighborhood_attention.yaml`. Baseline configs without `test_neighbor_counts` behave exactly as before. Smoke-tested: sampler runs per-k on GPU, samples remain identical at zero-init (as expected) and diverge progressively with k once out_proj is non-zero, confirming mask truncation works end-to-end through the CFG path.

**Prompt:** "I got the following error python plonk/train.py exp=osv_5m_neighborhood experiment_name=StreetCLIP_Multi_Images mode=eval … ValueError: kth(=4) out of bounds (4)"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Made `on_test_epoch_end` robust to `HaversineMetrics.manifold_metrics` failing on small buckets. `manifold_metrics` chunks samples into 20 splits and calls `np.argpartition(arr, k=manifold_k+1=4)` on each chunk, which raises `ValueError: kth(=4) out of bounds (4)` when a chunk has ≤4 samples — which now happens for the k=5 bucket on the r=100m test set since very few samples have 5+ neighbors. Wrapped `metric.compute()` in try/except; on failure we print a warning and fall back to `_basic_metrics(metric)`, a new static helper that reproduces the Haversine/Geoguessr/Accuracy section of `HaversineMetrics.compute` without the manifold step. Updated `_print_per_k_summary` to take the union of metric keys across rows and print '-' for missing cells, so the table renders correctly when different buckets have different schemas. No changes to the underlying `HaversineMetrics` — the existing code is left intact and the mitigation lives at the call site.

**Prompt:** "Filter per-k metrics to samples with only k real neighbors, not more or less. If there are more just sample from there to get k images. Also include an overall category which contains the overall results using the maximum number"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Changed per-k test bucketing in `test_step_vary_neighbors` (`plonk/models/module.py`). Previously every k bucket received all batch samples with their fusion capped to at most k available neighbors, which diluted signal because the majority of test samples have 0 neighbors. Now each k bucket receives only samples with `available >= k` (computed as `neighbor_mask.sum(-1)`) and always uses exactly k neighbors (the dataset's `random.sample` already returns a random subset when more than k are available, so taking the first k is a random pick). Added a new `"overall"` metric bucket that uses every sample with its full set of available neighbors — this is the honest end-to-end test metric. Factored bucket handling into a local `_run_bucket` helper and a static `_filter_batch` that indexes tensors / lists in the batch so ground-truth fields (gps, unique_country, etc.) stay aligned with the subsetted predictions. Updated `on_test_epoch_end` to iterate `[*str(k), "overall"]`, skip empty buckets, log as `test/<label>/<metric>`, and pass `(label, sample_count, metrics)` rows to `_print_per_k_summary`. Updated the summary table to include a label column ("0", "1", …, "overall") and a per-row sample count `n` so you can see both the trend and how many samples contributed. Smoke-tested on GPU with a hand-crafted batch of [0, 1, 3, 5] real neighbors: sample counts per bucket matched the expected (4, 3, 2, 2, 1, 1, 4) in both attention and averaging modes.

**Prompt:** "Below are the testing results for the average embeddings, why are the results the same for higher amounts of images? -> Yes do that"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Fixed a plateau in the per-k test sweep. `NeighborhoodWebdataset._fuse_neighbors` was randomly drawing `k_rand ∈ [min_neighbors, min(max_neighbors, available)]` on every call, including eval, so each sample only exposed a random subset of its real neighbors — which meant the per-k test couldn't exercise counts above that random cap for the majority of samples. Gated the random draw on `self.train`: training behavior unchanged (random k for augmentation); eval now deterministically picks `k = min(max_neighbors, available)` so the full neighbor set is loaded and the per-k sweep genuinely varies the count across all samples that have neighbors.

**Prompt:** "Do these test work for the averaging as well? -> Make that fix"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** Extended per-k testing to work for averaging-mode models as well. `NeighborhoodWebdataset._fuse_neighbors` now emits padded `neighbor_embs` + `neighbor_mask` in **both** fuse modes (previously only in attention mode), and additionally stores `anchor_emb` (the raw pre-average anchor) in average mode so test-time code can reconstruct per-k averages. Training in averaging mode is unchanged — `sample["emb"]` still carries the full weighted average. `test_step_vary_neighbors` now detects the model mode by checking `ema_network.neighbor_pooler`: with a pooler it truncates the mask (attention path); without one it rebuilds the weighted average `(anchor + w · Σneighbors[:k]) / (1 + w · effective_k)` using `batch["anchor_emb"]` and passes that as `emb`. Added `test_neighbor_weight` field on the lightning module (default 1.0) so the reconstruction weight can match `dataset.neighborhood.neighbor_weight`. Added `test_neighbor_counts: [0,1,2,3,4,5]` and `test_neighbor_weight: 1.0` to `osv_5m_neighborhood.yaml`. Smoke-tested both experiments on GPU — each produces distinct samples per k, confirming the averaging path correctly rebuilds from raw tensors.

---

### 2026-04-18

**Prompt:** "I tried to run the training function but it failed. [FileNotFoundError: ...index_street_clip_r100.pkl]"
**Model:** Claude Opus 4.7 (1M context)
**Summary:** The user changed `dataset.neighborhood.radius` from 500 to 100 but had no r=100 index built. Before building, refactored `plonk/scripts/build_spatial_index.py` to stream embeddings directly into the memory-mapped `.npy` as tars are scanned (prior version accumulated ~4.9M × 1024 float32 in a Python list ≈ 20 GB peak RAM, which would have OOMed on the 30 GB machine). Also switched neighbor queries from `tree.query_ball_tree(tree, r)` (materializes all pairs at once) to batched `tree.query_ball_point(coords[i:i+B], r)` with `--query_batch_size` knob, and changed `--radius` from float to int so filenames look like `r100.pkl` not `r100.0.pkl`. Added `peek_emb_dim` + `_total_samples_from_sizes` helpers to size the memmap up front from `sizes.json`. Smoke-tested on val split (49k samples, ~7s). Train + test builds completed after ~15 min. Neighbor coverage at r=100m: train 45.7% have ≥1 neighbor (mean=0.67, max=9), val 0.6% (max=1), test 14.8% (mean=0.18, max=8).

---

### 2026-04-17

**Prompt:** "You wrote code to find images near eachother so during the training the model could learn on multiple near by images. Can you find the code you wrote and verify its functioning and if not or if there are parts missing fix it."
**Model:** Claude Opus 4.7 (1M context)
**Summary (2):** Added `plonk/configs/exp/osv_5m_neighborhood.yaml` — experiment config mirroring `osv_5m_default_dino.yaml` but with `dataset: osv5m_emb_neighborhood`. Sets `radius=500, min_neighbors=0, max_neighbors=5, neighbor_weight=1.0`, `full_batch_size=1024`, `experiment_name_suffix=neighborhood_r500_k5`. Verified Hydra composes correctly (street_clip, 1024 cond_dim, `NeighborhoodWebdataset` target).

**Summary:** Verified neighborhood sampling stack (`plonk/scripts/build_spatial_index.py`, `plonk/data/neighborhood_dataset.py`, `plonk/configs/dataset/osv5m_emb_neighborhood.yaml`) end-to-end against the existing `street_clip_r500` index on `plonk/datasets/osv5m/train/`. Pipeline produces correct `(emb, gps)` tensors and actually fuses neighbor embeddings (29/50 samples modified in the smoke test). Removed dead code in `neighborhood_dataset.py`: the `_strip_internal_keys` step was a no-op because `wds.map` re-injects `__key__` into the sample after any mapper; also removed the redundant identity transform for `__key__` in `outputs_transforms`. Kept the `outputs_rename["__key__"] = "__key__"` entry because `filter_dict_keys` does drop keys not in its arg list. `__key__` now rides through to the collated batch as a list of strings, which the training module ignores.

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
