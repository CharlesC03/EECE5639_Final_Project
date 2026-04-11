# Change Ledger

Changes made by Claude, in reverse chronological order.

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
