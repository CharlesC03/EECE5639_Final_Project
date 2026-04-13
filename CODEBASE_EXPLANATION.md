# How PLONK Works: Complete Codebase Walkthrough

## What is PLONK?

PLONK is a **probabilistic geolocation** system. Given a photograph, it predicts *where on Earth* the photo was taken, outputting GPS coordinates (latitude/longitude). Unlike classification-based approaches that bucket the world into cells, PLONK uses **generative models** (diffusion, flow matching, von Mises-Fisher distributions) to model a full probability distribution over possible locations on the sphere.

---

## High-Level Architecture

```
Image  -->  [Feature Extractor]  -->  Embedding  -->  [Generative Model]  -->  GPS coordinates
             (StreetCLIP/DINOv2)       (1024-d)       (conditioned denoiser)     (lat, lon)
```

The system has these major components:

1. **Feature extraction** (frozen, pre-trained): StreetCLIP or DINOv2 encodes images into a 1024-d embedding vector
2. **Coordinate representation**: GPS (lat/lon) is converted to 3D Cartesian points on the unit sphere
3. **Generative backbone**: A conditioned MLP predicts vector fields/noise, trained with various loss functions
4. **Sampling**: Iterative denoising at inference to produce location predictions

---

## Data Pipeline

**`plonk/data/webdataset.py:29`** — `GPSWebdataset` loads data from sharded `.tar` files (WebDataset format). Each sample contains:
- An image (or pre-computed embedding `.npy` file)
- A JSON with `latitude` and `longitude`
- Optional metadata (country, region, city)

GPS coordinates are converted to radians via `get_gps()` at line 192. The datamodule (`plonk/data/datamodule.py`) wraps this with `wds.WebLoader` for distributed training.

**Preprocessing chain** (at training time, `module.py:46-47`):
1. `data_preprocessing` = `GPStoCartesian` (`preprocessing.py:25`): converts `(lat_rad, lon_rad)` to 3D Cartesian `(x, y, z)` on the unit sphere
2. `cond_preprocessing` = `PrecomputedPreconditioning`: passes through pre-computed embeddings as-is

---

## The Neural Network (Where to Modify Architecture)

The core network is **`GeoAdaLNMLP`** (`plonk/models/networks/mlp.py:73`). This is a **conditional MLP with Adaptive Layer Normalization (AdaLN)**:

```
Input: batch dict with keys "y" (noisy position), "gamma" (noise level), "emb" (image embedding)

1. Time embedding:  gamma -> PositionalEmbedding -> MLP -> t  (dim//4 -> dim*4)
2. Condition:       emb -> Linear -> cond
3. Combined cond:   cond = cond + t
4. Initial map:     y (3D) -> Linear -> x (dim)
5. N repeated AdaLN blocks:
     For each block:
       gamma, mu, sigma = AdaMap(cond)          # condition -> 3 modulation vectors
       x_res = (1 + gamma) * LayerNorm(x) + mu  # adaptive normalization
       x = x + MLP(x_res) * sigma               # gated residual
6. Final AdaLN + Linear -> output (3D)
```

**Key parameters** (set in config, e.g., `configs/exp/osv_5m_geoadalnmlp_r3_small_sigmoid_flow.yaml`):
- `dim: 512` — hidden dimension
- `depth: 12` — number of AdaLN blocks
- `expansion: 4` — MLP expansion factor
- `cond_dim: 1024` — embedding dimension from feature extractor
- `input_dim: 3` — 3D Cartesian coordinates

**`AdaLNMLPBlock`** (`mlp.py:54`): Each block does adaptive layer norm conditioning + a gated MLP residual. The `ada_map` produces 3 vectors (gamma, mu, sigma) from the condition, used to modulate, shift, and gate.

There are also **transformer building blocks** in `plonk/models/networks/transformers.py` (SelfAttentionBlock, CrossAttentionBlock) but they are **not currently used** by any experiment config — only the MLP variants are wired up.

### Variant Networks

- **`GeoAdaLNMLPVonFisher`** (`mlp.py:105`): Same AdaLN stack but no time input. Outputs `(mu, kappa)` — a mean direction and concentration parameter for a von Mises-Fisher distribution on the sphere.
- **`GeoAdaLNMLPVonFisherMixture`** (`mlp.py:145`): Outputs a mixture of von Mises-Fisher distributions: `(mu_mixture, kappa_mixture, weights)`.

---

## Training: Three Generative Paradigms

All training is orchestrated by PyTorch Lightning via Hydra configs. There are two entry points:
- **`plonk/train_random.py`** -> uses `DiffGeolocalizer` (diffusion / flow matching)
- **`plonk/train_von_fisher.py`** -> uses `VonFisherGeolocalizer`

### 1. Flow Matching (`FlowMatchingLoss`, `losses.py:40`)

The default and best-performing approach. Training:
```
t ~ Uniform(0,1)
gamma = scheduler(t)                    # sigmoid schedule
n ~ N(0, I)                             # Gaussian noise
y = gamma * x_0 + (1 - gamma) * n      # interpolate between data and noise
target = x_0 - n                        # velocity field
loss = ||network(y, gamma, cond) - target||^2
```

At inference (`flow_sampler.py`): start from noise, iteratively step `x_next = x_cur + dt * predicted_velocity` for 250 steps.

### 2. Riemannian Flow Matching (`RiemannianFlowMatchingLoss`, `losses.py:74`)

Same idea but respects the spherical geometry:
- Instead of linear interpolation, it uses **geodesic interpolation** on the sphere (`manifolds.py:29`)
- The loss is the Riemannian inner product of the prediction error
- The sampler (`riemannian_flow_sampler.py`) projects onto the sphere after each step

### 3. DDPM Diffusion (`DDPMLoss`, `losses.py:6`)

Standard denoising diffusion:
```
y = sqrt(gamma) * x_0 + sqrt(1-gamma) * n
loss = ||network(y, gamma, cond) - n||^2
```

### 4. Von Mises-Fisher (`VonFisherLoss`, `losses.py:119`)

A non-iterative approach. The network directly predicts the parameters of a von Mises-Fisher distribution:
```
mu, kappa = network(cond)
loss = -log p_vMF(x_0 | mu, kappa)
```

No noise schedules or iterative sampling needed — just evaluate the predicted distribution. Sampling is done via rejection sampling (`von_fisher_sampling.py`).

---

## Noise Schedulers (`schedulers.py`)

Control how much noise is present at each timestep:
- **`SigmoidScheduler`** (default, lines 4-33): maps `t in [0,1]` through a sigmoid curve with configurable `start=-7, end=3, tau=1`
- **`LinearScheduler`**: simple linear interpolation
- **`CosineScheduler`** / **`CosineSchedulerSimple`**: cosine-based schedules

These are critical for training stability. The sigmoid schedule with `start=-7, end=3` is used by the best models.

---

## EMA and Callbacks

- **`EMACallback`** (`callbacks/ema.py`): maintains an exponential moving average of the network weights (`ema_network`). Decay = 0.999. This EMA copy is used for inference, not the training weights.
- **`FixNANinGrad`** (`callbacks/fix_nans.py`): monitors for NaN gradients and rolls back if found.
- **`IncreaseDataEpoch`**: increments the WebDataset shared epoch counter for deterministic shuffling.

---

## Inference Pipeline (`pipe.py`)

`PlonkPipeline` wraps everything for easy inference:
1. Load pre-trained `Plonk` model from HuggingFace Hub
2. Extract image features with StreetCLIP/DINOv2
3. Run the sampler (flow/Riemannian/DDIM depending on model)
4. Convert Cartesian -> GPS -> degrees

Also supports `compute_likelihood()` (exact log-likelihood via ODE solving with `torchdiffeq`) and `compute_localizability()` (entropy-like measure via importance sampling).

---

## Metrics (`metrics/distance_based.py`)

- **Haversine distance**: great-circle distance in km
- **Accuracy at radius**: % of predictions within 1/25/200/750/2500 km
- **Area accuracy**: correct country/region/city prediction
- **GeoGuessr score**: `5000 * exp(-distance/1492.7)`
- **Precision/Recall/Density/Coverage**: manifold-based distribution quality metrics

---

## How to Modify the Model Architecture

Here are the key intervention points:

### 1. Change network depth/width
Easiest approach — just edit the experiment YAML:
```yaml
# configs/exp/your_experiment.yaml
model:
  network:
    dim: 768      # hidden dimension (default 512)
    depth: 16     # number of blocks (default 12)
    expansion: 4  # MLP expansion ratio
```

### 2. Modify the AdaLN block
Edit `plonk/models/networks/mlp.py:54` (`AdaLNMLPBlock`). For example, you could:
- Add attention within the block
- Change the conditioning mechanism (replace AdaLN with cross-attention)
- Add skip connections across blocks

### 3. Replace the backbone entirely
Create a new network class in `plonk/models/networks/`, then:
- Add a new YAML config in `configs/model/network/your_network.yaml` pointing to it with `_target_:`
- Create a new experiment YAML that overrides `/model/network: your_network`
- Your network must accept a batch dict with keys `"y"`, `"gamma"`, `"emb"` and return a tensor of shape `(batch, input_dim)`

### 4. Use the existing transformer blocks
`transformers.py` has ready-made `SelfAttentionBlock` and `CrossAttentionBlock`. You could build a transformer-based denoiser that uses cross-attention to condition on the image embedding, rather than AdaLN.

### 5. Change the generative paradigm
Switch between flow matching, Riemannian flow matching, DDPM, or von Mises-Fisher by changing the `loss`, `val_sampler`, and `test_sampler` overrides in your experiment config.

### 6. Change the feature extractor
To use a different image encoder, you'd modify `pipe.py` (for inference) and create new embedding extraction scripts in `plonk/data/extract_embeddings/` (for training data preparation, since embeddings are pre-computed and stored in the WebDataset shards).
