from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from plonk.utils.manifolds import Sphere
from torch.func import jacrev, vjp, vmap
from torchdiffeq import odeint
from geoopt import ProductManifold, Euclidean
from plonk.models.samplers.riemannian_flow_sampler import ode_riemannian_flow_sampler


class DiffGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = instantiate(cfg.network)
        # self.network = torch.compile(self.network, fullgraph=True)
        self.input_dim = cfg.network.input_dim
        self.train_noise_scheduler = instantiate(cfg.train_noise_scheduler)
        self.inference_noise_scheduler = instantiate(cfg.inference_noise_scheduler)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.ema_network.eval()
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)
        self.loss = instantiate(cfg.loss)(
            self.train_noise_scheduler,
        )
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)
        self.manifold = instantiate(cfg.manifold) if hasattr(cfg, "manifold") else None

        self.interpolant = cfg.interpolant

        # Optional: per-neighbor-count evaluation (for attention-mode neighbor experiments).
        # When set (e.g. [0, 1, 2, 3, 4, 5]) test_step runs once per k using the first k
        # neighbors from the dataset and logs/prints metrics for each k separately.
        self.test_neighbor_counts = (
            list(cfg.test_neighbor_counts)
            if getattr(cfg, "test_neighbor_counts", None) is not None
            else None
        )
        # Weight per neighbor when the model is in averaging mode. Should match
        # dataset.neighborhood.neighbor_weight used at training time.
        self.test_neighbor_weight = float(getattr(cfg, "test_neighbor_weight", 1.0))
        if self.test_neighbor_counts is not None:
            keys = [str(k) for k in self.test_neighbor_counts] + ["overall"]
            self.test_metrics_per_k = nn.ModuleDict(
                {key: instantiate(cfg.test_metrics) for key in keys}
            )

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.data_preprocessing(batch)
            batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(self.preconditioning, self.network, batch).mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print("Parameters without grad:")
                print(no_grad)

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(
            self.preconditioning,
            self.network,
            batch,
            generator=self.validation_generator,
        ).mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if hasattr(self, "ema_model"):
            loss_ema = self.loss(
                self.preconditioning,
                self.ema_network,
                batch,
                generator=self.validation_generator_ema,
            ).mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # nll = -self.compute_exact_loglikelihood(batch).mean()
        # self.log(
        #     "val/nll",
        #     nll,
        #     sync_dist=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=batch_size,
        # )

    # def on_validation_epoch_end(self):
    #     metrics = self.val_metrics.compute()
    #     for metric_name, metric_value in metrics.items():
    #         self.log(
    #             f"val/{metric_name}",
    #             metric_value,
    #             sync_dist=True,
    #             on_step=False,
    #             on_epoch=True,
    #         )

    def on_test_start(self):
        self.test_generator = torch.Generator(device=self.device).manual_seed(3407)

    def test_step_simple(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        if isinstance(self.manifold, Sphere):
            x_N = self.manifold.random_base(
                batch_size,
                self.input_dim,
                device=self.device,
            )
            x_N = x_N.reshape(batch_size, self.input_dim)
        else:
            x_N = torch.randn(
                batch_size,
                self.input_dim,
                device=self.device,
                generator=self.test_generator,
            )
        cond = batch[self.cfg.cond_preprocessing.output_key]

        samples = self.sample(
            x_N=x_N,
            cond=cond,
            stage="val",
            generator=self.test_generator,
            cfg=self.cfg.cfg_rate,
        )
        self.test_metrics.update({"gps": samples}, batch)
        if self.cfg.compute_nll:
            nll = -self.compute_exact_loglikelihood(batch, cfg=0).mean()
            self.log(
                "test/NLL",
                nll,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def test_best_nll(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        num_sample_per_cond = 32
        if isinstance(self.manifold, Sphere):
            x_N = self.manifold.random_base(
                batch_size * num_sample_per_cond,
                self.input_dim,
                device=self.device,
            )
            x_N = x_N.reshape(batch_size * num_sample_per_cond, self.input_dim)
        else:
            x_N = torch.randn(
                batch_size * num_sample_per_cond,
                self.input_dim,
                device=self.device,
                generator=self.test_generator,
            )
        cond = (
            batch[self.cfg.cond_preprocessing.output_key]
            .unsqueeze(1)
            .repeat(1, num_sample_per_cond, 1)
            .view(-1, batch[self.cfg.cond_preprocessing.output_key].shape[-1])
        )
        samples = self.sample_distribution(
            x_N,
            cond,
            sampling_batch_size=32768,
            stage="val",
            generator=self.test_generator,
            cfg=0,
        )
        samples = samples.view(batch_size * num_sample_per_cond, -1)
        batch_swarm = {"gps": samples, "emb": cond}
        nll_batch = -self.compute_exact_loglikelihood(batch_swarm, cfg=0)
        nll_batch = nll_batch.view(batch_size, num_sample_per_cond, -1)
        nll_best = nll_batch[
            torch.arange(batch_size), nll_batch.argmin(dim=1).squeeze(1)
        ]
        self.log(
            "test/best_nll",
            nll_best.mean(),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        samples = samples.view(batch_size, num_sample_per_cond, -1)[
            torch.arange(batch_size), nll_batch.argmin(dim=1).squeeze(1)
        ]
        self.test_metrics.update({"gps": samples}, batch)

    @staticmethod
    def _filter_batch(batch, idx):
        """Subset every per-sample field in batch by idx (tensor of indices)."""
        idx_list = idx.cpu().tolist() if torch.is_tensor(idx) else list(idx)
        out = {}
        for key, val in batch.items():
            if torch.is_tensor(val):
                out[key] = val[idx]
            elif isinstance(val, (list, tuple)):
                out[key] = [val[i] for i in idx_list]
            else:
                out[key] = val
        return out

    def test_step_vary_neighbors(self, batch, batch_idx):
        """Evaluate the model with a controlled number of neighbor images per sample.

        Each configured k runs on the subset of samples that have *at least* k real
        neighbors available (and takes k of them — the dataset's random.sample gives
        a random subset when the sample has more than k). The "overall" bucket uses
        every sample with whatever neighbors it happens to have, so it matches the
        end-to-end evaluation you'd get from normal inference.

        Works for both fusion modes:
          - Attention (model has NeighborhoodAttentionPooler): passes first k neighbors
            with mask=True to the pooler.
          - Averaging (no pooler): rebuilds the weighted average of anchor + first k
            neighbors and passes it as `emb`. Uses batch["anchor_emb"] so the
            dataset-side pre-average on `emb` is ignored.

        Same x_N noise is reused across k values so differences reflect the effect
        of added neighbors, not sampling variance.
        """
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        cond_key = self.cfg.cond_preprocessing.output_key

        if isinstance(self.manifold, Sphere):
            x_N = self.manifold.random_base(
                batch_size, self.input_dim, device=self.device
            )
            x_N = x_N.reshape(batch_size, self.input_dim)
        else:
            x_N = torch.randn(
                batch_size,
                self.input_dim,
                device=self.device,
                generator=self.test_generator,
            )

        has_pooler = getattr(self.ema_network, "neighbor_pooler", None) is not None
        full_mask = batch["neighbor_mask"]
        neighbor_embs = batch["neighbor_embs"]
        anchor_emb = batch["anchor_emb"] if "anchor_emb" in batch else batch[cond_key]
        available = full_mask.sum(dim=-1)  # (B,) — real neighbor count per sample

        def _run_bucket(label, filter_idx, anchor_i, neighbors_i, mask_i, xN_i, gt_i):
            if len(filter_idx) == 0:
                return
            if has_pooler:
                sample_batch = {
                    "y": xN_i,
                    cond_key: anchor_i,
                    "neighbor_embs": neighbors_i,
                    "neighbor_mask": mask_i,
                }
            else:
                valid = mask_i.to(anchor_i.dtype)  # (b, k)
                neighbor_sum = (neighbors_i * valid.unsqueeze(-1)).sum(dim=1)
                effective_k = valid.sum(dim=1, keepdim=True)
                total_weight = 1.0 + effective_k * self.test_neighbor_weight
                fused = torch.where(
                    effective_k > 0,
                    (anchor_i + self.test_neighbor_weight * neighbor_sum) / total_weight,
                    anchor_i,
                )
                sample_batch = {"y": xN_i, cond_key: fused}
            output = self.test_sampler(
                self.ema_model,
                sample_batch,
                conditioning_keys=cond_key,
                scheduler=self.inference_noise_scheduler,
                cfg_rate=self.cfg.cfg_rate,
                generator=self.test_generator,
            )
            samples = self.postprocessing(output)
            self.test_metrics_per_k[label].update({"gps": samples}, gt_i)

        # Per-k buckets: samples with at least k real neighbors, using exactly k.
        for k in self.test_neighbor_counts:
            filter_idx = torch.where(available >= k)[0]
            if len(filter_idx) == 0:
                continue
            gt_i = self._filter_batch(batch, filter_idx)
            anchor_i = anchor_emb[filter_idx]
            xN_i = x_N[filter_idx]
            if k == 0:
                neighbors_i = neighbor_embs[filter_idx, :0]
                mask_i = full_mask[filter_idx, :0]
            else:
                neighbors_i = neighbor_embs[filter_idx, :k]
                mask_i = full_mask[filter_idx, :k]  # all True since we filtered >= k
            _run_bucket(str(k), filter_idx, anchor_i, neighbors_i, mask_i, xN_i, gt_i)

        # Overall bucket: all samples, each with its full set of available neighbors.
        all_idx = torch.arange(batch_size, device=self.device)
        _run_bucket(
            "overall", all_idx, anchor_emb, neighbor_embs, full_mask, x_N, batch
        )

    def test_step(self, batch, batch_idx):
        if self.test_neighbor_counts is not None:
            self.test_step_vary_neighbors(batch, batch_idx)
        elif self.cfg.compute_swarms:
            self.test_best_nll(batch, batch_idx)
        else:
            self.test_step_simple(batch, batch_idx)

    def on_test_epoch_end(self):
        if self.test_neighbor_counts is not None:
            rows = []
            labels = [str(k) for k in self.test_neighbor_counts] + ["overall"]
            for label in labels:
                metric = self.test_metrics_per_k[label]
                count = int(metric.count.item())
                if count == 0:
                    continue
                try:
                    metrics = metric.compute()
                except Exception as e:
                    # HaversineMetrics.manifold_metrics chunks into 20 splits and calls
                    # np.argpartition(k=manifold_k+1) on each, which fails for small
                    # buckets. Basic distance/accuracy metrics are still valid — recompute
                    # them manually so the bucket isn't dropped from the table.
                    self.print(
                        f"[warn] bucket {label} (n={count}): manifold_metrics skipped ({e})"
                    )
                    metrics = self._basic_metrics(metric)
                self.log(
                    f"test/{label}/n_samples",
                    float(count),
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx="sum",
                )
                for metric_name, metric_value in metrics.items():
                    self.log(
                        f"test/{label}/{metric_name}",
                        metric_value,
                        sync_dist=True,
                        on_step=False,
                        on_epoch=True,
                    )
                rows.append((label, count, metrics))
            self._print_per_k_summary(rows)
            return

        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @staticmethod
    def _basic_metrics(metric):
        """Reproduce HaversineMetrics.compute() without the manifold_metrics step."""
        count = metric.count.float()
        out = {
            "Haversine": (metric.haversine_sum / count).item(),
            "Geoguessr": (metric.geoguessr_sum / count).item(),
        }
        for acc in metric.acc_radius:
            out[f"Accuracy_{acc}_km_radius"] = (
                metric.__dict__[f"close_enough_points_{acc}"] / count
            ).item()
        for acc in metric.acc_area:
            n_acc = metric.__dict__[f"count_{acc}"]
            if int(n_acc.item()) > 0:
                out[f"Accuracy_{acc}"] = (
                    metric.__dict__[f"close_enough_points_{acc}"] / n_acc
                ).item()
        return out

    def _print_per_k_summary(self, rows):
        """Format a per-k summary table. rows = [(label, n_samples, metrics_dict), ...].

        Columns are the union of metric keys across all rows so buckets that fell
        back to basic metrics (missing manifold precision/recall/density/coverage)
        don't break the schema — missing cells show as '-'.
        """
        if not rows:
            return
        metric_names = []
        seen = set()
        for _, _, m in rows:
            for name in m:
                if name not in seen:
                    seen.add(name)
                    metric_names.append(name)

        def cell(m, name):
            return f"{m[name]:.4f}" if name in m else "-"

        header = ["label", "n"] + metric_names
        label_w = max(len("label"), max(len(str(label)) for label, _, _ in rows))
        n_w = max(len("n"), max(len(str(n)) for _, n, _ in rows))
        metric_ws = [
            max(len(name), max(len(cell(m, name)) for _, _, m in rows))
            for name in metric_names
        ]
        col_widths = [label_w, n_w] + metric_ws

        def fmt_row(cells):
            return "  ".join(str(c).rjust(w) for c, w in zip(cells, col_widths))

        lines = ["", "=== Test metrics by neighbor count ==="]
        lines.append(fmt_row(header))
        lines.append("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
        for label, n, m in rows:
            lines.append(fmt_row([label, n] + [cell(m, name) for name in metric_names]))
        lines.append("")
        self.print("\n".join(lines))

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.network, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.network.parameters())
        if "lr_scheduler" in self.cfg:
            scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        cond=None,
        x_N=None,
        num_steps=None,
        stage="test",
        cfg=0,
        generator=None,
        return_trajectories=False,
        postprocessing=True,
    ):
        if x_N is None:
            assert batch_size is not None
            if isinstance(self.manifold, Sphere):
                x_N = self.manifold.random_base(
                    batch_size, self.input_dim, device=self.device
                )
                x_N = x_N.reshape(batch_size, self.input_dim)
            else:
                x_N = torch.randn(batch_size, self.input_dim, device=self.device)
        batch = {"y": x_N}
        if stage == "val":
            sampler = self.val_sampler
        elif stage == "test":
            sampler = self.test_sampler
        else:
            raise ValueError(f"Unknown stage {stage}")
        batch[self.cfg.cond_preprocessing.input_key] = cond
        batch = self.cond_preprocessing(batch, device=self.device)
        if num_steps is None:
            output = sampler(
                self.ema_model,
                batch,
                conditioning_keys=self.cfg.cond_preprocessing.output_key,
                scheduler=self.inference_noise_scheduler,
                cfg_rate=cfg,
                generator=generator,
                return_trajectories=return_trajectories,
            )
        else:
            output = sampler(
                self.ema_model,
                batch,
                conditioning_keys=self.cfg.cond_preprocessing.output_key,
                scheduler=self.inference_noise_scheduler,
                num_steps=num_steps,
                cfg_rate=cfg,
                generator=generator,
                return_trajectories=return_trajectories,
            )
        if return_trajectories:
            return (
                self.postprocessing(output[0]) if postprocessing else output[0],
                [
                    self.postprocessing(frame) if postprocessing else frame
                    for frame in output[1]
                ],
            )
        else:
            return self.postprocessing(output) if postprocessing else output

    def sample_distribution(
        self,
        x_N,
        cond,
        sampling_batch_size=2048,
        num_steps=None,
        stage="test",
        cfg=0,
        generator=None,
        return_trajectories=False,
    ):
        if return_trajectories:
            x_0 = []
            trajectories = []
            i = -1
            for i in range(x_N.shape[0] // sampling_batch_size):
                x_N_batch = x_N[i * sampling_batch_size : (i + 1) * sampling_batch_size]
                cond_batch = cond[
                    i * sampling_batch_size : (i + 1) * sampling_batch_size
                ]
                out, trajectories = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
                trajectories.append(trajectories)
            if x_N.shape[0] % sampling_batch_size != 0:
                x_N_batch = x_N[(i + 1) * sampling_batch_size :]
                cond_batch = cond[(i + 1) * sampling_batch_size :]
                out, trajectories = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
                trajectories.append(trajectories)
            x_0 = torch.cat(x_0, dim=1)
            trajectories = [torch.cat(frame, dim=1) for frame in trajectories]
            return x_0, trajectories
        else:
            x_0 = []
            i = -1
            for i in range(x_N.shape[0] // sampling_batch_size):
                x_N_batch = x_N[i * sampling_batch_size : (i + 1) * sampling_batch_size]
                cond_batch = cond[
                    i * sampling_batch_size : (i + 1) * sampling_batch_size
                ]
                out = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
            if x_N.shape[0] % sampling_batch_size != 0:
                x_N_batch = x_N[(i + 1) * sampling_batch_size :]
                cond_batch = cond[(i + 1) * sampling_batch_size :]
                out = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
            x_0 = torch.cat(x_0, dim=0)
            return x_0

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.ema_network, *args, **kwargs)

    def compute_exact_loglikelihood(
        self,
        batch=None,
        x_1=None,
        cond=None,
        t1=1.0,
        num_steps=1000,
        rademacher=False,
        data_preprocessing=True,
        cfg=0,
    ):
        nfe = [0]
        if batch is None:
            batch = {"x_0": x_1, "emb": cond}
        if data_preprocessing:
            batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        timesteps = self.inference_noise_scheduler(
            torch.linspace(0, t1, 2).to(batch["x_0"])
        )
        with torch.inference_mode(mode=False):

            def odefunc(t, tensor):
                nfe[0] += 1
                t = t.to(tensor)
                gamma = self.inference_noise_scheduler(t)
                x = tensor[..., : self.input_dim]
                y = batch["emb"]

                def vecfield(x, y):
                    if cfg > 0:
                        batch_vecfield = {
                            "y": x,
                            "emb": y,
                            "gamma": gamma.reshape(-1),
                        }
                        model_output_cond = self.ema_model(batch_vecfield)
                        batch_vecfield_uncond = {
                            "y": x,
                            "emb": torch.zeros_like(y),
                            "gamma": gamma.reshape(-1),
                        }
                        model_output_uncond = self.ema_model(batch_vecfield_uncond)
                        model_output = model_output_cond + cfg * (
                            model_output_cond - model_output_uncond
                        )

                    else:
                        batch_vecfield = {
                            "y": x,
                            "emb": y,
                            "gamma": gamma.reshape(-1),
                        }
                        model_output = self.ema_model(batch_vecfield)

                    if self.interpolant == "flow_matching":
                        d_gamma = self.inference_noise_scheduler.derivative(t).reshape(
                            -1, 1
                        )
                        return d_gamma * model_output
                    elif self.interpolant == "diffusion":
                        alpha_t = self.inference_noise_scheduler.alpha(t).reshape(-1, 1)
                        return (
                            -1 / 2 * (alpha_t * x - torch.abs(alpha_t) * model_output)
                        )
                    else:
                        raise ValueError(f"Unknown interpolant {self.interpolant}")

                if rademacher:
                    v = torch.randint_like(x, 2) * 2 - 1
                else:
                    v = None
                dx, div = output_and_div(vecfield, x, y, v=v)
                div = div.reshape(-1, 1)
                del t, x
                return torch.cat([dx, div], dim=-1)

            x_1 = batch["x_0"]
            state1 = torch.cat([x_1, torch.zeros_like(x_1[..., :1])], dim=-1)
            with torch.no_grad():
                if False and isinstance(self.manifold, Sphere):
                    self.print("Riemannian flow sampler")
                    product_man = ProductManifold(
                        (self.manifold, self.input_dim), (Euclidean(), 1)
                    )
                    state0 = ode_riemannian_flow_sampler(
                        odefunc,
                        state1,
                        manifold=product_man,
                        scheduler=self.inference_noise_scheduler,
                        num_steps=num_steps,
                    )
                else:
                    self.print("ODE solver")
                    state0 = odeint(
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, 2).to(batch["x_0"]),
                        atol=1e-6,
                        rtol=1e-6,
                        method="dopri5",
                        options={"min_step": 1e-5},
                    )[-1]
        x_0, logdetjac = state0[..., : self.input_dim], state0[..., -1]
        if self.manifold is not None:
            x_0 = self.manifold.projx(x_0)
            logp0 = self.manifold.base_logprob(x_0)
        else:
            logp0 = (
                -1 / 2 * (x_0**2).sum(dim=-1)
                - self.input_dim
                * torch.log(torch.tensor(2 * np.pi, device=x_0.device))
                / 2
            )
        self.print(f"nfe: {nfe[0]}")
        logp1 = logp0 + logdetjac
        logp1 = logp1 / (self.input_dim * np.log(2))
        return logp1


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


# for likelihood computation
def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u, argnums=0)
    return lambda x, y: torch.trace(J(x, y).squeeze(0))


def output_and_div(vecfield, x, y, v=None):
    if v is None:
        dx = vecfield(x, y)
        div = vmap(div_fn(vecfield))(x, y)
    else:
        vecfield_x = lambda x: vecfield(x, y)
        dx, vjpfunc = vjp(vecfield_x, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class VonFisherGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = instantiate(cfg.network)
        # self.network = torch.compile(self.network, fullgraph=True)
        self.input_dim = cfg.network.input_dim
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.ema_network.eval()
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)
        self.loss = instantiate(cfg.loss)()
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.data_preprocessing(batch)
            batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(self.preconditioning, self.network, batch).mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print("Parameters without grad:")
                print(no_grad)

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(
            self.preconditioning,
            self.network,
            batch,
            generator=self.validation_generator,
        ).mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if hasattr(self, "ema_model"):
            loss_ema = self.loss(
                self.preconditioning,
                self.ema_network,
                batch,
                generator=self.validation_generator_ema,
            ).mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def on_test_start(self):
        self.test_generator = torch.Generator(device=self.device).manual_seed(3407)

    def test_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        cond = batch[self.cfg.cond_preprocessing.output_key]

        samples = self.sample(cond=cond, stage="test")
        self.test_metrics.update({"gps": samples}, batch)
        nll = -self.compute_exact_loglikelihood(batch).mean()
        self.log(
            "test/NLL",
            nll,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.network, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.network.parameters())
        if "lr_scheduler" in self.cfg:
            scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        cond=None,
        postprocessing=True,
        stage="val",
    ):
        batch = {}
        if stage == "val":
            sampler = self.val_sampler
        elif stage == "test":
            sampler = self.test_sampler
        else:
            raise ValueError(f"Unknown stage {stage}")
        batch[self.cfg.cond_preprocessing.input_key] = cond
        batch = self.cond_preprocessing(batch, device=self.device)
        output = sampler(
            self.ema_model,
            batch,
        )
        return self.postprocessing(output) if postprocessing else output

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.ema_network, *args, **kwargs)

    def compute_exact_loglikelihood(
        self,
        batch=None,
    ):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        return -self.loss(self.preconditioning, self.ema_network, batch)


class RandomGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.test_metrics = instantiate(cfg.test_metrics)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.postprocessing = instantiate(cfg.postprocessing)

    def test_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        samples = torch.randn(batch_size, 3, device=self.device)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        samples = self.postprocessing(samples)
        self.test_metrics.update({"gps": samples}, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
