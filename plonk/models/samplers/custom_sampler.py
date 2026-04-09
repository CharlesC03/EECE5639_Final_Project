"""
Custom Riemannian flow sampler with configurable inference knobs.

Extends the standard riemannian_flow_sampler with options to control:
  - num_steps: number of integration steps (more = higher quality, slower)
  - step_schedule: "uniform" (equal dt) or "cosine" (smaller steps near t=0)
  - projection_interval: project onto sphere every N steps instead of every step
  - cfg_rate: classifier-free guidance strength
  - midpoint_correction: use midpoint method for more accurate ODE integration
"""

import torch
from plonk.utils.manifolds import Sphere


def custom_riemannian_flow_sampler(
    net,
    batch,
    manifold=Sphere(),
    conditioning_keys=None,
    scheduler=None,
    num_steps=250,
    cfg_rate=0,
    generator=None,
    return_trajectories=False,
    step_schedule="uniform",
    projection_interval=1,
    midpoint_correction=False,
):
    """
    Configurable Riemannian flow sampler.

    Args:
        net: The denoising network (wrapped with preconditioning).
        batch: Dict with "y" (initial noise) and conditioning keys.
        manifold: The manifold to project onto (Sphere by default).
        conditioning_keys: Key in batch for the conditioning embedding.
        scheduler: Noise schedule mapping t -> gamma.
        num_steps: Number of ODE integration steps.
        cfg_rate: Classifier-free guidance rate. 0 = no guidance.
        generator: Optional torch.Generator for reproducibility.
        return_trajectories: If True, also return intermediate states.
        step_schedule: "uniform" or "cosine". Controls step size distribution.
        projection_interval: Project onto manifold every N steps (1 = every step).
        midpoint_correction: Use midpoint method for more accurate integration.
    """
    if scheduler is None:
        raise ValueError("Scheduler must be provided")

    x_cur = batch["y"].to(torch.float32)
    if return_trajectories:
        traj = [x_cur.detach()]

    # --- Build step schedule ---
    step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)

    if step_schedule == "uniform":
        steps = 1 - step_indices / num_steps
    elif step_schedule == "cosine":
        # Cosine schedule: more steps concentrated near t=0 (data end)
        steps = 0.5 * (1 + torch.cos(step_indices / num_steps * torch.pi))
    else:
        raise ValueError(f"Unknown step_schedule: {step_schedule}")

    gammas = scheduler(steps)
    dtype = torch.float32

    # --- Pre-build stacked batch for CFG if needed ---
    if cfg_rate > 0 and conditioning_keys is not None:
        stacked_batch = {}
        stacked_batch[conditioning_keys] = torch.cat(
            [batch[conditioning_keys], torch.zeros_like(batch[conditioning_keys])],
            dim=0,
        )

    # --- Helper: evaluate network with optional CFG ---
    def eval_net(x, gamma_val):
        with torch.cuda.amp.autocast(dtype=dtype):
            if cfg_rate > 0 and conditioning_keys is not None:
                stacked_batch["y"] = torch.cat([x, x], dim=0)
                stacked_batch["gamma"] = gamma_val.expand(x.shape[0] * 2)
                denoised_all = net(stacked_batch)
                denoised_cond, denoised_uncond = denoised_all.chunk(2, dim=0)
                return denoised_cond * (1 + cfg_rate) - denoised_uncond * cfg_rate
            else:
                batch["y"] = x
                batch["gamma"] = gamma_val.expand(x.shape[0])
                return net(batch)

    # --- Integration loop ---
    for step, (gamma_now, gamma_next) in enumerate(zip(gammas[:-1], gammas[1:])):
        dt = gamma_next - gamma_now

        if midpoint_correction:
            # Midpoint method: evaluate at midpoint for better accuracy
            v1 = eval_net(x_cur, gamma_now)
            gamma_mid = (gamma_now + gamma_next) / 2
            x_mid = x_cur + (dt / 2) * v1
            if (step + 1) % projection_interval == 0:
                x_mid = manifold.projx(x_mid)
            v2 = eval_net(x_mid, gamma_mid)
            x_next = x_cur + dt * v2
        else:
            # Euler method (standard)
            denoised = eval_net(x_cur, gamma_now)
            x_next = x_cur + dt * denoised

        # Project onto sphere
        if (step + 1) % projection_interval == 0:
            x_next = manifold.projx(x_next)

        x_cur = x_next
        if return_trajectories:
            traj.append(x_cur.detach().to(torch.float32))

    # Always project at the very end
    x_cur = manifold.projx(x_cur)

    if return_trajectories:
        return x_cur.to(torch.float32), traj
    else:
        return x_cur.to(torch.float32)
