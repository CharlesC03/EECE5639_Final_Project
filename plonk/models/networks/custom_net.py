"""
CustomGeoNet — A modular geolocation network designed for easy experimentation.

Architecture is controlled via config. You can mix and match:
  - Block types: "adaln_mlp", "self_attention", "cross_attention"
  - Conditioning strategy: "add" (add cond to time emb) or "cross_attn" (cross-attend to cond)
  - Activation functions: "gelu", "silu", "relu"
  - Normalization: "layernorm", "rmsnorm"
  - Optional residual scaling, dropout, stochastic depth

The forward() interface is identical to GeoAdaLNMLP: accepts a batch dict with
keys "y" (noisy coords), "gamma" (noise level), "emb" (image embedding).
Returns a tensor of shape (batch_size, input_dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from plonk.models.positional_embeddings import PositionalEmbedding, FourierEmbedding
from plonk.models.networks.transformers import FusedMLP


# ---------------------------------------------------------------------------
# Utility: pick activation
# ---------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
    }[name]


# ---------------------------------------------------------------------------
# Utility: pick normalization
# ---------------------------------------------------------------------------
def get_norm(name: str, dim: int) -> nn.Module:
    if name == "layernorm":
        return nn.LayerNorm(dim, elementwise_affine=False)
    elif name == "rmsnorm":
        return RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm: {name}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ---------------------------------------------------------------------------
# Time embedding (reusable, same interface as original)
# ---------------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    def __init__(self, dim: int, time_scaling: float = 1000.0,
                 noise_embedding_type: str = "positional", expansion: int = 4):
        super().__init__()
        self.encode_time = (
            PositionalEmbedding(num_channels=dim, endpoint=True)
            if noise_embedding_type == "positional"
            else FourierEmbedding(num_channels=dim)
        )
        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        time = self.encode_time(t * self.time_scaling)
        time = (time - time.mean(dim=-1, keepdim=True)) / time.std(dim=-1, keepdim=True)
        return self.map_time(time)


# ---------------------------------------------------------------------------
# Block: Adaptive LayerNorm MLP (same idea as original, but configurable)
# ---------------------------------------------------------------------------
class AdaLNBlock(nn.Module):
    """MLP block with adaptive layer normalization from conditioning."""

    def __init__(self, dim: int, expansion: int = 4, activation: str = "gelu",
                 norm: str = "layernorm", dropout: float = 0.0,
                 residual_scale: float = 1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )
        # Modulation: condition -> (gamma, mu, gate)
        self.ada_map = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        self.norm = get_norm(norm, dim)
        self.residual_scale = residual_scale

        # Zero-init output for stable training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, mu, gate = self.ada_map(cond).chunk(3, dim=-1)
        x_norm = (1 + gamma) * self.norm(x) + mu
        x = x + self.residual_scale * self.mlp(x_norm) * gate
        return x


# ---------------------------------------------------------------------------
# Block: Self-Attention with AdaLN modulation
# ---------------------------------------------------------------------------
class SelfAttentionAdaLNBlock(nn.Module):
    """Self-attention block where tokens = [position_token, cond_token].

    Operates on a sequence of length 2: the position embedding and the
    conditioning embedding, so it's lightweight but lets the two interact.
    """

    def __init__(self, dim: int, num_heads: int = 4, expansion: int = 4,
                 activation: str = "gelu", norm: str = "layernorm",
                 dropout: float = 0.0, residual_scale: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.norm1 = get_norm(norm, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = get_norm(norm, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

        # AdaLN modulation for both sub-layers
        self.ada_map = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))
        self.residual_scale = residual_scale

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, mu1, gamma2, mu2 = self.ada_map(cond).chunk(4, dim=-1)

        # Self-attention
        x_norm = (1 + gamma1) * self.norm1(x) + mu1
        B, D = x_norm.shape
        # Treat as sequence of length 1 for attention (with cond as a second token)
        tokens = torch.stack([x_norm, cond], dim=1)  # (B, 2, D)
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = self.out_proj(attn_out[:, 0])  # Take position token output
        x = x + self.residual_scale * attn_out

        # FFN
        x_norm2 = (1 + gamma2) * self.norm2(x) + mu2
        x = x + self.residual_scale * self.mlp(x_norm2)
        return x


# ---------------------------------------------------------------------------
# Block: Cross-Attention (position attends to conditioning)
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """Position embedding cross-attends to the conditioning embedding."""

    def __init__(self, dim: int, num_heads: int = 4, expansion: int = 4,
                 activation: str = "gelu", norm: str = "layernorm",
                 dropout: float = 0.0, residual_scale: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.norm_q = get_norm(norm, dim)
        self.norm_kv = get_norm(norm, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_ffn = get_norm(norm, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )
        self.residual_scale = residual_scale

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Cross-attention: x attends to cond
        q = self.q_proj(self.norm_q(x)).unsqueeze(1)  # (B, 1, D)
        kv_input = cond.unsqueeze(1)  # (B, 1, D)
        kv = self.kv_proj(self.norm_kv(kv_input))
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)").squeeze(1)
        x = x + self.residual_scale * self.out_proj(attn_out)

        # FFN
        x = x + self.residual_scale * self.mlp(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Block registry — add new block types here
# ---------------------------------------------------------------------------
BLOCK_REGISTRY = {
    "adaln_mlp": AdaLNBlock,
    "self_attention": SelfAttentionAdaLNBlock,
    "cross_attention": CrossAttentionBlock,
}


def build_block(block_type: str, **kwargs) -> nn.Module:
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type '{block_type}'. "
            f"Available: {list(BLOCK_REGISTRY.keys())}"
        )
    return BLOCK_REGISTRY[block_type](**kwargs)


# ---------------------------------------------------------------------------
# Main network: CustomGeoNet
# ---------------------------------------------------------------------------
class CustomGeoNet(nn.Module):
    """
    A fully configurable denoising network for geolocation.

    All architecture choices are constructor args so you can change them
    from the Hydra YAML config without touching any Python code.

    Args:
        input_dim:   Coordinate dimensionality (3 for Cartesian on sphere).
        dim:         Hidden dimension throughout the network.
        depth:       Number of repeated blocks.
        expansion:   MLP expansion factor inside each block.
        cond_dim:    Dimension of the conditioning embedding (e.g. 1024 for StreetCLIP).
        block_type:  Which block to repeat. One of: "adaln_mlp", "self_attention",
                     "cross_attention".  You can also pass a list of block type names
                     (length must equal depth) to use different blocks at each layer.
        num_heads:   Number of attention heads (only used by attention blocks).
        activation:  Activation function name: "gelu", "silu", "relu".
        norm:        Normalization type: "layernorm", "rmsnorm".
        dropout:     Dropout rate inside blocks.
        residual_scale: Multiplicative scaling on residual connections.
        time_scaling: Scaling factor for the noise-level positional embedding.
        noise_embedding_type: "positional" or "fourier".
        cond_strategy: How image conditioning enters the network.
                       "add" — added to the time embedding (like the original).
                       "project" — projected separately, concatenated with time emb,
                                   then projected back to dim.
    """

    def __init__(
        self,
        input_dim: int = 3,
        dim: int = 512,
        depth: int = 12,
        expansion: int = 4,
        cond_dim: int = 1024,
        block_type: str = "adaln_mlp",
        num_heads: int = 4,
        activation: str = "gelu",
        norm: str = "layernorm",
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        time_scaling: float = 1000.0,
        noise_embedding_type: str = "positional",
        cond_strategy: str = "add",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.depth = depth
        self.cond_strategy = cond_strategy

        # --- Time embedding ---
        self.time_embedder = TimeEmbedder(
            dim=dim // 4,
            time_scaling=time_scaling,
            noise_embedding_type=noise_embedding_type,
            expansion=4,
        )

        # --- Conditioning projection ---
        if cond_strategy == "add":
            self.cond_mapper = nn.Linear(cond_dim, dim)
        elif cond_strategy == "project":
            self.cond_mapper = nn.Linear(cond_dim, dim)
            # Time embedder outputs dim//4 * 4 = dim, and cond is dim → concat is 2*dim
            self.cond_fuse = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.SiLU(),
            )
        else:
            raise ValueError(f"Unknown cond_strategy: {cond_strategy}")

        # --- Input projection ---
        self.initial_mapper = nn.Linear(input_dim, dim)

        # --- Core blocks ---
        # Allow block_type to be a single string or a list for per-layer control
        if isinstance(block_type, str):
            block_types = [block_type] * depth
        else:
            assert len(block_type) == depth, (
                f"block_type list length ({len(block_type)}) must equal depth ({depth})"
            )
            block_types = list(block_type)

        block_kwargs = dict(
            dim=dim,
            expansion=expansion,
            activation=activation,
            norm=norm,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        # Only pass num_heads to blocks that use it
        attn_kwargs = {**block_kwargs, "num_heads": num_heads}

        self.blocks = nn.ModuleList()
        for bt in block_types:
            if bt in ("self_attention", "cross_attention"):
                self.blocks.append(build_block(bt, **attn_kwargs))
            else:
                self.blocks.append(build_block(bt, **block_kwargs))

        # --- Final output projection (with AdaLN) ---
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        self.final_norm = get_norm(norm, dim)
        self.final_linear = nn.Linear(dim, input_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                "y"     — noisy coordinates, shape (B, input_dim)
                "gamma" — noise level, shape (B,)
                "emb"   — conditioning embedding, shape (B, cond_dim)
        Returns:
            Predicted velocity/noise, shape (B, input_dim)
        """
        x = self.initial_mapper(batch["y"])
        gamma = batch["gamma"]
        emb = batch["emb"]

        # Build conditioning vector
        t = self.time_embedder(gamma)
        c = self.cond_mapper(emb)

        if self.cond_strategy == "add":
            cond = c + t
        elif self.cond_strategy == "project":
            cond = self.cond_fuse(torch.cat([c, t], dim=-1))

        # Pass through blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final output
        gamma_out, mu_out = self.final_adaln(cond).chunk(2, dim=-1)
        x = (1 + gamma_out) * self.final_norm(x) + mu_out
        x = self.final_linear(x)
        return x
