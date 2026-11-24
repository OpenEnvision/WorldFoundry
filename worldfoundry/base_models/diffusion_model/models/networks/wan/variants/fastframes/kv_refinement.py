# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License).
"""KV refinement network for FastFrames (Sec. 3.3 of arXiv:2611.23226).

Replaces a frame's noisy KV cache entries after its final denoising step with
clean-context keys and values using a shared spatial stem and batched per-layer
``baddbmm`` heads, preserving exact released checkpoint parameter names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import (
    sinusoidal_embedding_1d,
)

__all__ = [
    "GroupGEMM",
    "GroupedKVRepairHeads",
    "KVRefinementNetwork",
    "Modulation",
    "StemBlock",
    "TimestepEmbedder",
]

_ALL_LAYERS = slice(None)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Evaluate ``[B, H, L, D]`` SDPA preferring cuDNN on CUDA BF16/FP16."""
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v)
        except RuntimeError:
            return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer SiLU MLP."""

    def __init__(self, d_a: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d_a), nn.SiLU(), nn.Linear(d_a, d_a))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding_1d(self.freq_dim, t.flatten().float()).float()
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Modulation(nn.Module):
    """Zero-initialized adaLN projection: ``[B, d_a]`` -> ``n`` tensors ``[B, 1, d_a]``."""

    def __init__(self, d_a: int, n: int) -> None:
        super().__init__()
        self.n = n
        self.lin = nn.Linear(d_a, n * d_a)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.lin(F.silu(e)).unsqueeze(1).chunk(self.n, dim=-1)


class StemBlock(nn.Module):
    """Pre-norm spatial transformer block with adaLN and zero-initialized gates."""

    def __init__(self, d_a: int, n_heads: int = 8, mlp_mult: int = 4) -> None:
        super().__init__()
        if d_a % n_heads != 0:
            raise ValueError(f"d_a ({d_a}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_a // n_heads
        self.norm1 = nn.LayerNorm(d_a, elementwise_affine=False)
        self.qkv = nn.Linear(d_a, 3 * d_a)
        self.proj = nn.Linear(d_a, d_a)
        self.norm2 = nn.LayerNorm(d_a, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_a, d_a * mlp_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_a * mlp_mult, d_a),
        )
        self.mod = Modulation(d_a, 6)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(e)
        h = self.norm1(x) * (1 + sc1) + sh1
        q, k, v = (
            self.qkv(h).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        x = x + g1 * self.proj(_sdpa(q, k, v).transpose(1, 2).reshape(batch_size, seq_len, -1))
        h = self.norm2(x) * (1 + sc2) + sh2
        return x + g2 * self.mlp(h)


class GroupGEMM(nn.Module):
    """Batched per-layer KV refinement heads evaluated via ``torch.baddbmm``."""

    def __init__(
        self,
        num_members: int,
        dim: int,
        d_a: int,
        d_v: int,
        d_k: int,
        num_heads: int,
        head_dim: int,
        mlp_mult: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.d_a = d_a
        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.head_dim = head_dim
        n = int(num_members)

        def _weight(in_dim: int, out_dim: int, *, zero: bool = False) -> nn.Parameter:
            tensor = torch.zeros(n, in_dim, out_dim) if zero else torch.randn(n, in_dim, out_dim) * (in_dim**-0.5)
            return nn.Parameter(tensor)

        def _bias(out_dim: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(n, 1, out_dim))

        self.proj_k_w, self.proj_k_b = _weight(dim, d_k), _bias(d_k)
        self.proj_v_w, self.proj_v_b = _weight(dim, d_v), _bias(d_v)
        self.fuse_w, self.fuse_b = _weight(d_k + 2 * d_v, d_v), _bias(d_v)
        self.m0_w, self.m0_b = _weight(d_v, d_v * mlp_mult), _bias(d_v * mlp_mult)
        self.m2_w, self.m2_b = _weight(d_v * mlp_mult, d_v), _bias(d_v)
        self.mod_w, self.mod_b = _weight(d_a, 2 * d_v, zero=True), _bias(2 * d_v)
        self.dK_w, self.dK_b = _weight(d_v, dim, zero=True), _bias(dim)
        self.dV_w, self.dV_b = _weight(d_v, dim, zero=True), _bias(dim)
        self.v_logscale = nn.Parameter(torch.zeros(n, 1, dim))

    def _renorm_per_head(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xh = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        rh = ref.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xh = xh * (rh.norm(dim=-1, keepdim=True) / (xh.norm(dim=-1, keepdim=True) + 1e-6))
        return xh.reshape(batch_size, seq_len, -1)

    def forward(
        self,
        nK: torch.Tensor,
        nV: torch.Tensor,
        phi_v: torch.Tensor,
        e: torch.Tensor,
        layers: slice | torch.Tensor = _ALL_LAYERS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine stacked noisy ``nK, nV: [M, S, dim]`` with ``phi_v: [M, S, d_v]``."""
        num_layers = nK.shape[0]
        pk = torch.baddbmm(self.proj_k_b[layers], nK, self.proj_k_w[layers])
        pv = torch.baddbmm(self.proj_v_b[layers], nV, self.proj_v_w[layers])
        h = torch.baddbmm(self.fuse_b[layers], torch.cat([pk, pv, phi_v], dim=-1), self.fuse_w[layers])
        e1 = F.silu(e).reshape(1, 1, self.d_a).expand(num_layers, 1, self.d_a).contiguous()
        shift, scale = torch.baddbmm(self.mod_b[layers], e1, self.mod_w[layers]).chunk(2, dim=-1)
        h = F.layer_norm(h, (self.d_v,)) * (1 + scale) + shift
        h = torch.baddbmm(self.m0_b[layers], h, self.m0_w[layers])
        h = torch.baddbmm(self.m2_b[layers], F.gelu(h, approximate="tanh"), self.m2_w[layers])
        k = self._renorm_per_head(nK + torch.baddbmm(self.dK_b[layers], h, self.dK_w[layers]), ref=nK)
        v = (nV + torch.baddbmm(self.dV_b[layers], h, self.dV_w[layers])) * torch.exp(self.v_logscale[layers])
        return k, v


class GroupedKVRepairHeads(nn.Module):
    """Per-layer refinement heads preserving ``heads_grouped.gemms.0.*`` and ``phi_proj.<d_a>.*`` keys."""

    def __init__(
        self,
        num_blocks: int,
        dim: int,
        d_a: int,
        num_heads: int,
        head_dim: int,
        mlp_mult: int = 2,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gemms = nn.ModuleList([GroupGEMM(num_blocks, dim, d_a, d_a, d_a, num_heads, head_dim, mlp_mult)])
        self.phi_proj = nn.ModuleDict({str(d_a): nn.Linear(d_a, d_a, bias=False)})

    def forward(
        self,
        nK_all: torch.Tensor,
        nV_all: torch.Tensor,
        phi: torch.Tensor,
        e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine all layers ``[L, S, dim]`` given shared stem features ``phi``."""
        phi2 = phi[0] if phi.dim() == 3 else phi
        gemm = self.gemms[0]
        phi_v = self.phi_proj[str(gemm.d_v)](phi2).unsqueeze(0).expand(self.num_blocks, -1, -1)
        return gemm(nK_all, nV_all, phi_v, e)


class KVRefinementNetwork(nn.Module):
    """Shared clean-latent stem plus one batched KV repair head per DiT block."""

    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 12,
        head_dim: int = 128,
        num_blocks: int = 30,
        d_a: int = 768,
        n_stem: int = 4,
        stem_n_heads: int = 8,
        stem_mlp_mult: int = 4,
        head_mlp_mult: int = 2,
        freq_dim: int = 256,
    ) -> None:
        super().__init__()
        if dim != num_heads * head_dim:
            raise ValueError(f"dim ({dim}) must equal num_heads * head_dim ({num_heads} * {head_dim})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.d_a = d_a
        self.n_stem = n_stem
        self.t_embed = TimestepEmbedder(d_a, freq_dim)
        self.stem_in = nn.Linear(dim, d_a)
        self.stem_blocks = nn.ModuleList([StemBlock(d_a, stem_n_heads, stem_mlp_mult) for _ in range(n_stem)])
        self.stem_norm = nn.LayerNorm(d_a, elementwise_affine=True)
        self.heads_grouped = GroupedKVRepairHeads(num_blocks, dim, d_a, num_heads, head_dim, head_mlp_mult)

    def embed_timestep(
        self,
        timestep: float | torch.Tensor,
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Embed the refinement timestep into ``[batch_size, d_a]``."""
        t_tensor = (
            timestep.to(device).float().reshape(-1)
            if torch.is_tensor(timestep)
            else torch.tensor([float(timestep)], device=device, dtype=torch.float32)
        )
        if t_tensor.numel() == 1:
            t_tensor = t_tensor.expand(batch_size)
        return self.t_embed(t_tensor).to(dtype)

    def compute_phi(self, stem_tokens: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Run the spatial stem over patch-embedded clean latents ``[B, S, dim]``."""
        x = self.stem_in(stem_tokens)
        for block in self.stem_blocks:
            x = block(x, e)
        return self.stem_norm(x)

    def repair_all(
        self,
        nK_all: torch.Tensor,
        nV_all: torch.Tensor,
        phi: torch.Tensor,
        e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine all layers' keys and values ``[L, S, dim]`` in a single grouped call."""
        return self.heads_grouped(nK_all, nV_all, phi, e)

    @property
    def repaired_block_indices(self) -> list[int]:
        """DiT block indices repaired by this network."""
        return list(range(self.num_blocks))
