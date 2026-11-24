# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License / Apache-2.0 Wan derivatives).
"""Frame-wise autoregressive causal Wan2.1 DiT for FastFrames inference.

Reuses WorldFoundry's canonical Wan normalization, RoPE frequency, and sinusoidal
timestep primitives from ``wan.variants.forcing.model`` while preserving the exact
FastFrames checkpoint key layout, persistent per-layer KV cache, pre-head hidden
state export for the pixel decoder, and optional ``FastFramesFastOps`` fused QKV /
FP32 real-valued RoPE / GEMM patch-embedding path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import (
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)

__all__ = [
    "FASTFRAMES_WAN_CONFIGS",
    "FastFramesCausalAttentionBlock",
    "FastFramesCausalHead",
    "FastFramesCausalWanModel",
    "FastFramesCrossAttention",
    "FastFramesFastOps",
    "FastFramesSelfAttention",
    "FastFramesWanConfig",
    "allocate_fastframes_kv_cache",
    "causal_rope_apply",
    "fastframes_attention",
    "fastframes_cross_attention",
    "rope_real",
    "set_fastframes_cross_attention_backend",
]

_cross_attention_backend: str = "cudnn"


def set_fastframes_cross_attention_backend(name: str) -> None:
    """Select the cross-attention SDPA dispatch mode (``"cudnn"`` or ``"auto"``)."""
    global _cross_attention_backend
    if name not in ("cudnn", "auto"):
        raise ValueError(f"Unknown FastFrames cross-attention backend: {name!r}")
    _cross_attention_backend = name


def fastframes_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    backend: str = "cudnn",
) -> torch.Tensor:
    """Multi-head SDPA on ``[B, L, H, D]`` tensors with cuDNN preference on CUDA.

    On CUDA BF16/FP16 tensors with ``backend="cudnn"``, attempts cuDNN SDPA first
    (matching upstream FastFrames) and falls back cleanly to standard PyTorch SDPA
    when cuDNN attention kernels are unavailable for the current device or shape.
    """
    if backend not in ("cudnn", "auto"):
        raise ValueError(f"Unknown FastFrames attention backend: {backend!r}")

    out_dtype = q.dtype
    compute_dtype = torch.bfloat16 if (q.is_cuda or out_dtype == torch.bfloat16) else out_dtype
    q_t = q.transpose(1, 2).to(compute_dtype)
    k_t = k.transpose(1, 2).to(compute_dtype)
    v_t = v.transpose(1, 2).to(compute_dtype)

    if backend == "cudnn" and q_t.is_cuda and compute_dtype in (torch.float16, torch.bfloat16):
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal, dropout_p=dropout_p)
        except RuntimeError:
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal, dropout_p=dropout_p)
    else:
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def fastframes_cross_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Text cross-attention using the configured FastFrames SDPA backend."""
    return fastframes_attention(q, k, v, backend=_cross_attention_backend)


def causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply 3-D RoPE on ``[B, L, H, D]`` tokens whose first frame is ``start_frame``."""
    num_heads, half_dim = x.size(2), x.size(3) // 2
    freq_t, freq_h, freq_w = freqs.split([half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3], dim=1)
    outputs: list[torch.Tensor] = []
    for index, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[index, :seq_len].to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        freqs_i = torch.cat(
            [
                freq_t[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        rotated = torch.view_as_real(x_i * freqs_i).flatten(2)
        outputs.append(torch.cat([rotated, x[index, seq_len:]]))
    return torch.stack(outputs).type_as(x)


def rope_real(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Apply RoPE as FP32 real rotations using ``table`` shaped ``[L, 1, D/2, 2]``."""
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = table[..., 0], table[..., 1]
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).reshape_as(x).to(x.dtype)


def _unpatchify(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    patch_size: tuple[int, int, int],
    out_dim: int,
) -> list[torch.Tensor]:
    """Unpack ``[B, L, prod(patch) * C]`` tokens to a list of ``[C, F, H, W]`` tensors."""
    outputs: list[torch.Tensor] = []
    for item, dims in zip(x, grid_sizes.tolist(), strict=True):
        token_count = math.prod(dims)
        unpacked = item[:token_count].view(*dims, *patch_size, out_dim)
        unpacked = torch.einsum("fhwpqrc->cfphqwr", unpacked)
        outputs.append(unpacked.reshape(out_dim, *(d * p for d, p in zip(dims, patch_size, strict=True))))
    return outputs


def _init_wan_weights(model: FastFramesCausalWanModel) -> None:
    """Match upstream Wan2.1 weight initialization."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(model.patch_embedding.weight.flatten(1))
    for module in model.text_embedding.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
    for module in model.time_embedding.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
    nn.init.zeros_(model.head.head.weight)


@dataclass(frozen=True)
class FastFramesWanConfig:
    """Architecture parameters for a FastFrames Wan2.1 causal DiT backbone."""

    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6


FASTFRAMES_WAN_CONFIGS: dict[str, FastFramesWanConfig] = {
    "Wan2.1-T2V-1.3B": FastFramesWanConfig(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30),
    "Wan2.1-T2V-14B": FastFramesWanConfig(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40),
}


class FastFramesSelfAttention(nn.Module):
    """Wan2.1 self-attention projections and QK RMSNorm matching checkpoint keys."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q: nn.Module = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k: nn.Module = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and normalize Q/K/V into ``[B, L, H, D]`` tensors."""
        batch_size, seq_len = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        return q, k, v


class FastFramesCrossAttention(FastFramesSelfAttention):
    """Wan2.1 text cross-attention with optional pre-projected text K/V cache."""

    def text_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project text context once per prompt into ``[B, L_text, H, D]`` K/V."""
        batch_size = context.size(0)
        k = self.norm_k(self.k(context)).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v(context).view(batch_size, -1, self.num_heads, self.head_dim)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.norm_q(self.q(x)).view(batch_size, -1, self.num_heads, self.head_dim)
        if cache is not None:
            k, v = cache["k"], cache["v"]
        else:
            if context is None:
                raise ValueError("context is required when cross-attention cache is None")
            k, v = self.text_kv(context)
        return self.o(fastframes_cross_attention(q, k, v).flatten(2))


class FastFramesCausalAttentionBlock(nn.Module):
    """Causal Wan2.1 transformer block updating one frame slot of the KV cache."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = FastFramesSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3: nn.Module = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = FastFramesCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward_frame(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        cross_kv: dict[str, torch.Tensor],
        kv_cache: dict[str, torch.Tensor],
        frame: int,
        fast: FastFramesFastOps | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        """Process one latent frame ``[B, L, C]`` attending to frames ``<= frame``."""
        tokens = x.shape[1]
        start, end = frame * tokens, (frame + 1) * tokens
        mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        h = (self.norm1(x).unflatten(1, (1, tokens)) * (1 + mod[1]) + mod[0]).flatten(1, 2)
        sa = self.self_attn
        shape = (x.shape[0], tokens, sa.num_heads, sa.head_dim)
        if fast is None:
            q, k, v = sa.qkv(h)
            q = causal_rope_apply(q, grid_sizes, freqs, start_frame=frame).type_as(v)
            k = causal_rope_apply(k, grid_sizes, freqs, start_frame=frame).type_as(v)
        else:
            q_raw, k_raw, v_raw = F.linear(h, *fast.qkv[layer]).chunk(3, dim=-1)
            q = rope_real(sa.norm_q(q_raw).view(shape), fast.rope[frame])
            k = rope_real(sa.norm_k(k_raw).view(shape), fast.rope[frame])
            v = v_raw.view(shape)
        kv_cache["k"][:, start:end].copy_(k)
        kv_cache["v"][:, start:end].copy_(v)
        y = sa.o(fastframes_attention(q, kv_cache["k"][:, :end], kv_cache["v"][:, :end]).flatten(2))
        x = x + (y.unflatten(1, (1, tokens)) * mod[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), None, cache=cross_kv)
        y = self.ffn((self.norm2(x).unflatten(1, (1, tokens)) * (1 + mod[4]) + mod[3]).flatten(1, 2))
        return x + (y.unflatten(1, (1, tokens)) * mod[5]).flatten(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        cross_kv: dict[str, torch.Tensor],
        kv_cache: dict[str, torch.Tensor],
        frame: int,
        fast: FastFramesFastOps | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        return self.forward_frame(
            x,
            e,
            grid_sizes,
            freqs,
            cross_kv,
            kv_cache,
            frame,
            fast=fast,
            layer=layer,
        )


class FastFramesCausalHead(nn.Module):
    """Per-frame AdaLN output head mapping pre-head tokens to flow patch voxels."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Project ``x: [B, F*L, C]`` modulated by per-frame ``e: [B, F, 1, C]``."""
        num_frames = e.shape[1]
        frame_tokens = x.shape[1] // num_frames
        shift, scale = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(self.norm(x).unflatten(1, (num_frames, frame_tokens)) * (1 + scale) + shift)


class FastFramesFastOps:
    """Precomputed inference fast paths matching upstream FastFrames ``FastOps``.

    * Patch embedding via reshape + linear GEMM instead of Conv3d.
    * One fused QKV GEMM per DiT block (concatenated Q, K, V weights/biases).
    * Precomputed per-frame FP32 real-valued ``(cos, sin)`` RoPE tables.

    Construct from a frozen model; rebuild if weights or spatial geometry change.
    """

    def __init__(self, model: FastFramesCausalWanModel, height: int, width: int, num_frames: int) -> None:
        patch = model.patch_embedding
        self.patch_size = model.patch_size
        self.patch_weight = patch.weight.detach().reshape(model.dim, -1)
        self.patch_bias = patch.bias
        self.qkv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in model.blocks:
            if not isinstance(block, FastFramesCausalAttentionBlock):
                raise TypeError(f"Expected FastFramesCausalAttentionBlock, got {type(block).__name__}")
            sa = block.self_attn
            weight = torch.cat([sa.q.weight, sa.k.weight, sa.v.weight], dim=0).detach().contiguous()
            if sa.q.bias is None or sa.k.bias is None or sa.v.bias is None:
                raise ValueError("FastFramesCausalSelfAttention projections must include biases.")
            bias = torch.cat([sa.q.bias, sa.k.bias, sa.v.bias], dim=0).detach().contiguous()
            self.qkv.append((weight, bias))
        h = height // model.patch_size[1]
        w = width // model.patch_size[2]
        half = model.dim // model.num_heads // 2
        model.freqs = model.freqs.to(patch.weight.device)
        temporal, vertical, horizontal = model.freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
        self.rope: list[torch.Tensor] = [
            torch.view_as_real(
                torch.cat(
                    (
                        temporal[frame : frame + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                        vertical[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                        horizontal[:w].view(1, 1, w, -1).expand(1, h, w, -1),
                    ),
                    dim=-1,
                ).reshape(h * w, 1, half)
            )
            .float()
            .contiguous()
            for frame in range(num_frames)
        ]

    def patch_embed(self, latent: torch.Tensor) -> torch.Tensor:
        """Project ``[B, 1, C, H, W]`` latent frame to ``[B, H*W/(p_h*p_w), dim]``."""
        _, p_h, p_w = self.patch_size
        batch_size, _, channels, height, width = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(batch_size, channels, 1, height // p_h, p_h, width // p_w, p_w)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch_size, (height // p_h) * (width // p_w), channels * p_h * p_w)
        return F.linear(x, self.patch_weight, self.patch_bias)


class FastFramesCausalWanModel(nn.Module):
    """Frame-wise autoregressive Wan2.1 T2V DiT returning flow and pre-head features."""

    def __init__(self, config: FastFramesWanConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        if dim % config.num_heads != 0 or (dim // config.num_heads) % 2 != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({config.num_heads}) with an even head_dim")
        self.patch_size = tuple(config.patch_size)
        self.text_len = config.text_len
        self.dim = dim
        self.freq_dim = config.freq_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.patch_embedding = nn.Conv3d(
            config.in_dim,
            dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                FastFramesCausalAttentionBlock(
                    dim,
                    config.ffn_dim,
                    config.num_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = FastFramesCausalHead(dim, config.out_dim, self.patch_size, config.eps)
        self.freqs = self.rope_table()
        _init_wan_weights(self)

    def rope_table(self) -> torch.Tensor:
        """Complex128 RoPE table kept outside buffers so ``.to(dtype)`` preserves precision."""
        head_dim = self.dim // self.num_heads
        with torch.device("cpu"):
            return torch.cat(
                [
                    rope_params(1024, head_dim - 4 * (head_dim // 6)),
                    rope_params(1024, 2 * (head_dim // 6)),
                    rope_params(1024, 2 * (head_dim // 6)),
                ],
                dim=1,
            )

    def embed_text(self, prompt_embeds: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Zero-pad T5 prompt embeddings to ``text_len`` tokens and project into ``dim``."""
        padded: list[torch.Tensor] = []
        for item in prompt_embeds:
            if item.size(0) > self.text_len:
                raise ValueError(f"prompt_embeds length {item.size(0)} exceeds text_len={self.text_len}")
            padded.append(torch.cat([item, item.new_zeros(self.text_len - item.size(0), item.size(1))], dim=0))
        return self.text_embedding(torch.stack(padded, dim=0))

    def time_conditioning(self, t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``t: [B, F]`` timesteps to ``(e0: [B, F, 6, C], e_head: [B, F, 1, C])``."""
        emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype))
        e0 = self.time_projection(emb).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
        return e0, emb.unflatten(0, t.shape).unsqueeze(2)

    def text_cache(self, prompt_embeds: torch.Tensor | Sequence[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Pre-project prompt embeddings once into per-layer cross-attention K/V."""
        context = self.embed_text(prompt_embeds)
        caches: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            if not isinstance(block, FastFramesCausalAttentionBlock):
                raise TypeError(f"Expected FastFramesCausalAttentionBlock, got {type(block).__name__}")
            k, v = block.cross_attn.text_kv(context)
            caches.append({"k": k, "v": v})
        return caches

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Unpack ``[B, L, prod(patch) * C]`` tokens to a list of ``[C, F, H, W]`` tensors."""
        return _unpatchify(x, grid_sizes, self.patch_size, self.out_dim)

    def forward_frame(
        self,
        latent: torch.Tensor,
        e0: torch.Tensor,
        e_head: torch.Tensor,
        text_cache: Sequence[dict[str, torch.Tensor]],
        kv_cache: Sequence[dict[str, torch.Tensor]],
        frame: int,
        fast: FastFramesFastOps | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise one latent frame ``[B, 1, C, H, W]`` attending to frames ``<= frame``.

        Returns ``(flow, pre_head_features)`` where ``flow`` has shape
        ``[B, 1, out_dim, H, W]`` and ``pre_head_features`` has shape ``[B, L, C]``.
        """
        if self.freqs.device != latent.device:
            self.freqs = self.freqs.to(latent.device)
        batch_size, _, _, height, width = latent.shape
        if fast is None:
            x = self.patch_embedding(latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        else:
            x = fast.patch_embed(latent)
        grid_sizes = torch.tensor(
            [[1, height // self.patch_size[1], width // self.patch_size[2]]] * batch_size,
            dtype=torch.long,
        )
        for index, block in enumerate(self.blocks):
            if not isinstance(block, FastFramesCausalAttentionBlock):
                raise TypeError(f"Expected FastFramesCausalAttentionBlock, got {type(block).__name__}")
            x = block.forward_frame(
                x,
                e0,
                grid_sizes,
                self.freqs,
                text_cache[index],
                kv_cache[index],
                frame,
                fast=fast,
                layer=index,
            )
        tokens = self.head(x, e_head)
        flow = torch.stack(self.unpatchify(tokens, grid_sizes), dim=0).permute(0, 2, 1, 3, 4)
        return flow, x

    def forward(
        self,
        latent: torch.Tensor,
        e0: torch.Tensor,
        e_head: torch.Tensor,
        text_cache: Sequence[dict[str, torch.Tensor]],
        kv_cache: Sequence[dict[str, torch.Tensor]],
        frame: int,
        fast: FastFramesFastOps | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_frame(latent, e0, e_head, text_cache, kv_cache, frame, fast=fast)


def allocate_fastframes_kv_cache(
    model: FastFramesCausalWanModel,
    num_frames: int,
    frame_tokens: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Allocate per-layer self-attention KV caches shaped ``[B, F * L, H, D]``."""
    target_device = model.patch_embedding.weight.device if device is None else torch.device(device)
    shape = (batch_size, num_frames * frame_tokens, model.num_heads, model.dim // model.num_heads)
    return [
        {
            "k": torch.zeros(shape, dtype=dtype, device=target_device),
            "v": torch.zeros(shape, dtype=dtype, device=target_device),
        }
        for _ in range(model.num_layers)
    ]
