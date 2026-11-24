# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, models/wan_model.py.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Bidirectional Astronex-World denoiser: Wan2.2 TI2V-5B with a PRoPE camera branch.

Every self-attention layer up to ``prope_num_layers`` runs a second attention over
PRoPE-projected Q/K/V and adds it back through ``self_attn.prope_o``. Timestep
modulation is per frame whenever the timestep is (TI2V conditions the reference
frame at ``t = 0``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..forcing.model import WanLayerNorm, WanRMSNorm, rope_apply, rope_params
from ..minwm.prope import prope_qkv
from .action import ActionEmbedder
from .attention import attention


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal timestep features for a timestep tensor of any shape (``[..., dim]``)."""
    half = dim // 2
    position = position.type(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half).to(position).div(half))
    sinusoid = position.unsqueeze(-1) * freqs
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)


def expand_cameras_to_tokens(
    viewmats: torch.Tensor, Ks: torch.Tensor, grid_sizes: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat per-frame ``[B, F, 4, 4]`` poses and ``[B, F, 3, 3]`` intrinsics over each frame's tokens."""
    expanded_viewmats, expanded_ks = [], []
    for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        tokens = frames * height * width
        viewmat = viewmats[index, :frames][:, None, None].expand(-1, height, width, -1, -1).reshape(tokens, 4, 4)
        intrinsics = Ks[index, :frames][:, None, None].expand(-1, height, width, -1, -1).reshape(tokens, 3, 3)
        pad = seq_len - tokens
        if pad > 0:
            identity4 = torch.eye(4, device=viewmat.device, dtype=viewmat.dtype).unsqueeze(0).expand(pad, -1, -1)
            identity3 = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype).unsqueeze(0).expand(pad, -1, -1)
            viewmat = torch.cat([viewmat, identity4])
            intrinsics = torch.cat([intrinsics, identity3])
        expanded_viewmats.append(viewmat)
        expanded_ks.append(intrinsics)
    return torch.stack(expanded_viewmats), torch.stack(expanded_ks)


def pad_text_context(context: torch.Tensor, text_len: int) -> torch.Tensor:
    """Zero-pad every ``[L, C]`` prompt embedding to ``text_len`` tokens."""
    return torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])


class AstronexSelfAttention(nn.Module):
    """Full-clip self-attention with 3D RoPE and an optional PRoPE camera branch."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6, use_camera: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.prope_o = nn.Linear(dim, dim) if use_camera else None

    def forward(self, x, seq_lens, grid_sizes, freqs, viewmats=None, Ks=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        out = attention(rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs), v, k_lens=seq_lens)
        out = self.o(out.flatten(2))
        if viewmats is not None and self.prope_o is not None:
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), viewmats=viewmats, Ks=Ks
            )
            camera = attention(
                q_p.permute(0, 2, 1, 3), k_p.permute(0, 2, 1, 3), v_p.permute(0, 2, 1, 3), k_lens=seq_lens
            )
            camera = apply_fn_o(camera.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            out = out + self.prope_o(camera.flatten(2))
        return out


class AstronexCrossAttention(nn.Module):
    """Text cross-attention whose key/value projections can be cached across causal chunks."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if crossattn_cache is not None and crossattn_cache["is_init"]:
            k, v = crossattn_cache["k"], crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            if crossattn_cache is not None:
                crossattn_cache.update(is_init=True, k=k, v=v)
        return self.o(attention(q, k, v).flatten(2))


class AstronexAttentionBlock(nn.Module):
    """Wan block with clip-level (``[B, 6, C]``) or per-frame (``[B, F, 6, C]``) modulation."""

    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6, use_camera=False):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = AstronexSelfAttention(dim, num_heads, qk_norm, eps, use_camera)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = AstronexCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, viewmats=None, Ks=None):
        if e.dim() == 4:
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

            def modulate(h, scale, shift):
                return (h.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[scale]) + e[shift]).flatten(1, 2)

            def gate(h, index):
                return (h.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[index]).flatten(1, 2)
        else:
            e = (self.modulation + e).chunk(6, dim=1)

            def modulate(h, scale, shift):
                return h * (1 + e[scale]) + e[shift]

            def gate(h, index):
                return h * e[index]

        y = self.self_attn(modulate(self.norm1(x), 1, 0), seq_lens, grid_sizes, freqs, viewmats=viewmats, Ks=Ks)
        x = x + gate(y, 2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(modulate(self.norm2(x), 4, 3))
        return x + gate(y, 5)


class AstronexHead(nn.Module):
    """Output head with clip-level (``[B, C]``) or per-frame (``[B, F, C]``) modulation."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        if e.dim() == 3:
            num_frames = e.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.norm(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            return self.head(x.flatten(1, 2))
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + e[1]) + e[0])


def unpatchify(x: torch.Tensor, grid_sizes: torch.Tensor, patch_size, out_dim: int) -> list[torch.Tensor]:
    """``[B, L, out_dim * prod(patch)]`` tokens -> per-sample ``[out_dim, F, H, W]`` latents."""
    out = []
    for u, grid in zip(x, grid_sizes.tolist(), strict=False):
        u = u[: math.prod(grid)].view(*grid, *patch_size, out_dim)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        out.append(u.reshape(out_dim, *[i * j for i, j in zip(grid, patch_size, strict=False)]))
    return out


def wan_rope_freqs(dim: int, num_heads: int) -> torch.Tensor:
    """Wan's complex 3D RoPE table split over (frame, height, width) head channels."""
    d = dim // num_heads
    return torch.cat(
        [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
        dim=1,
    )


class AstronexWanModel(nn.Module):
    """Bidirectional Astronex-World denoiser over a whole clip."""

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,
        num_heads=24,
        num_layers=30,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_camera=False,
        prope_num_layers=None,
        use_action=False,
        action_dim=64,
        num_embodiments=32,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.text_len = text_len
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        camera_layers = num_layers if prope_num_layers is None else min(int(prope_num_layers), num_layers)

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                AstronexAttentionBlock(
                    dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, use_camera and index < camera_layers
                )
                for index in range(num_layers)
            ]
        )
        self.head = AstronexHead(dim, out_dim, self.patch_size, eps)
        # The released checkpoint carries the action embedder for both sampling forms;
        # the bidirectional sampler does not condition on actions.
        self.action_embedder = ActionEmbedder(dim, action_dim, num_embodiments) if use_action else None
        # A plain attribute rather than a buffer, so ``.to(dtype)`` keeps the complex128 table.
        self.freqs = wan_rope_freqs(dim, num_heads)

    def forward(self, x, t, context, seq_len, viewmats=None, Ks=None):
        """``x``: ``[B, C, F, H, W]`` latents; ``t``: ``[B]`` or per-frame ``[B, F]``; returns the flow."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(-1 if t.dim() > 1 else 1, (6, self.dim))
        context = self.text_embedding(pad_text_context(context, self.text_len))
        if viewmats is not None and Ks is not None:
            viewmats, Ks = expand_cameras_to_tokens(viewmats, Ks, grid_sizes, seq_len)

        for block in self.blocks:
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, viewmats=viewmats, Ks=Ks)
        x = self.head(x, e)
        return torch.stack(unpatchify(x, grid_sizes, self.patch_size, self.out_dim))
