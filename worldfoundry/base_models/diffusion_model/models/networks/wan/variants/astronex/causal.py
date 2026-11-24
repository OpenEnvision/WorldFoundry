# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, models/causal_model.py.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Causal Astronex-World denoiser: chunked generation over sink + sliding-window KV caches.

The content cache stores raw keys; every call re-ropes the resident window with the
window start as frame 0, which pins the sink-to-query rotation distance. The PRoPE
cache likewise stores raw K/V when ``prope_chunk_anchor`` is set and re-projects the
whole window under the caller's anchor pose (``prope_window_viewmats``) at every call.
Eviction either drops the oldest non-sink frames or keeps the frames the caller names
(``cache_keep_frames``, chosen by frustum overlap); both caches track the global frame
held in each slot (``frame_ids``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..forcing.model import WanLayerNorm
from ..forcing.self_forcing import CausalHead, causal_rope_apply
from ..minwm.prope import _prepare_apply_fns_all_dim, prope_qkv
from .action import ActionEmbedder, apply_action_modulation
from .attention import attention
from .model import (
    AstronexCrossAttention,
    WanRMSNorm,
    expand_cameras_to_tokens,
    pad_text_context,
    sinusoidal_embedding_1d,
    unpatchify,
    wan_rope_freqs,
)


def roll_cache_by_frames(cache: dict, keep_frames, frame_seqlen: int, sink_size: int, num_rolled_tokens: int) -> None:
    """Compact the non-sink cache slots down to ``keep_frames`` (global frame ids), in the order given.

    Survivors land contiguously after the sink; the resident window is re-roped from
    slot zero at every call, so a slot's position comes from its slot, not its age.
    """
    ids = cache["frame_ids"]
    where = {frame: slot for slot, frame in enumerate(ids.tolist()) if frame >= 0}
    wanted = list(keep_frames)
    room = num_rolled_tokens // frame_seqlen
    if len(wanted) != room or len(set(wanted)) != room:
        raise RuntimeError(f"Expected {room} distinct history frames, got {wanted}")
    missing = [frame for frame in wanted if frame not in where or where[frame] < sink_size]
    if missing:
        raise RuntimeError(f"Requested history is not resident: {missing}; resident={ids.tolist()}")
    k, v = cache["k"], cache["v"]
    # Every source block is cloned before any write: a survivor may land where another is read from.
    sources_k = [k[:, where[f] * frame_seqlen : (where[f] + 1) * frame_seqlen].clone() for f in wanted]
    sources_v = [v[:, where[f] * frame_seqlen : (where[f] + 1) * frame_seqlen].clone() for f in wanted]
    for slot, (block_k, block_v) in enumerate(zip(sources_k, sources_v, strict=False)):
        start = (sink_size + slot) * frame_seqlen
        k[:, start : start + frame_seqlen] = block_k
        v[:, start : start + frame_seqlen] = block_v
    new_ids = ids.clone()
    for slot, frame in enumerate(wanted):
        new_ids[sink_size + slot] = frame
    new_ids[sink_size + len(wanted) :] = -1
    cache["frame_ids"] = new_ids


def _write_rolling_cache(
    cache: dict,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    *,
    current_start: int,
    frame_seqlen: int,
    sink_size: int,
    local_attn_size: int,
    cache_keep_frames,
) -> int:
    """Append this call's tokens to a sink + sliding-window cache; return the new local end index."""
    num_new_tokens = new_k.shape[1]
    current_end = current_start + num_new_tokens
    cache_size = cache["k"].shape[1]
    sink_tokens = sink_size * frame_seqlen
    if "frame_ids" not in cache:
        cache["frame_ids"] = torch.full((cache_size // frame_seqlen,), -1, dtype=torch.long, device=cache["k"].device)
    global_end = cache["global_end_index"].item()
    local_end = cache["local_end_index"].item()
    num_evicted_tokens = 0
    if local_attn_size != -1 and current_end > global_end and num_new_tokens + local_end > cache_size:
        num_evicted_tokens = num_new_tokens + local_end - cache_size
        num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
        if cache_keep_frames is not None:
            roll_cache_by_frames(cache, cache_keep_frames, frame_seqlen, sink_size, num_rolled_tokens)
        else:
            source = slice(sink_tokens + num_evicted_tokens, sink_tokens + num_evicted_tokens + num_rolled_tokens)
            cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["k"][:, source].clone()
            cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["v"][:, source].clone()
            keep = num_rolled_tokens // frame_seqlen
            drop = num_evicted_tokens // frame_seqlen
            ids = cache["frame_ids"]
            ids[sink_size : sink_size + keep] = ids[sink_size + drop : sink_size + drop + keep].clone()
            ids[sink_size + keep :] = -1
    local_end_index = local_end + current_end - global_end - num_evicted_tokens
    local_start_index = local_end_index - num_new_tokens
    cache["k"][:, local_start_index:local_end_index] = new_k
    cache["v"][:, local_start_index:local_end_index] = new_v
    return local_end_index


class AstronexCausalSelfAttention(nn.Module):
    """Causal self-attention over the KV caches, with an optional chunk-anchored PRoPE branch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_camera: bool = False,
        prope_chunk_anchor: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.prope_chunk_anchor = prope_chunk_anchor
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.prope_o = nn.Linear(dim, dim) if use_camera else None

    def forward(
        self,
        x,
        grid_sizes,
        freqs,
        kv_cache,
        current_start=0,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
        prope_window_viewmats=None,
        prope_window_Ks=None,
        cache_keep_frames=None,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        max_attention_size = 31200 if self.local_attn_size == -1 else self.local_attn_size * frame_seqlen
        num_new_frames = s // frame_seqlen
        rolling = {
            "current_start": current_start,
            "frame_seqlen": frame_seqlen,
            "sink_size": self.sink_size,
            "local_attn_size": self.local_attn_size,
            "cache_keep_frames": cache_keep_frames,
        }

        attn_end = _write_rolling_cache(kv_cache, k, v, **rolling)
        ids = kv_cache["frame_ids"]
        first_slot = (attn_end - s) // frame_seqlen
        first_frame = current_start // frame_seqlen
        for offset in range(num_new_frames):
            if first_slot + offset < ids.shape[0]:
                ids[first_slot + offset] = first_frame + offset
        kv_cache["global_end_index"].fill_(current_start + s)
        kv_cache["local_end_index"].fill_(attn_end)

        window_start = max(0, attn_end - max_attention_size)
        window_frames = (attn_end - window_start) // frame_seqlen
        window_grid = grid_sizes.clone()
        window_grid[:, 0] = window_frames
        roped_key = causal_rope_apply(kv_cache["k"][:, window_start:attn_end], window_grid, freqs, start_frame=0)
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=window_frames - num_new_frames)
        out = attention(roped_query.type_as(v), roped_key.type_as(v), kv_cache["v"][:, window_start:attn_end])
        out = self.o(out.flatten(2))

        if viewmats is None or self.prope_o is None or prope_kv_cache is None:
            return out

        chunk_anchor = self.prope_chunk_anchor and prope_window_viewmats is not None
        q_p, k_p, v_p, apply_fn_o = prope_qkv(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), viewmats=viewmats, Ks=Ks
        )
        q_p, k_p, v_p = (t.permute(0, 2, 1, 3) for t in (q_p, k_p, v_p))
        # Chunk-anchored caches hold raw K/V and are re-projected under the current anchor below.
        cached_k, cached_v = (k, v) if chunk_anchor else (k_p, v_p)
        p_end = _write_rolling_cache(prope_kv_cache, cached_k, cached_v, **rolling)
        prope_kv_cache["frame_ids"][(p_end - s) // frame_seqlen : (p_end - s) // frame_seqlen + num_new_frames] = (
            torch.arange(first_frame, first_frame + num_new_frames, device=prope_kv_cache["frame_ids"].device)
        )
        if cache_keep_frames is not None and not torch.equal(kv_cache["frame_ids"], prope_kv_cache["frame_ids"]):
            raise RuntimeError("Content and PRoPE cache frame order disagree")
        p_start = max(0, p_end - max_attention_size)
        if chunk_anchor:
            cached_frames = (p_end - p_start) // frame_seqlen
            if cached_frames != prope_window_viewmats.shape[1]:
                raise RuntimeError(
                    f"PRoPE window mismatch: cache holds {cached_frames} frames but "
                    f"prope_window_viewmats has {prope_window_viewmats.shape[1]}"
                )
            apply_fn_kv = _prepare_apply_fns_all_dim(
                head_dim=self.head_dim,
                viewmats=prope_window_viewmats,
                Ks=prope_window_Ks,
                patches_x=None,  # ty: ignore[invalid-argument-type]
                patches_y=None,  # ty: ignore[invalid-argument-type]
                image_width=None,  # ty: ignore[invalid-argument-type]
                image_height=None,  # ty: ignore[invalid-argument-type]
            )[1]
            window_k = apply_fn_kv(prope_kv_cache["k"][:, p_start:p_end].permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            window_v = apply_fn_kv(prope_kv_cache["v"][:, p_start:p_end].permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        else:
            window_k = prope_kv_cache["k"][:, p_start:p_end]
            window_v = prope_kv_cache["v"][:, p_start:p_end]
        camera = attention(q_p, window_k, window_v)
        prope_kv_cache["global_end_index"].fill_(current_start + s)
        prope_kv_cache["local_end_index"].fill_(p_end)
        camera = apply_fn_o(camera.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return out + self.prope_o(camera.flatten(2))


class AstronexCausalAttentionBlock(nn.Module):
    """Wan block with per-frame ``[B, F, 6, C]`` modulation over the causal caches."""

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_camera=False,
        prope_chunk_anchor=False,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = AstronexCausalSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, use_camera, prope_chunk_anchor
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = AstronexCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes, freqs, context, kv_cache, crossattn_cache, current_start, **camera):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        def frames(h):
            return h.unflatten(dim=1, sizes=(num_frames, frame_seqlen))

        y = self.self_attn(
            (frames(self.norm1(x)) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            **camera,
        )
        x = x + (frames(y) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache=crossattn_cache)
        y = self.ffn((frames(self.norm2(x)) * (1 + e[4]) + e[3]).flatten(1, 2))
        return x + (frames(y) * e[5]).flatten(1, 2)


class AstronexCausalWanModel(nn.Module):
    """Causal Astronex-World denoiser; every call denoises one chunk against the caches."""

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
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_camera=False,
        prope_num_layers=None,
        prope_chunk_anchor=False,
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
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        camera_layers = num_layers if prope_num_layers is None else min(int(prope_num_layers), num_layers)

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                AstronexCausalAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    use_camera and index < camera_layers,
                    prope_chunk_anchor,
                )
                for index in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, self.patch_size, eps)
        self.action_embedder = ActionEmbedder(dim, action_dim, num_embodiments) if use_action else None
        # A plain attribute rather than a buffer, so ``.to(dtype)`` keeps the complex128 table.
        self.freqs = wan_rope_freqs(dim, num_heads)

    def forward(
        self,
        x,
        t,
        context,
        kv_cache,
        crossattn_cache,
        current_start=0,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
        prope_window_viewmats=None,
        prope_window_Ks=None,
        cache_keep_frames=None,
        actions=None,
        embodiment_id=None,
    ):
        """``x``: ``[B, C, F, H, W]`` chunk latents; ``t``: per-frame ``[B, F]``; returns the chunk flow."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        e0 = apply_action_modulation(self.action_embedder, e0, actions, embodiment_id)
        context = self.text_embedding(pad_text_context(context, self.text_len))
        if viewmats is not None and Ks is not None:
            viewmats, Ks = expand_cameras_to_tokens(viewmats, Ks, grid_sizes, x.shape[1])

        camera = {
            "viewmats": viewmats,
            "Ks": Ks,
            "prope_window_viewmats": prope_window_viewmats,
            "prope_window_Ks": prope_window_Ks,
            "cache_keep_frames": cache_keep_frames,
        }
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                e0,
                grid_sizes,
                self.freqs,
                context,
                kv_cache[index],
                crossattn_cache[index],
                current_start,
                prope_kv_cache=prope_kv_cache[index] if prope_kv_cache is not None else None,
                **camera,
            )
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        return torch.stack(unpatchify(x, grid_sizes, self.patch_size, self.out_dim))
