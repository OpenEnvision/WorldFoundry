# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The lingbot-world Authors. Portions derived from:
# https://github.com/Robbyant/lingbot-world/blob/main/wan/distributed/sequence_parallel.py
#
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt at repo root).
#
# Modifications Copyright (c) 2026 SkyworkAI and contributors.
"""Memory-token RoPE under sequence parallel (padded Ulysses).

Memory / sink tokens have a different temporal length than the rolling
window. This module pads RoPE and uses
:mod:`worldfoundry.core.attention.parallel.padded_ulysses_attention` so ranks can
all-to-all unequal sequence tails without dropping the sink block.
"""

import torch
import torch.nn.functional as torch_F
from einops import rearrange

from worldfoundry.core.attention.parallel.padded_ulysses_attention import distributed_attention
from worldfoundry.core.attention.parallel.sequence_parallel_rope import make_sequence_parallel_rope_apply, pad_freqs
from worldfoundry.core.distributed.sequence_parallel.ops import gather_forward, get_rank, get_world_size
from worldfoundry.core.nn import sinusoidal_embedding_1d

rope_apply = make_sequence_parallel_rope_apply(
    get_world_size,
    get_rank,
    compute_dtype=torch.float32,
    match_frequency_dtype=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Memory-aware RoPE — sink frames and rolling frames use distinct indices
# ──────────────────────────────────────────────────────────────────────────


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_mem_sp(x, grid_sizes, freqs, memory_length, memory_latent_idx, predict_latent_idx):
    """
    Apply RoPE to input tensor using precomputed freqs and optional time indices, with SP support.
    x: [B, S_local, N, C]
    grid_sizes: [B, 3] (Global)
    freqs: [M, C/2]
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    if freqs.dim() == 3:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=2)
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        freqs_i_list = []

        if memory_length > 0:
            grid_sizes_mem = [memory_length, h, w]

            if memory_latent_idx is not None:
                mem_indices = memory_latent_idx
            else:
                mem_indices = list(range(memory_length))

            freqs_mem = _get_freqs_chunk(grid_sizes_mem, freqs, mem_indices)
            freqs_i_list.append(freqs_mem)

        # Pred part
        pred_len_frames = f - memory_length
        if pred_len_frames > 0:
            grid_sizes_pred = [pred_len_frames, h, w]

            if predict_latent_idx is not None:
                if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                    start_idx, end_idx = predict_latent_idx
                    pred_indices = list(range(start_idx, end_idx))
                else:
                    pred_indices = predict_latent_idx
            else:
                pred_indices = list(range(pred_len_frames))

            freqs_pred = _get_freqs_chunk(grid_sizes_pred, freqs, pred_indices)
            freqs_i_list.append(freqs_pred)

        freqs_i = torch.cat(freqs_i_list, dim=0)

        sp_size = get_world_size()
        sp_rank = get_rank()

        freqs_i = pad_freqs(freqs_i, s * sp_size)

        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]

        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(s, n, -1, 2))
        x_i = torch.view_as_real(x_i * freqs_i_rank.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        output.append(x_i)

    return torch.stack(output).float()


def _get_freqs_chunk(grid_sizes, freqs, t_indices):
    """Build a 3D cis table for ``t_indices`` frames at spatial ``(h, w)``.

    Supports both ``[M, C/2]`` and batched ``[N, M, C/2]`` frequency
    tables. Indices may be a tensor or a Python sequence.
    """
    f, h, w = grid_sizes
    seq_len = f * h * w

    if torch.is_tensor(t_indices):
        t_idx = t_indices
    else:
        t_idx = torch.tensor(t_indices, device=freqs[0].device)
    t_idx = t_idx.to(dtype=torch.long)

    if freqs[0].dim() == 3:
        n = freqs[0].size(0)
        t_freqs = freqs[0][:, t_idx, :]  # [n, f, c_t]
        h_freqs = freqs[1][:, :h, :]  # [n, h, c_h]
        w_freqs = freqs[2][:, :w, :]  # [n, w, c_w]

        freqs_i = torch.cat(
            [
                t_freqs.permute(1, 0, 2).view(f, 1, 1, n, -1).expand(f, h, w, n, -1),
                h_freqs.permute(1, 0, 2).view(1, h, 1, n, -1).expand(f, h, w, n, -1),
                w_freqs.permute(1, 0, 2).view(1, 1, w, n, -1).expand(f, h, w, n, -1),
            ],
            dim=-1,
        ).reshape(seq_len, n, -1)
    else:
        t_freqs = freqs[0][t_idx]
        h_freqs = freqs[1][:h]
        w_freqs = freqs[2][:w]

        freqs_i = torch.cat(
            [
                t_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs.view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

    return freqs_i


# ──────────────────────────────────────────────────────────────────────────
# Memory SP DiT — concatenate sink tokens, shard sequence, gather at head
# ──────────────────────────────────────────────────────────────────────────


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    mouse_cond=None,
    keyboard_cond=None,
    x_memory=None,
    timestep_memory=None,
    mouse_cond_memory=None,
    keyboard_cond_memory=None,
    plucker_emb=None,
    memory_latent_idx=None,
    predict_latent_idx=None,
    return_memory=False,
    fa_version=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    memory_length = 0
    if x_memory is not None:
        memory_length = x_memory.shape[2]
        x = torch.cat([x_memory, x], dim=2)
        t = torch.cat([timestep_memory, t], dim=1)

    if self.model_type == "i2v":
        assert y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    elif t.dim() == 2 and t.size(1) < seq_len:
        t = torch.cat([t, t.new_zeros(t.size(0), seq_len - t.size(1))], dim=1)

    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    if (
        getattr(self, "use_text_crossattn", True)
        and getattr(self, "text_embedding", None) is not None
        and context is not None
    ):
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
    else:
        context = None

    if plucker_emb is not None:
        if torch.is_tensor(plucker_emb):
            plucker_items = [u.unsqueeze(0) for u in plucker_emb]
        else:
            plucker_items = [u.unsqueeze(0) if u.dim() == 4 else u for u in plucker_emb]
        plucker_emb = [
            rearrange(
                i,
                "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            )
            for i in plucker_items
        ]
        plucker_emb = torch.cat(plucker_emb, dim=1)
        plucker_emb = self.patch_embedding_wancamctrl(plucker_emb)
        plucker_hidden = self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(plucker_emb)))
        plucker_emb = plucker_emb + plucker_hidden

        plucker_len = plucker_emb.size(1)
        if plucker_len < seq_len:
            pad_len = seq_len - plucker_len
            pad = plucker_emb.new_zeros(plucker_emb.size(0), pad_len, plucker_emb.size(2))
            plucker_emb = torch.cat([plucker_emb, pad], dim=1)
        elif plucker_len > seq_len:
            plucker_emb = plucker_emb[:, :seq_len, :]

        if get_world_size() > 1:
            plucker_emb = torch.chunk(plucker_emb, get_world_size(), dim=1)[get_rank()]

    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    if t.dim() == 2 and t.size(1) == seq_len:
        t = torch.chunk(t, get_world_size(), dim=1)[get_rank()]

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        mouse_cond=mouse_cond,
        keyboard_cond=keyboard_cond,
        plucker_emb=plucker_emb,
        mouse_cond_memory=mouse_cond_memory,
        keyboard_cond_memory=keyboard_cond_memory,
        memory_length=memory_length,
        memory_latent_idx=memory_latent_idx,
        predict_latent_idx=predict_latent_idx,
        fa_version=fa_version,
    )

    for block in self.blocks:
        x = block(x, **kwargs)
        if isinstance(x, tuple):
            x = x[0]

    x = self.head(x, e)

    x = gather_forward(x, dim=1)

    x = self.unpatchify(x, grid_sizes)

    if getattr(self, "use_memory", False):
        x = torch.stack([u.float() for u in x]).float()
        if return_memory:
            return x[:, :, :memory_length], x[:, :, memory_length:]
        return x[:, :, memory_length:]
    return [u.float() for u in x]


# ──────────────────────────────────────────────────────────────────────────
# Memory SP attention — padded Ulysses so unequal sink tails still exchange
# ──────────────────────────────────────────────────────────────────────────


def sp_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    memory_length=0,
    memory_latent_idx=None,
    predict_latent_idx=None,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Sequence-parallel self-attention with optional memory-token RoPE.

    When ``use_memory`` is set, memory and predict frames use distinct
    temporal indices so sink tokens do not inherit the rolling window's
    positions. QKV are cast to the linear's dtype when weights are not
    int8-packed. Attention itself runs through padded Ulysses.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        """Cast to ``dtype`` unless already float16/bfloat16."""
        return x if x.dtype in half_dtypes else x.to(dtype)

    def qkv_fn(x):
        """Project QKV, matching each linear's storage dtype (including int8)."""
        x = x.contiguous()
        q_in = x if hasattr(self.q, "weight_int8") else x.to(self.q.weight.dtype)
        k_in = x if hasattr(self.k, "weight_int8") else x.to(self.k.weight.dtype)
        v_in = x if hasattr(self.v, "weight_int8") else x.to(self.v.weight.dtype)

        q = self.norm_q(self.q(q_in)).view(b, s, n, d)
        k = self.norm_k(self.k(k_in)).view(b, s, n, d)
        v = self.v(v_in).view(b, s, n, d)

        return q.contiguous(), k.contiguous(), v.contiguous()

    q, k, v = qkv_fn(x)

    if getattr(self, "use_memory", False):
        q = rope_apply_mem_sp(q, grid_sizes, freqs, memory_length, memory_latent_idx, predict_latent_idx)
        k = rope_apply_mem_sp(k, grid_sizes, freqs, memory_length, memory_latent_idx, predict_latent_idx)
    else:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
        fa_version=fa_version,
    )

    x = x.flatten(2)
    x = self.o(x)
    return x
