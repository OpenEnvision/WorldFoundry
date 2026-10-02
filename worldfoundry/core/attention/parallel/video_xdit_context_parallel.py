# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Video DiT blocks using xFuser long-context attention and SP RoPE.

Requires xfuser. Applies :func:`make_sequence_parallel_rope_apply` so
frequencies match USP ranks, then runs ``xFuserLongContextAttention``.
The no-xfuser equivalent is
:mod:`worldfoundry.core.attention.parallel.patch_xdit_context_parallel`.

Not this module:
    Scope-style (non-VACE) xfuser forwards live in
    :mod:`.scope_xdit_context_parallel`. In-tree Ulysses without xfuser
    lives in :mod:`.patch_xdit_context_parallel`.

Public surface:

- :func:`usp_dit_forward` / :func:`usp_dit_forward_vace` /
  :func:`usp_attn_forward` — monkey-patch targets for Wan video + VACE.
"""

import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from worldfoundry.core.attention.parallel.sequence_parallel_rope import make_sequence_parallel_rope_apply
from worldfoundry.core.nn import sinusoidal_embedding_1d

rope_apply = make_sequence_parallel_rope_apply(
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
)


# ──────────────────────────────────────────────────────────────────────────
# Monkey-patch forwards — VACE hints, then USP shard / xfuser attn / gather
# ──────────────────────────────────────────────────────────────────────────


def usp_dit_forward_vace(self, x, vace_context, seq_len, kwargs):
    """Run VACE hint blocks on the local SP shard of the control tokens."""

    # embeddings
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])

    # arguments
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)

    # Context Parallel
    c = torch.chunk(c, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    hints = []
    for block in self.vace_blocks:
        c, c_skip = block(c, **new_kwargs)
        hints.append(c_skip)
    return hints


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    vace_context=None,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if self.model_type != "vace" and y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    if self.model_type != "vace" and clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
    )

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    if self.model_type == "vace":
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs["hints"] = hints
        kwargs["context_scale"] = vace_context_scale

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    """Self-attention with SP RoPE and xFuser long-context attention.

    ``seq_lens`` is accepted for call-site compatibility; the current
    path still attends the padded shard (see TODOs). Half-cast happens
    only for the xfuser kernel so the output projection stays in the
    module dtype.
    """

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        """Cast to the kernel dtype only when the tensor is not already half."""

        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        """Project, RMSNorm Q/K, and view as ``[B, S, H, D]``."""

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(None, query=half(q), key=half(k), value=half(v), window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
