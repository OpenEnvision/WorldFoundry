# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan self-attn: sequence-parallel RoPE plus Ulysses (non-causal default).

Thin adapter that binds
:func:`~worldfoundry.core.attention.parallel.sequence_parallel_rope.make_sequence_parallel_attention_forward`
to :func:`~worldfoundry.core.attention.parallel.ulysses_attention.distributed_attention`.

Not this module:
    Causal video should import :mod:`.causal_rope_sequence_parallel`.
    Memory-token padded Ulysses lives in :mod:`.padded_ulysses_attention`.
    This file does not implement RoPE or the all-to-all itself.

Public surface:

- :func:`rope_apply` — SP-sliced Wan 3D RoPE.
- :func:`sp_dit_forward` / :data:`sp_attn_forward` — monkey-patch
  targets for non-causal Wan DiT blocks.
"""

import torch

from worldfoundry.core.attention.parallel.sequence_parallel_rope import (
    make_sequence_parallel_attention_forward,
    make_sequence_parallel_rope_apply,
)
from worldfoundry.core.attention.parallel.ulysses_attention import distributed_attention
from worldfoundry.core.distributed.sequence_parallel.ops import gather_forward, get_rank, get_world_size
from worldfoundry.core.nn import sinusoidal_embedding_1d

# ──────────────────────────────────────────────────────────────────────────
# Bound factories — same callables Wan blocks already import by name
# ──────────────────────────────────────────────────────────────────────────

rope_apply = make_sequence_parallel_rope_apply(get_world_size, get_rank)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


sp_attn_forward = make_sequence_parallel_attention_forward(
    distributed_attention,
    rope_apply,
    get_world_size,
    get_rank,
)
