"""Shared RoPE and attention factories for sequence-parallel video inference.

Ulysses / USP ranks own a contiguous token shard. RoPE frequency
tables are usually built for the *global* length; applying them
naively on a short shard either truncates or broadcasts wrong
positions. :func:`pad_freqs` and
:func:`make_sequence_parallel_rope_apply` slice/pad the table so each
rank rotates with its global offsets.

Not this module:
    Causal vs non-causal attention is *not* chosen here — callers pick
    :mod:`.causal_rope_sequence_parallel` or
    :mod:`.rope_sequence_parallel` (and the padded-Ulysses variant for
    memory tokens). Does not own the SP process group.

Public surface:

- :func:`pad_freqs` / :func:`apply_sequence_parallel_rope` — identity-pad
  and per-sample 3D apply on a shard.
- :func:`make_sequence_parallel_rope_apply` /
  :func:`make_sequence_parallel_attention_forward` — bind rank providers.
- :func:`sequence_parallel_attention_forward` — Wan QKV / RoPE / out.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_3d


# ──────────────────────────────────────────────────────────────────────────
# Frequency pad / apply — global offsets on a contiguous token shard
# ──────────────────────────────────────────────────────────────────────────


def pad_freqs(original_tensor: Tensor, target_len: int) -> Tensor:
    """Pad a ``[sequence, *, *]`` RoPE table with identity rotations."""

    seq_len, dim1, dim2 = original_tensor.shape
    padding = original_tensor.new_ones((target_len - seq_len, dim1, dim2))
    return torch.cat((original_tensor, padding), dim=0)


@torch.amp.autocast("cuda", enabled=False)
def apply_sequence_parallel_rope(
    x: Tensor,
    grid_sizes: Tensor,
    freqs: Tensor,
    *,
    world_size: int,
    rank: int,
    compute_dtype: torch.dtype = torch.float64,
    match_frequency_dtype: bool = False,
) -> Tensor:
    """Apply Wan-style 3D RoPE to a sequence-parallel tensor shard."""

    sequence_len, num_heads, complex_dim = x.size(1), x.size(2), x.size(3) // 2
    if not freqs.is_complex():
        freqs = torch.view_as_complex(freqs.contiguous())
    freq_parts = freqs.split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )

    output = []
    for index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        full_sequence_len = frames * height * width
        value = torch.view_as_complex(x[index, :sequence_len].to(compute_dtype).reshape(sequence_len, num_heads, -1, 2))
        sample_freqs = torch.cat(
            (
                freq_parts[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freq_parts[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freq_parts[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ),
            dim=-1,
        ).reshape(full_sequence_len, 1, -1)
        sample_freqs = pad_freqs(sample_freqs, sequence_len * world_size)
        local_freqs = sample_freqs[
            rank * sequence_len : (rank + 1) * sequence_len,
            :,
            :,
        ]
        if match_frequency_dtype:
            local_freqs = local_freqs.to(value.dtype)
        value = torch.view_as_real(value * local_freqs).flatten(2)
        output.append(torch.cat((value, x[index, sequence_len:])))

    return torch.stack(output).float()


def make_sequence_parallel_rope_apply(
    get_world_size: Any,
    get_rank: Any,
    **apply_kwargs: Any,
) -> Any:
    """Bind runtime rank providers to the shared sequence-parallel RoPE path."""

    def rope_apply(x: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
        """Apply SP RoPE using the bound world-size / rank providers."""

        return apply_sequence_parallel_rope(
            x,
            grid_sizes,
            freqs,
            world_size=get_world_size(),
            rank=get_rank(),
            **apply_kwargs,
        )

    return rope_apply


# ──────────────────────────────────────────────────────────────────────────
# Wan attention forward — fused QK RMSNorm+RoPE when batch==1 and qk_norm
# ──────────────────────────────────────────────────────────────────────────


def sequence_parallel_attention_forward(
    module: Any,
    x: Tensor,
    seq_lens: Tensor,
    grid_sizes: Tensor,
    freqs: Tensor,
    *,
    distributed_attention: Any,
    rope_apply: Any,
    world_size: int = 1,
    rank: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run the shared Wan QKV/RoPE/output path with a selected attention backend."""

    batch, sequence_len = x.shape[:2]
    num_heads, head_dim = module.num_heads, module.head_dim
    q = module.q(x)
    k = module.k(x)
    v = module.v(x).view(batch, sequence_len, num_heads, head_dim)
    rope_is_fused = bool(getattr(module, "qk_norm", False)) and batch == 1
    if rope_is_fused:
        grid_size = tuple(int(value) for value in grid_sizes[0].tolist())
        q, k = hidden_qk_rmsnorm_rope_3d(
            q,
            k,
            module.norm_q.weight,
            module.norm_k.weight,
            freqs,
            num_heads=num_heads,
            grid_size=grid_size,
            eps=module.eps,
            sequence_offset=int(rank) * sequence_len,
            valid_tokens=grid_size[0] * grid_size[1] * grid_size[2],
        )
        q = q.view(batch, sequence_len, num_heads, head_dim)
        k = k.view(batch, sequence_len, num_heads, head_dim)
    else:
        q = module.norm_q(q).view(batch, sequence_len, num_heads, head_dim)
        k = module.norm_k(k).view(batch, sequence_len, num_heads, head_dim)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

    half_dtypes = (torch.float16, torch.bfloat16)
    q = q if q.dtype in half_dtypes else q.to(dtype)
    k = k if k.dtype in half_dtypes else k.to(dtype)
    v = v if v.dtype in half_dtypes else v.to(dtype)
    output = distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=module.window_size,
    )
    return module.o(output.flatten(2))


def make_sequence_parallel_attention_forward(
    distributed_attention: Any,
    rope_apply: Any,
    get_world_size: Any | None = None,
    get_rank: Any | None = None,
) -> Any:
    """Bind attention and RoPE backends to the shared Wan attention forward."""

    def attention_forward(
        module: Any,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """Wan-block ``forward`` bound to one attention + RoPE pair."""

        return sequence_parallel_attention_forward(
            module,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            distributed_attention=distributed_attention,
            rope_apply=rope_apply,
            world_size=1 if get_world_size is None else get_world_size(),
            rank=0 if get_rank is None else get_rank(),
            dtype=dtype,
        )

    return attention_forward


__all__ = [
    "apply_sequence_parallel_rope",
    "make_sequence_parallel_attention_forward",
    "make_sequence_parallel_rope_apply",
    "pad_freqs",
    "sequence_parallel_attention_forward",
]
