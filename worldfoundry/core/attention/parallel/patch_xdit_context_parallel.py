"""In-tree xDiT-style context parallel without requiring the xFuser package.

xFuser's SP ranks/world-size are reimplemented with torch.distributed so
Wan checkpoints can run Ulysses on a stock wheel. RoPE frequencies are
padded to the sharded length; attention goes through
:func:`~worldfoundry.core.attention.parallel.ulysses_attention.distributed_attention`.

When the real xFuser stack is installed, prefer
:mod:`worldfoundry.core.attention.parallel.scope_xdit_context_parallel`.

Not this module:
    Does not import xfuser / yunchang. Does not own process-group
    teardown. Video VACE hooks live in
    :mod:`.video_xdit_context_parallel`.

Public surface:

- :func:`get_sequence_parallel_world_size` / :func:`get_sequence_parallel_rank`
  — torch.distributed stand-ins for xFuser SP queries.
- :func:`sequence_parallel_all_gather` / :func:`initialize_usp` — gather
  and env:// init for a full-world Ulysses group.
- :func:`rope_apply` / :func:`usp_dit_forward` / :func:`usp_attn_forward`
  — monkey-patch targets for Wan DiT blocks.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from einops import rearrange

from worldfoundry.core.attention.parallel.sequence_parallel_rope import pad_freqs
from worldfoundry.core.attention.parallel.ulysses_attention import distributed_attention
from worldfoundry.core.distributed.sequence_parallel.ops import gather_forward
from worldfoundry.core.nn import sinusoidal_embedding_1d


# ──────────────────────────────────────────────────────────────────────────
# xFuser-free SP queries — same names so monkey-patches stay drop-in
# ──────────────────────────────────────────────────────────────────────────


def get_sequence_parallel_world_size() -> int:
    """Return the native Ulysses world size without requiring xFuser."""

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_sequence_parallel_rank() -> int:
    """Return the native Ulysses rank without requiring xFuser."""

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def sequence_parallel_all_gather(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Gather equal native Ulysses shards along ``dim``."""

    return gather_forward(value, dim) if get_sequence_parallel_world_size() > 1 else value


def initialize_usp(device_type: str = "cuda") -> None:
    """Initialize the torch-native full-world Ulysses process group."""

    if device_type == "cuda":
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def rope_apply(x, freqs, num_heads):
    """Rotate a local shard with the matching slice of a global freq table.

    Identity-pad first so short tables do not wrap or broadcast the last
    row onto later ranks. Complex multiply stays in float64 so Wan
    checkpoints remain bit-compatible with the single-GPU path.
    """

    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    s_per_rank = x.shape[1]

    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    freqs = pad_freqs(freqs, s_per_rank * sp_size)
    freqs_rank = freqs[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]

    x_out = torch.view_as_real(x_out * freqs_rank).flatten(2)
    return x_out.to(x.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Monkey-patch forwards — shard tokens, run local blocks, gather + unpad
# ──────────────────────────────────────────────────────────────────────────


def usp_dit_forward(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """Wan DiT forward that chunks the token axis instead of calling xFuser.

    Uneven last shards are zero-padded to the first chunk length so
    all-gather sees a rectangular tensor; the pad is cropped after the
    gather. Gradient-checkpoint closures exist only to keep
    ``use_reentrant=False`` callable on the original ``block``.
    """

    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    context = self.text_embedding(context)

    if self.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = self.patchify(x)

    freqs = (
        torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    def create_custom_forward(module):
        """Bind ``module`` so checkpoint sees a 1:1 ``*inputs`` callable."""

        def custom_forward(*inputs):
            """Forward ``module`` without capturing extra closure state."""

            return module(*inputs)

        return custom_forward

    # Context Parallel
    chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
    pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
    chunks = [
        torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0) for chunk in chunks
    ]
    x = chunks[get_sequence_parallel_rank()]

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
        else:
            x = block(x, context, t_mod, freqs)

    x = self.head(x, t)

    # Context Parallel
    x = sequence_parallel_all_gather(x, dim=1)
    x = x[:, :-pad_shape] if pad_shape > 0 else x

    # unpatchify
    x = self.unpatchify(x, (f, h, w))
    return x


def usp_attn_forward(self, x, freqs):
    """Self-attention block: local RoPE, then in-tree Ulysses attention.

    ``seq_lens=None`` because this path already padded tokens to a
    common shard length; a second length vector would re-introduce the
    host sync the pad was meant to avoid.
    """

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    x = distributed_attention(q, k, v, seq_lens=None)
    x = x.flatten(2)

    del q, k, v
    return self.o(x)
