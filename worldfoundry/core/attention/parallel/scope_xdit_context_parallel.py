"""xFuser-backed long-context attention (USP / yunchang kernels).

This is the *optional* path: it imports xfuser and yunchang. Use it only
when those packages are installed and ``initialize_usp`` has built the SP
group. The in-tree fallback is
:mod:`worldfoundry.core.attention.parallel.patch_xdit_context_parallel`.
NPU ranks pad RoPE on CPU because some NPU kernels reject host-side pad.

Not this module:
    Does not own the SP process group (see
    :mod:`worldfoundry.core.distributed.sequence_parallel.xfuser_parallel`). Does not
    implement Ulysses all-to-all in-tree — that is yunchang /
    xFuserLongContextAttention.

Public surface:

- :func:`pad_freqs` / :func:`rope_apply` — SP-sliced complex RoPE.
- :func:`usp_dit_forward` / :func:`usp_attn_forward` — monkey-patch
  targets for xfuser-enabled Wan DiT blocks.
"""

from typing import Optional

import torch
from einops import rearrange
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType

from worldfoundry.core.distributed.sequence_parallel.xfuser_parallel import initialize_usp  # noqa: F401 - public compatibility export
from worldfoundry.core.execution.device import is_npu_available
from worldfoundry.core.nn import sinusoidal_embedding_1d
from worldfoundry.core.nn.checkpointing.gradient import gradient_checkpoint_forward

# ──────────────────────────────────────────────────────────────────────────
# RoPE pad/apply — NPU host pad, then slice the rank's global offsets
# ──────────────────────────────────────────────────────────────────────────


def pad_freqs(original_tensor, target_len):
    """Pad a freq table with ones (identity rotation) to the SP global length.

    NPU tensors are copied to CPU for the cat because some NPU kernels
    reject a host-side ``torch.ones`` pad. The result is moved back so
    the subsequent complex multiply stays on-device.
    """

    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    original_tensor_device = original_tensor.device
    if original_tensor.device == "npu":
        original_tensor = original_tensor.cpu()
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0).to(device=original_tensor_device)
    return padded_tensor


def rope_apply(x, freqs, num_heads):
    """Rotate a local shard with the matching slice of a padded global table."""

    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    s_per_rank = x.shape[1]

    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    freqs = pad_freqs(freqs, s_per_rank * sp_size)
    freqs_rank = freqs[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]
    freqs_rank = freqs_rank.to(torch.complex64) if freqs_rank.device.type == "npu" else freqs_rank
    x_out = torch.view_as_real(x_out * freqs_rank).flatten(2)
    return x_out.to(x.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Monkey-patch forwards — xfuser SP chunk, yunchang attention, all-gather
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
    """Wan DiT forward that shards tokens via xfuser SP and gathers after the head.

    Last-rank padding matches the first chunk so ``all_gather`` stays
    rectangular; the pad is cropped before unpatchify. Training uses
    :func:`gradient_checkpoint_forward` so offload policy stays in one
    place rather than inlining ``save_on_cpu`` here.
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

    # Context Parallel
    chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
    pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
    chunks = [
        torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0) for chunk in chunks
    ]
    x = chunks[get_sequence_parallel_rank()]

    for block in self.blocks:
        if self.training:
            x = gradient_checkpoint_forward(
                block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = self.head(x, t)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :-pad_shape] if pad_shape > 0 else x

    # unpatchify
    x = self.unpatchify(x, (f, h, w))
    return x


def usp_attn_forward(self, x, freqs):
    """Self-attention via xFuser long-context kernels (FA on CUDA, NPU variant)."""

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    attn_type = AttnType.FA
    ring_impl_type = "basic"
    if is_npu_available():
        attn_type = AttnType.NPU
        ring_impl_type = "basic_npu"
    x = xFuserLongContextAttention(attn_type=attn_type, ring_impl_type=ring_impl_type)(
        None,
        query=q,
        key=k,
        value=v,
    )
    x = x.flatten(2)

    del q, k, v
    return self.o(x)
