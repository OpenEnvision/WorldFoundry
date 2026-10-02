"""Causal RoPE + Ulysses for sequence-parallel video generation.

Applies 3D RoPE on the local SP shard, then causal Ulysses attention.
Frequencies are padded to the shard length so ranks with unequal tail
tokens still broadcast. This is the LingBot causal path; non-causal Wan
self-attn uses :mod:`worldfoundry.core.attention.parallel.rope_sequence_parallel`.
"""

import math

import torch
import torch.nn.functional as torch_F
from einops import rearrange

from worldfoundry.core.attention.parallel.causal_ulysses_attention import distributed_attention
from worldfoundry.core.attention.parallel.sequence_parallel_rope import (
    make_sequence_parallel_attention_forward,
    make_sequence_parallel_rope_apply,
)
from worldfoundry.core.attention.sequence.varlen import flash_attention
from worldfoundry.core.distributed.sequence_parallel.ops import (
    all_to_all,
    all_to_all_many,
    gather_forward,
    get_rank,
    get_world_size,
)
from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_3d
from worldfoundry.core.nn.transformer.shape import sinusoidal_embedding_1d

# The chunked causal path is adapted from Robbyant/lingbot-world-v2 commit
# 94f43115 under CC BY-NC-SA 4.0; see the integration's upstream license file.


rope_apply = make_sequence_parallel_rope_apply(get_world_size, get_rank)

# ──────────────────────────────────────────────────────────────────────────
# 3D causal RoPE — rotate a shard starting at an arbitrary frame
# ──────────────────────────────────────────────────────────────────────────


@torch.amp.autocast("cuda", enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply 3D complex RoPE to a local shard, starting at ``start_frame``.

    Frequencies are split into time/height/width and expanded to the
    sample's ``(F, H, W)`` grid. Tokens past ``seq_len`` are left
    unrotated so padded tails stay identity. Autocast is disabled so
    the complex multiply stays in float64.
    """
    n, c = x.size(2), x.size(3) // 2
    if not freqs.is_complex():
        freqs = torch.view_as_complex(freqs.contiguous())

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ──────────────────────────────────────────────────────────────────────────
# Non-causal SP DiT — sequence-sharded tokens (Wan self-attn path)
# ──────────────────────────────────────────────────────────────────────────


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
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

    # cam
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i,
                "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            )
            for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
        )
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len < seq_len:
            pad_len = seq_len - cam_len
            pad = c2ws_plucker_emb.new_zeros(c2ws_plucker_emb.size(0), pad_len, c2ws_plucker_emb.size(2))
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
        elif cam_len > seq_len:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_len, :]

        if get_world_size() > 1:
            c2ws_plucker_emb = torch.chunk(c2ws_plucker_emb, get_world_size(), dim=1)[get_rank()]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict,
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


# ──────────────────────────────────────────────────────────────────────────
# Causal SP DiT — every rank holds the full sequence; heads are split
# ──────────────────────────────────────────────────────────────────────────


def sp_dit_forward_causal(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    cross_attn_first_call=None,
):
    """
    x:                  A list of videos each with shape [C, T, H, W].
    t:                  [B].
    context:            A list of text embeddings each with shape [L, C].
    seq_len:            Maximum sequence length for positional encoding.
    y:                  Conditional video inputs for image-to-video mode, same shape as x.
    dit_cond_dict:      Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                        with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
    kv_cache:           Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, kv_size, local_heads, head_dim]), ``global_end_index``,
                        and ``local_end_index`` (scalar Tensors tracking cache position).
    crossattn_cache:    Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
    current_start:      Token offset of the current chunk in the full sequence. Used to index
                        into the KV cache and compute positional embeddings correctly.
    max_attention_size: Maximum number of KV tokens each query can attend to. Limits the
                        effective context window of self-attention to control memory usage.

    This follows the official LingBot fast runtime: every rank processes the
    full token sequence and only attention heads are split across ranks.
    """

    assert len(x) == 1

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
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)[0]
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_lens)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_lens)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    # cam
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i,
                "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            )
            for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
        )
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len > seq_lens:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_lens, :]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict,
        max_attention_size=max_attention_size,
        frame_seqlen=frame_seqlen,
        cross_attn_first_call=cross_attn_first_call,
    )

    for block_index, block in enumerate(self.blocks):
        kwargs.update(
            {
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
            }
        )
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return [u.float() for u in x]


def sp_attn_forward_causal(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    kv_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    seq_lens_int=None,
):
    r"""
    Sequence-parallel causal self-attention using Ulysses head splitting.

    Args:
        x(Tensor):              Shape [B, L, C]
        seq_lens(Tensor):       Number of valid tokens per sample.
        grid_sizes(Tensor):     Shape [B, 3], the second dimension contains (F, H, W).
        freqs(Tensor):          Rope freqs, shape [1024, C / num_heads / 2].
        kv_cache(dict):         Self-attention KV cache. Contains keys ``k``, ``v``
                                (Tensor of shape [B, kv_size, local_heads, head_dim]),
                                ``global_end_index``, and ``local_end_index``
                                (scalar Tensors tracking cache position).
        current_start(int):     Token offset of the current chunk in the full sequence.
                                Used to index into the KV cache and compute positional
                                embeddings correctly.
        max_attention_size(int): Maximum number of KV tokens each query can attend to.
                                Limits the effective context window of self-attention
                                to control memory usage.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    sp_size = get_world_size()
    rank = get_rank()
    local_heads = n // sp_size

    del seq_lens_int
    if frame_seqlen is None:
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_start_frame = current_start // frame_seqlen
    q = self.q(x)
    k = self.k(x)
    v = self.v(x).view(b, s, n, d)
    rope_is_fused = self.qk_norm and b == 1
    if rope_is_fused:
        grid_size = tuple(int(value) for value in grid_sizes[0].tolist())
        q, k = hidden_qk_rmsnorm_rope_3d(
            q,
            k,
            self.norm_q.weight,
            self.norm_k.weight,
            freqs,
            num_heads=n,
            grid_size=grid_size,
            eps=self.eps,
            start_frame=current_start_frame,
            valid_tokens=math.prod(grid_size),
            head_start=rank * local_heads,
            head_end=(rank + 1) * local_heads,
        )
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
    else:
        q = self.norm_q(q).view(b, s, n, d)
        k = self.norm_k(k).view(b, s, n, d)

    q_local = q[:, :, rank * local_heads : (rank + 1) * local_heads, :]
    k_local = k[:, :, rank * local_heads : (rank + 1) * local_heads, :]
    v_local = v[:, :, rank * local_heads : (rank + 1) * local_heads, :]
    if rope_is_fused:
        query = q_local.type_as(v_local)
        key = k_local.type_as(v_local)
    else:
        query = causal_rope_apply(q_local, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_local)
        key = causal_rope_apply(k_local, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_local)

    current_end = current_start + seq_lens
    kv_cache_size = kv_cache["k"].shape[1]
    if (
        self.local_attn_size != -1
        and (current_end > kv_cache["global_end_index"].item())
        and (seq_lens + kv_cache["local_end_index"].item() > kv_cache_size)
    ):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        sink_tokens = self.sink_size * frame_seqlen
        num_evicted_tokens = seq_lens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
        ].clone()
        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
        ].clone()
        # Insert the new keys/values at the end
        local_end_index = (
            kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
        )
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v_local
    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v_local

    k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
    v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]

    # Attention on local heads, full key/value cache for this rank
    x_local = flash_attention(query, k_cache, v_cache)  # [B, s, local_heads, d]

    # Gather all head results across GPUs: [B, s, n, d]
    x_full = gather_forward(x_local, dim=2)

    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # output
    x_full = x_full.flatten(2)
    x_full = self.o(x_full)
    return x_full


# ──────────────────────────────────────────────────────────────────────────
# Chunked causal SP — sequence shards + all-to-all (LingBot-World-V2)
# ──────────────────────────────────────────────────────────────────────────


def sp_dit_forward_causal_chunked(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    cross_attn_first_call=None,
):
    """Run causal DiT with sequence-chunked Ulysses parallelism.

    Unlike :func:`sp_dit_forward_causal`, each rank owns a padded sequence
    shard. Attention temporarily redistributes it to head shards via
    all-to-all. This is the layout used by LingBot-World-V2 causal-fast.
    """
    if len(x) != 1:
        raise ValueError("Chunked causal sequence parallelism requires batch size 1.")
    if self.model_type == "i2v" and y is None:
        raise ValueError("Image-to-video models require conditional latents in 'y'.")

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    if y is not None:
        x = [torch.cat([sample, condition], dim=0) for sample, condition in zip(x, y)]

    x = [self.patch_embedding(sample.unsqueeze(0)) for sample in x]
    grid_sizes = torch.stack([torch.tensor(sample.shape[2:], dtype=torch.long) for sample in x])
    x = [sample.flatten(2).transpose(1, 2) for sample in x]
    seq_lens = torch.tensor([sample.size(1) for sample in x], dtype=torch.long)[0]
    if seq_lens.max() > seq_len:
        raise ValueError(f"Token sequence {int(seq_lens)} exceeds configured maximum {seq_len}.")
    x = torch.cat(x)

    sp_size = get_world_size()
    seq_lens_int = int(seq_lens)
    padded_seq_lens = math.ceil(seq_lens_int / sp_size) * sp_size
    if pad_len := padded_seq_lens - seq_lens_int:
        x = torch.cat([x, x.new_zeros(x.size(0), pad_len, x.size(2))], dim=1)

    if t.dim() == 1:
        t = t.expand(t.size(0), padded_seq_lens)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        batch_size = t.size(0)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, (batch_size, padded_seq_lens)).float()
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([item, item.new_zeros(self.text_len - item.size(0), item.size(1))]) for item in context])
    )

    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        camera = [
            rearrange(
                item,
                "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            )
            for item in dit_cond_dict["c2ws_plucker_emb"]
        ]
        camera = self.patch_embedding_wancamctrl(torch.cat(camera, dim=1))
        camera = camera + self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(camera)))
        if camera.size(1) < padded_seq_lens:
            camera = torch.cat(
                [camera, camera.new_zeros(camera.size(0), padded_seq_lens - camera.size(1), camera.size(2))],
                dim=1,
            )
        else:
            camera = camera[:, :padded_seq_lens]
        if sp_size > 1:
            camera = torch.chunk(camera, sp_size, dim=1)[get_rank()]
        dit_cond_dict = {**dit_cond_dict, "c2ws_plucker_emb": camera}

    rank = get_rank()
    x = torch.chunk(x, sp_size, dim=1)[rank]
    e = torch.chunk(e, sp_size, dim=1)[rank]
    e0 = torch.chunk(e0, sp_size, dim=1)[rank]
    block_kwargs = {
        "e": e0,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": self.freqs,
        "context": context,
        "context_lens": context_lens,
        "dit_cond_dict": dit_cond_dict,
        "max_attention_size": max_attention_size,
        "frame_seqlen": frame_seqlen,
        "cross_attn_first_call": cross_attn_first_call,
        "seq_lens_int": seq_lens_int,
    }
    for block_index, block in enumerate(self.blocks):
        x = block(
            x,
            kv_cache=kv_cache[block_index],
            crossattn_cache=crossattn_cache[block_index],
            current_start=current_start,
            **block_kwargs,
        )

    x = self.head(x, e)
    x = gather_forward(x, dim=1)
    return [sample.float() for sample in self.unpatchify(x, grid_sizes)]


def sp_attn_forward_causal_chunked(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    kv_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    frame_seqlen=None,
    seq_lens_int=None,
):
    """Run causal self-attention for a sequence-chunked Ulysses layout."""
    batch_size, local_seq_len, num_heads, head_dim = (
        *x.shape[:2],
        self.num_heads,
        self.head_dim,
    )

    seq_lens_int = int(seq_lens) if seq_lens_int is None else seq_lens_int
    if frame_seqlen is None:
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    start_frame = current_start // frame_seqlen
    query = self.q(x)
    key = self.k(x)
    value = self.v(x).view(batch_size, local_seq_len, num_heads, head_dim)
    rope_is_fused = self.qk_norm and batch_size == 1
    if rope_is_fused:
        grid_size = tuple(int(item) for item in grid_sizes[0].tolist())
        query, key = hidden_qk_rmsnorm_rope_3d(
            query,
            key,
            self.norm_q.weight,
            self.norm_k.weight,
            freqs,
            num_heads=num_heads,
            grid_size=grid_size,
            eps=self.eps,
            sequence_offset=get_rank() * local_seq_len,
            start_frame=start_frame,
            valid_tokens=seq_lens_int,
        )
        query = query.view(batch_size, local_seq_len, num_heads, head_dim)
        key = key.view(batch_size, local_seq_len, num_heads, head_dim)
    else:
        query = self.norm_q(query).view(batch_size, local_seq_len, num_heads, head_dim)
        key = self.norm_k(key).view(batch_size, local_seq_len, num_heads, head_dim)

    query, key, value = all_to_all_many(
        (query, key, value),
        scatter_dim=2,
        gather_dim=1,
    )

    padded_seq_len = local_seq_len * get_world_size()
    if rope_is_fused:
        query = query.type_as(value)
        key = key.type_as(value)
    else:
        query = causal_rope_apply(query, grid_sizes, freqs, start_frame=start_frame).type_as(value)
        key = causal_rope_apply(key, grid_sizes, freqs, start_frame=start_frame).type_as(value)
    query = query[:, :seq_lens_int]
    key = key[:, :seq_lens_int]
    value = value[:, :seq_lens_int]

    current_end = current_start + seq_lens_int
    cache_end = int(kv_cache["local_end_index"].item())
    global_end = int(kv_cache["global_end_index"].item())
    cache_size = kv_cache["k"].shape[1]
    if self.local_attn_size == -1:
        local_start = current_start
        local_end = current_end
    elif current_end > global_end and seq_lens_int + cache_end > cache_size:
        sink_tokens = self.sink_size * frame_seqlen
        evicted = seq_lens_int + cache_end - cache_size
        retained = cache_end - evicted - sink_tokens
        if retained > 0:
            kv_cache["k"][:, sink_tokens : sink_tokens + retained] = kv_cache["k"][
                :, sink_tokens + evicted : sink_tokens + evicted + retained
            ].clone()
            kv_cache["v"][:, sink_tokens : sink_tokens + retained] = kv_cache["v"][
                :, sink_tokens + evicted : sink_tokens + evicted + retained
            ].clone()
        local_end = cache_end + current_end - global_end - evicted
        local_start = local_end - seq_lens_int
    else:
        local_end = cache_end + current_end - global_end
        local_start = local_end - seq_lens_int

    kv_cache["k"][:, local_start:local_end] = key
    kv_cache["v"][:, local_start:local_end] = value
    cache_start = max(0, local_end - max_attention_size)
    output = flash_attention(
        query,
        kv_cache["k"][:, cache_start:local_end],
        kv_cache["v"][:, cache_start:local_end],
    )
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end)

    if pad_len := padded_seq_len - seq_lens_int:
        output = torch.cat(
            [output, output.new_zeros(batch_size, pad_len, output.size(2), head_dim)],
            dim=1,
        )
    output = all_to_all(output, scatter_dim=1, gather_dim=2)
    return self.o(output.flatten(2))
