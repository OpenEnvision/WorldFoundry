"""Wan + xFuser long-context attention adapter.

Binds xfuser SP ranks to Wan block forwards (AMP + rearrange). Requires
the xfuser package. The in-tree substitute is
:mod:`worldfoundry.core.attention.parallel.patch_xdit_context_parallel`.

Public surface: :func:`usp_attn_forward`, :func:`usp_attn_s2v_forward`,
:func:`sp_prope_forward`, and the RoPE helpers.
"""

import torch
import torch.cuda.amp as amp
from einops import rearrange

from worldfoundry.core.distributed.sequence_parallel.runtime import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
    xFuserLongContextAttention,
)

# ──────────────────────────────────────────────────────────────────────────
# RoPE + USP attention — freqs are sliced per SP rank after padding to full S
# ──────────────────────────────────────────────────────────────────────────


def pad_freqs(original_tensor, target_len):
    """Pad frequency rows with ones so a short RoPE table covers the full SP sequence.

    Ones are the complex multiplicative identity, so padded tokens keep their
    values instead of being rotated by uninitialized memory.
    """

    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
        dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output)

def rope_apply_qk(q, k, grid_sizes, freqs):
    """Apply :func:`rope_apply` to Q and K with the same grid / freqs."""

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k

def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16,
                     t=0):
    """Wan self-attention forward that routes Q/K/V through xFuser long-context.

    Bound onto a block as a method. RoPE is applied on the local SP shard
    after freqs are padded to the full sequence. Unpadded attention is still
    a TODO — padded tokens currently participate in the kernel.
    """

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        """Cast to ``dtype`` unless already float16/bfloat16 (AMP path)."""

        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        """Project and reshape Q/K/V to ``[B, S, H, D]`` on the local shard."""

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = rope_apply_qk(q, k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x

@amp.autocast(enabled=False)
@torch.compiler.disable()
def s2v_rope_apply(x, grid_sizes, freqs):
    """Speech-to-video RoPE: per-sample freqs, float64 complex multiply, then float32.

    Unlike :func:`rope_apply` this does not slice freqs by SP rank — the
    caller already packed a per-sample table that matches the local shard.
    """

    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i]
        freqs_i_rank = pad_freqs(freqs_i, s)
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

def s2v_rope_apply_qk(q, k, grid_sizes, freqs):
    """Apply :func:`s2v_rope_apply` to Q and K with the same per-sample freqs."""

    q = s2v_rope_apply(q, grid_sizes, freqs)
    k = s2v_rope_apply(k, grid_sizes, freqs)
    return q, k

def usp_attn_s2v_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16,
                     t=0):
    """S2V self-attention forward through xFuser, using per-sample RoPE tables."""

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        """Cast to ``dtype`` unless already float16/bfloat16 (AMP path)."""

        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        """Project and reshape Q/K/V to ``[B, S, H, D]`` on the local shard."""

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = s2v_rope_apply_qk(q, k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


# ──────────────────────────────────────────────────────────────────────────
# PRoPE + camera attention — head-parallel when N % sp == 0, else all-gather
# ──────────────────────────────────────────────────────────────────────────


def _all_to_all_4d(tensor, scatter_dim, gather_dim, sp_group_obj):
    """
    All-to-all for 4D tensors [B, S, N, D] using all_to_all_single.
    Mirrors UlysseseAlltoAll layout semantics for consistent behavior.
    Scatters along scatter_dim and gathers along gather_dim.
    """
    import torch.distributed as dist

    sp_size = get_sequence_parallel_world_size()
    assert tensor.shape[scatter_dim] % sp_size == 0, (
        f"scatter_dim {scatter_dim} size {tensor.shape[scatter_dim]} not divisible by sp_size {sp_size}"
    )

    # Extract underlying ProcessGroup from GroupCoordinator
    pg = getattr(sp_group_obj, 'device_group', sp_group_obj)

    # Step 1: permute scatter_dim to front.
    dims = list(range(tensor.dim()))
    dims.remove(scatter_dim)
    dims.insert(0, scatter_dim)
    tensor = tensor.permute(dims).contiguous()

    # Step 2: all_to_all_single on dim=0
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=pg)

    # Step 3: view and permute to rearrange.
    new_shape = (sp_size, -1) + output.shape[1:]
    output = output.view(new_shape)

    if scatter_dim == 2 and gather_dim == 1:
        # scatter heads, gather seq: (sp, N/sp, B, S_chunk, D) → (B, sp*S_chunk, N/sp, D)
        output = output.permute(2, 0, 3, 1, 4).contiguous()
        final_shape = (output.shape[0], -1, output.shape[3], output.shape[4])
    elif scatter_dim == 1 and gather_dim == 2:
        # scatter seq, gather heads: (sp, S_chunk, B, N/sp, D) → (B, S_chunk, sp*N/sp, D)
        output = output.permute(2, 1, 0, 3, 4).contiguous()
        final_shape = (output.shape[0], output.shape[1], -1, output.shape[4])
    else:
        raise ValueError(f"Unsupported scatter/gather combo: scatter_dim={scatter_dim}, gather_dim={gather_dim}")

    return output.view(final_shape)


def sp_prope_forward(self, x, cam_emb, seq_lens=None, sp_group=None):
    """
    Sequence-parallel inference version of PropeSelfAttention.forward.

    Automatically selects the best strategy based on num_heads vs sp_size:
    - If num_heads % sp_size == 0: head-parallel (Ulysses all-to-all), each GPU handles
      N/sp_size heads over the full sequence. More efficient.
    - Otherwise: all_gather full sequence, local attention with all heads, chunk back.
      Less communication-efficient but always works.

    Args:
        self: PropeSelfAttention instance
        x (Tensor): Shape [B, S_chunk, D] — chunked input tokens
        cam_emb (dict): Camera embedding dict with keys 'viewmats' [B, T, 4, 4] and 'K' [B, T, 3, 3]
        seq_lens (Tensor, optional): Shape [B], actual token count per sample (full, not chunked)
        sp_group: Not used in inference (None), kept for API compatibility
    Returns:
        Tensor: Shape [B, S_chunk, D]
    """
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.camera_attention import attention
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import prope_qkv

    batch_size, chunk_seq_len, dim = x.shape
    num_heads = self.num_heads
    head_dim = self.head_dim
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    sp_group_obj = get_sp_group()

    # q/k/v projections on chunked input: [B, S_chunk, N, D_head]
    q = self.norm_q(self.q_proj(x)).view(batch_size, chunk_seq_len, num_heads, head_dim)
    k = self.norm_k(self.k_proj(x)).view(batch_size, chunk_seq_len, num_heads, head_dim)
    v = self.v_proj(x).view(batch_size, chunk_seq_len, num_heads, head_dim)

    use_head_parallel = (num_heads % sp_size == 0)

    if use_head_parallel:
        # Head-parallel: all-to-all scatter heads, gather seq
        # [B, S_chunk, N, D] → [B, S_full, N/sp, D]
        q = _all_to_all_4d(q, scatter_dim=2, gather_dim=1, sp_group_obj=sp_group_obj)
        k = _all_to_all_4d(k, scatter_dim=2, gather_dim=1, sp_group_obj=sp_group_obj)
        v = _all_to_all_4d(v, scatter_dim=2, gather_dim=1, sp_group_obj=sp_group_obj)
    else:
        # Fallback: all_gather full sequence, keep all heads
        # [B, S_chunk, N, D] → [B, S_full, N, D]
        q = sp_group_obj.all_gather(q, dim=1)
        k = sp_group_obj.all_gather(k, dim=1)
        v = sp_group_obj.all_gather(v, dim=1)

    # Apply PRoPE positional encoding: prope_qkv expects [B, N, S, D_head]
    q, k, v, apply_fn_o = prope_qkv(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        viewmats=cam_emb['viewmats'],
        Ks=cam_emb['K'],
    )

    # Local attention: attention expects [B, S, N, D_head]
    out = attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v=v.transpose(1, 2),
    )

    # Apply inverse PRoPE transform
    out = apply_fn_o(out.transpose(1, 2)).transpose(1, 2)

    if use_head_parallel:
        # All-to-all back: scatter seq, gather heads
        # [B, S_full, N/sp, D] → [B, S_chunk, N, D]
        out = _all_to_all_4d(out, scatter_dim=1, gather_dim=2, sp_group_obj=sp_group_obj)
    else:
        # Chunk back to this rank's portion
        # [B, S_full, N, D] → [B, S_chunk, N, D]
        out = out.chunk(sp_size, dim=1)[sp_rank]

    # Project back to dim: [B, S_chunk, D]
    out = out.flatten(2)
    out = self.out_proj(out)

    return out
