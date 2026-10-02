"""MinWM causal Wan 2.1 transformer (chunk KV cache).

Causal official-layout Wan 2.1.  Action and ProPE live in sibling
modules.  Adds causal attention.
"""
from worldfoundry.core.attention.sequence.varlen import attention
from .action import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import torch.distributed as dist

try:
    from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21.causal import CausalWanConfig
    from worldfoundry.core.distributed.sequence_parallel.communication_op import (
        sequence_model_parallel_all_gather,
        sequence_model_parallel_all_to_all_4D,
    )
    from worldfoundry.core.distributed.sequence_parallel.parallel_states import get_parallel_state
    _HAS_CLEANCODE_INFRA = True
except ImportError as _e:
    _HAS_CLEANCODE_INFRA = False
    _CLEANCODE_IMPORT_ERROR = _e


def _require_cleancode_infra(context: str = ""):
    """Hard-fail if SP > 1 but CleanCode SP infra failed to import.

    Called at every SP decision point so that a silent fallback to
    sp_enabled=False can never happen when the user explicitly requested SP.
    """
    if _HAS_CLEANCODE_INFRA:
        return
    # Only crash when distributed training is active (world_size > 1).
    # Single-GPU / non-distributed usage can degrade gracefully.
    sp_size_env = int(os.environ.get("SP_SIZE", "1"))
    if dist.is_initialized() and dist.get_world_size() > 1 and sp_size_env > 1:
        msg = (
            f"[FATAL] CleanCode SP infra import failed but distributed "
            f"training is active (world_size={dist.get_world_size()}).\n"
            f"Import error: {_CLEANCODE_IMPORT_ERROR}\n"
            f"Context: {context}\n"
            f"SP > 1 requires wan_2p1.sp and wan_2p1.configs.causal_wan_config. "
            f"Fix the import or run with SP=1."
        )
        raise RuntimeError(msg)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Causal rope apply.

    Args:
        x: The x.
        grid_sizes: The grid sizes.
        freqs: The freqs.
        start_frame: The start frame.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache and frame-offset RoPE."""

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            local_attn_size: The local attn size.
            sink_size: The sink size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 31200 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            viewmats(Tensor, optional): Shape [B, L, 4, 4] camera extrinsics for PRoPE
            Ks(Tensor, optional): Shape [B, L, 3, 3] camera intrinsics for PRoPE
            prope_kv_cache(dict, optional): PRoPE KV cache for inference, same structure as kv_cache
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # PRoPE: apply BEFORE SP all-to-all so viewmats/Ks and q/k/v have matching lengths
        prope_enabled = (viewmats is not None) and hasattr(self, 'prope_o') and \
                        (kv_cache is None or prope_kv_cache is not None)
        if prope_enabled:
            from .prope import prope_qkv
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            q_p = q_p.permute(0, 2, 1, 3)
            k_p = k_p.permute(0, 2, 1, 3)
            v_p = v_p.permute(0, 2, 1, 3)

        # SP: scatter heads, gather sequence for attention
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanSelfAttention.forward SP check")
            sp_enabled = False

        if sp_enabled:
            q = sequence_model_parallel_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sequence_model_parallel_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sequence_model_parallel_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            if prope_enabled:
                q_p = sequence_model_parallel_all_to_all_4D(q_p, scatter_dim=2, gather_dim=1)
                k_p = sequence_model_parallel_all_to_all_4D(k_p, scatter_dim=2, gather_dim=1)
                v_p = sequence_model_parallel_all_to_all_4D(v_p, scatter_dim=2, gather_dim=1)

        is_tf = False
        sp_world_size = 0
        per_rank_half = 0
        if kv_cache is None:
            # if it is teacher forcing training?
            # Use q.shape[1] (post all-to-all full seq len) for TF detection
            # With SP padding, full_s may be slightly larger than seq_lens[0]*2,
            # so use > 1.5x threshold instead of exact equality.
            full_s = q.shape[1]
            is_tf = (full_s > seq_lens[0].item() * 1.5)
            if is_tf:
                # SP + TF: after all-to-all, sequence is interleaved:
                # [clean_r0, noisy_r0, clean_r1, noisy_r1, ...]
                # Reorder to contiguous [all_clean, all_noisy] for correct chunk(2).
                if sp_enabled:
                    sp_world_size = parallel_dims.sp
                    chunk_size = full_s // sp_world_size  # tokens per SP rank = 2 * (L/sp)
                    per_rank_half = chunk_size // 2

                    def _interleaved_to_contiguous(x):
                        """Helper function to interleaved to contiguous.

                        Args:
                            x: The x.
                        """
                        B, S, H, D = x.shape
                        return x.reshape(B, sp_world_size, 2, per_rank_half, H, D) \
                                .permute(0, 2, 1, 3, 4, 5) \
                                .reshape(B, S, H, D)

                    q = _interleaved_to_contiguous(q)
                    k = _interleaved_to_contiguous(k)
                    v = _interleaved_to_contiguous(v)
                    if prope_enabled:
                        q_p = _interleaved_to_contiguous(q_p)
                        k_p = _interleaved_to_contiguous(k_p)
                        v_p = _interleaved_to_contiguous(v_p)

                    # Strip SP padding before attention.
                    # After contiguous reorder: [clean_valid, clean_pad, noisy_valid, noisy_pad]
                    unpadded_half = seq_lens[0].item()
                    sp_pad_per_half = full_s // 2 - unpadded_half
                    if sp_pad_per_half > 0:
                        # Remove padding from each half
                        def _strip_sp_pad(t):
                            """Helper function to strip sp pad.

                            Args:
                                t: The t.
                            """
                            c = t[:, :unpadded_half]
                            n = t[:, full_s // 2:full_s // 2 + unpadded_half]
                            return torch.cat([c, n], dim=1)
                        q = _strip_sp_pad(q)
                        k = _strip_sp_pad(k)
                        v = _strip_sp_pad(v)
                        if prope_enabled:
                            q_p = _strip_sp_pad(q_p)
                            k_p = _strip_sp_pad(k_p)
                            v_p = _strip_sp_pad(v_p)

                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :q.shape[1]].transpose(2, 1) if padded_length > 0 else x.transpose(2, 1)

                # PRoPE attention path (TF mode)
                if prope_enabled:
                    padded_length_p = padded_length
                    if padded_length_p > 0:
                        pad_zeros_p = torch.zeros([q_p.shape[0], padded_length_p, q_p.shape[2], q_p.shape[3]],
                                                  device=q_p.device, dtype=q_p.dtype)
                        padded_q_p = torch.cat([q_p, pad_zeros_p], dim=1)
                        padded_k_p = torch.cat([k_p, pad_zeros_p], dim=1)
                        padded_v_p = torch.cat([v_p, pad_zeros_p], dim=1)
                    else:
                        padded_q_p, padded_k_p, padded_v_p = q_p, k_p, v_p

                    x_prope = flex_attention(
                        query=padded_q_p.transpose(2, 1),
                        key=padded_k_p.transpose(2, 1),
                        value=padded_v_p.transpose(2, 1),
                        block_mask=block_mask
                    )
                    x_prope = x_prope[:, :, :q_p.shape[1]].transpose(2, 1) if padded_length_p > 0 else x_prope.transpose(2, 1)

                # Restore SP padding after attention so reverse reorder works
                if sp_enabled and sp_pad_per_half > 0:
                    B_x, S_x, H_x, D_x = x.shape
                    half_valid = S_x // 2  # == unpadded_half
                    pad_t = x.new_zeros(B_x, sp_pad_per_half, H_x, D_x)
                    x = torch.cat([
                        x[:, :half_valid], pad_t,
                        x[:, half_valid:], pad_t
                    ], dim=1)
                    if prope_enabled:
                        x_prope = torch.cat([
                            x_prope[:, :half_valid], pad_t,
                            x_prope[:, half_valid:], pad_t
                        ], dim=1)

            else:
                # DF path removed — only Teacher Forcing is supported.
                assert False, "Diffusion Forcing is currently not supported. Only Teacher Forcing is supported."
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

            # PRoPE second attention path (inference mode with KV cache)
            if prope_enabled and prope_kv_cache is not None:
                # q_p, k_p, v_p were already computed above by prope_qkv
                # Store PRoPE-transformed k_p, v_p into prope_kv_cache
                # Use the same eviction logic as the RoPE cache
                prope_cache_size = prope_kv_cache["k"].shape[1]
                if self.local_attn_size != -1 and (current_end > prope_kv_cache["global_end_index"].item()) and (
                        num_new_tokens + prope_kv_cache["local_end_index"].item() > prope_cache_size):
                    p_num_evicted = num_new_tokens + prope_kv_cache["local_end_index"].item() - prope_cache_size
                    p_num_rolled = prope_kv_cache["local_end_index"].item() - p_num_evicted - sink_tokens
                    prope_kv_cache["k"][:, sink_tokens:sink_tokens + p_num_rolled] = \
                        prope_kv_cache["k"][:, sink_tokens + p_num_evicted:sink_tokens + p_num_evicted + p_num_rolled].clone()
                    prope_kv_cache["v"][:, sink_tokens:sink_tokens + p_num_rolled] = \
                        prope_kv_cache["v"][:, sink_tokens + p_num_evicted:sink_tokens + p_num_evicted + p_num_rolled].clone()
                    p_local_end = prope_kv_cache["local_end_index"].item() + current_end - \
                        prope_kv_cache["global_end_index"].item() - p_num_evicted
                    p_local_start = p_local_end - num_new_tokens
                    prope_kv_cache["k"][:, p_local_start:p_local_end] = k_p
                    prope_kv_cache["v"][:, p_local_start:p_local_end] = v_p
                else:
                    p_local_end = prope_kv_cache["local_end_index"].item() + current_end - prope_kv_cache["global_end_index"].item()
                    p_local_start = p_local_end - num_new_tokens
                    prope_kv_cache["k"][:, p_local_start:p_local_end] = k_p
                    prope_kv_cache["v"][:, p_local_start:p_local_end] = v_p

                x_prope = attention(
                    q_p,
                    prope_kv_cache["k"][:, max(0, p_local_end - self.max_attention_size):p_local_end],
                    prope_kv_cache["v"][:, max(0, p_local_end - self.max_attention_size):p_local_end]
                )
                prope_kv_cache["global_end_index"].fill_(current_end)
                prope_kv_cache["local_end_index"].fill_(p_local_end)

        # SP: scatter sequence, gather heads back
        if sp_enabled:
            # TF + SP: reorder from contiguous [all_clean, all_noisy] back to
            # interleaved [clean_r0, noisy_r0, ...] before reverse all-to-all
            if is_tf:
                B_x, S_x, H_x, D_x = x.shape
                x = x.reshape(B_x, 2, sp_world_size, per_rank_half, H_x, D_x) \
                      .permute(0, 2, 1, 3, 4, 5) \
                      .reshape(B_x, S_x, H_x, D_x)
                if prope_enabled:
                    x_prope = x_prope.reshape(B_x, 2, sp_world_size, per_rank_half, H_x, D_x) \
                                     .permute(0, 2, 1, 3, 4, 5) \
                                     .reshape(B_x, S_x, H_x, D_x)
            x = sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
            if prope_enabled:
                x_prope = sequence_model_parallel_all_to_all_4D(x_prope, scatter_dim=1, gather_dim=2)

        # output
        x = x.flatten(2)
        x = self.o(x)

        # PRoPE output path: fuse with zero-init projection
        if prope_enabled:
            # apply_fn_o correction: [B, L, H, D] → [B, H, L, D] → correct → [B, L, H, D]
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x_prope = self.prope_o(x_prope.flatten(2))
            x = x + x_prope

        return x


class CausalWanAttentionBlock(nn.Module):
    """Causal Wan block: cached self-attn, cross-attn, then gated FFN."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        """Init.

        Args:
            cross_attn_type: The cross attn type.
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            local_attn_size: The local attn size.
            sink_size: The sink size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C] (frame-level) or [B, L, 6, C] (token-level, SP mode)
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        token_level_e = (e.shape[1] == x.shape[1])
        e_num_frames = e.shape[1]  # save before chunk() turns e into a tuple
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        if token_level_e:
            # SP token-level: e[i] is [B, L, 1, C], squeeze to [B, L, C]
            y = self.self_attn(
                self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start,
                viewmats=viewmats, Ks=Ks, prope_kv_cache=prope_kv_cache)
            x = x + y * e[2].squeeze(2)

            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
                """Cross attn ffn.

                Args:
                    x: The x.
                    context: The context.
                    context_lens: The context lens.
                    e: The e.
                    crossattn_cache: The crossattn cache.
                """
                x = x + self.cross_attn(self.norm3(x), context,
                                        context_lens, crossattn_cache=crossattn_cache)
                y = self.ffn(
                    self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
                )
                x = x + y * e[5].squeeze(2)
                return x
        else:
            # Original frame-level path
            num_frames, frame_seqlen = e_num_frames, x.shape[1] // e_num_frames

            # self-attention
            y = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start,
                viewmats=viewmats, Ks=Ks, prope_kv_cache=prope_kv_cache)

            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
                """Cross attn ffn.

                Args:
                    x: The x.
                    context: The context.
                    context_lens: The context lens.
                    e: The e.
                    crossattn_cache: The crossattn cache.
                """
                x = x + self.cross_attn(self.norm3(x), context,
                                        context_lens, crossattn_cache=crossattn_cache)
                y = self.ffn(
                    (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
                )
                # with amp.autocast(dtype=torch.float32):
                x = x + (y.unflatten(dim=1, sizes=(num_frames,
                         frame_seqlen)) * e[5]).flatten(1, 2)
                return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    """AdaLN output head used by causal Wan graphs."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(nn.Module, NativeConfigMixin):
    r"""MinWM causal Wan 2.1 transformer (chunk KV cache)."""

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    # FSDP shard conditions — infra only, no model change
    _fsdp_shard_conditions = CausalWanConfig().arch_config._fsdp_shard_conditions if _HAS_CLEANCODE_INFRA else []

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module=None, value=False, enable=None, gradient_checkpointing_func=None):
        """Helper function to set gradient checkpointing.

        Args:
            module: The module.
            value: The value.
            enable: The enable.
            gradient_checkpointing_func: The gradient checkpointing func.
        """
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # SP: chunk sequence along token dimension (same pattern as _forward_train)
        # KV cache is stored in head-parallel domain (post all-to-all inside attention),
        # following CleanCode's DMD pipeline design.
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanModel._forward_infer SP check")
            sp_enabled = False
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            x = torch.chunk(x, sp_size, dim=1)[sp_rank]
            # NOTE: e0 is NOT chunked — it has shape [B, 1, 6, dim] (per-frame),
            # and block.forward derives frame_seqlen from x.shape[1] // e.shape[1].

        # PRoPE: expand viewmats/Ks from (B, F_chunk, *, *) to (B, seq_len, *, *)
        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f]  # (F_chunk, 4, 4)
                vm = vm[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                pad_len = single_seq_len - f * h * w
                if pad_len > 0:
                    vm = torch.cat([vm, torch.eye(4, device=vm.device, dtype=vm.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_vm.append(vm)

                ks = Ks[i, :f]  # (F_chunk, 3, 3)
                ks = ks[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                if pad_len > 0:
                    ks = torch.cat([ks, torch.eye(3, device=ks.device, dtype=ks.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_ks.append(ks)

            viewmats = torch.stack(expanded_vm)  # (B, single_seq_len, 4, 4)
            Ks = torch.stack(expanded_ks)        # (B, single_seq_len, 3, 3)

            # SP: chunk viewmats/Ks along sequence dim to match x's per-rank slice.
            # x was chunked above; viewmats/Ks must follow so prope_qkv sees matching seqlen.
            if sp_enabled:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )
        if viewmats is not None:
            kwargs['viewmats'] = viewmats
            kwargs['Ks'] = Ks

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """
            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prope_kv_cache": prope_kv_cache[block_index] if prope_kv_cache is not None else None
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prope_kv_cache": prope_kv_cache[block_index] if prope_kv_cache is not None else None
                    }
                )
                x = block(x, **kwargs)

        # SP: gather sequence from all ranks before head
        if sp_enabled:
            x = sequence_model_parallel_all_gather(x, dim=1)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        viewmats=None,
        Ks=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None: # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # PRoPE: expand viewmats/Ks from (B, F, *, *) to (B, seq_len, *, *)
        # Must happen before SP chunking so token counts align.
        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()  # tokens per video (noisy half)
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f]  # (F, 4, 4)
                vm = vm[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                pad_len = single_seq_len - f * h * w
                if pad_len > 0:
                    vm = torch.cat([vm, torch.eye(4, device=vm.device, dtype=vm.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_vm.append(vm)

                ks = Ks[i, :f]  # (F, 3, 3)
                ks = ks[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                if pad_len > 0:
                    ks = torch.cat([ks, torch.eye(3, device=ks.device, dtype=ks.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_ks.append(ks)

            viewmats = torch.stack(expanded_vm)  # (B, single_seq_len, 4, 4)
            Ks = torch.stack(expanded_ks)        # (B, single_seq_len, 3, 3)

            # TF mode: clean and noisy share the same camera trajectory
            if clean_x is not None:
                viewmats = torch.cat([viewmats, viewmats], dim=1)  # (B, 2*single_seq_len, 4, 4)
                Ks = torch.cat([Ks, Ks], dim=1)                    # (B, 2*single_seq_len, 3, 3)

        # SP: token-level chunk (like HunyuanVideo)
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanModel._forward_train SP check")
            sp_enabled = False
        sp_pad_len = 0
        sp_seq_len_orig = x.shape[1]  # before any SP padding (includes clean+noisy if TF)
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            num_frames_total = e0.shape[1]
            frame_seqlen = x.shape[1] // num_frames_total
            # Expand e0 from frame-level [B, F, 6, C] to token-level [B, L, 6, C]
            e0 = e0.unsqueeze(2).expand(-1, -1, frame_seqlen, -1, -1).flatten(1, 2)

            if clean_x is not None:
                # TF mode: chunk clean and noisy halves separately (like HunyuanVideo),
                # so each rank's local x stays [clean_chunk, noisy_chunk].
                half = sp_seq_len_orig // 2
                x_clean_half = x[:, :half]
                x_noisy_half = x[:, half:]
                e0_clean_half = e0[:, :half]
                e0_noisy_half = e0[:, half:]
                # Pad each half to sp_size multiple
                sp_pad_len = (sp_size - half % sp_size) % sp_size
                if sp_pad_len > 0:
                    x_clean_half = F.pad(x_clean_half, (0, 0, 0, sp_pad_len))
                    x_noisy_half = F.pad(x_noisy_half, (0, 0, 0, sp_pad_len))
                    e0_clean_half = F.pad(e0_clean_half, (0, 0, 0, 0, 0, sp_pad_len))
                    e0_noisy_half = F.pad(e0_noisy_half, (0, 0, 0, 0, 0, sp_pad_len))
                    if viewmats is not None:
                        vm_clean = F.pad(viewmats[:, :half], (0, 0, 0, 0, 0, sp_pad_len))
                        vm_noisy = F.pad(viewmats[:, half:], (0, 0, 0, 0, 0, sp_pad_len))
                        ks_clean = F.pad(Ks[:, :half], (0, 0, 0, 0, 0, sp_pad_len))
                        ks_noisy = F.pad(Ks[:, half:], (0, 0, 0, 0, 0, sp_pad_len))
                        viewmats = torch.cat([vm_clean, vm_noisy], dim=1)
                        Ks = torch.cat([ks_clean, ks_noisy], dim=1)
                # Chunk each half
                x_clean_half = torch.chunk(x_clean_half, sp_size, dim=1)[sp_rank]
                x_noisy_half = torch.chunk(x_noisy_half, sp_size, dim=1)[sp_rank]
                e0_clean_half = torch.chunk(e0_clean_half, sp_size, dim=1)[sp_rank]
                e0_noisy_half = torch.chunk(e0_noisy_half, sp_size, dim=1)[sp_rank]
                if viewmats is not None:
                    vm_clean_chunk = torch.chunk(viewmats[:, :half + sp_pad_len], sp_size, dim=1)[sp_rank]
                    vm_noisy_chunk = torch.chunk(viewmats[:, half + sp_pad_len:], sp_size, dim=1)[sp_rank]
                    ks_clean_chunk = torch.chunk(Ks[:, :half + sp_pad_len], sp_size, dim=1)[sp_rank]
                    ks_noisy_chunk = torch.chunk(Ks[:, half + sp_pad_len:], sp_size, dim=1)[sp_rank]
                    viewmats = torch.cat([vm_clean_chunk, vm_noisy_chunk], dim=1)
                    Ks = torch.cat([ks_clean_chunk, ks_noisy_chunk], dim=1)
                # Reassemble [clean_chunk, noisy_chunk]
                x = torch.cat([x_clean_half, x_noisy_half], dim=1)
                e0 = torch.cat([e0_clean_half, e0_noisy_half], dim=1)
            else:
                # DF mode: single chunk
                sp_pad_len = (sp_size - x.shape[1] % sp_size) % sp_size
                if sp_pad_len > 0:
                    x = F.pad(x, (0, 0, 0, sp_pad_len))
                    e0 = F.pad(e0, (0, 0, 0, 0, 0, sp_pad_len))
                x = torch.chunk(x, sp_size, dim=1)[sp_rank]
                e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]

        # arguments
        block_mask = self.block_mask
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask)
        if viewmats is not None:
            kwargs['viewmats'] = viewmats
            kwargs['Ks'] = Ks

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """
            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,31200,1536]

        # SP: gather sequence from all ranks and remove padding
        if sp_enabled:
            x = sequence_model_parallel_all_gather(x, dim=1)
            # Determine unpadded target length
            # TF mode: clean half was discarded, target = original_seq_len / 2
            # DF mode: target = original_seq_len
            sp_target_len = sp_seq_len_orig // 2 if clean_x is not None else sp_seq_len_orig
            if x.shape[1] > sp_target_len:
                x = x[:, :sp_target_len]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        """Forward."""
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
