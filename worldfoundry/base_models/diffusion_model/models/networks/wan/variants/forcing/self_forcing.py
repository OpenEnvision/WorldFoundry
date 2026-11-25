"""Self-Forcing causal Wan (chunk KV cache, teacher-forcing train).

Causal self-attention over generated prefixes.  Used by the shared
Self-Forcing runtime.  Adds causal attention; no VACE or linear attn.
"""

import math
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from worldfoundry.core.attention.sequence.varlen import attention
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config

from .fast_kernels import (
    sf_apply_rope_bshd,
    sf_cached_t2v_cross_attention,
    sf_exact_norms_enabled,
    sf_fast_kernels_enabled,
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
    sf_rmsnorm,
)
from .model import (
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def _build_3d_rope_cos_sin(
    freqs: torch.Tensor,
    f: int,
    h: int,
    w: int,
    start_frame: int = 0,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build real-valued ``(cos, sin)`` 3D RoPE factors of shape ``[1, f*h*w, 1, c]``.

    ``dtype=torch.float64`` is bitwise identical to ``causal_rope_apply`` while
    avoiding complex ops, host ``.tolist()`` syncs, and per-block table rebuilds.
    """
    c = freqs.shape[1]
    t_dim = c - 2 * (c // 3)
    s_dim = c // 3
    f_t, f_h, f_w = freqs.split([t_dim, s_dim, s_dim], dim=1)
    freqs_3d = torch.cat(
        [
            f_t[start_frame : start_frame + f].view(f, 1, 1, t_dim).expand(f, h, w, t_dim),
            f_h[:h].view(1, h, 1, s_dim).expand(f, h, w, s_dim),
            f_w[:w].view(1, 1, w, s_dim).expand(f, h, w, s_dim),
        ],
        dim=-1,
    ).reshape(1, f * h * w, 1, c)
    return freqs_3d.real.to(dtype).contiguous(), freqs_3d.imag.to(dtype).contiguous()


def _apply_real_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply precomputed real-valued RoPE ``(cos, sin)`` to ``[B, S, H, D]`` via fused kernel."""
    from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

    return apply_real_rope_bshd(x, cos, sin)


_DEFAULT_APPLY_REAL_ROPE = _apply_real_rope


def _use_sf_fast_kernels(module: nn.Module) -> bool:
    """Return True when the lossless Hopper Triton fast path should run for *module*."""
    if _apply_real_rope is not _DEFAULT_APPLY_REAL_ROPE:
        return False
    return sf_fast_kernels_enabled(getattr(module, "_worldfoundry_sf_fast_kernels", None))


def _use_sf_exact_norms(module: nn.Module, attention_backend: str | None = None) -> bool:
    """Return True when 0-ULP bit-exact norm reductions should be used for *module*."""
    return sf_exact_norms_enabled(
        attention_backend=attention_backend,
        override=getattr(module, "_worldfoundry_sf_exact_norms", None),
    )


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

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


_DEFAULT_CAUSAL_ROPE_APPLY = causal_rope_apply


def kv_cache_relative_rope_apply(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    query_start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` and the whole cache read with positions taken from cache slots.

    Cached keys are stored without RoPE. On every read, cache slot ``i`` takes
    temporal position ``i`` and the query block takes the positions of the slots
    it was written to, so positions stay inside the cache however long the
    rollout runs. Follows the KV-cache-relative RoPE of NVIDIA FlashDreams
    (Apache-2.0); see THIRD-PARTY-NOTICES.

    Args:
        q: Query block, shape ``[B, L, num_heads, head_dim]``.
        cached_k: Un-rotated keys of the valid cache prefix, shape ``[B, S, num_heads, head_dim]``.
        grid_sizes: Query ``(F, H, W)`` patch grid per sample, shape ``[B, 3]``.
        freqs: RoPE frequency table, shape ``[1024, head_dim / 2]``.
        query_start_frame: First cache slot frame of the query block.

    Returns:
        The rotated query and keys, in the dtype of ``q`` and ``cached_k``.
    """
    frame_tokens = math.prod(grid_sizes[0][1:].tolist())
    cache_grid_sizes = grid_sizes.clone()
    cache_grid_sizes[:, 0] = cached_k.shape[1] // frame_tokens
    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=query_start_frame)
    roped_key = causal_rope_apply(cached_k, cache_grid_sizes, freqs, start_frame=0)
    return roped_query, roped_key


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache and frame-offset RoPE."""

    supports_fused_qkv: ClassVar[bool] = True

    def __init__(
        self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6, kv_cache_relative_rope=False
    ):
        """Causal self-attention over a block KV cache.

        With ``kv_cache_relative_rope`` the cache stores keys without RoPE and every read rotates them by cache
        slot (see :func:`kv_cache_relative_rope_apply`).
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
        self.kv_cache_relative_rope = kv_cache_relative_rope
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attention_backend: str | None = None

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind a pipeline-scoped provider for the cached self-attention.

        ``None`` and ``"torch"`` keep the dense PyTorch SDPA path of the
        official runtime, which is bitwise exact. Any other provider runs
        through the shared attention dispatcher and is approximate: it changes
        the BF16 rounding of every call, so a rollout no longer reproduces the
        official samples.
        """

        self.attention_backend = None if backend is None else str(backend)

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
        *,
        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        precomputed_relative_rope: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
        kv_plan: tuple[int, int, int, int, int, int] | None = None,
        frame_seqlen: int | None = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function (supports fused QKV without extra copies)
        if hasattr(self, "qkv"):
            q_proj, k_proj, v_proj = self.qkv(x).chunk(3, dim=-1)
        else:
            q_proj = self.q(x)
            k_proj = self.k(x)
            v_proj = self.v(x)

        if kv_cache is None:
            q = self.norm_q(q_proj).view(b, s, n, d)
            k = self.norm_k(k_proj).view(b, s, n, d)
            v = v_proj.view(b, s, n, d)
            # if it is teacher forcing training?
            is_tf = s == seq_lens[0].item() * 2
            if is_tf:
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
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )[:, :, : q.shape[1]].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )[:, :, : q.shape[1]].transpose(2, 1)
        else:
            if frame_seqlen is None:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()

            # Resolve KV slice bounds and roll any evicted window slots before writing new K/V.
            current_end = current_start + s
            if kv_plan is not None:
                local_start_index, local_end_index, sink_tokens, num_evicted_tokens, num_rolled_tokens, sync_tensors = (
                    kv_plan
                )
            else:
                sink_tokens = self.sink_size * frame_seqlen
                kv_cache_size = kv_cache["k"].shape[1]
                num_new_tokens = s
                prev_global_end = kv_cache["global_end_index"].item()
                prev_local_end = kv_cache["local_end_index"].item()
                if (
                    self.local_attn_size != -1
                    and (current_end > prev_global_end)
                    and (num_new_tokens + prev_local_end > kv_cache_size)
                ):
                    num_evicted_tokens = num_new_tokens + prev_local_end - kv_cache_size
                    num_rolled_tokens = prev_local_end - num_evicted_tokens - sink_tokens
                    local_end_index = prev_local_end + current_end - prev_global_end - num_evicted_tokens
                else:
                    num_evicted_tokens = 0
                    num_rolled_tokens = 0
                    local_end_index = prev_local_end + current_end - prev_global_end
                local_start_index = local_end_index - num_new_tokens
                sync_tensors = 1

            if num_evicted_tokens > 0 and num_rolled_tokens > 0:
                src_k = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ]
                src_v = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ]
                # Clone only when source and destination overlap to avoid redundant allocations.
                if num_evicted_tokens < num_rolled_tokens:
                    src_k = src_k.clone()
                    src_v = src_v.clone()
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens].copy_(src_k)
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens].copy_(src_v)

            use_fast = _use_sf_fast_kernels(self)
            exact_norms = _use_sf_exact_norms(self, self.attention_backend) if use_fast else True
            is_wan_qk_norm = type(self.norm_q) is WanRMSNorm and type(self.norm_k) is WanRMSNorm
            is_id_qk_norm = type(self.norm_q) is nn.Identity and type(self.norm_k) is nn.Identity
            can_fuse_qk = use_fast and (is_wan_qk_norm or is_id_qk_norm)

            if can_fuse_qk and not self.kv_cache_relative_rope and precomputed_rope is not None:
                cos, sin = precomputed_rope
                roped_query = sf_qk_rmsnorm_rope_write_kv(
                    q_proj,
                    k_proj,
                    v_proj,
                    self.norm_q.weight if is_wan_qk_norm else None,
                    self.norm_k.weight if is_wan_qk_norm else None,
                    cos,
                    sin,
                    kv_cache["k"],
                    kv_cache["v"],
                    dst_start=local_start_index,
                    num_heads=n,
                    head_dim=d,
                    eps=float(self.norm_q.eps) if is_wan_qk_norm else float(self.eps),
                    rotate_k=True,
                    exact=exact_norms,
                    enabled=True,
                )
                cached_k = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size) : local_end_index]
                cached_v = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size) : local_end_index]
            elif can_fuse_qk and self.kv_cache_relative_rope and precomputed_relative_rope is not None:
                (q_cos, q_sin), (k_cos, k_sin) = precomputed_relative_rope
                roped_query = sf_qk_rmsnorm_rope_write_kv(
                    q_proj,
                    k_proj,
                    v_proj,
                    self.norm_q.weight if is_wan_qk_norm else None,
                    self.norm_k.weight if is_wan_qk_norm else None,
                    q_cos,
                    q_sin,
                    kv_cache["k"],
                    kv_cache["v"],
                    dst_start=local_start_index,
                    num_heads=n,
                    head_dim=d,
                    eps=float(self.norm_q.eps) if is_wan_qk_norm else float(self.eps),
                    rotate_k=False,
                    exact=exact_norms,
                    enabled=True,
                )
                cached_k = sf_apply_rope_bshd(kv_cache["k"][:, :local_end_index], k_cos, k_sin, enabled=True).type_as(
                    v_proj
                )
                cached_v = kv_cache["v"][:, :local_end_index]
            else:
                if use_fast and is_wan_qk_norm:
                    q = sf_rmsnorm(
                        q_proj, self.norm_q.weight, eps=float(self.norm_q.eps), exact=exact_norms, enabled=True
                    ).view(b, s, n, d)
                    k = sf_rmsnorm(
                        k_proj, self.norm_k.weight, eps=float(self.norm_k.eps), exact=exact_norms, enabled=True
                    ).view(b, s, n, d)
                else:
                    q = self.norm_q(q_proj).view(b, s, n, d)
                    k = self.norm_k(k_proj).view(b, s, n, d)
                v = v_proj.view(b, s, n, d)

                if self.kv_cache_relative_rope:
                    roped_key = k
                elif precomputed_rope is not None:
                    cos, sin = precomputed_rope
                    roped_query = _apply_real_rope(q, cos, sin)
                    roped_key = _apply_real_rope(k, cos, sin)
                else:
                    current_start_frame = current_start // frame_seqlen
                    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                    roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

                if self.kv_cache_relative_rope:
                    if precomputed_relative_rope is not None:
                        (q_cos, q_sin), (k_cos, k_sin) = precomputed_relative_rope
                        roped_query = _apply_real_rope(q, q_cos, q_sin).type_as(v)
                        cached_k = _apply_real_rope(kv_cache["k"][:, :local_end_index], k_cos, k_sin).type_as(v)
                    else:
                        roped_query, cached_k = kv_cache_relative_rope_apply(
                            q,
                            kv_cache["k"][:, :local_end_index],
                            grid_sizes,
                            freqs,
                            query_start_frame=local_start_index // frame_seqlen,
                        )
                        roped_query = roped_query.type_as(v)
                    cached_v = kv_cache["v"][:, :local_end_index]
                else:
                    cached_k = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size) : local_end_index]
                    cached_v = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size) : local_end_index]
            if self.attention_backend in (None, "torch"):
                x = attention(roped_query, cached_k, cached_v)
            elif (
                self.attention_backend == "flash_attention_3"
                and roped_query.device.type == "cuda"
                and roped_query.dtype in {torch.float16, torch.bfloat16}
                and not (
                    torch.is_grad_enabled()
                    and (roped_query.requires_grad or cached_k.requires_grad or cached_v.requires_grad)
                )
            ):
                from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
                    wan_flash_attn_3_bshd,
                )

                x = wan_flash_attn_3_bshd(roped_query, cached_k, cached_v)
            else:
                from worldfoundry.core.attention.backends.dispatch import attention_forward

                x = attention_forward(
                    roped_query,
                    cached_k,
                    cached_v,
                    q_pattern="b s n d",
                    k_pattern="b s n d",
                    v_pattern="b s n d",
                    out_pattern="b s n d",
                    backend=self.attention_backend,
                )
            if sync_tensors:
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    """Causal Wan block: cached self-attn, cross-attn, then gated FFN."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        kv_cache_relative_rope=False,
    ):
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
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, kv_cache_relative_rope
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

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
        *,
        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        precomputed_relative_rope: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
        kv_plan: tuple[int, int, int, int, int, int] | None = None,
        frame_seqlen: int | None = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seq = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32
        use_fast = _use_sf_fast_kernels(self)
        exact_norms = _use_sf_exact_norms(self, self.self_attn.attention_backend) if use_fast else True

        # self-attention
        if use_fast and type(self.norm1) is WanLayerNorm:
            sa_in = sf_layernorm_adaln(
                x,
                e[1],
                e[0],
                num_frames=num_frames,
                frame_seq=frame_seq,
                eps=float(self.norm1.eps),
                exact=exact_norms,
                enabled=True,
            )
        else:
            sa_in = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seq)) * (1 + e[1]) + e[0]).flatten(1, 2)

        y = self.self_attn(
            sa_in,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            precomputed_rope=precomputed_rope,
            precomputed_relative_rope=precomputed_relative_rope,
            kv_plan=kv_plan,
            frame_seqlen=frame_seq if frame_seqlen is None else frame_seqlen,
        )

        # post-self-attention residual + norm3 super-fusion
        if use_fast and type(self.norm3) is WanLayerNorm:
            x, cross_in = sf_residual_gate_add_layernorm(
                x,
                y,
                e[2],
                self.norm3.weight,
                self.norm3.bias,
                num_frames=num_frames,
                frame_seq=frame_seq,
                eps=float(self.norm3.eps),
                exact=exact_norms,
                enabled=True,
            )
        elif use_fast:
            x = sf_residual_gate_add(x, y, e[2], num_frames=num_frames, frame_seq=frame_seq, enabled=True)
            cross_in = self.norm3(x)
        else:
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seq)) * e[2]).flatten(1, 2)
            cross_in = self.norm3(x)

        # cross-attention
        if (
            use_fast
            and crossattn_cache is not None
            and crossattn_cache.get("is_init")
            and context_lens is None
            and type(self.cross_attn) is WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"]
            and not self.cross_attn._forward_hooks
            and not self.cross_attn._forward_pre_hooks
        ):
            cross_out = sf_cached_t2v_cross_attention(
                self.cross_attn, cross_in, crossattn_cache, exact=exact_norms, enabled=True
            )
        else:
            cross_out = self.cross_attn(cross_in, context, context_lens, crossattn_cache=crossattn_cache)

        # post-cross-attention residual + norm2 + AdaLN super-fusion
        if use_fast and type(self.norm2) is WanLayerNorm:
            x, ffn_in = sf_residual_add_layernorm_adaln(
                x,
                cross_out,
                e[4],
                e[3],
                num_frames=num_frames,
                frame_seq=frame_seq,
                eps=float(self.norm2.eps),
                exact=exact_norms,
                enabled=True,
            )
        else:
            x = x + cross_out
            ffn_in = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seq)) * (1 + e[4]) + e[3]).flatten(1, 2)

        # FFN + post-FFN residual gate add
        y = self.ffn(ffn_in)
        if use_fast:
            x = sf_residual_gate_add(x, y, e[5], num_frames=num_frames, frame_seq=frame_seq, enabled=True)
        else:
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seq)) * e[5]).flatten(1, 2)
        return x


class CausalHead(nn.Module):
    """AdaLN output head used by causal Wan graphs."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
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

    def forward(self, x, e, *, exact_norms: bool | None = None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        if _use_sf_fast_kernels(self) and type(self.norm) is WanLayerNorm:
            use_exact = _use_sf_exact_norms(self, None) if exact_norms is None else bool(exact_norms)
            h = sf_layernorm_adaln(
                x,
                e[1],
                e[0],
                num_frames=num_frames,
                frame_seq=frame_seqlen,
                eps=float(self.norm.eps),
                exact=use_exact,
                enabled=True,
            )
            return self.head(h.unflatten(dim=1, sizes=(num_frames, frame_seqlen)))
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(nn.Module, NativeConfigMixin):
    r"""Self-Forcing causal Wan (chunk KV cache, teacher-forcing train)."""

    ignore_for_config: ClassVar[list] = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules: ClassVar[list] = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
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
        eps=1e-6,
        kv_cache_relative_rope=False,
    ):
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
            kv_cache_relative_rope (`bool`, *optional*, defaults to False):
                Store un-rotated keys in the KV cache and rotate the cache read with
                cache-slot positions, so a rolling ``local_attn_size`` cache keeps
                bounded positions over an unbounded rollout
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
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
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    kv_cache_relative_rope,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (
                    q_idx == kv_idx
                )
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
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
        device: torch.device | str, num_frames: int = 21, frame_seqlen: int = 1560, num_frame_per_block=1
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
            start=0, end=num_frames * frame_seqlen, step=attention_block_size, device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start : start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length, step=attention_block_size, device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list, strict=False)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(
                attention_mask,
                B=None,
                H=None,
                Q_LEN=total_length + padded_length,
                KV_LEN=total_length + padded_length,
                device=device,
            )
            import cv2

            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255.0 * mask))  # noqa: UP031 - Keeps printf conversions that depend on argument types.

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=4,
        local_attn_size=-1,
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

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen, end=total_length, step=frame_seqlen * num_frame_per_block, device=device
        )

        for _idx, tmp in enumerate(frame_indices):
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (
                    q_idx == kv_idx
                )

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
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
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int = 0,
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

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        fast_batched = isinstance(x, torch.Tensor) and x.ndim == 5 and y is None
        if fast_batched:
            x_emb = self.patch_embedding(x)
            grid_tuple = (int(x_emb.shape[2]), int(x_emb.shape[3]), int(x_emb.shape[4]))
            f_grid, h_grid, w_grid = grid_tuple
            frame_seqlen = h_grid * w_grid
            batch_size = int(x_emb.shape[0])
            x = x_emb.flatten(2).transpose(1, 2).contiguous()
            # Zero CUDA syncs: precomputed real (cos, sin) covers both fp64 (bitwise exact) and fp32.
            grid_sizes = None
            seq_lens = None
        else:
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            grid_tuple = tuple(int(v) for v in x[0].shape[2:])
            f_grid, h_grid, w_grid = grid_tuple
            frame_seqlen = h_grid * w_grid
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat(x)

        # time embeddings — reuse per-device sinusoidal frequency table (zero H2D syncs, bitwise exact)
        sin_cache = self.__dict__.setdefault("_worldfoundry_sinusoid_freqs", {})
        sin_freqs = sin_cache.get(x.device)
        if sin_freqs is None:
            half = self.freq_dim // 2
            sin_freqs = torch.pow(10000.0, -torch.arange(half, device=x.device, dtype=torch.float64).div(half))
            sin_cache[x.device] = sin_freqs
        angles = torch.outer(t.flatten().to(dtype=torch.float64), sin_freqs)
        e = self.time_embedding(torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context — skip redundant text MLP once all cross-attention K/V caches are initialized
        context_lens = None
        cross_ready = (
            crossattn_cache is not None
            and len(crossattn_cache) == len(self.blocks)
            and all(bool(entry.get("is_init")) for entry in crossattn_cache)
        )
        if not cross_ready:
            if isinstance(context, torch.Tensor) and context.ndim == 3 and context.shape[1] == self.text_len:
                context = self.text_embedding(context)
            else:
                context = self.text_embedding(
                    torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
                )
            if clip_fea is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)

        # Plan KV cache slice indices once on host instead of 60x per-block .item() CUDA syncs.
        kv_plan: tuple[int, int, int, int, int, int] | None = None
        sync_after_blocks: tuple[int, int] | None = None
        if kv_cache is not None and len(kv_cache) == len(self.blocks):
            layer0 = kv_cache[0]
            kv_cache_size = int(layer0["k"].shape[1])
            num_new_tokens = int(x.shape[1])
            current_end = int(current_start) + num_new_tokens
            sink_tokens = int(self.blocks[0].self_attn.sink_size) * frame_seqlen
            local_attn_size = int(self.blocks[0].self_attn.local_attn_size)
            mirror_key = (id(kv_cache), layer0["k"].data_ptr())
            host_mirror = self.__dict__.setdefault("_worldfoundry_kv_host_mirror", {})
            cached_pair = host_mirror.get(mirror_key)
            if cached_pair is None:
                if int(current_start) == 0:
                    prev_global_end = 0
                    prev_local_end = 0
                else:
                    prev_global_end = int(layer0["global_end_index"].item())
                    prev_local_end = int(layer0["local_end_index"].item())
            else:
                prev_global_end, prev_local_end = cached_pair
                if int(current_start) == 0 and prev_global_end > current_end:
                    # Fresh rollout restarted at slot 0 reusing pooled KV buffers without reset_causal_host_cache().
                    prev_global_end, prev_local_end = 0, 0
            if (
                local_attn_size != -1
                and current_end > prev_global_end
                and num_new_tokens + prev_local_end > kv_cache_size
            ):
                num_evicted = num_new_tokens + prev_local_end - kv_cache_size
                num_rolled = prev_local_end - num_evicted - sink_tokens
                next_local_end = prev_local_end + current_end - prev_global_end - num_evicted
            else:
                num_evicted = 0
                num_rolled = 0
                next_local_end = prev_local_end + current_end - prev_global_end
            local_start_index = next_local_end - num_new_tokens
            kv_plan = (local_start_index, next_local_end, sink_tokens, num_evicted, num_rolled, 0)
            sync_after_blocks = (current_end, next_local_end)
            if len(host_mirror) >= 16:
                host_mirror.clear()
            host_mirror[mirror_key] = sync_after_blocks

        # Precompute real-valued 3D RoPE (cos, sin) once for all blocks (zero complex ops / zero syncs).
        # fp64 matches causal_rope_apply bitwise; fp32 trades minor ULP rounding for smaller tables.
        rope_precision = str(getattr(self, "_worldfoundry_rope_precision", "fp64")).strip().casefold()
        rope_dtype = torch.float32 if rope_precision == "fp32" else torch.float64
        rope_cache = self.__dict__.setdefault("_worldfoundry_causal_rope_cache", {})

        def _cached_cos_sin(num_f: int, start_f: int) -> tuple[torch.Tensor, torch.Tensor]:
            rope_key = (id(self.freqs), device, rope_dtype, num_f, h_grid, w_grid, start_f)
            cached = rope_cache.get(rope_key)
            if cached is None:
                cached = _build_3d_rope_cos_sin(
                    self.freqs,
                    num_f,
                    h_grid,
                    w_grid,
                    start_frame=start_f,
                    dtype=rope_dtype,
                )
                if len(rope_cache) >= 128:
                    rope_cache.clear()
                rope_cache[rope_key] = cached
            return cached

        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        precomputed_relative_rope: (
            tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None
        use_precomputed_rope = causal_rope_apply is _DEFAULT_CAUSAL_ROPE_APPLY
        if self.blocks[0].self_attn.kv_cache_relative_rope:
            if use_precomputed_rope and kv_plan is not None and frame_seqlen > 0:
                q_start_f = kv_plan[0] // frame_seqlen
                k_frames = kv_plan[1] // frame_seqlen
                precomputed_relative_rope = (
                    _cached_cos_sin(f_grid, q_start_f),
                    _cached_cos_sin(k_frames, 0),
                )
            elif grid_sizes is None:
                grid_sizes = torch.tensor([grid_tuple] * int(x.shape[0]), dtype=torch.long)
        elif use_precomputed_rope:
            start_frame = int(current_start) // frame_seqlen
            precomputed_rope = _cached_cos_sin(f_grid, start_frame)
        elif grid_sizes is None:
            grid_sizes = torch.tensor([grid_tuple] * int(x.shape[0]), dtype=torch.long)

        # arguments
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.block_mask,
            "precomputed_rope": precomputed_rope,
            "precomputed_relative_rope": precomputed_relative_rope,
            "kv_plan": kv_plan,
            "frame_seqlen": frame_seqlen,
        }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {"kv_cache": kv_cache[block_index], "current_start": current_start, "cache_start": cache_start}
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        if sync_after_blocks is not None and not (
            not torch.compiler.is_compiling() and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            end_global, end_local = sync_after_blocks
            for layer_cache in kv_cache:
                layer_cache["global_end_index"].fill_(end_global)
                layer_cache["local_end_index"].fill_(end_local)

        # head
        head_attn_backend = getattr(self.blocks[0].self_attn, "attention_backend", None) if self.blocks else None
        head_exact = _use_sf_exact_norms(self.head, head_attn_backend)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2), exact_norms=head_exact)
        # unpatchify without host sync when batched
        if fast_batched:
            c = self.out_dim
            pt, ph, pw = self.patch_size
            x = x[:, : f_grid * h_grid * w_grid].view(batch_size, f_grid, h_grid, w_grid, pt, ph, pw, c)
            x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            return x.view(batch_size, c, f_grid * pt, h_grid * ph, w_grid * pw)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def reset_causal_host_cache(self) -> None:
        """Clear host-side KV index mirror when a persistent KV buffer pool is reset."""
        mirror = self.__dict__.get("_worldfoundry_kv_host_mirror")
        if isinstance(mirror, dict):
            mirror.clear()

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
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x]
            )

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.block_mask,
        }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]

        # head
        head_attn_backend = getattr(self.blocks[0].self_attn, "attention_backend", None) if self.blocks else None
        head_exact = _use_sf_exact_norms(self.head, head_attn_backend)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2), exact_norms=head_exact)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache") is not None:
            return self._forward_inference(*args, **kwargs)
        else:
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
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
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
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
