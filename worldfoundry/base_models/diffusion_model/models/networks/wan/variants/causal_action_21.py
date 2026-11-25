"""Causal Wan 2.1 action DiT with chunk KV cache.

Streaming / teacher-forcing variant of action-Wan 2.1.  Self-attention
is causal over frames; ``causal_rope_apply`` offsets RoPE by
``start_frame`` so cached prefixes stay aligned.

Adds action + causal attention.  No VACE or linear attn.
"""

import math
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_21 import (
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_attention_21 import attention
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_conditioning_21 import ActionModule
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
    sf_fast_kernels_enabled,
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import project_fused_qkv
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config

from .causal_cache import commit_kv_cache_positions, read_kv_cache_positions


def _module_has_hooks(module: nn.Module) -> bool:
    return bool(getattr(module, "_forward_hooks", None) or getattr(module, "_forward_pre_hooks", None))


def _build_action21_rope_cos_sin(
    freqs: torch.Tensor,
    num_frames: int,
    h_grid: int,
    w_grid: int,
    start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = int(freqs.shape[1])
    t_dim = c - 2 * (c // 3)
    s_dim = c // 3
    f_t, f_h, f_w = freqs.split([t_dim, s_dim, s_dim], dim=1)
    freqs_3d = torch.cat(
        [
            f_t[start_frame : start_frame + num_frames]
            .view(num_frames, 1, 1, t_dim)
            .expand(num_frames, h_grid, w_grid, t_dim),
            f_h[:h_grid].view(1, h_grid, 1, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
            f_w[:w_grid].view(1, 1, w_grid, s_dim).expand(num_frames, h_grid, w_grid, s_dim),
        ],
        dim=-1,
    ).reshape(1, num_frames * h_grid * w_grid, 1, c)
    return freqs_3d.real.to(torch.float64).contiguous(), freqs_3d.imag.to(torch.float64).contiguous()


# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
#     flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    f, h, w = grid_sizes.tolist()

    for i in range(len(x)):
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


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache and frame-offset RoPE."""

    # Load-time fusion changes projection storage only. This forward retains
    # ownership of frame-offset RoPE, padding, and every causal cache mutation.
    supports_fused_qkv = True

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880
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
        *,
        precomputed_rope=None,
        frame_seqlen=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] # num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # Project QKV
        if getattr(self, "_qkv_fused", False):
            q_proj, k_proj, v_proj = project_fused_qkv(self.qkv, x)
        else:
            q_proj, k_proj, v_proj = self.q(x), self.k(x), self.v(x)

        if kv_cache is None:
            q = self.norm_q(q_proj).view(b, s, n, d)
            k = self.norm_k(k_proj).view(b, s, n, d)
            v = v_proj.view(b, s, n, d)
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype),
                ],
                dim=1,
            )

            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype),
                ],
                dim=1,
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)],
                dim=1,
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),  # after: B, HW, F, C
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask,
            )[:, :, : q.shape[1]].transpose(2, 1)
        else:
            assert grid_sizes.ndim == 1
            if frame_seqlen is None:
                frame_seqlen = math.prod(grid_sizes[1:]).item()
            current_start_frame = current_start // frame_seqlen
            current_end = current_start + s
            sink_tokens = self.sink_size * frame_seqlen

            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = s
            cached_global_end, cached_local_end = read_kv_cache_positions(kv_cache)

            if (current_end > cached_global_end) and (num_new_tokens + cached_local_end > kv_cache_size):
                num_evicted_tokens = num_new_tokens + cached_local_end - kv_cache_size
                num_rolled_tokens = cached_local_end - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                local_end_index = cached_local_end + current_end - cached_global_end - num_evicted_tokens
            else:
                local_end_index = cached_local_end + current_end - cached_global_end
            local_start_index = local_end_index - num_new_tokens

            use_fused_qk = (
                x.is_cuda
                and not torch.is_grad_enabled()
                and precomputed_rope is not None
                and not _module_has_hooks(self.norm_q)
                and not _module_has_hooks(self.norm_k)
                and (
                    (not self.qk_norm)
                    or (
                        hasattr(self.norm_q, "weight")
                        and hasattr(self.norm_k, "weight")
                        and hasattr(self.norm_q, "eps")
                    )
                )
                and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
            )
            if use_fused_qk:
                rope_cos, rope_sin = precomputed_rope
                norm_q_any: Any = self.norm_q
                norm_k_any: Any = self.norm_k
                q_w: torch.Tensor | None = norm_q_any.weight if self.qk_norm else None
                k_w: torch.Tensor | None = norm_k_any.weight if self.qk_norm else None
                roped_query = sf_qk_rmsnorm_rope_write_kv(
                    q_proj,
                    k_proj,
                    v_proj,
                    q_w,
                    k_w,
                    rope_cos,
                    rope_sin,
                    kv_cache["k"],
                    kv_cache["v"],
                    dst_start=local_start_index,
                    num_heads=n,
                    head_dim=d,
                    eps=float(self.eps),
                    rotate_k=True,
                )
            else:
                q = self.norm_q(q_proj).view(b, s, n, d)
                k = self.norm_k(k_proj).view(b, s, n, d)
                v = v_proj.view(b, s, n, d)
                roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
            )
            commit_kv_cache_positions(kv_cache, current_end, local_end_index)

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
        action_config={},  # noqa: B006 - Keeps the upstream default object and signature.
        block_idx=0,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if len(action_config) != 0 and block_idx in action_config["blocks"]:
            self.action_model = ActionModule(**action_config, local_attn_size=self.local_attn_size)
        else:
            self.action_model = None
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
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
        block_mask,
        block_mask_mouse,
        block_mask_keyboard,
        num_frame_per_block=3,
        use_rope_keyboard=False,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        context_lens=None,
        *,
        precomputed_rope=None,
        frame_seqlen_hint=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.ndim == 4
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        use_fast = (
            x.is_cuda
            and not torch.is_grad_enabled()
            and not self.training
            and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
        )

        if use_fast and not _module_has_hooks(self.norm1):
            sa_in = sf_layernorm_adaln(
                x, e[1], e[0], num_frames=num_frames, frame_seq=frame_seqlen, eps=float(self.norm1.eps)
            )
        else:
            sa_in = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)

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
            frame_seqlen=frame_seqlen_hint,
        )

        # Fuse x + y * e[2] + norm3(x) when eligible
        norm3_any: Any = self.norm3
        can_fuse_norm3 = (
            use_fast
            and not _module_has_hooks(self.norm3)
            and hasattr(norm3_any, "weight")
            and hasattr(norm3_any, "eps")
            and torch.promote_types(x.dtype, torch.promote_types(y.dtype, e[2].dtype)) == context.dtype
        )
        if can_fuse_norm3:
            x_out_dtype = context.dtype
            w3 = norm3_any.weight if norm3_any.weight.dtype == x_out_dtype else norm3_any.weight.to(x_out_dtype)
            b3_raw = getattr(norm3_any, "bias", None)
            b3 = None if b3_raw is None else (b3_raw if b3_raw.dtype == x_out_dtype else b3_raw.to(x_out_dtype))
            x, x_ca_in = sf_residual_gate_add_layernorm(
                x,
                y,
                e[2],
                w3,
                b3,
                num_frames=num_frames,
                frame_seq=frame_seqlen,
                eps=float(norm3_any.eps),
            )
        elif use_fast:
            x = sf_residual_gate_add(x, y, e[2], num_frames=num_frames, frame_seq=frame_seqlen)
            x_ca_in = self.norm3(x.to(context.dtype))
        else:
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
            x_ca_in = self.norm3(x.to(context.dtype))

        assert grid_sizes.ndim == 1
        ca_out = self.cross_attn(x_ca_in, context, crossattn_cache=crossattn_cache)
        if self.action_model is not None:
            assert mouse_cond is not None or keyboard_cond is not None
            x = x + ca_out
            start_f = (
                current_start // frame_seqlen_hint
                if frame_seqlen_hint is not None
                else current_start // math.prod(grid_sizes[1:]).item()
            )
            x = self.action_model(
                x.to(context.dtype),
                grid_sizes[0],
                grid_sizes[1],
                grid_sizes[2],
                mouse_cond,
                keyboard_cond,
                block_mask_mouse,
                block_mask_keyboard,
                is_causal=True,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                start_frame=start_f,
                use_rope_keyboard=use_rope_keyboard,
                num_frame_per_block=num_frame_per_block,
            )
            if use_fast and not _module_has_hooks(self.norm2):
                ffn_in = sf_layernorm_adaln(
                    x, e[4], e[3], num_frames=num_frames, frame_seq=frame_seqlen, eps=float(self.norm2.eps)
                )
            else:
                ffn_in = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(
                    1, 2
                )
        elif use_fast and not _module_has_hooks(self.norm2):
            x, ffn_in = sf_residual_add_layernorm_adaln(
                x, ca_out, e[4], e[3], num_frames=num_frames, frame_seq=frame_seqlen, eps=float(self.norm2.eps)
            )
        else:
            x = x + ca_out
            ffn_in = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(
                1, 2
            )

        y_ffn = self.ffn(ffn_in)
        if use_fast:
            x = sf_residual_gate_add(x, y_ffn, e[5], num_frames=num_frames, frame_seq=frame_seqlen)
        else:
            x = x + (y_ffn.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
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

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """

        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        if (
            x.is_cuda
            and not torch.is_grad_enabled()
            and not self.training
            and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
            and not _module_has_hooks(self.norm)
        ):
            h = sf_layernorm_adaln(
                x, e[1], e[0], num_frames=num_frames, frame_seq=frame_seqlen, eps=float(self.norm.eps)
            )
            return self.head(h)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(nn.Module, NativeConfigMixin):
    r"""Causal Wan 2.1 action DiT with chunk KV cache and frame-offset RoPE."""

    ignore_for_config: ClassVar[list] = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules: ClassVar[list] = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        action_config={},  # noqa: B006 - Keeps the upstream default object and signature.
        eps=1e-6,
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
        """

        super().__init__()

        assert model_type in ["i2v"]
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
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

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "i2v_cross_attn"
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
                    action_config=action_config,
                    eps=eps,
                    block_idx=idx,
                )
                for idx in range(num_layers)
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
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True
        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 880,
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

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_keyboard(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 880,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0, end=total_length2, step=frame_seqlen * num_frame_per_block, device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask2 = create_block_mask(
            attention_mask2,
            B=None,
            H=None,
            Q_LEN=total_length2 + padded_length2,
            KV_LEN=num_frames + padded_length_kv2,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_action(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 1,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0, end=total_length2, step=frame_seqlen * num_frame_per_block, device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask2 = create_block_mask(
            attention_mask2,
            B=None,
            H=None,
            Q_LEN=total_length2 + padded_length2,
            KV_LEN=num_frames + padded_length_kv2,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    def _forward_inference(
        self,
        x,
        t,
        visual_context,
        cond_concat,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache: dict | None = None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
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

        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module == True  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1)  # B C' F H W

        # embeddings
        x = self.patch_embedding(x)
        f_grid, h_grid, w_grid = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])
        frame_seqlen_hint = h_grid * w_grid
        grid_sizes = torch.tensor((f_grid, h_grid, w_grid), dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)  # B FHW C'
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        assert seq_lens[0] <= 15 * 1 * 880

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # context
        context_lens = None
        context = self.img_emb(visual_context)

        precomputed_rope = None
        if x.is_cuda and not torch.is_grad_enabled() and not self.training:
            start_frame = current_start // frame_seqlen_hint
            rope_cache = self.__dict__.setdefault("_worldfoundry_rope_cache", {})
            rope_key = (id(self.freqs), device, f_grid, h_grid, w_grid, start_frame)
            precomputed_rope = rope_cache.get(rope_key)
            if precomputed_rope is None:
                if len(rope_cache) >= 64:
                    rope_cache.clear()
                precomputed_rope = _build_action21_rope_cos_sin(self.freqs, f_grid, h_grid, w_grid, start_frame)
                rope_cache[rope_key] = precomputed_rope

        # arguments
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "mouse_cond": mouse_cond,
            "context_lens": context_lens,
            "keyboard_cond": keyboard_cond,
            "block_mask": self.block_mask,
            "block_mask_mouse": self.block_mask_mouse,
            "block_mask_keyboard": self.block_mask_keyboard,
            "use_rope_keyboard": self.use_rope_keyboard,
            "num_frame_per_block": self.num_frame_per_block,
            "precomputed_rope": precomputed_rope,
            "frame_seqlen_hint": frame_seqlen_hint,
        }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
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
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def _forward_train(
        self,
        x,
        t,
        visual_context,
        cond_concat,
        mouse_cond=None,
        keyboard_cond=None,
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
        # params
        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module == True  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x = torch.cat([x, cond_concat], dim=1)
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device,
                num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )
        if self.block_mask_keyboard is None:
            if self.use_rope_keyboard == False:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_keyboard(
                    device,
                    num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
            else:
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_action(
                    device,
                    num_frames=x.shape[2],
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
        if self.block_mask_mouse is None:
            self.block_mask_mouse = self._prepare_blockwise_causal_attn_mask_action(
                device,
                num_frames=x.shape[2],
                frame_seqlen=1,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.img_emb(visual_context)

        # arguments
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "mouse_cond": mouse_cond,
            "context_lens": context_lens,
            "keyboard_cond": keyboard_cond,
            "block_mask": self.block_mask,
            "block_mask_mouse": self.block_mask_mouse,
            "block_mask_keyboard": self.block_mask_keyboard,
            "use_rope_keyboard": self.use_rope_keyboard,
            "num_frame_per_block": self.num_frame_per_block,
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

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

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
                    shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size, strict=False)])
        return x

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

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.use_action_module == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            for m in self.blocks:
                try:
                    nn.init.zeros_(m.action_model.proj_mouse.weight)
                    if m.action_model.proj_mouse.bias is not None:
                        nn.init.zeros_(m.action_model.proj_mouse.bias)
                    nn.init.zeros_(m.action_model.proj_keyboard.weight)
                    if m.action_model.proj_keyboard.bias is not None:
                        nn.init.zeros_(m.action_model.proj_keyboard.bias)
                except BaseException:
                    pass
