"""Causal-Forcing Wan graph (streaming denoising with KV cache).

Sibling of :mod:`.self_forcing` with the Causal-Forcing cache contract.
Adds causal attention.
"""

import math
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from worldfoundry.core.attention.sequence.varlen import attention
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config

from .fast_kernels import (
    sf_fast_kernels_enabled,
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
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


def _module_has_hooks(module: nn.Module) -> bool:
    return bool(getattr(module, "_forward_hooks", None) or getattr(module, "_forward_pre_hooks", None))


def _tensor_version_safe(tensor: torch.Tensor) -> int | None:
    if tensor.is_inference():
        return None
    return int(tensor._version)


def _build_forcing_rope_cos_sin(
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
flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")


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


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache and frame-offset RoPE."""

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
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

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
        frame_seqlen_hint=None,
        kv_plan=None,
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
            frame_seqlen = (
                int(frame_seqlen_hint) if frame_seqlen_hint is not None else math.prod(grid_sizes[0][1:]).item()
            )
            current_start_frame = current_start // frame_seqlen
            current_end = current_start + s
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = s

            if kv_plan is not None:
                prev_global_end, prev_local_end = kv_plan
            else:
                prev_global_end = int(kv_cache["global_end_index"].item())
                prev_local_end = int(kv_cache["local_end_index"].item())

            if (
                self.local_attn_size != -1
                and (current_end > prev_global_end)
                and (num_new_tokens + prev_local_end > kv_cache_size)
            ):
                num_evicted_tokens = num_new_tokens + prev_local_end - kv_cache_size
                num_rolled_tokens = prev_local_end - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                local_end_index = prev_local_end + current_end - prev_global_end - num_evicted_tokens
            else:
                local_end_index = prev_local_end + current_end - prev_global_end
            local_start_index = local_end_index - num_new_tokens

            use_fused_qk = (
                x.is_cuda
                and not torch.is_grad_enabled()
                and precomputed_rope is not None
                and not _module_has_hooks(self.norm_q)
                and not _module_has_hooks(self.norm_k)
                and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
                and ((not self.qk_norm) or (hasattr(self.norm_q, "weight") and hasattr(self.norm_k, "weight")))
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
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
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
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        *,
        precomputed_rope=None,
        frame_seqlen_hint=None,
        kv_plan=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
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

        # self-attention
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
            frame_seqlen_hint=frame_seqlen_hint,
            kv_plan=kv_plan,
        )

        # 2-stage super-fusion: x = x + y * e[2] AND x_ca_in = norm3(x)
        norm3_any: Any = self.norm3
        if (
            use_fast
            and not _module_has_hooks(self.norm3)
            and hasattr(norm3_any, "weight")
            and hasattr(norm3_any, "eps")
        ):
            x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, e[2].dtype))
            w3 = norm3_any.weight if norm3_any.weight.dtype == x_out_dtype else norm3_any.weight.to(x_out_dtype)
            b3_raw = getattr(norm3_any, "bias", None)
            b3 = None if b3_raw is None else (b3_raw if b3_raw.dtype == x_out_dtype else b3_raw.to(x_out_dtype))
            x, x_ca_in = sf_residual_gate_add_layernorm(
                x, y, e[2], w3, b3, num_frames=num_frames, frame_seq=frame_seqlen, eps=float(norm3_any.eps)
            )
        elif use_fast:
            x = sf_residual_gate_add(x, y, e[2], num_frames=num_frames, frame_seq=frame_seqlen)
            x_ca_in = self.norm3(x)
        else:
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
            x_ca_in = self.norm3(x)

        ca_out = self.cross_attn(x_ca_in, context, context_lens, crossattn_cache=crossattn_cache)

        # 2-stage super-fusion: x = x + ca_out AND ffn_in = norm2(x) * (1 + e[4]) + e[3]
        if use_fast and not _module_has_hooks(self.norm2):
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
    r"""Causal-Forcing Wan graph (streaming denoising with KV cache)."""

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
                    cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps
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

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_tuples = [tuple(int(v) for v in u.shape[2:]) for u in x]
        grid_sizes = torch.tensor(grid_tuples, dtype=torch.long)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        f_grid, h_grid, w_grid = grid_tuples[0]
        frame_seqlen_hint = h_grid * w_grid
        all_same_grid = all(gt == grid_tuples[0] for gt in grid_tuples)
        precomputed_rope = None
        if x.is_cuda and not torch.is_grad_enabled() and not self.training and all_same_grid:
            start_frame = current_start // frame_seqlen_hint
            rope_cache = self.__dict__.setdefault("_worldfoundry_rope_cache", {})
            rope_key = (id(self.freqs), device, f_grid, h_grid, w_grid, start_frame)
            precomputed_rope = rope_cache.get(rope_key)
            if precomputed_rope is None:
                if len(rope_cache) >= 64:
                    rope_cache.clear()
                precomputed_rope = _build_forcing_rope_cos_sin(self.freqs, f_grid, h_grid, w_grid, start_frame)
                rope_cache[rope_key] = precomputed_rope

        # Version-guarded host KV mirror eliminates ~180 GPU->CPU syncs/step (0 syncs on
        # normal caches, 1 batched sync on inference-mode caches or after external resets).
        kv_plan = None
        kv_mirror = None
        kv_mirror_key = None
        if kv_cache is not None and len(kv_cache) == len(self.blocks) and not torch.is_grad_enabled():
            layer0 = kv_cache[0]
            g_vers = tuple(_tensor_version_safe(layer["global_end_index"]) for layer in kv_cache)
            l_vers = tuple(_tensor_version_safe(layer["local_end_index"]) for layer in kv_cache)
            vers_trackable = all(v is not None for v in g_vers) and all(v is not None for v in l_vers)
            kv_mirror = self.__dict__.setdefault("_worldfoundry_kv_host_mirror", {})
            kv_mirror_key = (id(kv_cache), layer0["k"].data_ptr())
            cached = kv_mirror.get(kv_mirror_key)
            if vers_trackable and cached is not None and cached[2] == g_vers and cached[3] == l_vers:
                kv_plan = (cached[0], cached[1])
            elif len(kv_cache) == 1 or all(
                layer["global_end_index"].device == layer0["global_end_index"].device
                and layer["local_end_index"].device == layer0["local_end_index"].device
                for layer in kv_cache
            ):
                stacked = torch.stack(
                    [layer["global_end_index"].reshape(()) for layer in kv_cache]
                    + [layer["local_end_index"].reshape(()) for layer in kv_cache]
                ).tolist()
                n_layers = len(kv_cache)
                g_vals = stacked[:n_layers]
                l_vals = stacked[n_layers:]
                if all(v == g_vals[0] for v in g_vals) and all(v == l_vals[0] for v in l_vals):
                    kv_plan = (int(g_vals[0]), int(l_vals[0]))

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
            "frame_seqlen_hint": frame_seqlen_hint,
            "kv_plan": kv_plan,
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

        if kv_plan is not None and kv_mirror is not None and kv_mirror_key is not None and kv_cache is not None:
            g_vers_after = tuple(_tensor_version_safe(layer["global_end_index"]) for layer in kv_cache)
            l_vers_after = tuple(_tensor_version_safe(layer["local_end_index"]) for layer in kv_cache)
            if all(v is not None for v in g_vers_after) and all(v is not None for v in l_vers_after):
                prev_g, prev_l = kv_plan
                s_tok = int(x.shape[1])
                cur_end = current_start + s_tok
                kv_cap = int(kv_cache[0]["k"].shape[1])
                sa0: Any = self.blocks[0].self_attn
                if sa0.local_attn_size != -1 and cur_end > prev_g and s_tok + prev_l > kv_cap:
                    num_evicted = s_tok + prev_l - kv_cap
                    next_l = prev_l + cur_end - prev_g - num_evicted
                else:
                    next_l = prev_l + cur_end - prev_g
                if len(kv_mirror) >= 16:
                    kv_mirror.clear()
                kv_mirror[kv_mirror_key] = (cur_end, next_l, g_vers_after, l_vers_after)

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
            if clean_x is not None:  # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                    )
            else:  # DF?
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
            # clean_x.detach()
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
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache") is not None:
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
