"""LingBot causal-fast DiT (Self-Forcing-style chunk KV cache).

Borrowed patterns from Self-Forcing.  Frames are generated in chunks
with causal self-attention and a persistent KV cache.  Host layers come
from :mod:`.model`.

Adds causal / fast-attention hooks.  No VACE or linear attn.
"""

import math
from typing import Any, ClassVar

import torch
import torch.nn as nn

from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config

# isort: split
from worldfoundry.core.kernels import (
    hidden_qk_rmsnorm_rope_3d,
    layer_norm_scale_shift,
    residual_gate_add,
)

# isort: split

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
    sf_fast_kernels_enabled,
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
    sf_rmsnorm,
)
from worldfoundry.base_models.diffusion_model.optimizations.camera_condition import (
    apply_camera_modulation,
    camera_condition_for_block,
    embed_wan_camera_condition,
    prepare_wan_camera_condition,
)

from .attention import attention, flash_attention
from .model import (
    WanLayerNorm,
    WanRMSNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d,
)


def _module_has_hooks(module: nn.Module) -> bool:
    return bool(getattr(module, "_forward_hooks", None) or getattr(module, "_forward_pre_hooks", None))


def _tensor_version_safe(tensor: torch.Tensor) -> int | None:
    if tensor.is_inference():
        return None
    return int(tensor._version)


def _build_lingbot_rope_cos_sin(
    freqs: torch.Tensor,
    num_frames: int,
    h_grid: int,
    w_grid: int,
    start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_c = torch.view_as_complex(freqs.contiguous()) if not freqs.is_complex() else freqs
    c = int(freqs_c.shape[1])
    t_dim = c - 2 * (c // 3)
    s_dim = c // 3
    f_t, f_h, f_w = freqs_c.split([t_dim, s_dim, s_dim], dim=1)
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


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
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
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        seq_lens_int=None,
        *,
        grid_size=None,
        precomputed_rope=None,
        kv_plan=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        del seq_lens_int
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if frame_seqlen is None:
            frame_seqlen = grid_size[1] * grid_size[2] if grid_size is not None else math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        q_proj = self.q(x)
        k_proj = self.k(x)
        v_proj = self.v(x)

        current_end = current_start + s
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = s

        # Read cursor indices at most once (0 syncs when kv_plan is supplied by WanModelFast)
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

        use_superfused_qk = (
            x.is_cuda
            and not torch.is_grad_enabled()
            and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
            and precomputed_rope is not None
            and not _module_has_hooks(self.norm_q)
            and not _module_has_hooks(self.norm_k)
            and ((not self.qk_norm) or (hasattr(self.norm_q, "weight") and hasattr(self.norm_k, "weight")))
            and q_proj.dtype == k_proj.dtype == v_proj.dtype == kv_cache["k"].dtype == kv_cache["v"].dtype
        )
        if use_superfused_qk:
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
            v = v_proj.view(b, s, n, d)
            rope_is_fused = (
                self.qk_norm and b == 1 and not _module_has_hooks(self.norm_q) and not _module_has_hooks(self.norm_k)
            )
            if rope_is_fused:
                gs = grid_size if grid_size is not None else tuple(int(value) for value in grid_sizes[0].tolist())
                q_r, k_r = hidden_qk_rmsnorm_rope_3d(
                    q_proj,
                    k_proj,
                    self.norm_q.weight,
                    self.norm_k.weight,
                    freqs,
                    num_heads=n,
                    grid_size=gs,
                    eps=self.eps,
                    start_frame=current_start_frame,
                    valid_tokens=math.prod(gs),
                )
                roped_query = q_r.view(b, s, n, d).type_as(v)
                roped_key = k_r.view(b, s, n, d).type_as(v)
            else:
                q_n = self.norm_q(q_proj).view(b, s, n, d)
                k_n = self.norm_k(k_proj).view(b, s, n, d)
                roped_query = causal_rope_apply(q_n, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(k_n, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
        v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]
        backend = getattr(self, "attention_backend", "auto")
        if backend == "scaled_int8_fp8":
            from worldfoundry.core.attention.sequence.key_centering import streaming_scaled_attention

            x = streaming_scaled_attention(
                roped_query,
                k_cache,
                v_cache,
                kv_cache,
                current_start=current_start,
                local_start=local_start_index,
                local_end=local_end_index,
                maximum=max_attention_size,
            )
        elif backend == "torch":
            from worldfoundry.core.attention.backends.dispatch import attention_forward

            x = attention_forward(
                roped_query,
                k_cache,
                v_cache,
                q_pattern="b s n d",
                k_pattern="b s n d",
                v_pattern="b s n d",
                out_pattern="b s n d",
                backend="torch",
            )
        else:
            x = attention(roped_query, k_cache, v_cache)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    """Wan text cross-attention (optional KV cache)."""

    _worldfoundry_sf_fast_kernels: bool | None = None

    def forward(
        self,
        x,
        context,
        context_lens,
        crossattn_cache=None,
        cross_attn_first_call=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        use_fast_rms = (
            x.is_cuda
            and not torch.is_grad_enabled()
            and not self.training
            and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
        )
        norm_q_any: Any = self.norm_q
        norm_k_any: Any = self.norm_k

        # compute query, key, value
        q_raw = self.q(x)
        if (
            use_fast_rms
            and hasattr(norm_q_any, "weight")
            and hasattr(norm_q_any, "eps")
            and not _module_has_hooks(self.norm_q)
        ):
            q = sf_rmsnorm(q_raw, norm_q_any.weight, eps=float(norm_q_any.eps)).view(b, -1, n, d)
        else:
            q = self.norm_q(q_raw).view(b, -1, n, d)

        if crossattn_cache is not None:
            if cross_attn_first_call is None:
                is_first = not bool(crossattn_cache.get("is_init", False))
            else:
                is_first = bool(cross_attn_first_call)
            if is_first:
                crossattn_cache["is_init"] = True
                k_raw = self.k(context)
                if (
                    use_fast_rms
                    and hasattr(norm_k_any, "weight")
                    and hasattr(norm_k_any, "eps")
                    and not _module_has_hooks(self.norm_k)
                ):
                    k = sf_rmsnorm(k_raw, norm_k_any.weight, eps=float(norm_k_any.eps)).view(b, -1, n, d)
                else:
                    k = self.norm_k(k_raw).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k_raw = self.k(context)
            if (
                use_fast_rms
                and hasattr(norm_k_any, "weight")
                and hasattr(norm_k_any, "eps")
                and not _module_has_hooks(self.norm_k)
            ):
                k = sf_rmsnorm(k_raw, norm_k_any.weight, eps=float(norm_k_any.eps)).view(b, -1, n, d)
            else:
                k = self.norm_k(k_raw).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    """Causal Wan block: cached self-attn, cross-attn, then gated FFN."""

    def __init__(
        self, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6
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
            dim=dim, num_heads=num_heads, local_attn_size=local_attn_size, sink_size=sink_size, qk_norm=qk_norm, eps=eps
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        seq_lens_int=None,
        *,
        grid_size=None,
        precomputed_rope=None,
        kv_plan=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        mod_frames = int(e.shape[1])
        seq_len_tok = int(x.shape[1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        use_fast = (
            x.is_cuda
            and not torch.is_grad_enabled()
            and not self.training
            and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
            and mod_frames > 0
            and (seq_len_tok % mod_frames == 0)
        )
        mod_frame_seq = seq_len_tok // mod_frames if mod_frames > 0 else seq_len_tok

        # self-attention
        if _module_has_hooks(self.norm1):
            with torch.amp.autocast("cuda", dtype=torch.float32):
                sa_in = (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype)
        elif use_fast:
            sa_in = sf_layernorm_adaln(
                x, e[1], e[0], num_frames=mod_frames, frame_seq=mod_frame_seq, eps=float(self.norm1.eps)
            )
        else:
            sa_in = layer_norm_scale_shift(
                x,
                e[1].squeeze(2),
                e[0].squeeze(2),
                eps=self.norm1.eps,
                upcast=True,
            )
        y = self.self_attn(
            sa_in,
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            max_attention_size,
            frame_seqlen=frame_seqlen,
            seq_lens_int=seq_lens_int,
            grid_size=grid_size,
            precomputed_rope=precomputed_rope,
            kv_plan=kv_plan,
        )

        has_cam = dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict
        norm3_any: Any = self.norm3
        if (
            use_fast
            and not has_cam
            and not _module_has_hooks(self.norm3)
            and hasattr(norm3_any, "weight")
            and hasattr(norm3_any, "eps")
        ):
            x_out_dtype = torch.promote_types(x.dtype, torch.promote_types(y.dtype, e[2].dtype))
            w3 = norm3_any.weight if norm3_any.weight.dtype == x_out_dtype else norm3_any.weight.to(x_out_dtype)
            b3_raw = getattr(norm3_any, "bias", None)
            b3 = None if b3_raw is None else (b3_raw if b3_raw.dtype == x_out_dtype else b3_raw.to(x_out_dtype))
            x, x_ca_in = sf_residual_gate_add_layernorm(
                x,
                y,
                e[2],
                w3,
                b3,
                num_frames=mod_frames,
                frame_seq=mod_frame_seq,
                eps=float(norm3_any.eps),
            )
        else:
            if use_fast:
                x = sf_residual_gate_add(x, y, e[2], num_frames=mod_frames, frame_seq=mod_frame_seq)
            else:
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    x = residual_gate_add(x, y, e[2].squeeze(2))
            x = apply_camera_modulation(self, x, dit_cond_dict)
            x_ca_in = self.norm3(x)

        ca_out = self.cross_attn(
            x_ca_in,
            context,
            context_lens,
            crossattn_cache=crossattn_cache,
            cross_attn_first_call=cross_attn_first_call,
        )

        if _module_has_hooks(self.norm2):
            x = x + ca_out
            with torch.amp.autocast("cuda", dtype=torch.float32):
                ffn_in = (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(x.dtype)
        elif use_fast:
            x, ffn_in = sf_residual_add_layernorm_adaln(
                x,
                ca_out,
                e[4],
                e[3],
                num_frames=mod_frames,
                frame_seq=mod_frame_seq,
                eps=float(self.norm2.eps),
            )
        else:
            x = x + ca_out
            ffn_in = layer_norm_scale_shift(
                x,
                e[4].squeeze(2),
                e[3].squeeze(2),
                eps=self.norm2.eps,
                upcast=True,
            )

        y_ffn = self.ffn(ffn_in)
        if use_fast:
            x = sf_residual_gate_add(x, y_ffn, e[5], num_frames=mod_frames, frame_seq=mod_frame_seq)
        else:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = residual_gate_add(x, y_ffn, e[5].squeeze(2))
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
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        mod_frames = int(e.shape[1])
        seq_len_tok = int(x.shape[1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            if _module_has_hooks(self.norm):
                h = (self.norm(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype)
            elif (
                x.is_cuda
                and not torch.is_grad_enabled()
                and not self.training
                and sf_fast_kernels_enabled(self.__dict__.get("_worldfoundry_sf_fast_kernels"))
                and mod_frames > 0
                and (seq_len_tok % mod_frames == 0)
            ):
                h = sf_layernorm_adaln(
                    x,
                    e[1],
                    e[0],
                    num_frames=mod_frames,
                    frame_seq=seq_len_tok // mod_frames,
                    eps=float(self.norm.eps),
                )
            else:
                h = layer_norm_scale_shift(
                    x,
                    e[1].squeeze(2),
                    e[0].squeeze(2),
                    eps=self.norm.eps,
                    upcast=True,
                )
            x = self.head(h)
        return x


class WanModelFast(nn.Module, NativeConfigMixin):
    r"""LingBot causal-fast DiT (Self-Forcing-style chunk KV cache)."""

    ignore_for_config: ClassVar[list] = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules: ClassVar[list] = ["WanAttentionBlock"]
    prepare_camera_condition = prepare_wan_camera_condition

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        control_type="cam",
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
            control_type (`str`, *optional*, defaults to 'cam'):
               Type of conditioning control signal - 'cam' (6-dim camera Plucker
               embeddings) or 'act' (7-dim action embeddings including WASD movement)
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

        if control_type == "cam":
            control_dim = 6
        elif control_type == "act":
            control_dim = 7

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.view_as_real(
            torch.cat(
                [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
                dim=1,
            )
        )

        # initialize weights
        self.init_weights()

    def forward(
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
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == "i2v":
            assert y is not None

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

        # time embeddings — broadcast scalar [B] timesteps as [B, 1, ...] to avoid
        # 4680x redundant MLP compute and ~7.4 GB/step FP32 e0 materialization.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if t.dim() == 1:
                bt = int(t.size(0))
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float()).unflatten(0, (bt, 1))
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                bt, t_len = int(t.size(0)), int(t.size(1))
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, (bt, t_len)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context — skip redundant text MLP when every crossattn_cache slot is already populated
        context_lens = None
        all_cross_cached = (
            crossattn_cache is not None
            and len(crossattn_cache) == len(self.blocks)
            and cross_attn_first_call is not True
            and not torch.is_grad_enabled()
            and not self.training
            and not _module_has_hooks(self.text_embedding)
            and all(bool(entry.get("is_init", False)) for entry in crossattn_cache)
        )
        if all_cross_cached:
            context = None
        else:
            context = self.text_embedding(
                torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
            )

        # Prepared camera tensors belong to this chunk, not to the model.
        if (
            dit_cond_dict is not None
            and "c2ws_plucker_emb" in dit_cond_dict
            and "_camera_modulations" not in dit_cond_dict
        ):
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = embed_wan_camera_condition(self, dit_cond_dict["c2ws_plucker_emb"])

        grid_size_0 = grid_tuples[0]
        frame_seqlen_eff = int(frame_seqlen) if frame_seqlen is not None else grid_size_0[1] * grid_size_0[2]
        all_same_grid = all(gt == grid_size_0 for gt in grid_tuples)
        precomputed_rope = None
        if x.is_cuda and not torch.is_grad_enabled() and not self.training and all_same_grid:
            start_frame = current_start // frame_seqlen_eff
            rope_cache = self.__dict__.setdefault("_worldfoundry_rope_cache", {})
            rope_key = (id(self.freqs), device, grid_size_0, start_frame)
            precomputed_rope = rope_cache.get(rope_key)
            if precomputed_rope is None:
                if len(rope_cache) >= 64:
                    rope_cache.clear()
                precomputed_rope = _build_lingbot_rope_cos_sin(
                    self.freqs, grid_size_0[0], grid_size_0[1], grid_size_0[2], start_frame
                )
                rope_cache[rope_key] = precomputed_rope

        # Version-guarded host KV mirror eliminates ~192 GPU->CPU syncs/step (0 syncs on
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
            "dit_cond_dict": dit_cond_dict,
            "max_attention_size": max_attention_size,
            "frame_seqlen": frame_seqlen_eff,
            "cross_attn_first_call": cross_attn_first_call,
            "grid_size": grid_size_0 if all_same_grid else None,
            "precomputed_rope": precomputed_rope,
            "kv_plan": kv_plan,
        }

        for block_index, block in enumerate(self.blocks):
            kwargs["dit_cond_dict"] = camera_condition_for_block(dit_cond_dict, block_index)
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
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
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

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
