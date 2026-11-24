"""MiniMax H3 packed-token audio-video DiT (single-GPU PyTorch).

Ported from SGLang's Apache-2.0
``runtime/models/dits/minimax_h3.py``. This is a faithful single-GPU port:

* ``ColumnParallelLinear`` / ``RowParallelLinear`` / ``MergedColumnParallelLinear``
  collapse to plain ``nn.Linear`` (world_size = 1).
* Ulysses sequence-parallel all-to-all, tensor-parallel all-gather, breakable
  CUDA graphs, and the layerwise-offload mixin are removed — every path here is
  rank-local.
* Dense variable-length non-causal attention is computed with
  ``F.scaled_dot_product_attention`` per packed document (``cu_seqlens``).

The numerics and the mixed fp32/bf16 parameter split are preserved exactly:
patch projections, the timestep embedder, the output heads, and
``rope.inv_freq`` stay fp32; transformer blocks are bf16. The fused CUDA/Triton
operators are represented by the portable fallbacks in :mod:`.ops` (Stage 8
swaps in the fused kernels behind the ``can_use_fused_*`` probes).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from .ops import (
    apply_qk_norm,
    indexed_gate,
    make_rms_norm,
    qk_norm_rope,
    rms_norm_indexed_scale_shift,
    rope_cos_sin_cache,
    silu_mul,
)

_BF16 = torch.bfloat16
_FP32 = torch.float32

_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "force_dense_attention",
    }
)


# --------------------------------------------------------------------------- #
# Grouped-QKV checkpoint reorder (dense MHA: per-head [q,k,v] -> [q,k,v] all).
# --------------------------------------------------------------------------- #
def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder a grouped ``[q,k,v]`` checkpoint tensor to ``[q_all,k_all,v_all]``."""

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


# --------------------------------------------------------------------------- #
# Variable-length non-causal attention over a packed sequence.
# --------------------------------------------------------------------------- #
# Safe non-clipping fallbacks if cudnn_fp8 is captured without a prior warmup
# calibration pass (covers |Q|,|K| <= 224 and |V| <= 896 in E4M3).
_DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK = 0.5
_DEFAULT_CUDNN_FP8_STATIC_DESCALE_V = 2.0
_DEFAULT_CUDNN_FP8_CALIBRATION_HEADROOM = 1.25


def _normalize_host_bounds(bounds: Sequence[int] | None) -> tuple[int, ...] | None:
    if bounds is None:
        return None
    if isinstance(bounds, tuple):
        return bounds
    return tuple(int(v) for v in bounds)


def _compute_prefix_sink_tokens(
    *,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    infer_out_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    refined_prompt_embeds_length: int | torch.Tensor | None = None,
    refiner_cu_seqlens_host: tuple[int, ...] | None = None,
) -> int:
    """Compute the live ``[text | cond/ref | audio]`` prefix length before video targets.

    In MiniMax H3 packed layouts (``t2va``, ``fl2va``, ``ref2va``), non-target
    conditioning tokens (text, keyframe/reference blocks, and audio) occupy the
    leading prefix strictly before the generated video target suffix. Protecting
    the full prefix under sparse ``sol_attn`` prevents compressing keyframe
    anchors and cross-modal audio tokens.
    """

    text_len = int(text_pos.shape[0])
    if refined_prompt_embeds_length is not None and not torch.is_tensor(refined_prompt_embeds_length):
        text_len = min(text_len, max(0, int(refined_prompt_embeds_length)))
    elif refiner_cu_seqlens_host is not None and len(refiner_cu_seqlens_host) >= 2:
        text_len = min(text_len, max(0, int(refiner_cu_seqlens_host[-1])))
    cond_rows = max(0, int(img_pos.shape[0]) - int(infer_out_pos.shape[0]))
    return text_len + cond_rows + int(audio_pos.shape[0])


def _cudnn_fp8_segment_attention(
    q_seg: torch.Tensor,
    k_seg: torch.Tensor,
    v_seg: torch.Tensor,
    *,
    softmax_scale: float,
    backend_options: Mapping[str, Any] | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
) -> torch.Tensor:
    """Explicit E4M3 quantization + native cuDNN9 FP8 SDPA for ``[L, nh, hd]``."""

    from worldfoundry.core.attention.backends.dispatch import _record_provider_runtime
    from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

    opts = backend_options or {}
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    min_scale = 1e-6

    q_bhsd = q_seg.transpose(0, 1).unsqueeze(0).contiguous()
    k_bhsd = k_seg.transpose(0, 1).unsqueeze(0).contiguous()
    v_bhsd = v_seg.transpose(0, 1).unsqueeze(0).contiguous()

    capturing = q_bhsd.is_cuda and torch.cuda.is_current_stream_capturing()
    has_explicit_scale = any(k in opts for k in ("descale_q", "descale_k", "descale_v", "static_descale"))
    scale_o = opts.get("scale_o")
    if has_explicit_scale:
        default_qk = float(opts.get("static_descale", _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK))
        default_v = float(opts.get("static_descale", _DEFAULT_CUDNN_FP8_STATIC_DESCALE_V))
        dq = float(opts.get("descale_q", default_qk))
        dk = float(opts.get("descale_k", default_qk))
        dv = float(opts.get("descale_v", default_v))
    elif capturing:
        if calibrated_descales is not None:
            dq, dk, dv = calibrated_descales
        else:
            dq = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK
            dk = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK
            dv = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_V
    else:
        # Batch all 3 reductions into one device-to-host sync (3x fewer syncs
        # than per-tensor .item()) and record calibrated per-layer descales for
        # subsequent CUDA Graph capture.
        amax_q, amax_k, amax_v = torch.stack((q_bhsd.abs().amax(), k_bhsd.abs().amax(), v_bhsd.abs().amax())).tolist()
        dq = max(float(amax_q) / fp8_max, min_scale)
        dk = max(float(amax_k) / fp8_max, min_scale)
        dv = max(float(amax_v) / fp8_max, min_scale)
        if on_calibrate is not None:
            headroom = max(1.0, float(opts.get("calibration_headroom", _DEFAULT_CUDNN_FP8_CALIBRATION_HEADROOM)))
            on_calibrate((dq * headroom, dk * headroom, dv * headroom))

    qf = (q_bhsd * (1.0 / dq)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    kf = (k_bhsd * (1.0 / dk)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    vf = (v_bhsd * (1.0 / dv)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    try:
        attended = native_cudnn_fp8_sdpa(
            qf,
            kf,
            vf,
            scale=softmax_scale,
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
            scale_o=None if scale_o is None else float(scale_o),
        )
    except Exception:
        _record_provider_runtime("cudnn_fp8", attempts=1, errors=1)
        raise
    _record_provider_runtime("cudnn_fp8", attempts=1, successes=1)
    return attended.squeeze(0).transpose(0, 1)


def _dispatch_attention_segment(
    q_seg: torch.Tensor,
    k_seg: torch.Tensor,
    v_seg: torch.Tensor,
    *,
    softmax_scale: float,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    prefix_sink_tokens: int | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
) -> torch.Tensor:
    """Execute dense or sparse non-causal attention on one ``[L, nh, hd]`` segment."""

    if backend is None or backend == "torch":
        q_bhsd = q_seg.transpose(0, 1).unsqueeze(0)
        k_bhsd = k_seg.transpose(0, 1).unsqueeze(0)
        v_bhsd = v_seg.transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=False, scale=softmax_scale)
        return attended.squeeze(0).transpose(0, 1)

    if backend == "cudnn_fp8":
        return _cudnn_fp8_segment_attention(
            q_seg,
            k_seg,
            v_seg,
            softmax_scale=softmax_scale,
            backend_options=backend_options,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )

    from worldfoundry.core.attention.backends.dispatch import attention_forward

    resolved_options = dict(backend_options) if backend_options else None
    if backend == "sol_attn":
        seq_len = int(q_seg.shape[0])
        if prefix_sink_tokens is not None and 0 < int(prefix_sink_tokens) < seq_len:
            opts = {} if resolved_options is None else resolved_options
            if opts.get("compress_kv", True) and "sink_tokens" not in opts:
                opts["sink_tokens"] = int(prefix_sink_tokens)
                opts.setdefault("sink_start", 0)
                resolved_options = opts
        if resolved_options is not None and int(resolved_options.get("kv_splits", 1)) > 1:
            req_splits = int(resolved_options["kv_splits"])
            route_groups = ((seq_len + 63) // 64 + 63) // 64
            is_sm90 = q_seg.is_cuda and tuple(torch.cuda.get_device_capability(q_seg.device)) == (9, 0)
            if not is_sm90:
                resolved_options["kv_splits"] = 1
                resolved_options.pop("allow_multi_split", None)
            elif req_splits > route_groups:
                clamped = 2 if route_groups >= 2 and req_splits >= 2 else 1
                resolved_options["kv_splits"] = clamped
                if clamped == 1:
                    resolved_options.pop("allow_multi_split", None)

    attended = attention_forward(
        q_seg.unsqueeze(0),
        k_seg.unsqueeze(0),
        v_seg.unsqueeze(0),
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        scale=softmax_scale,
        backend=backend,
        backend_options=resolved_options,
    )
    return attended.squeeze(0)


def _can_use_varlen_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return whether ``aten._flash_attention_forward`` can run zero-sync packed varlen."""

    return (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device == k.device == v.device
        and torch.backends.cuda.is_flash_attention_available()
        and torch.cuda.get_device_capability(q.device) >= (8, 0)
        and hasattr(torch.ops.aten, "_flash_attention_forward")
    )


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_host: tuple[int, ...] | None = None,
    max_seqlen: int | None = None,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    ignore_trailing_pad: bool = False,
    prefix_sink_tokens: int | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
) -> torch.Tensor:
    """Dense or backend-routed non-causal attention per packed document.

    ``q``/``k``/``v``: ``[T, num_heads, head_dim]`` packed rows.
    ``cu_seqlens``: cumulative sequence boundaries ``[num_docs + 1]``.

    Execution tiers (zero GPU→CPU sync on all hot paths):
    1. Single document (or single document + ignored alignment pad) with
       ``cu_seqlens_host``: dispatches directly to Hopper cuDNN9 SDPA or the
       configured backend (``sol_attn``, ``cudnn_fp8``, etc.) without copying.
    2. Multi-document packed varlen on SM80+ CUDA bf16/fp16 when ``max_seqlen``
       is supplied and ``backend in (None, "torch")``: single fused
       ``aten._flash_attention_forward`` launch with zero ``.tolist()`` sync.
    3. Portable per-segment fallback using ``cu_seqlens_host`` or ``cu_seqlens``.
    """

    total = int(q.shape[0])
    if total == 0:
        return torch.empty_like(q)

    # Single-document unpadded fast path: Hopper cuDNN9 SDPA / configured backend.
    if (
        cu_seqlens_host is not None
        and len(cu_seqlens_host) == 2
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[1] == total
    ):
        return _dispatch_attention_segment(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            prefix_sink_tokens=prefix_sink_tokens,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )

    # Single-document + alignment-padding tail fast path: skip wasted
    # attention launches over padding rows that are never read downstream.
    if (
        ignore_trailing_pad
        and cu_seqlens_host is not None
        and len(cu_seqlens_host) == 3
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[2] == total
    ):
        used = cu_seqlens_host[1]
        if used == total:
            return _dispatch_attention_segment(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
            )
        out = torch.empty_like(q)
        if used > 0:
            out[:used] = _dispatch_attention_segment(
                q[:used],
                k[:used],
                v[:used],
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
            )
        if used < total:
            out[used:].zero_()
        return out

    # Multi-document packed varlen zero-sync fast path via aten._flash_attention_forward.
    if (
        (backend is None or backend == "torch")
        and max_seqlen is not None
        and int(max_seqlen) > 0
        and cu_seqlens.is_cuda
        and cu_seqlens.dtype == torch.int32
        and _can_use_varlen_flash_attention(q, k, v)
    ):
        cu_cont = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()
        max_len = int(max_seqlen)
        return torch.ops.aten._flash_attention_forward(
            q_c,
            k_c,
            v_c,
            cu_cont,
            cu_cont,
            max_len,
            max_len,
            0.0,
            False,
            False,
            scale=softmax_scale,
        )[0]

    bounds = cu_seqlens_host if cu_seqlens_host is not None else tuple(int(v) for v in cu_seqlens.tolist())
    if len(bounds) == 2 and bounds[0] == 0 and bounds[1] == total:
        return _dispatch_attention_segment(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )

    tight_coverage = bool(bounds) and bounds[0] == 0 and bounds[-1] == total
    out = torch.empty_like(q) if tight_coverage else torch.zeros_like(q)
    for start, stop in itertools.pairwise(bounds):
        length = int(stop) - int(start)
        if length <= 0:
            continue
        out[start:stop] = _dispatch_attention_segment(
            q[start:stop],
            k[start:stop],
            v[start:stop],
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )
    return out


# --------------------------------------------------------------------------- #
# Timestep embedder (fp32 sinusoidal, cos-before-sin).
# --------------------------------------------------------------------------- #
class MiniMaxH3TimeEmbedder(nn.Module):
    """fp32 sinusoidal timestep MLP; concatenates ``cos`` before ``sin``."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = nn.Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, bias=True, dtype=_FP32)
        self.proj_out = nn.Linear(arch.time_embed_hidden_size, arch.time_embed_dim, bias=True, dtype=_FP32)
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32 (cos concatenated before sin)."""

        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=_FP32, device=t.device) / half)
            self._frequency_cache = freqs
        args = t.to(_FP32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq)
        hidden = F.silu(hidden)
        return self.proj_out(hidden)


# --------------------------------------------------------------------------- #
# 3D RoPE frequency builder.
# --------------------------------------------------------------------------- #
class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary 0.75)."""

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, inv_freq_len, dtype=_FP32) / float(inv_freq_len)))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


# --------------------------------------------------------------------------- #
# Attention (fused qkv -> per-head qk RMSNorm -> RoPE -> varlen SDPA -> out).
# --------------------------------------------------------------------------- #
class MiniMaxH3Attention(nn.Module):
    """Fused-QKV self-attention with per-head RMSNorm, 3D RoPE, and varlen SDPA."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.attention_fn = _varlen_attention
        self.attention_backend: str | None = None
        self.attention_backend_options: dict[str, Any] | None = None
        self._cudnn_fp8_calibrated_descales: tuple[float, float, float] | None = None
        # Fused qkv: three logical matrices stored as one tensor.
        self.qkv_proj = nn.Linear(arch.hidden_size, 3 * self.inner_dim, bias=False, dtype=_BF16)
        self.q_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, arch.hidden_size, bias=False, dtype=_BF16)

    def _update_cudnn_fp8_calibration(self, descales: tuple[float, float, float]) -> None:
        if self._cudnn_fp8_calibrated_descales is None:
            self._cudnn_fp8_calibrated_descales = descales
        else:
            prev_q, prev_k, prev_v = self._cudnn_fp8_calibrated_descales
            self._cudnn_fp8_calibrated_descales = (
                max(prev_q, descales[0]),
                max(prev_k, descales[1]),
                max(prev_v, descales[2]),
            )

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a request-scoped attention provider (``AttentionBackendConfigurable``)."""

        self._cudnn_fp8_calibrated_descales = None
        if backend is None:
            self.attention_backend = None
            self.attention_backend_options = None
            return
        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        normalized = normalize_attention_backend(str(backend))
        self.attention_backend = normalized
        self.attention_backend_options = dict(backend_options) if backend_options else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed rows -> [T, hidden]."""

        total = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        if rope_cache is None:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        else:
            cos_sin_cache, positions = rope_cache
            q, k = qk_norm_rope(q, k, self.q_norm, self.k_norm, cos_sin_cache, positions)
        if attention_groups is None:
            if self.attention_fn is _varlen_attention:
                eff_backend = "torch" if force_dense_attention else self.attention_backend
                eff_options = None if force_dense_attention else self.attention_backend_options
                out = _varlen_attention(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    softmax_scale=self.softmax_scale,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    backend=eff_backend,
                    backend_options=eff_options,
                    ignore_trailing_pad=ignore_trailing_pad,
                    prefix_sink_tokens=prefix_sink_tokens,
                    calibrated_descales=self._cudnn_fp8_calibrated_descales,
                    on_calibrate=self._update_cudnn_fp8_calibration,
                )
            else:
                out = self.attention_fn(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
        else:
            # Variants can expose global media with document-local text without
            # materializing a quadratic attention mask or duplicating projections.
            out = torch.zeros_like(q)
            for queries, keys in attention_groups:
                attended = F.scaled_dot_product_attention(
                    q.index_select(0, queries).transpose(0, 1).unsqueeze(0),
                    k.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    v.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    scale=self.softmax_scale,
                )
                out.index_copy_(0, queries, attended.squeeze(0).transpose(0, 1))
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU MLP.
# --------------------------------------------------------------------------- #
class MiniMaxH3MLP(nn.Module):
    """SwiGLU MLP with fused ``gate|up`` projection (``fc1``) and ``fc2`` down-proj."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        # gate and up fused into fc1.
        self.fc1 = nn.Linear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False, dtype=_BF16)
        self.fc2 = nn.Linear(arch.ffn_hidden_size, arch.hidden_size, bias=False, dtype=_BF16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = silu_mul(hidden)
        return self.fc2(hidden)


# --------------------------------------------------------------------------- #
# AdaLN projection (SiLU + zero-init linear over unique condition embeddings).
# --------------------------------------------------------------------------- #
class MiniMaxH3AdalnProj(nn.Module):
    """[M, t_dim] -> expand_ratio tensors of [M * modality_num, H]."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = nn.Linear(arch.time_embed_dim, out_features, bias=True, dtype=_BF16)

    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        return self.linear(adaln_input)

    def split_output(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.split_output(self.project_local(adaln_input))


# --------------------------------------------------------------------------- #
# Token refiner (pre-norm transformer, no AdaLN/RoPE).
# --------------------------------------------------------------------------- #
class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Pre-norm transformer block used only on text tokens (no AdaLN / RoPE)."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)

    @property
    def attention_backend(self) -> str | None:
        return self.attn.attention_backend

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.attn.set_attention_backend(backend, backend_options=backend_options)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    """Shallow text refiner run once per request before packing into the DiT stream."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(arch) for _ in range(arch.token_refiner_num_layers)])
        self.final_norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.attention_backend: str | None = None

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        for block in self.blocks:
            block.set_attention_backend(backend, backend_options=backend_options)
        self.attention_backend = self.blocks[0].attention_backend if self.blocks else backend

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, cu_seqlens_host=cu_seqlens_host)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# DiT block (AdaLN-modulated attention + MLP with gated residuals).
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTBlock(nn.Module):
    """AdaLN-modulated attention + MLP with per-row gated residuals.

    ``combined_indices`` select among ``num_unique_timesteps × 3`` modality
    vectors (video / audio / text) produced by :class:`MiniMaxH3AdalnProj`.
    """

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    @property
    def attention_backend(self) -> str | None:
        return self.attn.attention_backend

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.attn.set_attention_backend(backend, backend_options=backend_options)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        ffn_conditioner: Any = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = rms_norm_indexed_scale_shift(x, self.norm1, shift_msa, scale_msa, combined_indices, dtype=x.dtype)
        if attention_groups is None:
            h = self.attn(
                h,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
            )
        else:
            h = self.attn(
                h,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                attention_groups=attention_groups,
            )
        x = indexed_gate(residual, gate_msa, h, combined_indices, dtype=x.dtype)

        residual = x
        if ffn_conditioner is not None:
            x = ffn_conditioner(x)
        h = rms_norm_indexed_scale_shift(x, self.norm2, shift_mlp, scale_mlp, combined_indices, dtype=x.dtype)
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices, dtype=x.dtype)


# --------------------------------------------------------------------------- #
# Final layer (single-modality AdaLN -> fp32 -> dual video/audio heads).
# --------------------------------------------------------------------------- #
class MiniMaxH3FinalLayer(nn.Module):
    """Final AdaLN (timestep-only) then fp32 video-patch and audio heads."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = nn.Linear(arch.hidden_size, video_patch_dim, bias=True, dtype=_FP32)
        self.audio_out = nn.Linear(arch.hidden_size, arch.audio_latents_dim, bias=True, dtype=_FP32)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = rms_norm_indexed_scale_shift(x, self.norm, shift, scale, inverse_indices, dtype=x.dtype)
        h = h.to(_FP32)  # full precision through both output projections
        return self.video_out(h), self.audio_out(h)


# --------------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTModel(nn.Module):
    """Single-GPU MiniMax H3 packed-token audio-video DiT."""

    def __init__(self, config: MiniMaxH3DiTArchConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        arch = config if config is not None else MiniMaxH3DiTArchConfig(**overrides)
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self.attention_backend: str | None = None
        self._block_graph_runner: Any = None
        self._piecewise_graph: Any = None
        self._piecewise_segment_names: tuple[str, ...] = ()
        self._piecewise_segment_bounds: dict[str, tuple[int, int]] = {}
        self._dense_initial_steps: int = 0
        self._graph_captured_receipts: dict[tuple[int, int, bool], dict[str, Any]] = {}
        self._graph_replayed_window: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
        }
        self._graph_replayed_lifetime: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
        }
        self._acceleration_metadata: dict[str, Any] = {}

        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.video_patch_proj = nn.Linear(video_patch_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.audio_patch_proj = nn.Linear(arch.audio_latents_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.condition_proj = nn.Linear(arch.text_dim, arch.hidden_size, bias=True, dtype=_BF16)
        self.time_embedder = MiniMaxH3TimeEmbedder(arch)
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch) for _ in range(arch.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(arch)

    def invalidate_cuda_graphs(self) -> None:
        """Drop any captured block-stack CUDA graphs after policy or weight changes."""

        if self._block_graph_runner is not None:
            self._block_graph_runner.invalidate()
        if self._piecewise_graph is not None:
            self._piecewise_graph.invalidate()
        self._graph_captured_receipts.clear()

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
        dense_layers: Sequence[int] = (),
        include_refiner: bool = False,
    ) -> None:
        """Configure the attention backend across DiT blocks (``AttentionBackendConfigurable``).

        ``dense_layers`` pins sensitive blocks (e.g. ``(0, 1, -1)`` under sparse
        ``sol_attn``) to exact ``"torch"`` SDPA while routing the rest to
        ``backend``. ``token_refiner`` stays on exact ``"torch"`` SDPA unless
        ``include_refiner=True`` (and resets cleanly if previously enabled).
        """

        self.invalidate_cuda_graphs()
        num_layers = len(self.blocks)
        pinned = {(idx + num_layers if idx < 0 else idx) for idx in dense_layers}
        for idx, block in enumerate(self.blocks):
            if backend is not None and idx in pinned:
                block.set_attention_backend("torch", backend_options=None)
            else:
                block.set_attention_backend(backend, backend_options=backend_options)
        if include_refiner:
            self.token_refiner.set_attention_backend(backend, backend_options=backend_options)
        elif backend is None or self.token_refiner.attention_backend is not None:
            self.token_refiner.set_attention_backend(None, backend_options=None)
        if backend is None:
            self.attention_backend = None
        else:
            from worldfoundry.core.attention.backends.probe import normalize_attention_backend

            self.attention_backend = normalize_attention_backend(str(backend))

    def _snapshot_slice_counters(self, start: int, stop: int) -> tuple[int, int, dict[str, int]]:
        """Snapshot lifetime FP8/FFN/attention-provider counters for one block slice."""

        from worldfoundry.core.attention.backends.dispatch import attention_provider_runtime_report

        from .triton_fp8_swiglu import FusedFP8SwiGLUMLP

        fp8_total = 0
        ffn_total = 0
        for idx in range(start, stop):
            for mod in self.blocks[idx].modules():
                if getattr(mod, "_worldfoundry_quantization_layer", False):
                    fp8_total += int(getattr(mod, "low_precision_kernel_calls", 0))
                elif isinstance(mod, FusedFP8SwiGLUMLP):
                    ffn_total += int(mod.fused_calls)
        prov_report = attention_provider_runtime_report()
        prov_counts = {
            str(name): int(stats.get("successes", 0))
            for name, stats in prov_report.items()
            if isinstance(stats, Mapping)
        }
        return fp8_total, ffn_total, prov_counts

    def _record_graph_replay_receipt(self, key: tuple[int, int, bool]) -> None:
        """Credit baked-in kernel counts for one verified CUDA Graph replay."""

        baked = self._graph_captured_receipts.get(key)
        if not baked:
            return
        for bucket in (self._graph_replayed_window, self._graph_replayed_lifetime):
            bucket["fp8_kernel_calls"] += int(baked.get("fp8_kernel_calls", 0))
            bucket["fused_ffn_calls"] += int(baked.get("fused_ffn_calls", 0))
            prov_dst = bucket["attention_provider_calls"]
            for name, count in baked.get("attention_provider_calls", {}).items():
                prov_dst[name] = prov_dst.get(name, 0) + int(count)

    def reset_acceleration_window(self) -> None:
        """Reset per-request telemetry counters on graphs, FP8 linears, and fused FFNs."""

        from worldfoundry.core.acceleration.quantization.linear import reset_quantization_runtime_window
        from worldfoundry.core.attention.backends.dispatch import reset_attention_provider_runtime

        reset_quantization_runtime_window(self)
        reset_attention_provider_runtime()
        for module in self.modules():
            reset_fn = getattr(module, "reset_request_window", None)
            if callable(reset_fn) and not getattr(module, "_worldfoundry_quantization_layer", False):
                reset_fn()
        if self._block_graph_runner is not None:
            self._block_graph_runner.begin_request_window()
        if self._piecewise_graph is not None:
            self._piecewise_graph.begin_request_window()
        self._graph_replayed_window = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
        }

    def acceleration_report(self) -> dict[str, Any]:
        """Return auditable runtime receipts for quantization, attention, fusion, and graphs."""

        from worldfoundry.core.acceleration.quantization.linear import quantization_runtime_report
        from worldfoundry.core.attention.backends.dispatch import attention_provider_runtime_report

        from .triton_fp8_swiglu import FusedFP8SwiGLUMLP

        fused_ffn_layers = 0
        fused_ffn_calls = 0
        fused_ffn_fallbacks = 0
        for module in self.modules():
            if isinstance(module, FusedFP8SwiGLUMLP):
                fused_ffn_layers += 1
                if module._request_window_active:
                    fused_ffn_calls += int(module.request_fused_calls)
                    fused_ffn_fallbacks += int(module.request_fallback_calls)
                else:
                    fused_ffn_calls += int(module.fused_calls)
                    fused_ffn_fallbacks += int(module.fallback_calls)

        graph_report: Mapping[str, Any] | None = None
        if self._block_graph_runner is not None:
            graph_report = self._block_graph_runner.report()
        elif self._piecewise_graph is not None:
            graph_report = self._piecewise_graph.report()

        block_backends: dict[str, int] = {}
        for block in self.blocks:
            name = str(block.attention_backend or "torch")
            block_backends[name] = block_backends.get(name, 0) + 1

        quant_report = quantization_runtime_report(self)
        replayed_fp8 = int(self._graph_replayed_window["fp8_kernel_calls"])
        replayed_ffn = int(self._graph_replayed_window["fused_ffn_calls"])
        replayed_prov = dict(self._graph_replayed_window["attention_provider_calls"])
        if isinstance(quant_report, dict):
            quant_report = dict(quant_report)
            quant_report["graph_replayed_low_precision_kernel_calls"] = replayed_fp8
            quant_report["effective_low_precision_kernel_calls"] = (
                int(quant_report.get("low_precision_kernel_calls", 0)) + replayed_fp8
            )

        prov_runtime = attention_provider_runtime_report()
        prov_effective: dict[str, int] = {}
        for name, stats in prov_runtime.items():
            if isinstance(stats, Mapping):
                prov_effective[str(name)] = int(stats.get("successes", 0))
        for name, count in replayed_prov.items():
            prov_effective[name] = prov_effective.get(name, 0) + int(count)

        return {
            "attention_backend": self.attention_backend or "torch",
            "block_attention_backends": block_backends,
            "dense_initial_steps": int(self._dense_initial_steps),
            "attention_provider_runtime": prov_runtime,
            "attention_provider_runtime_effective": prov_effective,
            "quantization": quant_report,
            "fused_fp8_swiglu": {
                "layers": fused_ffn_layers,
                "fused_calls": fused_ffn_calls,
                "graph_replayed_fused_calls": replayed_ffn,
                "effective_fused_calls": fused_ffn_calls + replayed_ffn,
                "fallback_calls": fused_ffn_fallbacks,
            },
            "cuda_graph": graph_report,
            "graph_replayed_receipts": {
                "window": {
                    "fp8_kernel_calls": replayed_fp8,
                    "fused_ffn_calls": replayed_ffn,
                    "attention_provider_calls": replayed_prov,
                },
                "lifetime": {
                    "fp8_kernel_calls": int(self._graph_replayed_lifetime["fp8_kernel_calls"]),
                    "fused_ffn_calls": int(self._graph_replayed_lifetime["fused_ffn_calls"]),
                    "attention_provider_calls": dict(self._graph_replayed_lifetime["attention_provider_calls"]),
                },
            },
            "metadata": dict(self._acceleration_metadata),
        }

    # -- weight-dtype guard ------------------------------------------------- #
    def post_load_weights(self) -> None:
        """Assert the mixed fp32/bf16 parameter split survived weight loading."""

        for name in _MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER:
            param = self.get_parameter(name)
            if param.dtype != _FP32:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        if self.rope.inv_freq.dtype != _FP32:
            raise ValueError(f"rope.inv_freq must stay fp32 after load, got {self.rope.inv_freq.dtype}.")

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        ids = pos_info.get("position_ids") if isinstance(pos_info, dict) else getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        value = psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    @staticmethod
    def _psp_optional_field(psp: Any, field: str) -> Any:
        return psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
        refiner_cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Project and refine request-static text conditioning once."""

        if refiner_cu_seqlens.ndim != 1 or int(refiner_cu_seqlens.numel()) < 2:
            raise ValueError(
                f"refiner cu_seqlens must be 1-D with at least 2 entries, got {list(refiner_cu_seqlens.shape)}"
            )
        bounds = (
            list(refiner_cu_seqlens_host)
            if refiner_cu_seqlens_host is not None
            else [int(value) for value in refiner_cu_seqlens.tolist()]
        )
        if bounds[0] != 0 or any(right < left for left, right in itertools.pairwise(bounds)):
            raise ValueError(f"refiner cu_seqlens must start at 0 and be non-decreasing, got {bounds}")
        text_len = bounds[-1]
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError(
                f"refiner cu_seqlens live text length must be in [1, {int(prompt_embeds.shape[0])}], got {text_len}"
            )
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=self.condition_proj.weight.dtype)
        # Preserve caller-provided document boundaries (single document [0, text_len]
        # or per-segment bounds such as H3-World's isolated action annotations).
        refiner_cu = refiner_cu_seqlens.to(device=device, dtype=torch.int32)
        text_embed = self.condition_proj(text_rows)
        return self.token_refiner(text_embed, cu_seqlens=refiner_cu, cu_seqlens_host=tuple(bounds))

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the request-static RoPE cos/sin cache and position index."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        seq_len = int(img_position_ids.shape[1])
        rope_freqs = self.rope(img_position_ids).to(device)
        return (
            rope_cos_sin_cache(rope_freqs, dtype=self.condition_proj.weight.dtype),
            torch.arange(seq_len, device=device, dtype=torch.long),
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None = None,
        refiner_cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the packed embedding sequence [S, H] bf16 and t_emb [M, t_dim] fp32."""

        if refined_prompt_embeds_length is None:
            text_len = (
                int(refiner_cu_seqlens_host[-1])
                if refiner_cu_seqlens_host is not None
                else int(refiner_cu_seqlens[-1].item())
            )
        elif torch.is_tensor(refined_prompt_embeds_length):
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(text_embeddings_selected.shape[0])}], got {text_len}"
            )
        text_pos = text_pos[:text_len]
        dtype = self.condition_proj.weight.dtype
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(device=device, dtype=dtype)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "refined prompt embeddings must have hidden width "
                    f"{self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                text_embeddings_selected,
                refiner_cu_seqlens,
                device=device,
                refiner_cu_seqlens_host=refiner_cu_seqlens_host,
            )

        seq_len = int(x.shape[1])
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)

        if text_pos.numel():
            embeddings.index_add_(0, text_pos, text_embed[:text_len].to(dtype))

        # Latent embedders stay fp32; only scatter into their packed positions.
        if img_pos.numel():
            x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(_FP32)
            video_embed = self.video_patch_proj(x_rows)
            embeddings.index_add_(0, img_pos, video_embed.to(dtype))

        if audio_pos.numel():
            audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(_FP32)
            audio_embed = self.audio_patch_proj(audio_rows)
            embeddings.index_add_(0, audio_pos, audio_embed.to(dtype))

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward returning ``(video_logits, audio_logits)``."""

        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: {sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")
        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = self._pos_ids(_required_kwarg(kwargs, "text_pos_info"), "text_pos_info")
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"), "img_pos_for_infer_output_info"
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        cu_seqlens_host = _normalize_host_bounds(self._psp_optional_field(psp, "cu_seqlens_q_host"))
        raw_max_seqlen = self._psp_optional_field(psp, "max_seqlen_q")
        max_seqlen = None if raw_max_seqlen is None else int(raw_max_seqlen)
        ignore_trailing_pad = bool(self._psp_optional_field(psp, "ignore_trailing_pad"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_cu_host = _normalize_host_bounds(self._psp_optional_field(refiner_psp, "cu_seqlens_q_host"))

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device

        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids).to(device)
            rope_cache = (
                rope_cos_sin_cache(rope_freqs, dtype=self.condition_proj.weight.dtype),
                torch.arange(seq_len, device=device, dtype=torch.long),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        adaln_input = F.silu(t_emb).to(self.condition_proj.weight.dtype)
        inverse_indices = inverse_indices.to(device)
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags.clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != seq_len:
                raise ValueError(
                    f"block_token_tags must cover the packed sequence ({seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(block_token_tags, inverse_indices, alpha=MINIMAX_H3_ADALN_MODALITY_NUM)

        prefix_sink_tokens = _compute_prefix_sink_tokens(
            text_pos=text_pos,
            img_pos=img_pos,
            infer_out_pos=infer_out_pos,
            audio_pos=audio_pos,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        force_dense_attention = bool(kwargs.get("force_dense_attention", False))
        video_logits, audio_logits = self.forward_blocks(
            decoder_input,
            adaln_input=adaln_input,
            combined_indices=block_combined,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens.to(device),
            inverse_indices=inverse_indices,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
        )

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits

    def _run_block_slice(
        self,
        start: int,
        stop: int,
        hidden: torch.Tensor,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int | None,
        ignore_trailing_pad: bool,
        prefix_sink_tokens: int | None,
        force_dense_attention: bool = False,
    ) -> torch.Tensor:
        """Pure-tensor block slice callable for whole/piecewise CUDA Graph capture."""

        capturing = hidden.is_cuda and torch.cuda.is_current_stream_capturing()
        before_snapshot = self._snapshot_slice_counters(start, stop) if capturing else None
        rope_cache = (cos_sin_cache, positions)
        for index in range(start, stop):
            hidden = self.blocks[index](
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
            )
        if before_snapshot is not None:
            fp8_after, ffn_after, prov_after = self._snapshot_slice_counters(start, stop)
            fp8_before, ffn_before, prov_before = before_snapshot
            prov_delta = {
                name: count - prov_before.get(name, 0)
                for name, count in prov_after.items()
                if count > prov_before.get(name, 0)
            }
            self._graph_captured_receipts[(start, stop, bool(force_dense_attention))] = {
                "fp8_kernel_calls": max(0, fp8_after - fp8_before),
                "fused_ffn_calls": max(0, ffn_after - ffn_before),
                "attention_provider_calls": prov_delta,
            }
        return hidden

    def _run_transformer_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
    ) -> torch.Tensor:
        """Execute ``self.blocks`` eagerly or via whole/piecewise CUDA Graph replay."""

        # Resolve host bounds once outside graph capture/replay so no block
        # ever triggers a GPU->CPU synchronization inside a captured graph.
        if (
            (self._block_graph_runner is not None or self._piecewise_graph is not None)
            and cu_seqlens_host is None
            and max_seqlen is None
        ):
            cu_seqlens_host = tuple(int(v) for v in cu_seqlens.tolist())

        cos_sin_cache, positions = rope_cache
        if self._block_graph_runner is not None:
            prev_replays = int(self._block_graph_runner.stats["replay"])
            out = self._block_graph_runner(
                hidden,
                adaln_input,
                combined_indices,
                cos_sin_cache,
                positions,
                cu_seqlens,
                cu_seqlens_host,
                max_seqlen,
                ignore_trailing_pad,
                prefix_sink_tokens,
                bool(force_dense_attention),
            )
            if int(self._block_graph_runner.stats["replay"]) > prev_replays:
                self._record_graph_replay_receipt((0, len(self.blocks), bool(force_dense_attention)))
            return out
        if self._piecewise_graph is not None:
            for seg_name in self._piecewise_segment_names:
                seg_runner = self._piecewise_graph._segments[seg_name]
                prev_replays = int(seg_runner.stats["replay"])
                hidden = self._piecewise_graph.run_segment(
                    seg_name,
                    hidden,
                    adaln_input,
                    combined_indices,
                    cos_sin_cache,
                    positions,
                    cu_seqlens,
                    cu_seqlens_host,
                    max_seqlen,
                    ignore_trailing_pad,
                    prefix_sink_tokens,
                    bool(force_dense_attention),
                )
                if int(seg_runner.stats["replay"]) > prev_replays:
                    bounds = self._piecewise_segment_bounds.get(seg_name)
                    if bounds is not None:
                        self._record_graph_replay_receipt((bounds[0], bounds[1], bool(force_dense_attention)))
            return hidden

        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
            )
        return hidden

    def forward_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        inverse_indices: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the packed residual stream through the shared blocks and output heads."""

        hidden = self._run_transformer_blocks(
            hidden,
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
        )
        return self.final_layer(hidden, adaln_input=adaln_input, inverse_indices=inverse_indices)


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "reorder_grouped_qkv_to_qkv",
]
