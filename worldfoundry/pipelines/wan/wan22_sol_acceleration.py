# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for Wan2.2 TI2V-5B and Wan2.2 T2V/I2V-A14B.

Non-invasively composes WorldFoundry's native Wan acceleration seams without
modifying ``networks/wan/model.py`` or ``denoisers/wan.py``:

1. **Kernel Fusion & Hopper FA3**:
   - Fused QKV projections via :func:`fuse_qkv_projections` with processor-owned
     dispatch (:func:`restore_fused_qkv_processor_dispatch`).
   - Compile-safe FP64 Triton QK-RMSNorm + 3D RoPE custom ops
     (``worldfoundry::wan22_hidden_qk_rmsnorm_rope_3d`` and
     ``worldfoundry::wan22_hidden_qk_rmsnorm_rope_freqs``) preserving FP64
     rotary precision with zero TorchDynamo graph breaks.
   - Triton AdaLN / residual fusion (``_worldfoundry_fused_adaln``).
   - Compile-safe Hopper FlashAttention-3 custom op
     (``worldfoundry::wan22_fa3_attn``) across dense self- and cross-attention.
   - Static Cross-KV caching (:func:`install_static_cross_kv_cache`) with
     denoiser-level pre-resolution (``_prepare_cross_kv``) outside compiled and
     graphed DiT blocks.
   - Wan VideoVAE ``channels_last_3d`` + BF16 resident weights + BF16 decode
     autocast.

2. **Sparse Attention (In-Tree PISA & 3D-Sol-Attn)**:
   - In-tree Hopper TMA PISA (:func:`piecewise_attention`) wired into Wan2.2
     self-attention blocks with configurable density, block size, dense edge
     blocks ``(0, 1, -1)``, dense warmup/tail steps, optional ``(1, 8, 8)`` 3D
     spatial patch tiling, and strict or audited fallback semantics.
   - Direct ``3D-Sol-Attn`` (:func:`sol_attention`) parity path for apples-to-apples
     comparison against PISA.

3. **Calibrated Single- & Dual-Expert EasyCache**:
   - Strided online transform-vector reuse with warmup, dense-last, and
     max-consecutive-skip guards plus optional CFG=2 lockstep synchronization.
   - Explicit expert-boundary cache reset on Wan2.2-A14B so high-noise cached
     residuals are wiped immediately on transition to the low-noise expert and
     both experts run independent local warmup/tail windows.

4. **Regional ``torch.compile`` & Piecewise CUDA Graph + Optional Rowwise FP8**:
   - Regional compilation of ``DiTBlock`` modules (zero graph breaks / zero
     recompiles across steps) and optional :class:`PiecewiseBlockGraphRunner`.
   - Optional Hopper rowwise FP8 W8A8 (strictly enforcing
     ``use_fast_accum=False`` and ``keep_dense_fallback=False``) with fused
     Triton GELU+FP8 FFN.
"""

from __future__ import annotations

import json
import math
import types
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from threading import Lock, RLock
from typing import Any, Literal, cast
from weakref import WeakValueDictionary

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    AttentionModule,
    CrossAttention,
    CrossAttentionProcessor,
    SelfAttention,
    WanModel,
    apply_wan_qk_norm_rope,
)
from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
    FusedRoPERuntimeState,
    install_fused_rope_runtime,
    record_compiled_fused_rope_graph_trace,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    fuse_qkv_projections,
    project_fused_qkv,
    qkv_fusion_report,
    restore_fused_qkv_processor_dispatch,
)
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    StaticCrossKVProcessor,
    install_static_cross_kv_cache,
)
from worldfoundry.base_models.diffusion_model.optimizations.wan.expert_schedule import WanExpertSchedule
from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.acceleration.quantization.fused_ffn import (
    FusedFP8GELUFeedForward,
    fuse_fp8_gelu_feed_forwards,
)
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    quantization_runtime_report,
    replace_linear_with_float8,
    set_low_precision_enabled,
)
from worldfoundry.core.attention.backends.native import native_sdpa_priority, scaled_dot_product_attention
from worldfoundry.core.attention.backends.probe import attention_backend_capability
from worldfoundry.core.attention.backends.sol import sol_attention
from worldfoundry.core.attention.sparse.piecewise import piecewise_attention, piecewise_attention_available
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

_MISSING = object()

_DEFAULT_FP8_EXCLUDES: tuple[str, ...] = (
    "time_embedding",
    "time_projection",
    "text_embedding",
    "head",
    "img_emb",
    "cross_attn",
)

# =============================================================================
# 1. Compile-Safe Hopper FA3 & FP64 Triton QK-RMSNorm+3D-RoPE Custom Ops
# =============================================================================


@torch.library.custom_op("worldfoundry::wan22_fa3_attn", mutates_args=(), device_types="cuda")
def _wan22_fa3_attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import flash_attn_interface as fa3

    out = fa3.flash_attn_func(q, k, v, causal=False)
    return out[0] if isinstance(out, tuple) else out


@_wan22_fa3_attn_op.register_fake
def _wan22_fa3_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    del k
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def wan22_fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, num_heads: int) -> torch.Tensor:
    """Run compile-safe Hopper FlashAttention-3 on packed ``[B, S, H*D]`` tensors."""
    batch, seq_q, hidden = q.shape
    head_dim = hidden // int(num_heads)
    q_4d = q.view(batch, seq_q, int(num_heads), head_dim).contiguous()
    k_4d = k.view(batch, k.shape[1], int(num_heads), head_dim).contiguous()
    v_4d = v.view(batch, v.shape[1], int(num_heads), head_dim).contiguous()
    out_4d = _wan22_fa3_attn_op(q_4d, k_4d, v_4d)
    return out_4d.reshape(batch, seq_q, hidden)


@torch.library.custom_op("worldfoundry::wan22_hidden_qk_rmsnorm_rope_3d", mutates_args=(), device_types="cuda")
def _wan22_hidden_qk_rmsnorm_rope_3d_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_table: torch.Tensor,
    num_heads: int,
    frames: int,
    height: int,
    width: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from worldfoundry.core.kernels import triton_diffusion

    return triton_diffusion.hidden_qk_rmsnorm_rope_3d(
        q,
        k,
        q_weight,
        k_weight,
        freqs_table,
        num_heads=int(num_heads),
        grid_size=(int(frames), int(height), int(width)),
        eps=float(eps),
        sequence_offset=0,
        start_frame=0,
        valid_tokens=int(frames) * int(height) * int(width),
        head_start=0,
        head_end=int(num_heads),
    )


@_wan22_hidden_qk_rmsnorm_rope_3d_op.register_fake
def _wan22_hidden_qk_rmsnorm_rope_3d_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_table: torch.Tensor,
    num_heads: int,
    frames: int,
    height: int,
    width: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, freqs_table, num_heads, frames, height, width, eps
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("worldfoundry::wan22_hidden_qk_rmsnorm_rope_freqs", mutates_args=(), device_types="cuda")
def _wan22_hidden_qk_rmsnorm_rope_freqs_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    eps: float,
    use_fp64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from worldfoundry.core.kernels import triton_diffusion

    return triton_diffusion.hidden_qk_rmsnorm_rope_freqs(
        q,
        k,
        q_weight,
        k_weight,
        freqs,
        num_heads=int(num_heads),
        eps=float(eps),
        precision="fp64" if bool(use_fp64) else "fp32",
    )


@_wan22_hidden_qk_rmsnorm_rope_freqs_op.register_fake
def _wan22_hidden_qk_rmsnorm_rope_freqs_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    eps: float,
    use_fp64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, freqs, num_heads, eps, use_fp64
    return torch.empty_like(q), torch.empty_like(k)


def _sdpa_packed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, num_heads: int) -> torch.Tensor:
    batch, seq_q, hidden = q.shape
    head_dim = hidden // int(num_heads)
    q_4d = q.view(batch, seq_q, int(num_heads), head_dim).transpose(1, 2)
    k_4d = k.view(batch, k.shape[1], int(num_heads), head_dim).transpose(1, 2)
    v_4d = v.view(batch, v.shape[1], int(num_heads), head_dim).transpose(1, 2)
    if torch.compiler.is_compiling():
        out_4d = F.scaled_dot_product_attention(q_4d, k_4d, v_4d)
    else:
        out_4d = scaled_dot_product_attention(
            q_4d,
            k_4d,
            v_4d,
            backends=native_sdpa_priority(q.device),
        )
    return out_4d.transpose(1, 2).reshape(batch, seq_q, hidden)


def _resolve_fa3_usable(device: torch.device | None, sm_major: int | None = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    if sm_major is not None and int(sm_major) < 9:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(device) if device is not None else torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError, ValueError):
        return False
    if cap[0] < 9:
        return False
    return bool(attention_backend_capability("flash_attention_3", device).usable)


def _resolve_sol_attn_usable(device: torch.device | None, sm_major: int | None = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    if sm_major is not None and int(sm_major) < 8:
        return False
    if not torch.cuda.is_available():
        return False
    return bool(attention_backend_capability("sol_attn", device).usable)


# =============================================================================
# 2. Declarative SOL-Engine Configuration & Presets
# =============================================================================


@dataclass(frozen=True, slots=True)
class Wan22SolConfig:
    """Validated SOL-Engine acceleration configuration for Wan2.2 5B & A14B."""

    # Kernel fusion
    fuse_qkv: bool = True
    qkv_strategy: Literal["auto", "packed", "split"] = "auto"
    qkv_split_threshold: int = 8192
    fused_rope: bool = True
    rope_precision: Literal["fp64", "fp32"] = "fp64"
    fused_adaln: bool = True
    inplace_residual: bool = False
    static_cross_kv: bool = True

    # Attention (dense + sparse)
    attention_backend: Literal["flash_attention_3", "torch"] = "flash_attention_3"
    sparse_attention: Literal["none", "pisa", "sol_attn"] = "none"
    pisa_density: float = 0.15
    pisa_block_size: int = 64
    pisa_min_sequence_length: int = 1024
    pisa_dense_steps: int = 4
    pisa_dense_last_steps: int = 2
    pisa_dense_blocks: tuple[int, ...] = (0, 1, -1)
    pisa_tile_3d: tuple[int, int, int] | None = (1, 8, 8)
    pisa_strict: bool = False
    sol_tau: float = 1.0
    sol_thresh_type: Literal["diag", "exact"] = "diag"
    sol_kv_splits: int = 1
    sol_dense_steps: int = 4
    sol_dense_last_steps: int = 2
    sol_dense_blocks: tuple[int, ...] = (0, 1, -1)
    sol_tile_3d: tuple[int, int, int] | None = (1, 8, 8)

    # Calibrated EasyCache (with optional per-expert A14B low-noise overrides)
    easycache: bool = False
    easycache_threshold: float = 0.12
    easycache_warmup_steps: int = 4
    easycache_dense_last: int = 2
    easycache_max_skip_steps: int = 3
    easycache_subsample_stride: int = 64
    easycache_low_noise_threshold: float | None = None
    easycache_low_noise_warmup_steps: int | None = None
    easycache_low_noise_dense_last: int | None = None
    easycache_low_noise_max_skip_steps: int | None = None
    sync_cfg_parallel_cache: bool = True

    # Execution: regional torch.compile & piecewise CUDA Graph
    compile: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    cuda_graph: Literal["off", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True

    # Optional calibrated Rowwise FP8 (strictly use_fast_accum=False)
    fp8: bool = False
    fp8_scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    fp8_use_fast_accum: bool = False
    fp8_fuse_ffn: bool = True
    fp8_keep_dense_fallback: bool = False
    fp8_min_features: int = 1024
    fp8_exclude_edge_blocks: bool = True
    fp8_exclude: tuple[str, ...] = _DEFAULT_FP8_EXCLUDES

    # VAE acceleration
    vae_channels_last_3d: bool = True
    vae_channels_last_2d: bool = True
    vae_decode_autocast: bool = True
    vae_bf16: bool = True
    vae_compile: bool = True

    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.qkv_strategy not in {"auto", "packed", "split"}:
            raise ValueError(f"unsupported qkv_strategy: {self.qkv_strategy!r}")
        if self.qkv_split_threshold <= 0:
            raise ValueError("qkv_split_threshold must be positive")
        if self.rope_precision not in {"fp64", "fp32"}:
            raise ValueError(f"unsupported rope_precision: {self.rope_precision!r}")
        if self.attention_backend not in {"flash_attention_3", "torch"}:
            raise ValueError(f"unsupported attention_backend: {self.attention_backend!r}")
        if self.sparse_attention not in {"none", "pisa", "sol_attn"}:
            raise ValueError(f"unsupported sparse_attention: {self.sparse_attention!r}")
        if not (0.0 < float(self.pisa_density) <= 1.0):
            raise ValueError("pisa_density must be in (0, 1]")
        if self.pisa_block_size not in {16, 32, 64, 128}:
            raise ValueError("pisa_block_size must be one of 16, 32, 64, 128")
        if self.pisa_min_sequence_length < 0:
            raise ValueError("pisa_min_sequence_length must be non-negative")
        if self.pisa_dense_steps < 0 or self.pisa_dense_last_steps < 0:
            raise ValueError("pisa dense step counts must be non-negative")
        if self.sol_tau < 0 or not math.isfinite(self.sol_tau):
            raise ValueError("sol_tau must be finite and non-negative")
        if self.sol_kv_splits not in {1, 2, 4}:
            raise ValueError("sol_kv_splits must be 1, 2, or 4")
        if self.sol_dense_steps < 0 or self.sol_dense_last_steps < 0:
            raise ValueError("sol dense step counts must be non-negative")
        if not math.isfinite(self.easycache_threshold) or self.easycache_threshold < 0:
            raise ValueError("easycache_threshold must be finite and non-negative")
        if self.easycache_warmup_steps < 0 or self.easycache_dense_last < 0:
            raise ValueError("easycache boundary step counts must be non-negative")
        if self.easycache_max_skip_steps < 1 or self.easycache_subsample_stride < 1:
            raise ValueError("easycache max_skip_steps and subsample_stride must be >= 1")
        if self.easycache_low_noise_threshold is not None and (
            not math.isfinite(self.easycache_low_noise_threshold) or self.easycache_low_noise_threshold < 0
        ):
            raise ValueError("easycache_low_noise_threshold must be finite and non-negative")
        if self.easycache_low_noise_warmup_steps is not None and self.easycache_low_noise_warmup_steps < 0:
            raise ValueError("easycache_low_noise_warmup_steps must be non-negative")
        if self.easycache_low_noise_dense_last is not None and self.easycache_low_noise_dense_last < 0:
            raise ValueError("easycache_low_noise_dense_last must be non-negative")
        if self.easycache_low_noise_max_skip_steps is not None and self.easycache_low_noise_max_skip_steps < 1:
            raise ValueError("easycache_low_noise_max_skip_steps must be >= 1")
        if self.cuda_graph not in {"off", "piecewise"}:
            raise ValueError(f"unsupported cuda_graph mode: {self.cuda_graph!r}")
        if self.sparse_attention == "sol_attn" and self.compile:
            raise ValueError("sol_attn cannot be combined with torch.compile; use cuda_graph='piecewise' instead")
        if self.sparse_attention == "pisa" and self.compile:
            raise ValueError("pisa TMA kernels should run eager or with cuda_graph='piecewise', not torch.compile")
        if self.fp8 and self.fp8_use_fast_accum:
            raise ValueError(
                "Hopper FP8 requires use_fast_accum=False (FP32 accumulation) to prevent deep-DiT norm drift"
            )
        for name in ("pisa_tile_3d", "sol_tile_3d"):
            tile = getattr(self, name)
            if tile is not None and (len(tile) != 3 or any(int(v) <= 0 for v in tile)):
                raise ValueError(f"{name} must be a 3-tuple of positive integers or None")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pisa_dense_blocks"] = list(self.pisa_dense_blocks)
        payload["sol_dense_blocks"] = list(self.sol_dense_blocks)
        if self.pisa_tile_3d is not None:
            payload["pisa_tile_3d"] = list(self.pisa_tile_3d)
        if self.sol_tile_3d is not None:
            payload["sol_tile_3d"] = list(self.sol_tile_3d)
        payload["fp8_exclude"] = list(self.fp8_exclude)
        payload["notes"] = list(self.notes)
        return payload


def wan22_sol_profile(
    preset: str,
    *,
    sm_major: int | None = None,
    **overrides: Any,
) -> Wan22SolConfig:
    """Return a calibrated SOL-Engine preset for Wan2.2-5B or Wan2.2-A14B."""
    normalized = str(preset).strip().lower().replace("-", "_")
    fa3_ok = _resolve_fa3_usable(None, sm_major=sm_major)
    attn_backend: Literal["flash_attention_3", "torch"] = "flash_attention_3" if fa3_ok else "torch"

    presets: dict[str, dict[str, Any]] = {
        "wan22_5b_exact": {
            "fuse_qkv": True,
            "qkv_strategy": "packed",
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": True,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "none",
            "easycache": False,
            "compile": True,
            "compile_mode": "default",
            "cuda_graph": "off",
            "fp8": False,
            "notes": (f"Wan2.2-5B L1: lossless QKV+FP64-RoPE+AdaLN+StaticXKV+{attn_backend} + regional torch.compile",),
        },
        "wan22_5b_sol": {
            "fuse_qkv": True,
            "qkv_strategy": "packed",
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": False,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "none",
            "easycache": True,
            "easycache_threshold": 0.24,
            "easycache_warmup_steps": 4,
            "easycache_dense_last": 2,
            "easycache_max_skip_steps": 5,
            "easycache_subsample_stride": 64,
            "compile": True,
            "compile_mode": "default",
            "cuda_graph": "off",
            "fp8": False,
            "notes": (
                f"Wan2.2-5B L2: EasyCache(0.24, dense_last=2) + QKV+FP64-RoPE+AdaLN+StaticXKV+{attn_backend} + regional compile",
            ),
        },
        "wan22_5b_sol_fp8": {
            "fuse_qkv": True,
            "qkv_strategy": "packed",
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": False,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "none",
            "easycache": True,
            "easycache_threshold": 0.22,
            "easycache_warmup_steps": 4,
            "easycache_dense_last": 2,
            "easycache_max_skip_steps": 5,
            "easycache_subsample_stride": 64,
            "compile": True,
            "compile_mode": "default",
            "cuda_graph": "off",
            "fp8": True,
            "fp8_scaling": "rowwise",
            "fp8_use_fast_accum": False,
            "fp8_fuse_ffn": True,
            "fp8_keep_dense_fallback": False,
            "fp8_exclude_edge_blocks": True,
            "notes": (
                f"Wan2.2-5B L2+FP8: EasyCache(0.22, dense_last=2) + Rowwise FP8(fast_accum=False) + {attn_backend} + regional compile",
            ),
        },
        "wan22_a14b_exact": {
            "fuse_qkv": True,
            "qkv_strategy": "auto",
            "qkv_split_threshold": 8192,
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": True,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "none",
            "easycache": False,
            "compile": False,
            "cuda_graph": "piecewise",
            "fp8": False,
            "notes": (f"Wan2.2-A14B L1: dual-expert lossless QKV+FP64-RoPE+AdaLN+StaticXKV+{attn_backend}",),
        },
        "wan22_a14b_sol": {
            "fuse_qkv": True,
            "qkv_strategy": "auto",
            "qkv_split_threshold": 8192,
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": False,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "pisa",
            "pisa_density": 0.15,
            "pisa_block_size": 64,
            "pisa_min_sequence_length": 1024,
            "pisa_dense_steps": 5,
            "pisa_dense_last_steps": 2,
            "pisa_dense_blocks": (0, 1, -1),
            "pisa_tile_3d": (1, 8, 8),
            "pisa_strict": False,
            "easycache": True,
            "easycache_threshold": 0.40,
            "easycache_warmup_steps": 4,
            "easycache_dense_last": 1,
            "easycache_max_skip_steps": 3,
            "easycache_low_noise_threshold": 0.30,
            "easycache_low_noise_warmup_steps": 3,
            "easycache_low_noise_dense_last": 2,
            "easycache_low_noise_max_skip_steps": 2,
            "easycache_subsample_stride": 64,
            "compile": False,
            "cuda_graph": "piecewise",
            "fp8": False,
            "notes": (
                f"Wan2.2-A14B L2: dual-expert fusion + {attn_backend} + PISA(0.15, 1x8x8, edge 0/1/-1) "
                "+ boundary-reset EasyCache(0.40/0.30) + piecewise graph",
            ),
        },
        "wan22_a14b_sol_attn": {
            "fuse_qkv": True,
            "qkv_strategy": "auto",
            "qkv_split_threshold": 8192,
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": False,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "sol_attn",
            "sol_tau": 1.0,
            "sol_thresh_type": "diag",
            "sol_kv_splits": 1,
            "sol_dense_steps": 5,
            "sol_dense_last_steps": 2,
            "sol_dense_blocks": (0, 1, -1),
            "sol_tile_3d": (1, 8, 8),
            "easycache": True,
            "easycache_threshold": 0.40,
            "easycache_warmup_steps": 4,
            "easycache_dense_last": 1,
            "easycache_max_skip_steps": 3,
            "easycache_low_noise_threshold": 0.30,
            "easycache_low_noise_warmup_steps": 3,
            "easycache_low_noise_dense_last": 2,
            "easycache_low_noise_max_skip_steps": 2,
            "easycache_subsample_stride": 64,
            "compile": False,
            "cuda_graph": "piecewise",
            "fp8": False,
            "notes": (
                f"Wan2.2-A14B L2 (3D-Sol-Attn): dual-expert fusion + {attn_backend} + Sol-Attn(tau=1.0, 1x8x8) "
                "+ boundary-reset EasyCache(0.40/0.30) + piecewise graph",
            ),
        },
        "wan22_a14b_sol_fp8": {
            "fuse_qkv": True,
            "qkv_strategy": "packed",
            "fused_rope": True,
            "rope_precision": "fp64",
            "fused_adaln": True,
            "inplace_residual": False,
            "static_cross_kv": True,
            "attention_backend": attn_backend,
            "sparse_attention": "pisa",
            "pisa_density": 0.15,
            "pisa_block_size": 64,
            "pisa_min_sequence_length": 1024,
            "pisa_dense_steps": 5,
            "pisa_dense_last_steps": 2,
            "pisa_dense_blocks": (0, 1, -1),
            "pisa_tile_3d": (1, 8, 8),
            "easycache": True,
            "easycache_threshold": 0.36,
            "easycache_warmup_steps": 4,
            "easycache_dense_last": 1,
            "easycache_max_skip_steps": 3,
            "easycache_low_noise_threshold": 0.28,
            "easycache_low_noise_warmup_steps": 3,
            "easycache_low_noise_dense_last": 2,
            "easycache_low_noise_max_skip_steps": 2,
            "compile": False,
            "cuda_graph": "piecewise",
            "fp8": True,
            "fp8_scaling": "rowwise",
            "fp8_use_fast_accum": False,
            "fp8_fuse_ffn": True,
            "fp8_keep_dense_fallback": False,
            "fp8_exclude_edge_blocks": True,
            "notes": (
                "Wan2.2-A14B L2+FP8: dual-expert fusion + PISA(0.15) + EasyCache + Rowwise FP8(fast_accum=False)",
            ),
        },
    }
    aliases = {
        "5b_exact": "wan22_5b_exact",
        "5b_sol": "wan22_5b_sol",
        "5b_sol_fp8": "wan22_5b_sol_fp8",
        "a14b_exact": "wan22_a14b_exact",
        "a14b_sol": "wan22_a14b_sol",
        "a14b_sol_attn": "wan22_a14b_sol_attn",
        "a14b_sol_fp8": "wan22_a14b_sol_fp8",
    }
    resolved_key = aliases.get(normalized, normalized)
    if resolved_key not in presets:
        allowed = ", ".join(sorted(presets))
        raise ValueError(f"unsupported Wan2.2 SOL preset {preset!r}; expected one of: {allowed}")
    params = dict(presets[resolved_key])
    params.update(overrides)
    return Wan22SolConfig(**params)


# =============================================================================
# 3. Compile-Safe Telemetry Registry & Attention Processors
# =============================================================================


@dataclass
class Wan22AttentionCounters:
    """Shared per-expert attention, RoPE & FP8 runtime counters."""

    fa3_self_calls: int = 0
    fa3_cross_calls: int = 0
    sdpa_self_calls: int = 0
    sdpa_cross_calls: int = 0
    pisa_calls: int = 0
    pisa_fallback_calls: int = 0
    sol_attn_calls: int = 0
    sol_attn_fallback_calls: int = 0
    triton_rope_calls: int = 0
    torch_rope_calls: int = 0
    compiled_fa3_self_traces: int = 0
    compiled_fa3_cross_traces: int = 0
    compiled_rope_traces: int = 0
    compiled_fp8_traces: int = 0
    fa3_graph_replay_blocks: int = 0
    sdpa_self_graph_replay_blocks: int = 0
    sdpa_cross_graph_replay_blocks: int = 0
    pisa_graph_replay_blocks: int = 0
    pisa_fallback_graph_replay_blocks: int = 0
    sol_attn_graph_replay_blocks: int = 0
    sol_attn_fallback_graph_replay_blocks: int = 0
    cross_fa3_graph_replay_blocks: int = 0
    rope_graph_replay_blocks: int = 0
    torch_rope_graph_replay_blocks: int = 0
    fp8_graph_replay_calls: int = 0
    fallback_reasons: set[str] = field(default_factory=set)

    def reset_request_window(self) -> None:
        self.fa3_self_calls = 0
        self.fa3_cross_calls = 0
        self.sdpa_self_calls = 0
        self.sdpa_cross_calls = 0
        self.pisa_calls = 0
        self.pisa_fallback_calls = 0
        self.sol_attn_calls = 0
        self.sol_attn_fallback_calls = 0
        self.triton_rope_calls = 0
        self.torch_rope_calls = 0
        self.fa3_graph_replay_blocks = 0
        self.sdpa_self_graph_replay_blocks = 0
        self.sdpa_cross_graph_replay_blocks = 0
        self.pisa_graph_replay_blocks = 0
        self.pisa_fallback_graph_replay_blocks = 0
        self.sol_attn_graph_replay_blocks = 0
        self.sol_attn_fallback_graph_replay_blocks = 0
        self.cross_fa3_graph_replay_blocks = 0
        self.rope_graph_replay_blocks = 0
        self.torch_rope_graph_replay_blocks = 0
        self.fp8_graph_replay_calls = 0
        self.fallback_reasons.clear()

    def snapshot_eager(self) -> dict[str, Any]:
        return {
            "fa3_self_calls": self.fa3_self_calls,
            "fa3_cross_calls": self.fa3_cross_calls,
            "sdpa_self_calls": self.sdpa_self_calls,
            "sdpa_cross_calls": self.sdpa_cross_calls,
            "pisa_calls": self.pisa_calls,
            "pisa_fallback_calls": self.pisa_fallback_calls,
            "sol_attn_calls": self.sol_attn_calls,
            "sol_attn_fallback_calls": self.sol_attn_fallback_calls,
            "triton_rope_calls": self.triton_rope_calls,
            "torch_rope_calls": self.torch_rope_calls,
            "fallback_reasons": frozenset(self.fallback_reasons),
        }

    def diff_from_snapshot(
        self,
        before: Mapping[str, Any],
        *,
        fp8_lp_delta: int = 0,
        divisor: int = 1,
    ) -> dict[str, Any]:
        prev_reasons = cast(frozenset[str], before["fallback_reasons"])
        div = max(int(divisor), 1)
        return {
            "fa3_self": max(self.fa3_self_calls - int(before["fa3_self_calls"]), 0) // div,
            "fa3_cross": max(self.fa3_cross_calls - int(before["fa3_cross_calls"]), 0) // div,
            "sdpa_self": max(self.sdpa_self_calls - int(before["sdpa_self_calls"]), 0) // div,
            "sdpa_cross": max(self.sdpa_cross_calls - int(before["sdpa_cross_calls"]), 0) // div,
            "pisa": max(self.pisa_calls - int(before["pisa_calls"]), 0) // div,
            "pisa_fallback": max(self.pisa_fallback_calls - int(before["pisa_fallback_calls"]), 0) // div,
            "sol_attn": max(self.sol_attn_calls - int(before["sol_attn_calls"]), 0) // div,
            "sol_attn_fallback": max(self.sol_attn_fallback_calls - int(before["sol_attn_fallback_calls"]), 0) // div,
            "triton_rope": max(self.triton_rope_calls - int(before["triton_rope_calls"]), 0) // div,
            "torch_rope": max(self.torch_rope_calls - int(before["torch_rope_calls"]), 0) // div,
            "fp8_lp_calls": max(int(fp8_lp_delta), 0) // div,
            "fallback_reasons": tuple(sorted(self.fallback_reasons - prev_reasons)),
        }

    def trim_capture_warmup_overhead(self, per_call_delta: Mapping[str, Any], *, extra_passes: int) -> None:
        if extra_passes <= 0:
            return
        self.fa3_self_calls = max(self.fa3_self_calls - int(per_call_delta.get("fa3_self", 0)) * extra_passes, 0)
        self.fa3_cross_calls = max(self.fa3_cross_calls - int(per_call_delta.get("fa3_cross", 0)) * extra_passes, 0)
        self.sdpa_self_calls = max(self.sdpa_self_calls - int(per_call_delta.get("sdpa_self", 0)) * extra_passes, 0)
        self.sdpa_cross_calls = max(self.sdpa_cross_calls - int(per_call_delta.get("sdpa_cross", 0)) * extra_passes, 0)
        self.pisa_calls = max(self.pisa_calls - int(per_call_delta.get("pisa", 0)) * extra_passes, 0)
        self.pisa_fallback_calls = max(
            self.pisa_fallback_calls - int(per_call_delta.get("pisa_fallback", 0)) * extra_passes, 0
        )
        self.sol_attn_calls = max(self.sol_attn_calls - int(per_call_delta.get("sol_attn", 0)) * extra_passes, 0)
        self.sol_attn_fallback_calls = max(
            self.sol_attn_fallback_calls - int(per_call_delta.get("sol_attn_fallback", 0)) * extra_passes, 0
        )
        self.triton_rope_calls = max(
            self.triton_rope_calls - int(per_call_delta.get("triton_rope", 0)) * extra_passes, 0
        )
        self.torch_rope_calls = max(self.torch_rope_calls - int(per_call_delta.get("torch_rope", 0)) * extra_passes, 0)

    def apply_captured_delta(self, delta: Mapping[str, Any]) -> None:
        self.fa3_graph_replay_blocks += int(delta.get("fa3_self", 0))
        self.cross_fa3_graph_replay_blocks += int(delta.get("fa3_cross", 0))
        self.sdpa_self_graph_replay_blocks += int(delta.get("sdpa_self", 0))
        self.sdpa_cross_graph_replay_blocks += int(delta.get("sdpa_cross", 0))
        self.pisa_graph_replay_blocks += int(delta.get("pisa", 0))
        self.pisa_fallback_graph_replay_blocks += int(delta.get("pisa_fallback", 0))
        self.sol_attn_graph_replay_blocks += int(delta.get("sol_attn", 0))
        self.sol_attn_fallback_graph_replay_blocks += int(delta.get("sol_attn_fallback", 0))
        self.rope_graph_replay_blocks += int(delta.get("triton_rope", 0))
        self.torch_rope_graph_replay_blocks += int(delta.get("torch_rope", 0))
        self.fp8_graph_replay_calls += int(delta.get("fp8_lp_calls", 0))
        for reason in delta.get("fallback_reasons", ()):
            self.fallback_reasons.add(str(reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fa3_self_calls": self.fa3_self_calls + self.fa3_graph_replay_blocks,
            "fa3_cross_calls": self.fa3_cross_calls + self.cross_fa3_graph_replay_blocks,
            "fa3_total_calls": (
                self.fa3_self_calls
                + self.fa3_cross_calls
                + self.fa3_graph_replay_blocks
                + self.cross_fa3_graph_replay_blocks
            ),
            "sdpa_self_calls": self.sdpa_self_calls + self.sdpa_self_graph_replay_blocks,
            "sdpa_cross_calls": self.sdpa_cross_calls + self.sdpa_cross_graph_replay_blocks,
            "pisa_calls": self.pisa_calls + self.pisa_graph_replay_blocks,
            "pisa_fallback_calls": self.pisa_fallback_calls + self.pisa_fallback_graph_replay_blocks,
            "sol_attn_calls": self.sol_attn_calls + self.sol_attn_graph_replay_blocks,
            "sol_attn_fallback_calls": self.sol_attn_fallback_calls + self.sol_attn_fallback_graph_replay_blocks,
            "triton_rope_calls": self.triton_rope_calls + self.rope_graph_replay_blocks,
            "torch_rope_calls": self.torch_rope_calls + self.torch_rope_graph_replay_blocks,
            "compiled_fa3_self_traces": self.compiled_fa3_self_traces,
            "compiled_fa3_cross_traces": self.compiled_fa3_cross_traces,
            "compiled_rope_traces": self.compiled_rope_traces,
            "compiled_fp8_traces": self.compiled_fp8_traces,
            "fallback_reasons": sorted(self.fallback_reasons),
        }


_COUNTER_REGISTRY: WeakValueDictionary[str, Wan22AttentionCounters] = WeakValueDictionary()
_COUNTER_LOCK = Lock()


def _register_counters(counters: Wan22AttentionCounters) -> str:
    token = str(id(counters))
    with _COUNTER_LOCK:
        _COUNTER_REGISTRY[token] = counters
    return token


@torch.compiler.assume_constant_result
def _record_compiled_trace(token: str, kind: str) -> int:
    with _COUNTER_LOCK:
        counters = _COUNTER_REGISTRY.get(token)
        if counters is None:
            return 0
        if kind == "fa3_self":
            counters.compiled_fa3_self_traces += 1
            return counters.compiled_fa3_self_traces
        if kind == "fa3_cross":
            counters.compiled_fa3_cross_traces += 1
            return counters.compiled_fa3_cross_traces
        if kind == "rope":
            counters.compiled_rope_traces += 1
            return counters.compiled_rope_traces
        if kind == "fp8":
            counters.compiled_fp8_traces += 1
            return counters.compiled_fp8_traces
        return 0


class Wan22SolSelfAttentionProcessor:
    """Fused QKV + FP64 Triton RoPE + FA3 / PISA / 3D-Sol-Attn processor for Wan2.2.

    Declares ``supports_fused_qkv = True`` so WorldFoundry's load-time
    :func:`fuse_qkv_projections` and :func:`restore_fused_qkv_processor_dispatch`
    preserve processor ownership while executing merged QKV projections.
    """

    supports_fused_qkv: bool = True

    def __init__(
        self,
        *,
        block_index: int,
        is_dense_block: bool,
        dense_backend: Literal["flash_attention_3", "torch"],
        sparse_backend: Literal["none", "pisa", "sol_attn"],
        fused_rope: bool,
        rope_precision: Literal["fp64", "fp32"],
        pisa_density: float,
        pisa_block_size: int,
        pisa_min_sequence_length: int,
        pisa_strict: bool,
        sol_tau: float,
        sol_thresh_type: Literal["diag", "exact"],
        sol_kv_splits: int,
        counters: Wan22AttentionCounters,
        counter_token: str,
        step_state: dict[str, Any],
    ) -> None:
        self.block_index = int(block_index)
        self.is_dense_block = bool(is_dense_block)
        self.dense_backend = dense_backend
        self.sparse_backend = sparse_backend
        self.fused_rope = bool(fused_rope)
        self.rope_precision = rope_precision
        self.pisa_density = float(pisa_density)
        self.pisa_block_size = int(pisa_block_size)
        self.pisa_min_sequence_length = int(pisa_min_sequence_length)
        self.pisa_strict = bool(pisa_strict)
        self.sol_tau = float(sol_tau)
        self.sol_thresh_type = sol_thresh_type
        self.sol_kv_splits = int(sol_kv_splits)
        self._counters = counters
        self._counter_token = counter_token
        self._step_state = step_state

    def _apply_qk_norm_rope(
        self,
        attention: SelfAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
        *,
        fused_table: torch.Tensor | None,
        fused_grid: tuple[int, int, int] | None,
        fused_indexed: bool,
        precision: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_triton = (
            self.fused_rope
            and q.device.type == "cuda"
            and k.device == q.device
            and not torch.is_grad_enabled()
            and q.dtype in {torch.bfloat16, torch.float16, torch.float32}
            and q.shape[-1] <= 8192
        )
        runtime_state = getattr(attention, "_worldfoundry_fused_rope_runtime", None)
        rope_token = getattr(attention, "_worldfoundry_fused_rope_token", None)

        grid_matches_seq = (
            fused_table is not None
            and fused_grid is not None
            and q.shape[1] == int(fused_grid[0]) * int(fused_grid[1]) * int(fused_grid[2])
            and max(int(fused_grid[0]), int(fused_grid[1]), int(fused_grid[2])) <= int(fused_table.shape[0])
        )
        can_use_3d = use_triton and grid_matches_seq
        if can_use_3d and fused_table is not None and fused_grid is not None:
            f_dim, h_dim, w_dim = (int(v) for v in fused_grid)
            q_out, k_out = _wan22_hidden_qk_rmsnorm_rope_3d_op(
                q.contiguous(),
                k.contiguous(),
                attention.norm_q.weight.contiguous(),
                attention.norm_k.weight.contiguous(),
                fused_table,
                int(attention.num_heads),
                f_dim,
                h_dim,
                w_dim,
                float(attention.norm_q.eps),
            )
            if torch.compiler.is_compiling():
                _record_compiled_trace(self._counter_token, "rope")
                if runtime_state is not None:
                    record_compiled_fused_rope_graph_trace(rope_token if isinstance(rope_token, str) else runtime_state)
            else:
                self._counters.triton_rope_calls += 1
                if isinstance(runtime_state, FusedRoPERuntimeState):
                    runtime_state.eager_calls += 1
                    runtime_state.provider_calls += 1
                    runtime_state.lifetime_provider_calls += 1
                    runtime_state.provider_paths.add("triton_hidden_qk_rmsnorm_rope_3d")
            return q_out, k_out

        can_use_freqs = (
            use_triton
            and (fused_indexed or (fused_table is not None and not grid_matches_seq))
            and freqs.ndim >= 2
            and int(freqs.shape[0]) == int(q.shape[1])
        )
        if can_use_freqs:
            q_out, k_out = _wan22_hidden_qk_rmsnorm_rope_freqs_op(
                q.contiguous(),
                k.contiguous(),
                attention.norm_q.weight.contiguous(),
                attention.norm_k.weight.contiguous(),
                freqs,
                int(attention.num_heads),
                float(attention.norm_q.eps),
                precision == "fp64",
            )
            if torch.compiler.is_compiling():
                _record_compiled_trace(self._counter_token, "rope")
                if runtime_state is not None:
                    record_compiled_fused_rope_graph_trace(rope_token if isinstance(rope_token, str) else runtime_state)
            else:
                self._counters.triton_rope_calls += 1
                if isinstance(runtime_state, FusedRoPERuntimeState):
                    runtime_state.eager_calls += 1
                    runtime_state.provider_calls += 1
                    runtime_state.lifetime_provider_calls += 1
                    runtime_state.provider_paths.add("triton_hidden_qk_rmsnorm_rope_freqs")
            return q_out, k_out

        if not torch.compiler.is_compiling():
            self._counters.torch_rope_calls += 1
        # If sequence length does not match fused_grid (e.g. prefix/memory tokens),
        # fall back to exact indexed RoPE rather than passing mismatched 3D grid args.
        safe_table = fused_table if grid_matches_seq else None
        safe_grid = fused_grid if grid_matches_seq else None
        return apply_wan_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=safe_table,
            fused_grid=safe_grid,
            fused_indexed=fused_indexed,
            precision=precision,
        )

    def _run_dense_attention(
        self,
        attention: SelfAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        use_fa3 = (
            self.dense_backend == "flash_attention_3"
            and q.device.type == "cuda"
            and q.dtype in {torch.bfloat16, torch.float16}
            and not attention.attn.compatibility_mode
        )
        if use_fa3:
            if torch.compiler.is_compiling():
                _record_compiled_trace(self._counter_token, "fa3_self")
            else:
                self._counters.fa3_self_calls += 1
            return wan22_fa3_attention(q, k, v, num_heads=attention.num_heads)
        if not torch.compiler.is_compiling():
            self._counters.sdpa_self_calls += 1
        return _sdpa_packed(q, k, v, num_heads=attention.num_heads)

    def _run_sparse_attention(
        self,
        attention: SelfAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, hidden = q.shape
        num_heads = int(attention.num_heads)
        head_dim = hidden // num_heads

        if self.sparse_backend == "pisa":
            eligible = (
                not torch.is_grad_enabled()
                and q.device.type == "cuda"
                and q.dtype in {torch.bfloat16, torch.float16}
                and seq_len >= self.pisa_min_sequence_length
                and 0 < head_dim <= 256
                and piecewise_attention_available(q.device)
            )
            if not eligible:
                if self.pisa_strict:
                    raise RuntimeError(
                        f"PISA sparse attention is ineligible on block {self.block_index} "
                        f"(device={q.device}, dtype={q.dtype}, seq={seq_len}, min_seq={self.pisa_min_sequence_length})"
                    )
                self._counters.pisa_fallback_calls += 1
                self._counters.fallback_reasons.add(
                    f"pisa:ineligible(device={q.device.type},dtype={q.dtype},seq={seq_len})"
                )
                return self._run_dense_attention(attention, q, k, v)
            q_bhsd = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
            k_bhsd = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
            v_bhsd = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
            out_bhsd = piecewise_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                density=self.pisa_density,
                block_size=self.pisa_block_size,
                min_sequence_length=self.pisa_min_sequence_length,
                strict=True,
            )
            self._counters.pisa_calls += 1
            return out_bhsd.transpose(1, 2).reshape(batch, seq_len, hidden)

        if self.sparse_backend == "sol_attn":
            eligible = (
                not torch.is_grad_enabled()
                and q.device.type == "cuda"
                and q.dtype == torch.bfloat16
                and head_dim == 128
                and _resolve_sol_attn_usable(q.device)
            )
            if not eligible:
                self._counters.sol_attn_fallback_calls += 1
                self._counters.fallback_reasons.add(
                    f"sol_attn:ineligible(device={q.device.type},dtype={q.dtype},head_dim={head_dim})"
                )
                return self._run_dense_attention(attention, q, k, v)
            q_bshd = q.view(batch, seq_len, num_heads, head_dim).contiguous()
            k_bshd = k.view(batch, seq_len, num_heads, head_dim).contiguous()
            v_bshd = v.view(batch, seq_len, num_heads, head_dim).contiguous()
            sink_tokens = getattr(attention.attn, "_worldfoundry_sol_sink_tokens", None)
            out_bshd = sol_attention(
                q_bshd,
                k_bshd,
                v_bshd,
                tau=self.sol_tau,
                thresh_type=self.sol_thresh_type,
                kv_splits=self.sol_kv_splits,
                allow_multi_split=self.sol_kv_splits > 1,
                sink_tokens=int(sink_tokens) if sink_tokens is not None else None,
            )
            self._counters.sol_attn_calls += 1
            return out_bshd.reshape(batch, seq_len, hidden)

        return self._run_dense_attention(attention, q, k, v)

    def __call__(
        self,
        attention: SelfAttention,
        x: torch.Tensor,
        freqs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hasattr(attention, "qkv"):
            q, k, v = project_fused_qkv(
                cast(nn.Module, attention.qkv),
                x,
                fusion_state=getattr(attention, "_worldfoundry_qkv_fusion_state", None),
                receipt_token=getattr(attention, "_worldfoundry_qkv_receipt_token", None),
            )
        else:
            q = attention.q(x)
            k = attention.k(x)
            v = attention.v(x)

        fused_table = kwargs.pop("_worldfoundry_rope_table", None)
        fused_grid = kwargs.pop("_worldfoundry_rope_grid", None)
        fused_indexed = bool(kwargs.pop("_worldfoundry_fused_rope_indexed", False))
        precision = str(kwargs.pop("_worldfoundry_rope_precision", self.rope_precision))
        sizes = kwargs.pop("_worldfoundry_request_batch_sizes", ())

        q, k = self._apply_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=fused_table,
            fused_grid=fused_grid,
            fused_indexed=fused_indexed,
            precision=precision,
        )

        force_dense = (
            self.sparse_backend == "none"
            or self.is_dense_block
            or bool(getattr(attention.attn, "_worldfoundry_force_dense", False))
            or bool(self._step_state.get("force_dense", False))
        )
        if sizes:
            if not force_dense:
                if self.sparse_backend == "pisa" and self.pisa_strict:
                    raise RuntimeError(
                        f"PISA sparse attention does not support merged request_batch_sizes on block {self.block_index}"
                    )
                if self.sparse_backend == "pisa":
                    self._counters.pisa_fallback_calls += 1
                elif self.sparse_backend == "sol_attn":
                    self._counters.sol_attn_fallback_calls += 1
                self._counters.fallback_reasons.add(f"{self.sparse_backend}:request-batching-dense-fallback")
            output = torch.cat(
                [
                    self._run_dense_attention(attention, qi, ki, vi)
                    for qi, ki, vi in zip(q.split(sizes), k.split(sizes), v.split(sizes), strict=False)
                ]
            )
            return attention.o(output)

        if force_dense:
            output = self._run_dense_attention(attention, q, k, v)
        else:
            output = self._run_sparse_attention(attention, q, k, v)
        return attention.o(output)


def _install_fast_cross_attn_forward(
    attn_module: AttentionModule,
    *,
    dense_backend: Literal["flash_attention_3", "torch"],
    counters: Wan22AttentionCounters,
    counter_token: str,
) -> Callable[..., torch.Tensor]:
    """Install a compile-safe FA3/SDPA forward on one Wan CrossAttention's ``attn``."""
    original_forward = attn_module.forward

    def _fast_cross_forward(
        self: AttentionModule,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        request_batch_sizes: tuple[int, ...] = (),
    ) -> torch.Tensor:
        if request_batch_sizes:
            return torch.cat(
                [
                    _fast_cross_forward(self, qi, ki, vi)
                    for qi, ki, vi in zip(
                        q.split(request_batch_sizes),
                        k.split(request_batch_sizes),
                        v.split(request_batch_sizes),
                        strict=False,
                    )
                ]
            )
        use_fa3 = (
            dense_backend == "flash_attention_3"
            and q.device.type == "cuda"
            and q.dtype in {torch.bfloat16, torch.float16}
            and k.dtype == q.dtype
            and v.dtype == q.dtype
            and not self.compatibility_mode
        )
        if use_fa3:
            if torch.compiler.is_compiling():
                _record_compiled_trace(counter_token, "fa3_cross")
            else:
                counters.fa3_cross_calls += 1
            return wan22_fa3_attention(q, k, v, num_heads=self.num_heads)
        if not torch.compiler.is_compiling():
            counters.sdpa_cross_calls += 1
        return _sdpa_packed(q, k, v, num_heads=self.num_heads)

    attn_module.forward = types.MethodType(_fast_cross_forward, attn_module)  # type: ignore[method-assign]
    return original_forward


# =============================================================================
# 4. Calibrated Wan2.2 EasyCache (Single-Expert, Dual-Expert & Piecewise-Safe)
# =============================================================================


def _sync_cfg_parallel_hit(hit: bool, device: torch.device) -> bool:
    """Optionally intersect EasyCache hit decisions across 2-rank CFG replicas."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return hit
    from worldfoundry.core.distributed.sequence_parallel.runtime import (
        get_cfg_parallel_group,
        get_cfg_parallel_world_size,
    )

    if get_cfg_parallel_world_size() != 2:
        return hit
    cfg_group = get_cfg_parallel_group()
    if cfg_group is None:
        return hit
    flag = torch.tensor([1 if hit else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cfg_group)
    return bool(int(flag.item()) == 1)


class Wan22SolEasyCache:
    """Calibrated online strided EasyCache for Wan2.2 block stacks.

    Supports both:
    - Direct block-stack execution via :meth:`run_blocks` (compatible with
      regional ``torch.compile`` on ``DiTBlock`` modules and bitwise-exact on
      every dense step).
    - Residual-seam execution via :class:`_Wan22PiecewiseEasyCacheBridge`
      (composing cleanly with ``WanModel``'s native ``PiecewiseBlockGraphRunner``
      and ``tile_3d`` permutation).
    - Explicit :meth:`clear_residuals` at Wan2.2-A14B expert-switch boundaries
      so cached residuals are released immediately and never bleed across experts.
    """

    algorithm = "easycache"

    def __init__(
        self,
        *,
        threshold: float = 0.12,
        total_steps: int = 50,
        warmup_steps: int = 4,
        dense_last: int = 2,
        max_skip_steps: int = 3,
        subsample_stride: int = 64,
        eps: float = 1e-8,
        sync_cfg_parallel: bool = False,
        expert_name: str = "main",
        branch: str = "positive",
        request_id: str | None = None,
    ) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Wan22SolEasyCache threshold must be finite and non-negative")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("Wan22SolEasyCache total_steps must be a positive integer")
        if warmup_steps < 0 or dense_last < 0 or max_skip_steps < 1 or subsample_stride < 1:
            raise ValueError("Wan22SolEasyCache step/stride limits are invalid")
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_skip_steps = int(max_skip_steps)
        self.subsample_stride = int(subsample_stride)
        self.eps = float(eps)
        self.sync_cfg_parallel = bool(sync_cfg_parallel)
        self.expert_name = str(expert_name)
        self.branch = str(branch)
        self.request_id = request_id
        self.reset()

    def clear_residuals(self) -> None:
        """Drop all cached tensors immediately while retaining audit events."""
        self._previous_input: torch.Tensor | None = None
        self._dense_input: torch.Tensor | None = None
        self._dense_output: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._factor: torch.Tensor | None = None
        self._accumulator: float = 0.0
        self._skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device, int] | None = None

    def reset(self) -> None:
        """Full reset including event history."""
        self.clear_residuals()
        self._previous_step: int = -1
        self.events: list[BlockCacheEvent] = []

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.detach().reshape(value.shape[0], -1)
        if self.subsample_stride > 1 and flat.shape[1] >= self.subsample_stride:
            flat = flat[:, :: self.subsample_stride]
        return flat.float().clone()

    def _decide(
        self,
        *,
        step: int,
        count: int,
        value: torch.Tensor,
        block_count: int,
    ) -> tuple[bool, str | None, torch.Tensor | None, tuple[tuple[int, ...], torch.dtype, torch.device, int]]:
        layout = (tuple(value.shape), value.dtype, value.device, int(block_count))
        reason: str | None = None
        probe = self._probe(value)
        if torch.is_grad_enabled():
            self.clear_residuals()
            reason = "autograd"
        elif not bool(torch.isfinite(probe).all()):
            self.clear_residuals()
            reason = "nonfinite-input"
        elif self._layout is not None and layout != self._layout:
            self.clear_residuals()
            reason = "layout-change"

        if reason is None and self._previous_step >= 0 and step <= self._previous_step:
            self.clear_residuals()
            reason = "step-restart"

        if reason is None:
            if step < self.warmup_steps:
                reason = "warmup"
            elif step >= max(count - self.dense_last, 0):
                reason = "dense-last"
            elif self.threshold <= 0.0:
                reason = "zero-threshold"
            elif (
                self._residual is None
                or self._factor is None
                or self._previous_input is None
                or self._dense_output is None
            ):
                reason = "seed"
            elif self._skips >= self.max_skip_steps:
                reason = "max-skip"

        if reason is None:
            delta = (probe - cast(torch.Tensor, self._previous_input)).abs().mean(dim=1)
            norm = cast(torch.Tensor, self._dense_output).abs().mean(dim=1).clamp_min(self.eps)
            estimate = float((cast(torch.Tensor, self._factor) * delta / norm).max().item())
            self._accumulator += estimate
            if not math.isfinite(self._accumulator) or self._accumulator >= self.threshold:
                reason = "threshold"

        hit = reason is None
        if self.sync_cfg_parallel:
            synced_hit = _sync_cfg_parallel_hit(hit, value.device)
            if hit and not synced_hit:
                hit = False
                reason = "cfg-sync-dense"
        return hit, reason, probe, layout

    def _record_dense_update(
        self,
        *,
        value: torch.Tensor,
        output: torch.Tensor,
        probe: torch.Tensor | None,
        layout: tuple[tuple[int, ...], torch.dtype, torch.device, int],
    ) -> None:
        if not torch.is_grad_enabled() and probe is not None:
            out_probe = self._probe(output)
            if bool(torch.isfinite(out_probe).all()):
                if self._dense_input is not None and self._dense_output is not None:
                    in_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.eps)
                    self._factor = (out_probe - self._dense_output).abs().mean(dim=1) / in_delta
                self._dense_input = probe
                self._dense_output = out_probe
                self._residual = (output - value).detach().clone()
                self._layout = layout
            else:
                self.clear_residuals()
        self._accumulator = 0.0
        self._skips = 0

    def run_blocks(
        self,
        step: int,
        value: torch.Tensor,
        run_block: Callable[[int, torch.Tensor], torch.Tensor],
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Run or reuse the full DiT block stack (used when Piecewise Graph is off)."""
        count = self.total_steps if total_steps is None else int(total_steps)
        hit, reason, probe, layout = self._decide(
            step=int(step),
            count=count,
            value=value,
            block_count=int(block_count),
        )
        if hit:
            output = value + cast(torch.Tensor, self._residual)
            self._skips += 1
        else:
            output = value
            for index in range(int(block_count)):
                output = run_block(index, output)
            self._record_dense_update(value=value, output=output, probe=probe, layout=layout)

        if not torch.is_grad_enabled():
            self._previous_input = None if reason == "nonfinite-input" else probe
        self._previous_step = int(step)
        indices = tuple(range(int(block_count)))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return output

    def run_residual_seam(
        self,
        *,
        step: int,
        block_input: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Run or reuse the block residual (composes with Piecewise Graph & 3D tiling)."""
        count = self.total_steps if total_steps is None else int(total_steps)
        hit, reason, probe, layout = self._decide(
            step=int(step),
            count=count,
            value=block_input,
            block_count=int(block_count),
        )
        if hit:
            residual = cast(torch.Tensor, self._residual)
            self._skips += 1
        else:
            residual = compute_residual()
            output = block_input + residual
            if not torch.is_grad_enabled() and probe is not None:
                out_probe = self._probe(output)
                if bool(torch.isfinite(out_probe).all()):
                    if self._dense_input is not None and self._dense_output is not None:
                        in_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.eps)
                        self._factor = (out_probe - self._dense_output).abs().mean(dim=1) / in_delta
                    self._dense_input = probe
                    self._dense_output = out_probe
                    self._residual = residual.detach().clone()
                    self._layout = layout
                else:
                    self.clear_residuals()
            self._accumulator = 0.0
            self._skips = 0

        if not torch.is_grad_enabled():
            self._previous_input = None if reason == "nonfinite-input" else probe
        self._previous_step = int(step)
        indices = tuple(range(int(block_count)))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return residual

    @property
    def hits(self) -> int:
        return sum(1 for ev in self.events if ev.hit)

    @property
    def dense_block_calls(self) -> int:
        return sum(ev.dense_block_calls for ev in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(ev.skipped_block_calls for ev in self.events)

    def receipt(self) -> dict[str, Any]:
        total = len(self.events)
        hits = self.hits
        reasons: dict[str, int] = {}
        for ev in self.events:
            reasons[ev.reason] = reasons.get(ev.reason, 0) + 1
        return {
            "algorithm": self.algorithm,
            "expert": self.expert_name,
            "branch": self.branch,
            "request_id": self.request_id,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_skip_steps": self.max_skip_steps,
            "subsample_stride": self.subsample_stride,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "reasons": reasons,
            "residuals_cleared": self._residual is None,
            "fallbacks": [],
        }


class _Wan22PiecewiseEasyCacheBridge:
    """Adapter exposing ``run(...)`` for ``WanModel``'s Piecewise Graph + 3D tile seam."""

    algorithm = "easycache"
    signal_kind = "timestep-modulation"

    def __init__(
        self,
        cache: Wan22SolEasyCache,
        *,
        block_count: int,
        block_input_holder: list[torch.Tensor | None],
    ) -> None:
        self._cache = cache
        self._block_count = int(block_count)
        self._holder = block_input_holder

    def run(
        self,
        step: int,
        cache_signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        block_input = self._holder[0]
        self._holder[0] = None
        probe_tensor = block_input if block_input is not None else cache_signal
        return self._cache.run_residual_seam(
            step=int(step),
            block_input=probe_tensor,
            compute_residual=compute_residual,
            block_count=self._block_count,
            total_steps=total_steps,
        )


# =============================================================================
# 5. VAE Acceleration
# =============================================================================


def _install_fast_wan_vae_decode(
    vae: nn.Module,
    *,
    channels_last_3d: bool,
    compile_decoder: bool,
    is_cuda: bool,
) -> tuple[bool, bool]:
    """Install O(T) list-concat + channels_last_3d + optional compiled decoder on Wan VAE."""
    vae_model = getattr(vae, "model", None)
    if not isinstance(vae_model, nn.Module):
        return False, False
    raw_decoder = getattr(vae_model, "decoder", None)
    if not isinstance(raw_decoder, nn.Module):
        return False, False
    if not hasattr(vae_model, "conv2") or not hasattr(vae_model, "clear_cache"):
        return False, False

    if not hasattr(vae_model, "_worldfoundry_sol_orig_decode"):
        vae_model._worldfoundry_sol_orig_decode = vae_model.decode

    compiled_decoder: Any = None
    compile_applied = False
    if compile_decoder and is_cuda:
        cached = getattr(vae_model, "_worldfoundry_sol_compiled_decoder", None)
        if cached is not None:
            compiled_decoder = cached
        else:
            compiled_decoder = torch.compile(raw_decoder, mode="default", fullgraph=False)
            vae_model._worldfoundry_sol_compiled_decoder = compiled_decoder
        compile_applied = True

    is_vae38 = type(vae_model).__name__ == "VideoVAE38_"
    memory_format = torch.channels_last_3d if channels_last_3d else torch.contiguous_format

    def _fast_wan_vae_decode(self_model: Any, z: torch.Tensor, scale: Any) -> torch.Tensor:
        self_model.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            sc = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / sc[1].view(1, self_model.z_dim, 1, 1, 1) + sc[0].view(1, self_model.z_dim, 1, 1, 1)
        else:
            sc = scale.to(dtype=z.dtype, device=z.device)
            z = z / sc[1] + sc[0]
        z = z.contiguous(memory_format=memory_format)
        x = self_model.conv2(z)
        dec_fn = compiled_decoder if compiled_decoder is not None else self_model.decoder
        chunks: list[torch.Tensor] = []
        for idx in range(x.shape[2]):
            self_model._conv_idx = [0]
            frame = x[:, :, idx : idx + 1, :, :].contiguous(memory_format=memory_format)
            if is_vae38 and idx == 0:
                out_i, self_model._feat_map, self_model._conv_idx = dec_fn(
                    frame,
                    feat_cache=self_model._feat_map,
                    feat_idx=self_model._conv_idx,
                    first_chunk=True,
                )
            else:
                out_i, self_model._feat_map, self_model._conv_idx = dec_fn(
                    frame,
                    feat_cache=self_model._feat_map,
                    feat_idx=self_model._conv_idx,
                )
            chunks.append(out_i)
        out = torch.cat(chunks, dim=2)
        if is_vae38:
            from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import unpatchify

            out = unpatchify(out, patch_size=2)
        self_model.clear_cache()
        return out

    vae_model.decode = types.MethodType(_fast_wan_vae_decode, vae_model)  # type: ignore[method-assign]
    return True, compile_applied


def accelerate_wan22_vae(
    decoder: Any,
    *,
    channels_last_3d: bool = True,
    channels_last_2d: bool = True,
    decode_autocast: bool = True,
    cast_bf16: bool = True,
    compile_decoder: bool = True,
) -> dict[str, Any]:
    """Accelerate Wan2.2's VideoVAE with BF16, channels_last_3d/2d, O(T) cat, and compiled decode."""
    vae = getattr(decoder, "vae", decoder)
    if not isinstance(vae, nn.Module):
        return {
            "enabled": False,
            "channels_last_3d": False,
            "channels_last_2d": False,
            "conv3d_layers_converted": 0,
            "conv2d_layers_converted": 0,
            "vae_bf16": False,
            "decode_autocast": False,
            "vae_compile": False,
            "fallbacks": ["vae:unsupported-module"],
        }
    param = next(vae.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    bf16_applied = False
    if cast_bf16 and is_cuda:
        vae.to(dtype=torch.bfloat16)
        bf16_applied = True
        if hasattr(decoder, "weight_dtype_requested"):
            decoder.weight_dtype_requested = torch.bfloat16

    conv3d_total = 0
    conv3d_converted = 0
    conv2d_total = 0
    conv2d_converted = 0
    if channels_last_3d or channels_last_2d:
        with torch.no_grad():
            for child in vae.modules():
                if channels_last_3d and isinstance(child, nn.Conv3d):
                    conv3d_total += 1
                    child.weight.data = child.weight.data.contiguous(memory_format=torch.channels_last_3d)
                    if child.weight.is_contiguous(memory_format=torch.channels_last_3d):
                        conv3d_converted += 1
                elif channels_last_2d and isinstance(child, nn.Conv2d):
                    conv2d_total += 1
                    child.weight.data = child.weight.data.contiguous(memory_format=torch.channels_last)
                    if child.weight.is_contiguous(memory_format=torch.channels_last):
                        conv2d_converted += 1
        if channels_last_3d and hasattr(decoder, "channels_last_3d_requested"):
            decoder.channels_last_3d_requested = True
            decoder.channels_last_3d_effective = conv3d_converted
            decoder.conv3d_count = conv3d_total
        if channels_last_2d and hasattr(decoder, "channels_last_requested"):
            decoder.channels_last_requested = True
            decoder.channels_last_effective = conv2d_converted
            decoder.conv2d_count = conv2d_total

    autocast_applied = False
    if decode_autocast and is_cuda:
        vae.decode_autocast_dtype = torch.bfloat16
        autocast_applied = True

    fast_decode_installed, vae_compiled = _install_fast_wan_vae_decode(
        vae,
        channels_last_3d=channels_last_3d,
        compile_decoder=compile_decoder,
        is_cuda=is_cuda,
    )

    return {
        "enabled": True,
        "channels_last_3d": bool(channels_last_3d),
        "channels_last_2d": bool(channels_last_2d),
        "conv3d_layers_total": conv3d_total,
        "conv3d_layers_converted": conv3d_converted,
        "conv2d_layers_total": conv2d_total,
        "conv2d_layers_converted": conv2d_converted,
        "vae_bf16": bf16_applied,
        "decode_autocast": autocast_applied,
        "fast_decode_installed": fast_decode_installed,
        "vae_compile": vae_compiled,
        "fallbacks": [],
    }


# =============================================================================
# 6. Single-Expert Controller (Wan2.2 TI2V-5B & Individual A14B Experts)
# =============================================================================


class Wan22SolExpertController:
    """Non-invasive SOL-Engine controller attached to one ``WanDenoiser``."""

    def __init__(
        self,
        denoiser: Any,
        config: Wan22SolConfig,
        *,
        expert_name: str = "main",
        easycache_threshold: float | None = None,
        easycache_warmup_steps: int | None = None,
        easycache_dense_last: int | None = None,
        easycache_max_skip_steps: int | None = None,
    ) -> None:
        model = getattr(denoiser, "model", None)
        if not isinstance(model, WanModel):
            raise TypeError(f"Wan22SolExpertController requires a WanDenoiser wrapping WanModel, got {type(model)}")
        self.denoiser = denoiser
        self.model: WanModel = model
        self.config = config
        self.expert_name = str(expert_name)

        self._ec_threshold = (
            float(config.easycache_threshold) if easycache_threshold is None else float(easycache_threshold)
        )
        self._ec_warmup = (
            int(config.easycache_warmup_steps) if easycache_warmup_steps is None else int(easycache_warmup_steps)
        )
        self._ec_dense_last = (
            int(config.easycache_dense_last) if easycache_dense_last is None else int(easycache_dense_last)
        )
        self._ec_max_skip = (
            int(config.easycache_max_skip_steps) if easycache_max_skip_steps is None else int(easycache_max_skip_steps)
        )

        self._lock = RLock()
        self._current_request: ContextVar[str | None] = ContextVar(
            f"worldfoundry_wan22_sol_request_{id(self)}",
            default=None,
        )
        self._active_request_window: str | None = None
        self._request_caches: dict[str, dict[str, Wan22SolEasyCache]] = {}
        self._request_snapshots: dict[str, str] = {}
        self._local_step_counters: dict[tuple[str, str], int] = {}
        self._block_input_holder: list[torch.Tensor | None] = [None]
        self._step_state: dict[str, Any] = {
            "force_dense": False,
            "local_step": 0,
            "local_total_steps": 0,
        }
        self._counters = Wan22AttentionCounters()
        self._counter_token = _register_counters(self._counters)
        self._setup_fallbacks: list[str] = []
        self._captured_segment_profiles: dict[str, dict[str, Any]] = {}

        # Saved originals for clean uninstall()
        self._orig_self_processors: list[tuple[SelfAttention, Any]] = []
        self._orig_cross_attn_forwards: list[tuple[AttentionModule, Callable[..., torch.Tensor]]] = []
        self._orig_block_forwards: list[tuple[nn.Module, Callable[..., Any]]] = []
        self._fp8_hook_handles: list[Any] = []
        self._orig_prepare_token_sequence: Any = None
        self._orig_feature_cache_kwargs: Any = None
        self._orig_run_network: Any = None
        self._orig_end_request: Any = None
        self._orig_runtime_report: Any = None
        self._orig_begin_opt_state: Any = None
        self._orig_graph_runner: Any = getattr(denoiser, "_graph_runner", _MISSING)
        self._orig_piecewise_graph: Any = getattr(denoiser, "_piecewise_graph", _MISSING)
        self._orig_prepare_cross_kv: Any = getattr(denoiser, "_prepare_cross_kv", _MISSING)
        self._orig_inplace_residual: Any = getattr(denoiser, "inplace_residual", _MISSING)
        inplace_rt_init = getattr(denoiser, "_inplace_residual_runtime", None)
        self._orig_inplace_runtime_requested: Any = (
            inplace_rt_init.get("requested", _MISSING) if isinstance(inplace_rt_init, dict) else _MISSING
        )
        self._orig_model_attrs: dict[str, Any] = {
            name: getattr(model, name, _MISSING)
            for name in (
                "_worldfoundry_fused_rope",
                "_worldfoundry_rope_precision",
                "_worldfoundry_fused_adaln",
                "_worldfoundry_inplace_residual",
                "_worldfoundry_sol_schedule",
                "_worldfoundry_compile_runtime",
                "_worldfoundry_compile_config",
            )
        }

        self._effective_dense_backend: Literal["flash_attention_3", "torch"] = "torch"
        self._effective_sparse_backend: Literal["none", "pisa", "sol_attn"] = "none"
        self._dense_block_indices: tuple[int, ...] = ()
        self._qkv_fused_blocks: int = 0
        self._fp8_layers: int = 0
        self._fp8_block_layers: int = 0
        self._fp8_fused_ffns: int = 0
        self._compiled_blocks: int = 0
        self._compile_runtime: dict[str, Any] | None = None
        self._use_residual_bridge: bool = False
        self._installed: bool = False

        self._install()

    def _device(self) -> torch.device:
        param = next(self.model.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    def _count_fp8_low_precision_calls(self) -> int:
        return sum(
            int(getattr(mod, "low_precision_kernel_calls", 0))
            for mod in self.model.modules()
            if isinstance(mod, Float8Linear)
        )

    @staticmethod
    def _snapshot_piecewise_segments(piecewise: Any) -> dict[str, tuple[int, int]]:
        report = piecewise.report()
        raw_segments = report.get("segments", {})
        if not isinstance(raw_segments, Mapping):
            return {}
        out: dict[str, tuple[int, int]] = {}
        for seg_name, seg_info in raw_segments.items():
            if not isinstance(seg_info, Mapping):
                continue
            lifetime = seg_info.get("lifetime")
            src = lifetime if isinstance(lifetime, Mapping) else seg_info
            out[str(seg_name)] = (int(src.get("capture", 0)), int(src.get("replay", 0)))
        return out

    def _install(self) -> None:
        device = self._device()
        is_cuda = device.type == "cuda"
        num_blocks = len(self.model.blocks)

        # 0. Pre-flight fail-closed checks before mutating any model or denoiser state.
        existing_fp8 = [mod for mod in self.model.modules() if isinstance(mod, Float8Linear)]
        if not self.config.fp8 and existing_fp8 and any(mod.weight is None for mod in existing_fp8):
            raise RuntimeError(
                "cannot install fp8=False on a model whose weights were already quantized to "
                "Float8Linear(keep_dense_fallback=False); reload BF16 weights first"
            )
        if self.config.fp8 and any(mod.use_fast_accum for mod in existing_fp8):
            raise ValueError(
                "existing Float8Linear layers have use_fast_accum=True; Hopper FP8 requires use_fast_accum=False"
            )
        if (
            self.config.sparse_attention == "pisa"
            and self.config.pisa_strict
            and is_cuda
            and not piecewise_attention_available(device)
        ):
            raise RuntimeError("PISA TMA kernel requested with pisa_strict=True but is unavailable")

        # 1. Resolve dense & sparse attention backends with explicit fallback tracking.
        if self.config.attention_backend == "flash_attention_3":
            if is_cuda and _resolve_fa3_usable(device):
                self._effective_dense_backend = "flash_attention_3"
            else:
                self._effective_dense_backend = "torch"
                if is_cuda:
                    self._setup_fallbacks.append("attention:flash_attention_3->torch")
        else:
            self._effective_dense_backend = "torch"

        if self.config.sparse_attention == "pisa":
            if is_cuda and piecewise_attention_available(device):
                self._effective_sparse_backend = "pisa"
            else:
                self._effective_sparse_backend = "pisa"  # keep processor-level eligibility/fallback tracking
                if is_cuda:
                    self._setup_fallbacks.append("sparse_attn:pisa-unavailable-on-device")
        elif self.config.sparse_attention == "sol_attn":
            if is_cuda and _resolve_sol_attn_usable(device):
                self._effective_sparse_backend = "sol_attn"
            else:
                self._effective_sparse_backend = "sol_attn"
                if is_cuda:
                    self._setup_fallbacks.append("sparse_attn:sol_attn-unavailable-on-device")
        else:
            self._effective_sparse_backend = "none"

        raw_dense_blocks = (
            self.config.pisa_dense_blocks
            if self.config.sparse_attention == "pisa"
            else (self.config.sol_dense_blocks if self.config.sparse_attention == "sol_attn" else ())
        )
        dense_set: set[int] = set()
        for raw_idx in raw_dense_blocks:
            norm = num_blocks + int(raw_idx) if int(raw_idx) < 0 else int(raw_idx)
            if 0 <= norm < num_blocks:
                dense_set.add(norm)
        self._dense_block_indices = tuple(sorted(dense_set))

        # 2. Install Wan22SolSelfAttentionProcessor on every block, then fuse QKV.
        for idx, block in enumerate(self.model.blocks):
            self_attn = getattr(block, "self_attn", None)
            if isinstance(self_attn, SelfAttention):
                self._orig_self_processors.append((self_attn, self_attn.get_processor()))
                proc = Wan22SolSelfAttentionProcessor(
                    block_index=idx,
                    is_dense_block=idx in dense_set,
                    dense_backend=self._effective_dense_backend,
                    sparse_backend=self._effective_sparse_backend,
                    fused_rope=self.config.fused_rope,
                    rope_precision=self.config.rope_precision,
                    pisa_density=self.config.pisa_density,
                    pisa_block_size=self.config.pisa_block_size,
                    pisa_min_sequence_length=self.config.pisa_min_sequence_length,
                    pisa_strict=self.config.pisa_strict,
                    sol_tau=self.config.sol_tau,
                    sol_thresh_type=self.config.sol_thresh_type,
                    sol_kv_splits=self.config.sol_kv_splits,
                    counters=self._counters,
                    counter_token=self._counter_token,
                    step_state=self._step_state,
                )
                self_attn.set_processor(proc)
                if getattr(self_attn, "_qkv_fused", False):
                    restore_fused_qkv_processor_dispatch(self_attn)

            cross_attn = getattr(block, "cross_attn", None)
            if isinstance(cross_attn, CrossAttention) and isinstance(cross_attn.attn, AttentionModule):
                orig_cross_fwd = _install_fast_cross_attn_forward(
                    cross_attn.attn,
                    dense_backend=self._effective_dense_backend,
                    counters=self._counters,
                    counter_token=self._counter_token,
                )
                self._orig_cross_attn_forwards.append((cross_attn.attn, orig_cross_fwd))

        if self.config.fuse_qkv:
            fuse_qkv_projections(
                self.model,
                strategy=self.config.qkv_strategy,
                split_threshold=self.config.qkv_split_threshold,
            )
            qkv_rep = qkv_fusion_report(self.model)
            self._qkv_fused_blocks = int(qkv_rep["fused_blocks"]) if qkv_rep is not None else 0

        # 3. Configure FP64 Triton RoPE + AdaLN fusion + inplace residual safety.
        model_any = cast(Any, self.model)
        model_any._worldfoundry_fused_rope = bool(self.config.fused_rope)
        model_any._worldfoundry_rope_precision = self.config.rope_precision
        model_any._worldfoundry_fused_adaln = bool(self.config.fused_adaln)
        if self.config.fused_rope:
            install_fused_rope_runtime(self.model)

        effective_inplace = bool(self.config.inplace_residual and not self.config.easycache)
        model_any._worldfoundry_inplace_residual = effective_inplace
        self.denoiser.inplace_residual = effective_inplace
        inplace_rt = getattr(self.denoiser, "_inplace_residual_runtime", None)
        if isinstance(inplace_rt, dict):
            inplace_rt["requested"] = effective_inplace

        # 4. Install Static Cross-KV + enable pre-resolution outside compiled/graphed blocks.
        if self.config.static_cross_kv:
            if getattr(self.model, "_worldfoundry_static_cross_kv", None) is None:
                install_static_cross_kv_cache(self.model)
            all_static = True
            for block in self.model.blocks:
                cross_attn = getattr(block, "cross_attn", None)
                if isinstance(cross_attn, CrossAttention):
                    proc = cross_attn.get_processor()
                    if not (isinstance(proc, StaticCrossKVProcessor) and type(proc._inner) is CrossAttentionProcessor):
                        all_static = False
                        break
            if all_static and (self.config.compile or self.config.cuda_graph == "piecewise"):
                self.denoiser._prepare_cross_kv = True

        # 5. Optional calibrated Rowwise FP8 (strictly use_fast_accum=False).
        if not self.config.fp8 and existing_fp8:
            set_low_precision_enabled(self.model, False)
        elif self.config.fp8:
            if is_cuda and torch.cuda.get_device_capability(device)[0] >= 9:
                excludes = list(self.config.fp8_exclude)
                if self.config.fp8_exclude_edge_blocks and num_blocks >= 2:
                    excludes.extend(["blocks.0.", f"blocks.{num_blocks - 1}."])
                replace_linear_with_float8(
                    self.model,
                    scaling=self.config.fp8_scaling,
                    use_fast_accum=False,
                    keep_dense_fallback=self.config.fp8_keep_dense_fallback,
                    min_features=self.config.fp8_min_features,
                    exclude=tuple(dict.fromkeys(excludes)),
                )
                set_low_precision_enabled(self.model, True)
                if self.config.fp8_fuse_ffn:
                    fuse_fp8_gelu_feed_forwards(self.model)
                self._fp8_layers = sum(1 for mod in self.model.modules() if isinstance(mod, Float8Linear))
                self._fp8_block_layers = sum(1 for mod in self.model.blocks.modules() if isinstance(mod, Float8Linear))
                self._fp8_fused_ffns = sum(
                    1 for mod in self.model.modules() if isinstance(mod, FusedFP8GELUFeedForward)
                )
            elif is_cuda:
                self._setup_fallbacks.append("quantization:fp8-requires-sm90")

        # 6. Configure 3D tile permutation & dense/sparse segment schedule for PISA / Sol-Attn.
        tile_3d: tuple[int, int, int] | None = None
        if self.config.sparse_attention == "pisa":
            tile_3d = self.config.pisa_tile_3d
        elif self.config.sparse_attention == "sol_attn":
            tile_3d = self.config.sol_tile_3d

        if self.config.sparse_attention != "none":
            # ``dense_steps`` is fixed to 1 so our _run_network hook can toggle
            # dense (step_index=0 -> _sd1_st0) vs sparse (step_index=1 -> _sd0_st{0,1})
            # deterministically for warmup and tail steps across both A14B experts.
            model_any._worldfoundry_sol_schedule = {
                "dense_steps": 1,
                "dense_blocks": self._dense_block_indices,
                "tile_3d": tile_3d,
            }
        else:
            model_any._worldfoundry_sol_schedule = None

        # 7. Configure CUDA Graph ownership cleanly (clear any pre-existing whole/piecewise runner).
        self.denoiser._graph_runner = None
        if self.config.cuda_graph == "piecewise" and is_cuda:
            self.denoiser._piecewise_graph = PiecewiseBlockGraphRunner(
                warmup=self.config.cuda_graph_warmup,
                max_graphs_per_segment=self.config.cuda_graph_max_graphs,
                clone_outputs=self.config.cuda_graph_clone_outputs,
                enabled=True,
            )
        else:
            self.denoiser._piecewise_graph = None

        # 8. Decide EasyCache seam: use residual bridge when Piecewise Graph or tile_3d
        #    is active so WanModel runs its native graphed/tiled block stack cleanly.
        self._use_residual_bridge = bool(
            self.config.easycache and ((self.config.cuda_graph == "piecewise" and is_cuda) or tile_3d is not None)
        )
        if self._use_residual_bridge:
            self._orig_prepare_token_sequence = self.model.prepare_token_sequence
            orig_pts = self._orig_prepare_token_sequence
            holder = self._block_input_holder

            def _observing_prepare_token_sequence(
                _self_model: WanModel,
                x: torch.Tensor,
                freqs: torch.Tensor,
                t_mod: torch.Tensor,
                t: torch.Tensor,
                grid_size: tuple[int, int, int],
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
                out = orig_pts(x, freqs, t_mod, t, grid_size, **kwargs)
                holder[0] = out[0]
                return out

            self.model.prepare_token_sequence = types.MethodType(  # type: ignore[method-assign]
                _observing_prepare_token_sequence,
                self.model,
            )

        # 9. Regional torch.compile on DiT blocks (zero graph breaks / recompiles).
        if self.config.compile:
            self._install_fp8_compile_trace_hooks()
            self._install_regional_compile()
        else:
            self.model.__dict__.pop("_worldfoundry_compile_runtime", None)
            self.model.__dict__.pop("_worldfoundry_compile_config", None)

        # 10. Install non-invasive denoiser method hooks.
        self._install_denoiser_hooks()
        self.denoiser._worldfoundry_wan22_sol_controller = self
        self._installed = True

    def _install_fp8_compile_trace_hooks(self) -> None:
        if not self.config.fp8 or self._fp8_block_layers <= 0:
            return
        token = self._counter_token

        def _fp8_trace_pre_hook(_mod: nn.Module, _args: Any) -> None:
            if torch.compiler.is_compiling():
                _record_compiled_trace(token, "fp8")

        for mod in self.model.blocks.modules():
            if isinstance(mod, (Float8Linear, FusedFP8GELUFeedForward)):
                self._fp8_hook_handles.append(mod.register_forward_pre_hook(_fp8_trace_pre_hook))

    def reset_expert_request_window(self, request_id: str | None = None) -> None:
        """Reset per-request attention/compile counters (used by dual-expert coordination)."""
        with self._lock:
            self._counters.reset_request_window()
            self._active_request_window = request_id
            if request_id is not None:
                self._current_request.set(request_id)
            if self._compile_runtime is not None:
                self._compile_runtime["request_calls"] = 0
                self._compile_runtime["request_failures"] = 0
                self._compile_runtime["request_last_error"] = None

    def _install_regional_compile(self) -> None:
        compile_runtime: dict[str, Any] = {
            "wrapper_installed": True,
            "calls": 0,
            "failures": 0,
            "last_error": None,
            "request_calls": 0,
            "request_failures": 0,
            "request_last_error": None,
            "compiled_blocks": 0,
            "mode": self.config.compile_mode,
            "fullgraph": self.config.compile_fullgraph,
            "dynamic": self.config.compile_dynamic,
        }
        compiled_count = 0
        for block in self.model.blocks:
            orig_fwd = block.forward
            self._orig_block_forwards.append((block, orig_fwd))
            compiled_fwd = torch.compile(
                orig_fwd,
                mode=self.config.compile_mode,
                fullgraph=self.config.compile_fullgraph,
                dynamic=self.config.compile_dynamic,
            )

            def _make_audited(fn: Callable[..., Any]) -> Callable[..., Any]:
                def _audited(*args: Any, **kwargs: Any) -> Any:
                    try:
                        out = fn(*args, **kwargs)
                    except BaseException as exc:
                        compile_runtime["failures"] += 1
                        compile_runtime["request_failures"] += 1
                        compile_runtime["last_error"] = f"{type(exc).__name__}: {exc}"
                        compile_runtime["request_last_error"] = compile_runtime["last_error"]
                        raise
                    compile_runtime["calls"] += 1
                    compile_runtime["request_calls"] += 1
                    return out

                return _audited

            block.forward = _make_audited(compiled_fwd)  # type: ignore[method-assign]
            compiled_count += 1

        compile_runtime["compiled_blocks"] = compiled_count
        self._compiled_blocks = compiled_count
        self._compile_runtime = compile_runtime
        model_any = cast(Any, self.model)
        model_any._worldfoundry_compile_runtime = compile_runtime
        model_any._worldfoundry_compile_config = {
            "scope": "regional",
            "mode": self.config.compile_mode,
            "fullgraph": self.config.compile_fullgraph,
            "dynamic": self.config.compile_dynamic,
            "regions": [f"blocks.{i}" for i in range(compiled_count)],
        }

    def _install_denoiser_hooks(self) -> None:
        self._orig_feature_cache_kwargs = self.denoiser.feature_cache_kwargs
        self._orig_run_network = self.denoiser._run_network
        self._orig_end_request = self.denoiser.end_request
        self._orig_runtime_report = self.denoiser.runtime_optimization_report
        self._orig_begin_opt_state = self.denoiser._begin_request_optimization_state

        controller = self

        def _hooked_begin_opt_state(_denoiser: Any, model_input: DenoiserInput) -> None:
            controller._orig_begin_opt_state(model_input)
            req_id = model_input.request_id
            if isinstance(req_id, str) and req_id.strip():
                with controller._lock:
                    if controller._active_request_window != req_id:
                        controller._counters.reset_request_window()
                        controller._active_request_window = req_id
                controller._current_request.set(req_id)
            elif model_input.step_index == 0 and (
                model_input.branch == "positive"
                or (
                    isinstance(model_input.conditioning, Mapping)
                    and bool(model_input.conditioning.get("_worldfoundry_cfg_parallel_request", False))
                )
            ):
                with controller._lock:
                    controller._counters.reset_request_window()
                    controller._active_request_window = None

        def _hooked_feature_cache_kwargs(_denoiser: Any, model_input: DenoiserInput) -> dict[str, Any]:
            return controller.prepare_step_kwargs(model_input)

        def _hooked_run_network(_denoiser: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(controller.model, "_worldfoundry_sol_schedule", None) is not None:
                kwargs["_worldfoundry_sol_step_index"] = 0 if controller._step_state.get("force_dense", False) else 1
            piecewise = getattr(_denoiser, "_piecewise_graph", None)
            segs_before = controller._snapshot_piecewise_segments(piecewise) if piecewise is not None else {}
            eager_before = controller._counters.snapshot_eager() if piecewise is not None else None
            lp_before = (
                controller._count_fp8_low_precision_calls() if (piecewise is not None and controller.config.fp8) else 0
            )
            try:
                out = controller._orig_run_network(*args, **kwargs)
            finally:
                controller._block_input_holder[0] = None
            if piecewise is not None and eager_before is not None:
                segs_after = controller._snapshot_piecewise_segments(piecewise)
                lp_after = controller._count_fp8_low_precision_calls() if controller.config.fp8 else 0
                captured_segs = [
                    name for name, (cap_a, _) in segs_after.items() if cap_a > segs_before.get(name, (0, 0))[0]
                ]
                replayed_segs = [
                    name for name, (_, rep_a) in segs_after.items() if rep_a > segs_before.get(name, (0, 0))[1]
                ]
                captured_set = set(captured_segs)
                if captured_segs:
                    # InferenceCUDAGraphRunner executes `warmup` eager passes + 1 capture pass
                    # before immediately replaying the newly captured graph on the same call.
                    capture_passes = max(int(controller.config.cuda_graph_warmup), 0) + 1
                    per_call_delta = controller._counters.diff_from_snapshot(
                        eager_before,
                        fp8_lp_delta=lp_after - lp_before,
                        divisor=capture_passes,
                    )
                    controller._counters.trim_capture_warmup_overhead(
                        per_call_delta,
                        extra_passes=capture_passes - 1,
                    )
                    for seg_name in captured_segs:
                        controller._captured_segment_profiles[seg_name] = per_call_delta
                for seg_name in replayed_segs:
                    if seg_name in captured_set:
                        continue
                    profile = controller._captured_segment_profiles.get(seg_name)
                    if profile is not None:
                        controller._counters.apply_captured_delta(profile)
            return out

        def _hooked_end_request(_denoiser: Any, request_id: str, *, error: BaseException | None = None) -> None:
            try:
                controller.finalize_request(request_id, error=error)
            finally:
                controller._orig_end_request(request_id, error=error)

        def _hooked_runtime_report(_denoiser: Any, request_id: str | None = None) -> dict[str, Any]:
            return controller.runtime_report(request_id=request_id)

        self.denoiser._begin_request_optimization_state = types.MethodType(_hooked_begin_opt_state, self.denoiser)
        self.denoiser.feature_cache_kwargs = types.MethodType(_hooked_feature_cache_kwargs, self.denoiser)
        self.denoiser._run_network = types.MethodType(_hooked_run_network, self.denoiser)
        self.denoiser.end_request = types.MethodType(_hooked_end_request, self.denoiser)
        self.denoiser.runtime_optimization_report = types.MethodType(_hooked_runtime_report, self.denoiser)

    def _is_sparse_dense_step(self, local_step: int, local_total: int) -> bool:
        if self.config.sparse_attention == "pisa":
            dense_first = int(self.config.pisa_dense_steps)
            dense_last = int(self.config.pisa_dense_last_steps)
        elif self.config.sparse_attention == "sol_attn":
            dense_first = int(self.config.sol_dense_steps)
            dense_last = int(self.config.sol_dense_last_steps)
        else:
            return True
        if local_step < dense_first:
            return True
        return local_total > 0 and dense_last > 0 and local_step >= max(local_total - dense_last, 0)

    def prepare_step_kwargs(
        self,
        model_input: DenoiserInput,
        *,
        local_step: int | None = None,
        local_total_steps: int | None = None,
    ) -> dict[str, Any]:
        """Configure sparse step guards and return EasyCache kwargs for one call."""
        req_id = (
            model_input.request_id
            if isinstance(model_input.request_id, str) and model_input.request_id.strip()
            else "__direct__"
        )
        branch = str(model_input.branch or "positive")
        self._current_request.set(req_id)

        if local_step is None or local_total_steps is None:
            resolved_step = int(model_input.step_index)
            resolved_total = int(model_input.total_steps)
        else:
            resolved_step = int(local_step)
            resolved_total = int(local_total_steps)

        self._step_state["local_step"] = resolved_step
        self._step_state["local_total_steps"] = resolved_total
        self._step_state["force_dense"] = self._is_sparse_dense_step(resolved_step, resolved_total)

        if not self.config.easycache:
            return {}

        is_cfg_parallel_call = bool(
            isinstance(model_input.conditioning, Mapping)
            and model_input.conditioning.get("_worldfoundry_cfg_parallel_request", False)
        )
        step_sync_cfg = bool(self.config.sync_cfg_parallel_cache and is_cfg_parallel_call)

        with self._lock:
            branch_map = self._request_caches.setdefault(req_id, {})
            cache = branch_map.get(branch)
            if cache is None or (resolved_step == 0 and cache._previous_step >= 0):
                cache = Wan22SolEasyCache(
                    threshold=self._ec_threshold,
                    total_steps=max(resolved_total, 1),
                    warmup_steps=self._ec_warmup,
                    dense_last=self._ec_dense_last,
                    max_skip_steps=self._ec_max_skip,
                    subsample_stride=self.config.easycache_subsample_stride,
                    sync_cfg_parallel=step_sync_cfg,
                    expert_name=self.expert_name,
                    branch=branch,
                    request_id=req_id,
                )
                branch_map[branch] = cache
            else:
                cache.total_steps = max(resolved_total, resolved_step + 1)
                cache.sync_cfg_parallel = step_sync_cfg

        cache_arg: Any = (
            _Wan22PiecewiseEasyCacheBridge(
                cache,
                block_count=len(self.model.blocks),
                block_input_holder=self._block_input_holder,
            )
            if self._use_residual_bridge
            else cache
        )
        return {
            "feature_cache": cache_arg,
            "feature_cache_step": resolved_step,
            "feature_cache_total_steps": resolved_total,
        }

    def clear_request_residuals(self, request_id: str) -> list[str]:
        """Immediately wipe cached residuals for ``request_id`` (expert boundary guard)."""
        cleared: list[str] = []
        with self._lock:
            branch_map = self._request_caches.get(request_id)
            if branch_map:
                for branch, cache in branch_map.items():
                    cache.clear_residuals()
                    cleared.append(branch)
        return sorted(cleared)

    def easycache_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return auditable EasyCache telemetry for ``request_id``."""
        if not self.config.easycache:
            return {"enabled": False, "expert": self.expert_name, "hits": 0, "events": 0, "hit_rate": 0.0}
        target_id = request_id if request_id is not None else self._current_request.get()
        with self._lock:
            branch_map = self._request_caches.get(target_id) if isinstance(target_id, str) else None
            if branch_map is None and isinstance(target_id, str):
                encoded = self._request_snapshots.get(target_id)
                if isinstance(encoded, str):
                    return cast(dict[str, Any], json.loads(encoded))
            if branch_map is None and self._request_caches:
                last_id = next(reversed(self._request_caches))
                branch_map = self._request_caches[last_id]
                target_id = last_id
            branches = {branch: cache.receipt() for branch, cache in (branch_map or {}).items()}
        total_events = sum(int(row["events"]) for row in branches.values())
        total_hits = sum(int(row["hits"]) for row in branches.values())
        total_dense = sum(int(row["dense_block_calls"]) for row in branches.values())
        total_skipped = sum(int(row["skipped_block_calls"]) for row in branches.values())
        return {
            "enabled": True,
            "algorithm": "easycache",
            "expert": self.expert_name,
            "request_id": target_id,
            "threshold": self._ec_threshold,
            "warmup_steps": self._ec_warmup,
            "dense_last": self._ec_dense_last,
            "max_skip_steps": self._ec_max_skip,
            "events": total_events,
            "hits": total_hits,
            "misses": total_events - total_hits,
            "hit_rate": total_hits / total_events if total_events else 0.0,
            "dense_block_calls": total_dense,
            "skipped_block_calls": total_skipped,
            "branches": branches,
            "fallbacks": [],
        }

    def finalize_request(self, request_id: str, *, error: BaseException | None = None) -> None:
        """Snapshot EasyCache receipt and release all live cached tensors."""
        if not isinstance(request_id, str) or not request_id.strip():
            return
        with self._lock:
            if self._active_request_window == request_id:
                self._active_request_window = None
            branch_map = self._request_caches.get(request_id)
            if branch_map is None:
                return
            for cache in branch_map.values():
                cache.clear_residuals()
            snapshot = self.easycache_report(request_id)
            snapshot["finalized"] = True
            snapshot["release_reason"] = "error" if error is not None else "completed"
            self._request_snapshots[request_id] = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            while len(self._request_snapshots) > 32:
                oldest = next(iter(self._request_snapshots))
                self._request_snapshots.pop(oldest, None)
            self._request_caches.pop(request_id, None)
            self._local_step_counters = {k: v for k, v in self._local_step_counters.items() if k[0] != request_id}

    def installation_receipt(self) -> dict[str, Any]:
        return {
            "engine": "worldfoundry-sol-wan22",
            "expert": self.expert_name,
            "config": self.config.to_dict(),
            "fusion": {
                "fuse_qkv_blocks": self._qkv_fused_blocks,
                "qkv_strategy": self.config.qkv_strategy,
                "fused_rope": bool(self.config.fused_rope),
                "rope_precision": self.config.rope_precision,
                "fused_adaln": bool(self.config.fused_adaln),
                "static_cross_kv": bool(self.config.static_cross_kv),
                "prepare_cross_kv": bool(getattr(self.denoiser, "_prepare_cross_kv", False)),
            },
            "attention": {
                "dense_backend": self._effective_dense_backend,
                "sparse_backend": self._effective_sparse_backend,
                "dense_blocks": list(self._dense_block_indices),
                "num_blocks": len(self.model.blocks),
                "pisa_density": self.config.pisa_density if self._effective_sparse_backend == "pisa" else None,
                "pisa_block_size": self.config.pisa_block_size if self._effective_sparse_backend == "pisa" else None,
                "pisa_tile_3d": (
                    list(self.config.pisa_tile_3d)
                    if self._effective_sparse_backend == "pisa" and self.config.pisa_tile_3d is not None
                    else None
                ),
                "sol_tau": self.config.sol_tau if self._effective_sparse_backend == "sol_attn" else None,
            },
            "easycache": {
                "enabled": bool(self.config.easycache),
                "threshold": self._ec_threshold,
                "warmup_steps": self._ec_warmup,
                "dense_last": self._ec_dense_last,
                "max_skip_steps": self._ec_max_skip,
                "residual_bridge": self._use_residual_bridge,
            },
            "compile": {
                "enabled": bool(self.config.compile),
                "scope": "regional" if self.config.compile else "off",
                "compiled_blocks": self._compiled_blocks,
                "mode": self.config.compile_mode if self.config.compile else None,
            },
            "cuda_graph": {
                "mode": self.config.cuda_graph
                if getattr(self.denoiser, "_piecewise_graph", None) is not None
                else "off",
            },
            "quantization": {
                "fp8_enabled": bool(self.config.fp8),
                "fp8_layers": self._fp8_layers,
                "fp8_fused_ffns": self._fp8_fused_ffns,
                "use_fast_accum": False,
                "scaling": self.config.fp8_scaling if self.config.fp8 else None,
            },
            "fallbacks": list(self._setup_fallbacks),
        }

    def runtime_report(self, request_id: str | None = None) -> dict[str, Any]:
        """Return merged native + SOL-Engine auditable runtime optimization report."""
        base = dict(self._orig_runtime_report(request_id=request_id))
        fallbacks = list(base.get("fallbacks", []))
        for fb in self._setup_fallbacks:
            if fb not in fallbacks:
                fallbacks.append(fb)
        for fb in sorted(self._counters.fallback_reasons):
            if fb not in fallbacks:
                fallbacks.append(fb)
        if self._compile_runtime is not None and int(self._compile_runtime.get("request_failures", 0)) > 0:
            fb = f"compile:failures={self._compile_runtime['request_failures']}"
            if fb not in fallbacks:
                fallbacks.append(fb)

        ec_report = self.easycache_report(request_id=request_id)
        attn_counters = self._counters.to_dict()
        compile_req_calls = (
            int(self._compile_runtime.get("request_calls", 0)) if self._compile_runtime is not None else 0
        )
        cross_mult = 2 if getattr(self.model, "has_image_input", False) else 1
        if self._compile_runtime is not None and compile_req_calls > 0:
            if attn_counters["compiled_fa3_self_traces"] > 0 and attn_counters["fa3_self_calls"] == 0:
                attn_counters["fa3_self_calls"] = compile_req_calls
            if attn_counters["compiled_fa3_cross_traces"] > 0 and attn_counters["fa3_cross_calls"] == 0:
                attn_counters["fa3_cross_calls"] = compile_req_calls * cross_mult
            attn_counters["fa3_total_calls"] = attn_counters["fa3_self_calls"] + attn_counters["fa3_cross_calls"]
            if attn_counters["compiled_rope_traces"] > 0 and attn_counters["triton_rope_calls"] == 0:
                attn_counters["triton_rope_calls"] = compile_req_calls

        raw_quant = quantization_runtime_report(self.model) if self.config.fp8 else None
        quant_report = dict(raw_quant) if isinstance(raw_quant, Mapping) else None
        qkv_report = qkv_fusion_report(self.model)
        graph_report = self.denoiser.graph_report() if hasattr(self.denoiser, "graph_report") else None

        if quant_report is not None:
            quant_report["compiled_fp8_traces"] = self._counters.compiled_fp8_traces
            quant_report["graph_replay_low_precision_calls"] = self._counters.fp8_graph_replay_calls
            if quant_report.get("effective") == "pending":
                if self.config.compile and compile_req_calls > 0 and self._counters.compiled_fp8_traces > 0:
                    num_blocks = max(len(self.model.blocks), 1)
                    estimated_lp = (compile_req_calls * self._fp8_block_layers) // num_blocks
                    quant_report["effective"] = "compiled-fp8-executed"
                    quant_report["compiled_estimated_low_precision_calls"] = estimated_lp
                elif self._counters.fp8_graph_replay_calls > 0:
                    quant_report["effective"] = "cuda-graph-replayed"
                    quant_report["low_precision_kernel_calls"] = self._counters.fp8_graph_replay_calls

        # Check compile trace execution proof (no stale false fallback on QKV/RoPE under compile)
        if self.config.compile and compile_req_calls > 0:
            fallbacks = [fb for fb in fallbacks if not fb.startswith("fuse_qkv: merged projections were installed")]

        active_attention = (
            f"{self._effective_sparse_backend}+{self._effective_dense_backend}"
            if self._effective_sparse_backend != "none"
            else self._effective_dense_backend
        )
        effective = dict(base.get("effective", {}))
        effective.update(
            {
                "wan22_sol_engine": True,
                "attention": active_attention,
                "attention_dense_backend": self._effective_dense_backend,
                "attention_sparse_backend": self._effective_sparse_backend,
                "rope_precision": (
                    f"fused-{self.config.rope_precision}"
                    if self.config.fused_rope
                    else f"complex-{self.config.rope_precision}"
                ),
                "fused_rope_precision": self.config.rope_precision if self.config.fused_rope else "off",
                "fused_adaln": bool(self.config.fused_adaln),
                "easycache": (f"hit_rate={ec_report['hit_rate']:.2%}" if ec_report.get("enabled") else "disabled"),
            }
        )
        if (
            self.config.compile
            and compile_req_calls > 0
            and isinstance(qkv_report, Mapping)
            and int(qkv_report.get("lifetime_compiled_graph_traces", 0)) > 0
        ):
            effective["fuse_qkv_runtime"] = "compiled-graph-executed"
        if quant_report is not None:
            effective["quantization"] = str(quant_report["effective"])
            effective["quantization_runtime"] = quant_report

        runtime = dict(base.get("runtime", {}))
        runtime["wan22_sol"] = {
            "expert": self.expert_name,
            "attention": attn_counters,
            "easycache": ec_report,
            "qkv_fusion": qkv_report,
            "quantization": quant_report,
            "compile": dict(self._compile_runtime) if self._compile_runtime is not None else None,
            "cuda_graph": dict(graph_report) if isinstance(graph_report, Mapping) else None,
        }
        quality_tier = str(base.get("quality_tier", "exact"))
        if self.config.easycache or self.config.sparse_attention != "none" or self.config.fp8:
            quality_tier = "approximate"
        return {
            **base,
            "engine": "worldfoundry-sol-wan22",
            "effective": effective,
            "fallbacks": fallbacks,
            "quality_tier": quality_tier,
            "runtime": runtime,
        }

    def uninstall(self) -> None:
        """Restore original denoiser methods, model flags, and attention processors."""
        if not self._installed:
            return
        from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
            _fused_self_attention_forward,
        )

        for handle in self._fp8_hook_handles:
            handle.remove()
        self._fp8_hook_handles.clear()

        for self_attn, orig_proc in self._orig_self_processors:
            self_attn.set_processor(orig_proc)
            if getattr(self_attn, "_qkv_fused", False):
                if getattr(orig_proc, "supports_fused_qkv", False) is True:
                    restore_fused_qkv_processor_dispatch(self_attn)
                else:
                    self_attn.forward = types.MethodType(  # type: ignore[method-assign]
                        _fused_self_attention_forward,
                        self_attn,
                    )
        for attn_mod, orig_fwd in self._orig_cross_attn_forwards:
            attn_mod.forward = orig_fwd  # type: ignore[method-assign]
        for block, orig_fwd in self._orig_block_forwards:
            block.forward = orig_fwd  # type: ignore[method-assign]
        if self._orig_prepare_token_sequence is not None:
            self.model.prepare_token_sequence = self._orig_prepare_token_sequence
        self.denoiser._begin_request_optimization_state = self._orig_begin_opt_state
        self.denoiser.feature_cache_kwargs = self._orig_feature_cache_kwargs
        self.denoiser._run_network = self._orig_run_network
        self.denoiser.end_request = self._orig_end_request
        self.denoiser.runtime_optimization_report = self._orig_runtime_report
        for attr_name, orig_val in (
            ("_graph_runner", self._orig_graph_runner),
            ("_piecewise_graph", self._orig_piecewise_graph),
            ("_prepare_cross_kv", self._orig_prepare_cross_kv),
            ("inplace_residual", self._orig_inplace_residual),
        ):
            if orig_val is _MISSING:
                self.denoiser.__dict__.pop(attr_name, None)
            else:
                setattr(self.denoiser, attr_name, orig_val)
        inplace_rt = getattr(self.denoiser, "_inplace_residual_runtime", None)
        if isinstance(inplace_rt, dict) and self._orig_inplace_runtime_requested is not _MISSING:
            inplace_rt["requested"] = self._orig_inplace_runtime_requested
        for attr_name, orig_val in self._orig_model_attrs.items():
            if orig_val is _MISSING:
                self.model.__dict__.pop(attr_name, None)
            else:
                setattr(self.model, attr_name, orig_val)
        self.denoiser.__dict__.pop("_worldfoundry_wan22_sol_controller", None)
        with self._lock:
            for branch_map in self._request_caches.values():
                for cache in branch_map.values():
                    cache.reset()
            self._request_caches.clear()
            self._captured_segment_profiles.clear()
            self._active_request_window = None
        self._installed = False


# =============================================================================
# 7. Dual-Expert Controller (Wan2.2 T2V/I2V-A14B with Boundary Cache Reset)
# =============================================================================


class Wan22SolDualExpertController:
    """Dual-expert SOL-Engine controller with strict high->low boundary cache reset."""

    def __init__(self, denoiser: Any, config: Wan22SolConfig) -> None:
        if not hasattr(denoiser, "high_noise") or not hasattr(denoiser, "low_noise"):
            raise TypeError(f"Wan22SolDualExpertController requires a dual-expert denoiser, got {type(denoiser)}")
        self.denoiser = denoiser
        self.config = config
        self._lock = RLock()
        self._schedules: dict[str, WanExpertSchedule] = {}
        self._active_expert: dict[str, str] = {}
        self._active_dual_request: str | None = None
        self._dynamic_local_steps: dict[tuple[str, str, str], int] = {}
        self._boundary_reset_events: dict[str, list[dict[str, Any]]] = {}
        self._boundary_reset_snapshots: dict[str, list[dict[str, Any]]] = {}
        self._total_boundary_resets: int = 0
        self._current_request: ContextVar[str | None] = ContextVar(
            f"worldfoundry_wan22_dual_sol_request_{id(self)}",
            default=None,
        )

        # Build independent controllers for BOTH 14B experts.
        self.high_controller = Wan22SolExpertController(
            denoiser.high_noise,
            config,
            expert_name="high-noise",
            easycache_threshold=config.easycache_threshold,
            easycache_warmup_steps=config.easycache_warmup_steps,
            easycache_dense_last=config.easycache_dense_last,
            easycache_max_skip_steps=config.easycache_max_skip_steps,
        )
        self.low_controller = Wan22SolExpertController(
            denoiser.low_noise,
            config,
            expert_name="low-noise",
            easycache_threshold=(
                config.easycache_threshold
                if config.easycache_low_noise_threshold is None
                else config.easycache_low_noise_threshold
            ),
            easycache_warmup_steps=(
                config.easycache_warmup_steps
                if config.easycache_low_noise_warmup_steps is None
                else config.easycache_low_noise_warmup_steps
            ),
            easycache_dense_last=(
                config.easycache_dense_last
                if config.easycache_low_noise_dense_last is None
                else config.easycache_low_noise_dense_last
            ),
            easycache_max_skip_steps=(
                config.easycache_max_skip_steps
                if config.easycache_low_noise_max_skip_steps is None
                else config.easycache_low_noise_max_skip_steps
            ),
        )

        self._orig_prepare_schedule = getattr(denoiser, "prepare_request_schedule", None)
        self._orig_end_request = denoiser.end_request
        self._orig_runtime_report = denoiser.runtime_optimization_report
        self._orig_high_fc = denoiser.high_noise.feature_cache_kwargs
        self._orig_low_fc = denoiser.low_noise.feature_cache_kwargs
        self._installed = False
        self._install_dual_hooks()

    def _reset_both_experts_for_new_request(self, req_id: str | None) -> None:
        self._active_dual_request = req_id
        self.high_controller.reset_expert_request_window(req_id)
        self.low_controller.reset_expert_request_window(req_id)

    def _install_dual_hooks(self) -> None:
        controller = self

        def _hooked_prepare_schedule(_denoiser: Any, request_id: str, schedule: Any) -> None:
            controller.prepare_request_schedule(request_id, schedule)
            if callable(controller._orig_prepare_schedule):
                controller._orig_prepare_schedule(request_id, schedule)

        def _make_expert_fc_hook(expert_name: str, expert_ctrl: Wan22SolExpertController):
            def _hooked_expert_fc(_expert_denoiser: Any, model_input: DenoiserInput) -> dict[str, Any]:
                return controller._prepare_expert_step(expert_name, expert_ctrl, model_input)

            return _hooked_expert_fc

        def _hooked_end_request(_denoiser: Any, request_id: str, *, error: BaseException | None = None) -> None:
            try:
                controller.finalize_request(request_id, error=error)
            finally:
                controller._orig_end_request(request_id, error=error)

        def _hooked_runtime_report(_denoiser: Any, request_id: str | None = None) -> dict[str, Any]:
            return controller.runtime_report(request_id=request_id)

        self.denoiser.prepare_request_schedule = types.MethodType(_hooked_prepare_schedule, self.denoiser)
        self.denoiser.high_noise.feature_cache_kwargs = types.MethodType(
            _make_expert_fc_hook("high-noise", self.high_controller),
            self.denoiser.high_noise,
        )
        self.denoiser.low_noise.feature_cache_kwargs = types.MethodType(
            _make_expert_fc_hook("low-noise", self.low_controller),
            self.denoiser.low_noise,
        )
        self.denoiser.end_request = types.MethodType(_hooked_end_request, self.denoiser)
        self.denoiser.runtime_optimization_report = types.MethodType(_hooked_runtime_report, self.denoiser)
        self.denoiser._worldfoundry_wan22_sol_controller = self
        self._installed = True

    def prepare_request_schedule(self, request_id: str, schedule: Any) -> None:
        """Precompute exact high-noise and low-noise local step ranges."""
        plan = WanExpertSchedule.build(
            schedule,
            boundary_ratio=float(self.denoiser.boundary_ratio),
            num_train_timesteps=int(self.denoiser.num_train_timesteps),
        )
        with self._lock:
            self._schedules[request_id] = plan
            self._active_expert.pop(request_id, None)
            self._boundary_reset_events[request_id] = []
            self._reset_both_experts_for_new_request(request_id)
        self._current_request.set(request_id)

    def _prepare_expert_step(
        self,
        expert_name: str,
        expert_ctrl: Wan22SolExpertController,
        model_input: DenoiserInput,
    ) -> dict[str, Any]:
        req_id = (
            model_input.request_id
            if isinstance(model_input.request_id, str) and model_input.request_id.strip()
            else "__direct__"
        )
        branch = str(model_input.branch or "positive")
        is_cfg_parallel = bool(
            isinstance(model_input.conditioning, Mapping)
            and model_input.conditioning.get("_worldfoundry_cfg_parallel_request", False)
        )
        self._current_request.set(req_id)

        # Detect new requests and enforce strict expert-boundary EasyCache reset!
        with self._lock:
            if req_id == "__direct__":
                if int(model_input.step_index) == 0 and (branch == "positive" or is_cfg_parallel):
                    self._active_expert.pop("__direct__", None)
                    self._boundary_reset_events["__direct__"] = []
                    self._dynamic_local_steps = {
                        k: v for k, v in self._dynamic_local_steps.items() if k[0] != "__direct__"
                    }
                    self.high_controller.clear_request_residuals("__direct__")
                    self.low_controller.clear_request_residuals("__direct__")
                    self._reset_both_experts_for_new_request("__direct__")
            elif self._active_dual_request != req_id:
                self._active_expert.pop(req_id, None)
                self._boundary_reset_events.setdefault(req_id, [])
                self._reset_both_experts_for_new_request(req_id)

            prev_expert = self._active_expert.get(req_id)
            if prev_expert is not None and prev_expert != expert_name:
                outgoing = self.high_controller if prev_expert == "high-noise" else self.low_controller
                cleared_branches = outgoing.clear_request_residuals(req_id)
                self._total_boundary_resets += 1
                self._boundary_reset_events.setdefault(req_id, []).append(
                    {
                        "from_expert": prev_expert,
                        "to_expert": expert_name,
                        "global_step": int(model_input.step_index),
                        "branch": branch,
                        "cleared_branches": cleared_branches,
                    }
                )
            self._active_expert[req_id] = expert_name

            plan = self._schedules.get(req_id)
            if plan is not None and 0 <= int(model_input.step_index) < len(plan.steps):
                step_info = plan.steps[int(model_input.step_index)]
                local_step = int(step_info.local_index)
                local_total = int(step_info.local_count)
            else:
                key = (req_id, expert_name, branch)
                if int(model_input.step_index) == 0:
                    self._dynamic_local_steps[key] = 0
                local_step = self._dynamic_local_steps.get(key, 0)
                self._dynamic_local_steps[key] = local_step + 1
                remaining = max(int(model_input.total_steps) - int(model_input.step_index), 1)
                local_total = local_step + remaining

        return expert_ctrl.prepare_step_kwargs(
            model_input,
            local_step=local_step,
            local_total_steps=local_total,
        )

    def finalize_request(self, request_id: str, *, error: BaseException | None = None) -> None:
        del error
        with self._lock:
            if self._active_dual_request == request_id:
                self._active_dual_request = None
            events = list(self._boundary_reset_events.pop(request_id, ()))
            self._boundary_reset_snapshots[request_id] = events
            while len(self._boundary_reset_snapshots) > 32:
                oldest = next(iter(self._boundary_reset_snapshots))
                self._boundary_reset_snapshots.pop(oldest, None)
            self._schedules.pop(request_id, None)
            self._active_expert.pop(request_id, None)
            self._dynamic_local_steps = {k: v for k, v in self._dynamic_local_steps.items() if k[0] != request_id}

    def installation_receipt(self) -> dict[str, Any]:
        high_r = self.high_controller.installation_receipt()
        low_r = self.low_controller.installation_receipt()
        fallbacks = [
            *(f"high-noise: {fb}" for fb in high_r.get("fallbacks", ())),
            *(f"low-noise: {fb}" for fb in low_r.get("fallbacks", ())),
        ]
        return {
            "engine": "worldfoundry-sol-wan22",
            "mode": "dual-expert",
            "boundary_ratio": float(self.denoiser.boundary_ratio),
            "config": self.config.to_dict(),
            "experts": {
                "high-noise": high_r,
                "low-noise": low_r,
            },
            "fallbacks": fallbacks,
        }

    def runtime_report(self, request_id: str | None = None) -> dict[str, Any]:
        base = dict(self._orig_runtime_report())
        req_id = (
            request_id
            or base.get("runtime", {}).get("dual_expert", {}).get("request_id")
            or self._current_request.get()
        )
        with self._lock:
            if isinstance(req_id, str):
                boundary_events = list(
                    self._boundary_reset_events.get(req_id, self._boundary_reset_snapshots.get(req_id, ()))
                )
            else:
                boundary_events = []

        high_ec = self.high_controller.easycache_report(request_id=req_id if isinstance(req_id, str) else None)
        low_ec = self.low_controller.easycache_report(request_id=req_id if isinstance(req_id, str) else None)
        total_events = int(high_ec.get("events", 0)) + int(low_ec.get("events", 0))
        total_hits = int(high_ec.get("hits", 0)) + int(low_ec.get("hits", 0))

        experts_map = base.get("runtime", {}).get("dual_expert", {}).get("experts", {})
        high_attn = (
            dict(experts_map["high-noise"]["runtime"]["wan22_sol"]["attention"])
            if isinstance(experts_map, Mapping)
            and isinstance(experts_map.get("high-noise"), Mapping)
            and isinstance(experts_map["high-noise"].get("runtime", {}).get("wan22_sol", {}).get("attention"), Mapping)
            else self.high_controller._counters.to_dict()
        )
        low_attn = (
            dict(experts_map["low-noise"]["runtime"]["wan22_sol"]["attention"])
            if isinstance(experts_map, Mapping)
            and isinstance(experts_map.get("low-noise"), Mapping)
            and isinstance(experts_map["low-noise"].get("runtime", {}).get("wan22_sol", {}).get("attention"), Mapping)
            else self.low_controller._counters.to_dict()
        )

        runtime = dict(base.get("runtime", {}))
        runtime["wan22_sol_dual_expert"] = {
            "boundary_ratio": float(self.denoiser.boundary_ratio),
            "expert_boundary_resets": len(boundary_events),
            "lifetime_expert_boundary_resets": self._total_boundary_resets,
            "boundary_reset_events": boundary_events,
            "easycache": {
                "enabled": bool(self.config.easycache),
                "total_events": total_events,
                "total_hits": total_hits,
                "total_hit_rate": total_hits / total_events if total_events else 0.0,
                "high-noise": high_ec,
                "low-noise": low_ec,
            },
            "attention": {
                "high-noise": high_attn,
                "low-noise": low_attn,
                "total_pisa_calls": int(high_attn["pisa_calls"]) + int(low_attn["pisa_calls"]),
                "total_pisa_fallback_calls": (
                    int(high_attn["pisa_fallback_calls"]) + int(low_attn["pisa_fallback_calls"])
                ),
                "total_sol_attn_calls": int(high_attn["sol_attn_calls"]) + int(low_attn["sol_attn_calls"]),
                "total_fa3_calls": int(high_attn["fa3_total_calls"]) + int(low_attn["fa3_total_calls"]),
            },
        }
        return {
            **base,
            "engine": "worldfoundry-sol-wan22",
            "runtime": runtime,
        }

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._orig_prepare_schedule is not None:
            self.denoiser.prepare_request_schedule = self._orig_prepare_schedule
        self.denoiser.end_request = self._orig_end_request
        self.denoiser.runtime_optimization_report = self._orig_runtime_report
        self.denoiser.high_noise.feature_cache_kwargs = self._orig_high_fc
        self.denoiser.low_noise.feature_cache_kwargs = self._orig_low_fc
        self.denoiser.__dict__.pop("_worldfoundry_wan22_sol_controller", None)
        self.high_controller.uninstall()
        self.low_controller.uninstall()
        self._installed = False


# =============================================================================
# 8. Public Entry Points
# =============================================================================


def accelerate_wan22_denoiser(
    denoiser: Any,
    config: Wan22SolConfig | str,
    *,
    sm_major: int | None = None,
    **overrides: Any,
) -> tuple[Wan22SolExpertController | Wan22SolDualExpertController, dict[str, Any]]:
    """Install the Wan2.2 SOL-Engine controller onto a single or dual-expert denoiser."""
    resolved = (
        wan22_sol_profile(config, sm_major=sm_major, **overrides)
        if isinstance(config, str)
        else (Wan22SolConfig(**{**asdict(config), **overrides}) if overrides else config)
    )
    existing = getattr(denoiser, "_worldfoundry_wan22_sol_controller", None)
    if existing is not None and hasattr(existing, "uninstall"):
        existing.uninstall()

    if hasattr(denoiser, "high_noise") and hasattr(denoiser, "low_noise"):
        controller: Wan22SolExpertController | Wan22SolDualExpertController = Wan22SolDualExpertController(
            denoiser, resolved
        )
    else:
        controller = Wan22SolExpertController(denoiser, resolved)
    return controller, controller.installation_receipt()


def accelerate_wan22_pipeline(
    pipeline: Any,
    config: Wan22SolConfig | str | None = None,
    *,
    sm_major: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Apply the Wan2.2 SOL-Engine stack to a ``Wan2p2Pipeline`` or ``Wan2p2T2VA14BPipeline``."""
    native = getattr(pipeline, "native_pipeline", pipeline)
    components = getattr(native, "components", None)
    denoiser = getattr(components, "denoiser", None) if components is not None else getattr(native, "denoiser", None)
    decoder = getattr(components, "decoder", None) if components is not None else getattr(native, "decoder", None)
    if denoiser is None:
        raise RuntimeError("Wan2.2 pipeline does not expose a native denoiser")

    is_dual = hasattr(denoiser, "high_noise") and hasattr(denoiser, "low_noise")
    selected_config = config if config is not None else ("wan22_a14b_sol" if is_dual else "wan22_5b_sol")
    controller, receipt = accelerate_wan22_denoiser(
        denoiser,
        selected_config,
        sm_major=sm_major,
        **overrides,
    )
    resolved_cfg = controller.config
    if decoder is not None:
        vae_receipt = accelerate_wan22_vae(
            decoder,
            channels_last_3d=resolved_cfg.vae_channels_last_3d,
            channels_last_2d=resolved_cfg.vae_channels_last_2d,
            decode_autocast=resolved_cfg.vae_decode_autocast,
            cast_bf16=resolved_cfg.vae_bf16,
            compile_decoder=resolved_cfg.vae_compile,
        )
        receipt["vae"] = vae_receipt
        for fb in vae_receipt.get("fallbacks", ()):
            if fb not in receipt["fallbacks"]:
                receipt["fallbacks"].append(fb)
    receipt["model_id"] = getattr(pipeline, "model_id", None)
    pipeline._wan22_sol_controller = controller
    pipeline._wan22_sol_receipt = receipt
    return receipt


def wan22_sol_runtime_report(target: Any, *, request_id: str | None = None) -> dict[str, Any]:
    """Collect the live auditable SOL-Engine receipt from a Wan2.2 pipeline or denoiser."""
    controller = getattr(target, "_wan22_sol_controller", None)
    if controller is None:
        controller = getattr(target, "_worldfoundry_wan22_sol_controller", None)
    if controller is None:
        native = getattr(target, "native_pipeline", None)
        components = getattr(native, "components", None) if native is not None else None
        denoiser = getattr(components, "denoiser", None) if components is not None else None
        controller = getattr(denoiser, "_worldfoundry_wan22_sol_controller", None) if denoiser is not None else None
    if controller is not None:
        report = controller.runtime_report(request_id=request_id)
        install_receipt = getattr(target, "_wan22_sol_receipt", None)
        if isinstance(install_receipt, Mapping) and "vae" in install_receipt:
            report.setdefault("runtime", {})["vae_sol"] = install_receipt["vae"]
        return report
    reporter = getattr(target, "runtime_optimization_report", None)
    if callable(reporter):
        return cast(dict[str, Any], reporter())
    return {"engine": "worldfoundry-sol-wan22", "installed": False, "fallbacks": []}


@contextmanager
def wan22_sol_warmup_scope(pipeline_or_denoiser: Any):
    """Context manager delegating to denoiser CUDA Graph warmup scope if present."""
    native = getattr(pipeline_or_denoiser, "native_pipeline", pipeline_or_denoiser)
    components = getattr(native, "components", None)
    denoiser = getattr(components, "denoiser", pipeline_or_denoiser) if components is not None else pipeline_or_denoiser
    warmup = getattr(denoiser, "execution_warmup", None)
    if callable(warmup):
        with warmup():
            yield
    else:
        yield


__all__ = [
    "Wan22AttentionCounters",
    "Wan22SolConfig",
    "Wan22SolDualExpertController",
    "Wan22SolEasyCache",
    "Wan22SolExpertController",
    "Wan22SolSelfAttentionProcessor",
    "accelerate_wan22_denoiser",
    "accelerate_wan22_pipeline",
    "accelerate_wan22_vae",
    "wan22_fa3_attention",
    "wan22_sol_profile",
    "wan22_sol_runtime_report",
    "wan22_sol_warmup_scope",
]
