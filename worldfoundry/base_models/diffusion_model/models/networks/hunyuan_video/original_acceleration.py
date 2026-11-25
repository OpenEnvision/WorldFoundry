# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for original HunyuanVideo-13B (``HunyuanVideoDiT``).

Implements the full HunyuanVideo-13B SOL-Engine recipe with zero silent
fallbacks and auditable receipts:

1. **Lossless Projection Fusion & Text Padding Compaction**:
   - Fuses ``MMSingleStreamBlock`` ``to_qkv`` (``C->3C``) and ``ff[0]``
     (``C->4C``) back into a single ``linear1`` (``C->7C``) projection while
     keeping double-stream fused ``to_qkv`` intact.
   - Bitwise-losslessly compacts right-padded text tokens from ``text_mask``
     outside CUDA Graph capture, eliminating dense attention masks and
     unlocking maskless Hopper FlashAttention-3 and SM90 Sol-Attn.
2. **Triton Fused Diffusion Kernels** (CUDA-Graph-safe ``custom_op``s):
   - Single-launch fused QKV-split + per-head QK-RMSNorm + interleaved 3D-RoPE
     for both single-stream (strided zero-copy view of ``linear1``) and
     double-stream (zero-``torch.cat`` direct write into joint ``[B, L, H, D]``).
   - Fused residual-gate + affine-free LayerNorm + AdaLN scale/shift, fused
     dual residual-gate add, and fused monotone GELU-tanh + rowwise FP8
     quantization (supporting strided ``linear1`` FFN slices with zero copy).
3. **3D-Tiled SM90 ``Sol-Attn``**:
   - Bijective ``tile_3d=(1, 8, 8)`` video permutation with 64-token leftover
     strip carving, contiguous suffix sink covering leftover video tokens plus
     ALL compacted text tokens, dense warmup steps (``dense_steps``), and
     protected dense edge blocks (double front + single transition/tail).
4. **Calibrated Rowwise FP8 W8A8 (``use_fast_accum=False``) + Fused FP8 FFN**:
   - Quantizes interior heavy video/joint projections while keeping embeddings,
     AdaLN modulations, text refiner, double-stream text pathway, final layer,
     and optional edge blocks in native BF16.
5. **Official HunyuanVideo-13B Polynomial TeaCache & Online EasyCache**:
   - Tracks modulated input drift (``double_blocks[0].component_a`` ``img_mod1``)
     using the published 4th-degree HunyuanVideo TeaCache polynomial with
     warmup, dense-last, and max-consecutive-skip guards, plus online EasyCache.
   - Composes cleanly with :class:`PiecewiseBlockGraphRunner` (zero capture
     failures) and optional 2x H200 Ulysses SP=2.
6. **HunyuanVideo 3D Causal VAE Acceleration**:
   - Converts Conv3d weights to ``channels_last_3d``, casts to BF16, and wraps
     decode with BF16 autocast and optional ``torch.compile``.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    _fp8_hardware_eligible,
    _fp8_linear_eligible,
    _is_compiling,
)
from worldfoundry.core.attention.backends.probe import attention_backend_capability
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner
from worldfoundry.core.kernels.triton_dit_block_ops import (
    fused_dit_dual_res_gate_add,
    fused_dit_layernorm_adaln,
    fused_dit_res_gate_ln_adaln,
    fused_hunyuan_double_qkv_rmsnorm_rope_hopper,
    fused_hunyuan_qkv_rmsnorm_rope_hopper,
)

# Published 4th-degree HunyuanVideo TeaCache rescaling polynomial (highest degree first).
OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS: tuple[float, ...] = (
    7.33226126e2,
    -4.01131952e2,
    6.75869174e1,
    -3.14987800e0,
    9.61237896e-2,
)

DEFAULT_HUNYUAN13B_QUANT_EXCLUDES: tuple[str, ...] = (
    "img_in",
    "txt_in",
    "time_in",
    "vector_in",
    "guidance_in",
    ".mod.",
    ".modulation.",
    "final_layer",
)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _validate_float(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive and parsed <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {parsed}")
    if not strictly_positive and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


# ──────────────────────────────────────────────────────────────────────────
# Configuration & SOL-Engine Presets
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Hunyuan13BSolAttnConfig:
    """Validated 3D-tiled SM90 Sol-Attn policy for HunyuanVideo-13B."""

    enabled: bool = False
    tau: float = 1.0
    thresh_type: Literal["diag", "exact"] = "diag"
    kv_splits: Literal[1, 2, 4] = 1
    tile_3d: tuple[int, int, int] | None = (1, 8, 8)
    dense_steps: int = 6
    dense_double_blocks: tuple[int, ...] = (0, 1)
    dense_single_blocks: tuple[int, ...] = (0, -2, -1)
    compress_kv: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"sol_attn.enabled must be a bool, got {self.enabled!r}")
        _validate_float(self.tau, name="sol_attn.tau", minimum=0.0)
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError(f"sol_attn.thresh_type must be 'diag' or 'exact', got {self.thresh_type!r}")
        if isinstance(self.kv_splits, bool) or self.kv_splits not in (1, 2, 4):
            raise ValueError(f"sol_attn.kv_splits must be 1, 2, or 4, got {self.kv_splits!r}")
        if self.tile_3d is not None and (
            len(self.tile_3d) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.tile_3d)
        ):
            raise ValueError(f"sol_attn.tile_3d must be 3 positive ints or None, got {self.tile_3d!r}")
        _validate_int(self.dense_steps, name="sol_attn.dense_steps", minimum=0)
        if not isinstance(self.compress_kv, bool):
            raise TypeError(f"sol_attn.compress_kv must be a bool, got {self.compress_kv!r}")

    def resolve_dense_double_indices(self, num_double: int) -> frozenset[int]:
        resolved: set[int] = set()
        for idx in self.dense_double_blocks:
            norm = num_double + idx if idx < 0 else idx
            if 0 <= norm < num_double:
                resolved.add(norm)
        return frozenset(resolved)

    def resolve_dense_single_indices(self, num_single: int) -> frozenset[int]:
        resolved: set[int] = set()
        for idx in self.dense_single_blocks:
            norm = num_single + idx if idx < 0 else idx
            if 0 <= norm < num_single:
                resolved.add(norm)
        return frozenset(resolved)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tile_3d"] = list(self.tile_3d) if self.tile_3d is not None else None
        payload["dense_double_blocks"] = list(self.dense_double_blocks)
        payload["dense_single_blocks"] = list(self.dense_single_blocks)
        return payload


@dataclass(frozen=True, slots=True)
class Hunyuan13BAccelerationConfig:
    """Declarative SOL-Engine acceleration recipe for original HunyuanVideo-13B."""

    attention_backend: Literal["torch", "flash_attention_3", "sol_attn"] = "flash_attention_3"
    compact_text_tokens: bool = True
    fused_ops: bool = True
    fuse_projections: bool = True
    sol_attn: Hunyuan13BSolAttnConfig = field(default_factory=Hunyuan13BSolAttnConfig)
    quantization: Literal["none", "fp8"] = "none"
    fp8_scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    fp8_use_fast_accum: bool = False
    fuse_fp8_ffn: bool = True
    keep_dense_fallback: bool = False
    quant_min_features: int = 1024
    quant_exclude_edge_blocks: bool = True
    quant_txt_stream: bool = False
    quant_exclude: tuple[str, ...] = DEFAULT_HUNYUAN13B_QUANT_EXCLUDES
    cache_algorithm: Literal["none", "teacache", "easycache"] = "none"
    cache_threshold: float = 0.15
    cache_warmup_steps: int = 6
    cache_dense_last: int = 2
    cache_max_consecutive_skips: int = 2
    cache_teacache_coefficients: tuple[float, ...] = OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS
    cuda_graph: Literal["off", "piecewise"] = "piecewise"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True
    sequence_parallel: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.attention_backend not in ("torch", "flash_attention_3", "sol_attn"):
            raise ValueError(f"unsupported attention_backend {self.attention_backend!r}")
        if self.quantization not in ("none", "fp8"):
            raise ValueError(f"unsupported quantization {self.quantization!r}")
        if self.fp8_scaling not in ("rowwise", "tensorwise"):
            raise ValueError(f"unsupported fp8_scaling {self.fp8_scaling!r}")
        if self.cache_algorithm not in ("none", "teacache", "easycache"):
            raise ValueError(f"unsupported cache_algorithm {self.cache_algorithm!r}")
        if self.cuda_graph not in ("off", "piecewise"):
            raise ValueError(f"unsupported cuda_graph {self.cuda_graph!r}")
        _validate_float(self.cache_threshold, name="cache_threshold", minimum=0.0)
        _validate_int(self.cache_warmup_steps, name="cache_warmup_steps", minimum=0)
        _validate_int(self.cache_dense_last, name="cache_dense_last", minimum=0)
        _validate_int(self.cache_max_consecutive_skips, name="cache_max_consecutive_skips", minimum=0)
        _validate_int(self.cuda_graph_warmup, name="cuda_graph_warmup", minimum=0)
        _validate_int(self.cuda_graph_max_graphs, name="cuda_graph_max_graphs", minimum=1)
        _validate_int(self.quant_min_features, name="quant_min_features", minimum=16)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sol_attn"] = self.sol_attn.to_dict()
        payload["quant_exclude"] = list(self.quant_exclude)
        payload["cache_teacache_coefficients"] = list(self.cache_teacache_coefficients)
        payload["notes"] = list(self.notes)
        return payload


def _fa3_usable(sm_major: int) -> bool:
    if int(sm_major) < 9:
        return False
    return bool(attention_backend_capability("flash_attention_3").usable)


def hunyuan13b_sol_profile(
    tier: str,
    *,
    sm_major: int = 9,
    cuda_graph: Literal["off", "piecewise"] | None = None,
    kv_splits: Literal[1, 2, 4] = 1,
    sol_tau: float = 1.0,
    sol_dense_steps: int | None = None,
    cache_warmup_steps: int | None = None,
    teacache_threshold: float | None = None,
    sequence_parallel: bool = False,
) -> Hunyuan13BAccelerationConfig:
    """Return a validated SOL-Engine preset for HunyuanVideo-13B."""
    normalized = str(tier).strip().lower()
    fa3_ok = _fa3_usable(sm_major)
    dense_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3" if fa3_ok else "torch"

    if normalized in {"l0_ref", "reference", "baseline"}:
        return Hunyuan13BAccelerationConfig(
            attention_backend="torch",
            compact_text_tokens=False,
            fused_ops=False,
            fuse_projections=False,
            sol_attn=Hunyuan13BSolAttnConfig(enabled=False),
            quantization="none",
            cache_algorithm="none",
            cuda_graph=cuda_graph or "off",
            sequence_parallel=sequence_parallel,
            notes=("L0 exact PyTorch baseline",),
        )
    if normalized in {"l1_exact", "exact"}:
        return Hunyuan13BAccelerationConfig(
            attention_backend=dense_backend,
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(enabled=False),
            quantization="none",
            cache_algorithm="none",
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(
                f"L1 lossless text compaction + {dense_backend} + fused linear1 + Triton kernels + piecewise graph",
            ),
        )
    if normalized in {"l2_fp8", "fp8"}:
        return Hunyuan13BAccelerationConfig(
            attention_backend=dense_backend,
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(enabled=False),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="none",
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(
                f"L2 {dense_backend} + calibrated rowwise FP8 (use_fast_accum=False) + fused FP8 FFN + piecewise graph",
            ),
        )
    if normalized in {"l3_sol_attn", "sol_attn"}:
        ds = 5 if sol_dense_steps is None else int(sol_dense_steps)
        return Hunyuan13BAccelerationConfig(
            attention_backend="sol_attn",
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(
                enabled=True,
                tau=sol_tau,
                thresh_type="diag",
                kv_splits=kv_splits,
                tile_3d=(1, 8, 8),
                dense_steps=ds,
                dense_double_blocks=(0, 1),
                dense_single_blocks=(0, -2, -1),
            ),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="none",
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(
                f"L3 3D-tiled SM90 Sol-Attn (tau={sol_tau}, kv_splits={kv_splits}, dense_steps={ds}) "
                "+ rowwise FP8 (use_fast_accum=False) + fused FP8 FFN + piecewise graph",
            ),
        )
    if normalized in {"l4_sol_high_fidelity", "sol_high_fidelity"}:
        ds = 8 if sol_dense_steps is None else max(int(sol_dense_steps), 8)
        cw = ds if cache_warmup_steps is None else int(cache_warmup_steps)
        thresh = 0.10 if teacache_threshold is None else float(teacache_threshold)
        return Hunyuan13BAccelerationConfig(
            attention_backend="sol_attn",
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(
                enabled=True,
                tau=min(sol_tau, 0.8),
                thresh_type="diag",
                kv_splits=kv_splits,
                tile_3d=(1, 8, 8),
                dense_steps=ds,
                dense_double_blocks=(0, 1),
                dense_single_blocks=(0, -2, -1),
            ),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="teacache",
            cache_threshold=thresh,
            cache_warmup_steps=cw,
            cache_dense_last=3,
            cache_max_consecutive_skips=1,
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(f"L4 high-fidelity Sol-Attn + TeaCache({thresh}) + rowwise FP8(exact accum) + piecewise graph",),
        )
    if normalized in {"l4_sol_balanced", "sol_balanced"}:
        ds = 6 if sol_dense_steps is None else int(sol_dense_steps)
        cw = ds if cache_warmup_steps is None else int(cache_warmup_steps)
        thresh = 0.15 if teacache_threshold is None else float(teacache_threshold)
        return Hunyuan13BAccelerationConfig(
            attention_backend="sol_attn",
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(
                enabled=True,
                tau=sol_tau,
                thresh_type="diag",
                kv_splits=kv_splits,
                tile_3d=(1, 8, 8),
                dense_steps=ds,
                dense_double_blocks=(0, 1),
                dense_single_blocks=(0, -2, -1),
            ),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="teacache",
            cache_threshold=thresh,
            cache_warmup_steps=cw,
            cache_dense_last=2,
            cache_max_consecutive_skips=2,
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(
                f"L4 balanced SOL-Engine: Sol-Attn(1x8x8, ds={ds}) + TeaCache({thresh}, ms=2) "
                "+ rowwise FP8(exact accum) + piecewise graph",
            ),
        )
    if normalized in {"l4_sol_full", "sol_full"}:
        ds = 5 if sol_dense_steps is None else int(sol_dense_steps)
        cw = (8 if sol_dense_steps is None else ds) if cache_warmup_steps is None else int(cache_warmup_steps)
        thresh = 0.22 if teacache_threshold is None else float(teacache_threshold)
        return Hunyuan13BAccelerationConfig(
            attention_backend="sol_attn",
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(
                enabled=True,
                tau=sol_tau,
                thresh_type="diag",
                kv_splits=kv_splits,
                tile_3d=(1, 8, 8),
                dense_steps=ds,
                dense_double_blocks=(0, 1),
                dense_single_blocks=(0, -2, -1),
            ),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="teacache",
            cache_threshold=thresh,
            cache_warmup_steps=cw,
            cache_dense_last=2,
            cache_max_consecutive_skips=3,
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(
                f"L4 full SOL-Engine: Sol-Attn(1x8x8, ds={ds}) + TeaCache({thresh}, w={cw}, ms=3) "
                "+ rowwise FP8(exact accum) + piecewise graph",
            ),
        )
    if normalized in {"l4_easycache", "easycache"}:
        ds = 6 if sol_dense_steps is None else int(sol_dense_steps)
        cw = ds if cache_warmup_steps is None else int(cache_warmup_steps)
        thresh = 0.08 if teacache_threshold is None else float(teacache_threshold)
        return Hunyuan13BAccelerationConfig(
            attention_backend="sol_attn",
            compact_text_tokens=True,
            fused_ops=True,
            fuse_projections=True,
            sol_attn=Hunyuan13BSolAttnConfig(
                enabled=True,
                tau=sol_tau,
                thresh_type="diag",
                kv_splits=kv_splits,
                tile_3d=(1, 8, 8),
                dense_steps=ds,
                dense_double_blocks=(0, 1),
                dense_single_blocks=(0, -2, -1),
            ),
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fuse_fp8_ffn=True,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_txt_stream=False,
            cache_algorithm="easycache",
            cache_threshold=thresh,
            cache_warmup_steps=cw,
            cache_dense_last=2,
            cache_max_consecutive_skips=2,
            cuda_graph=cuda_graph or "piecewise",
            sequence_parallel=sequence_parallel,
            notes=(f"L4 EasyCache({thresh}) + Sol-Attn(1x8x8) + rowwise FP8(exact accum) + piecewise graph",),
        )
    allowed = "l0_ref, l1_exact, l2_fp8, l3_sol_attn, l4_sol_high_fidelity, l4_sol_balanced, l4_sol_full, l4_easycache"
    raise ValueError(f"unsupported HunyuanVideo-13B SOL profile {tier!r}; expected one of: {allowed}")


# ──────────────────────────────────────────────────────────────────────────
# 3D-Tiled Sol-Attn Permutation (1x8x8 + 64-token strip carving)
# ──────────────────────────────────────────────────────────────────────────


def build_hunyuan_sol_tile_permutation(
    t: int,
    h: int,
    w: int,
    tile_3d: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a bijective 3D tile permutation, its inverse, and leftover video sink count.

    Primary tiles pack ``(tf, th, tw)`` blocks. For 64-token 2D spatial tiling
    (``tf == 1, th * tw == 64``), leftover bottom and right strips are carved
    into exact 64-token intra-frame 2D patches so the forced-dense suffix sink
    drops to ~0.4% at 720p while keeping all video tiles spatially coherent.
    """
    tf, th, tw = (int(v) for v in tile_3d)
    total = int(t) * int(h) * int(w)
    if total == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, 0
    t_full, h_full, w_full = (t // tf) * tf, (h // th) * th, (w // tw) * tw
    if t_full == 0 or h_full == 0 or w_full == 0:
        identity = torch.arange(total, device=device, dtype=torch.long)
        return identity, identity, total

    grid_idx = torch.arange(total, device=device, dtype=torch.long).view(t, h, w)
    parts: list[torch.Tensor] = [
        grid_idx[:t_full, :h_full, :w_full]
        .view(t_full // tf, tf, h_full // th, th, w_full // tw, tw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
    ]
    mask = torch.zeros((t, h, w), dtype=torch.bool, device=device)
    mask[:t_full, :h_full, :w_full] = True

    if tf == 1 and th * tw == 64:
        row_cursor = h_full
        for bh, bw in ((4, 16), (2, 32), (1, 64)):
            rows_fit = ((h - row_cursor) // bh) * bh
            cols_fit = (w // bw) * bw
            if rows_fit > 0 and cols_fit > 0:
                parts.append(
                    grid_idx[:t_full, row_cursor : row_cursor + rows_fit, :cols_fit]
                    .view(t_full, rows_fit // bh, bh, cols_fit // bw, bw)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(-1)
                )
                mask[:t_full, row_cursor : row_cursor + rows_fit, :cols_fit] = True
                row_cursor += rows_fit
        col_cursor = w_full
        for bh, bw in ((16, 4), (32, 2), (64, 1)):
            rows_fit = (h_full // bh) * bh
            cols_fit = ((w - col_cursor) // bw) * bw
            if rows_fit > 0 and cols_fit > 0:
                parts.append(
                    grid_idx[:t_full, :rows_fit, col_cursor : col_cursor + cols_fit]
                    .view(t_full, rows_fit // bh, bh, cols_fit // bw, bw)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(-1)
                )
                mask[:t_full, :rows_fit, col_cursor : col_cursor + cols_fit] = True
                col_cursor += cols_fit

    rem = grid_idx[~mask]
    video_rem_sink = int(rem.numel())
    if video_rem_sink > 0:
        parts.append(rem)
    perm = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(total, device=device, dtype=torch.long)
    return perm, inv_perm, video_rem_sink


# ──────────────────────────────────────────────────────────────────────────
# Triton Fused Kernels (CUDA-Graph-Safe custom_ops + exact PyTorch fallbacks)
# ──────────────────────────────────────────────────────────────────────────


def _triton_cuda_ok(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    first = tensors[0]
    if first.device.type != "cuda" or bool(getattr(torch.version, "hip", None)):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    return all(t.device == first.device and t.dtype in _FLOAT_DTYPES and t.stride(-1) == 1 for t in tensors)


@lru_cache(maxsize=1)
def _load_hunyuan_triton_kernels() -> dict[str, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _qkv_rmsnorm_interleaved_rope_kernel(
        qkv_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        qkv_stride_b,
        qkv_stride_s,
        cos_stride_s,
        sin_stride_s,
        out_stride_b,
        out_stride_s,
        out_stride_h,
        seq_len,
        rope_len,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_DIM: tl.constexpr,
        HAS_ROPE: tl.constexpr,
    ):
        """Fused QKV split + per-head QK RMSNorm + interleaved 3D-RoPE for [B, S, 3*H*D].

        Handles arbitrary ``qkv_stride_s`` (e.g. direct strided slice of single-stream
        ``linear1`` output) and applies RoPE strictly to prefix ``s < rope_len`` while
        copying normalized text tokens ``s >= rope_len`` in the same kernel launch.
        """
        pid = tl.program_id(0).to(tl.int64)
        h = pid % NUM_HEADS
        bs = pid // NUM_HEADS
        b = bs // seq_len
        s = bs % seq_len

        hidden_dim = NUM_HEADS * HEAD_DIM
        base_q = b * qkv_stride_b + s * qkv_stride_s + h * HEAD_DIM
        base_k = base_q + hidden_dim
        base_v = base_k + hidden_dim

        pair_idx = tl.arange(0, HALF_DIM)
        even_idx = pair_idx * 2
        odd_idx = even_idx + 1

        # Load Q, K, V as interleaved (even, odd) pairs in one coalesced pass.
        q_e_in = tl.load(qkv_ptr + base_q + even_idx)
        q_o_in = tl.load(qkv_ptr + base_q + odd_idx)
        k_e_in = tl.load(qkv_ptr + base_k + even_idx)
        k_o_in = tl.load(qkv_ptr + base_k + odd_idx)
        v_e_in = tl.load(qkv_ptr + base_v + even_idx)
        v_o_in = tl.load(qkv_ptr + base_v + odd_idx)

        in_dtype = q_e_in.dtype
        q_ef = q_e_in.to(tl.float32)
        q_of = q_o_in.to(tl.float32)
        k_ef = k_e_in.to(tl.float32)
        k_of = k_o_in.to(tl.float32)

        q_var = (tl.sum(q_ef * q_ef, axis=0) + tl.sum(q_of * q_of, axis=0)) / HEAD_DIM
        k_var = (tl.sum(k_ef * k_ef, axis=0) + tl.sum(k_of * k_of, axis=0)) / HEAD_DIM
        q_rstd = tl.rsqrt(q_var + eps)
        k_rstd = tl.rsqrt(k_var + eps)

        qw_e = tl.load(qw_ptr + even_idx).to(tl.float32)
        qw_o = tl.load(qw_ptr + odd_idx).to(tl.float32)
        kw_e = tl.load(kw_ptr + even_idx).to(tl.float32)
        kw_o = tl.load(kw_ptr + odd_idx).to(tl.float32)

        # Match worldfoundry.core.nn.RMSNorm exact dtype rounding boundary.
        q_ne = ((q_ef * q_rstd).to(in_dtype).to(tl.float32) * qw_e).to(in_dtype)
        q_no = ((q_of * q_rstd).to(in_dtype).to(tl.float32) * qw_o).to(in_dtype)
        k_ne = ((k_ef * k_rstd).to(in_dtype).to(tl.float32) * kw_e).to(in_dtype)
        k_no = ((k_of * k_rstd).to(in_dtype).to(tl.float32) * kw_o).to(in_dtype)

        if HAS_ROPE:  # noqa: SIM102 — constexpr branch must remain outer for Triton DCE
            if s < rope_len:
                c = tl.load(cos_ptr + s * cos_stride_s + even_idx).to(tl.float32)
                sn = tl.load(sin_ptr + s * sin_stride_s + even_idx).to(tl.float32)
                q_ne_f = q_ne.to(tl.float32)
                q_no_f = q_no.to(tl.float32)
                k_ne_f = k_ne.to(tl.float32)
                k_no_f = k_no.to(tl.float32)
                q_ne = (q_ne_f * c - q_no_f * sn).to(in_dtype)
                q_no = (q_no_f * c + q_ne_f * sn).to(in_dtype)
                k_ne = (k_ne_f * c - k_no_f * sn).to(in_dtype)
                k_no = (k_no_f * c + k_ne_f * sn).to(in_dtype)

        out_base = b * out_stride_b + s * out_stride_s + h * out_stride_h
        tl.store(q_out_ptr + out_base + even_idx, q_ne)
        tl.store(q_out_ptr + out_base + odd_idx, q_no)
        tl.store(k_out_ptr + out_base + even_idx, k_ne)
        tl.store(k_out_ptr + out_base + odd_idx, k_no)
        tl.store(v_out_ptr + out_base + even_idx, v_e_in)
        tl.store(v_out_ptr + out_base + odd_idx, v_o_in)

    @triton.jit
    def _double_qkv_rmsnorm_interleaved_rope_kernel(
        qkv_img_ptr,
        qkv_txt_ptr,
        qw_img_ptr,
        kw_img_ptr,
        qw_txt_ptr,
        kw_txt_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        img_stride_b,
        img_stride_s,
        txt_stride_b,
        txt_stride_s,
        cos_stride_s,
        sin_stride_s,
        out_stride_b,
        out_stride_s,
        out_stride_h,
        img_len,
        total_len,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HALF_DIM: tl.constexpr,
        HAS_ROPE: tl.constexpr,
    ):
        """Fused double-stream QKV -> joint [B, img_len+txt_len, H, D] with zero torch.cat."""
        pid = tl.program_id(0).to(tl.int64)
        h = pid % NUM_HEADS
        bs = pid // NUM_HEADS
        b = bs // total_len
        s = bs % total_len

        hidden_dim = NUM_HEADS * HEAD_DIM
        pair_idx = tl.arange(0, HALF_DIM)
        even_idx = pair_idx * 2
        odd_idx = even_idx + 1
        out_base = b * out_stride_b + s * out_stride_s + h * out_stride_h
        in_dtype: tl.constexpr = qkv_img_ptr.dtype.element_ty

        if s < img_len:
            base_q = b * img_stride_b + s * img_stride_s + h * HEAD_DIM
            base_k = base_q + hidden_dim
            base_v = base_k + hidden_dim
            q_e_in = tl.load(qkv_img_ptr + base_q + even_idx)
            q_o_in = tl.load(qkv_img_ptr + base_q + odd_idx)
            k_e_in = tl.load(qkv_img_ptr + base_k + even_idx)
            k_o_in = tl.load(qkv_img_ptr + base_k + odd_idx)
            v_e_in = tl.load(qkv_img_ptr + base_v + even_idx)
            v_o_in = tl.load(qkv_img_ptr + base_v + odd_idx)

            q_ef = q_e_in.to(tl.float32)
            q_of = q_o_in.to(tl.float32)
            k_ef = k_e_in.to(tl.float32)
            k_of = k_o_in.to(tl.float32)
            q_rstd = tl.rsqrt((tl.sum(q_ef * q_ef, axis=0) + tl.sum(q_of * q_of, axis=0)) / HEAD_DIM + eps)
            k_rstd = tl.rsqrt((tl.sum(k_ef * k_ef, axis=0) + tl.sum(k_of * k_of, axis=0)) / HEAD_DIM + eps)

            qw_e = tl.load(qw_img_ptr + even_idx).to(tl.float32)
            qw_o = tl.load(qw_img_ptr + odd_idx).to(tl.float32)
            kw_e = tl.load(kw_img_ptr + even_idx).to(tl.float32)
            kw_o = tl.load(kw_img_ptr + odd_idx).to(tl.float32)

            q_ne = ((q_ef * q_rstd).to(in_dtype).to(tl.float32) * qw_e).to(in_dtype)
            q_no = ((q_of * q_rstd).to(in_dtype).to(tl.float32) * qw_o).to(in_dtype)
            k_ne = ((k_ef * k_rstd).to(in_dtype).to(tl.float32) * kw_e).to(in_dtype)
            k_no = ((k_of * k_rstd).to(in_dtype).to(tl.float32) * kw_o).to(in_dtype)

            if HAS_ROPE:
                c = tl.load(cos_ptr + s * cos_stride_s + even_idx).to(tl.float32)
                sn = tl.load(sin_ptr + s * sin_stride_s + even_idx).to(tl.float32)
                q_ne_f = q_ne.to(tl.float32)
                q_no_f = q_no.to(tl.float32)
                k_ne_f = k_ne.to(tl.float32)
                k_no_f = k_no.to(tl.float32)
                q_ne = (q_ne_f * c - q_no_f * sn).to(in_dtype)
                q_no = (q_no_f * c + q_ne_f * sn).to(in_dtype)
                k_ne = (k_ne_f * c - k_no_f * sn).to(in_dtype)
                k_no = (k_no_f * c + k_ne_f * sn).to(in_dtype)
        else:
            st = s - img_len
            base_q = b * txt_stride_b + st * txt_stride_s + h * HEAD_DIM
            base_k = base_q + hidden_dim
            base_v = base_k + hidden_dim
            q_e_in = tl.load(qkv_txt_ptr + base_q + even_idx)
            q_o_in = tl.load(qkv_txt_ptr + base_q + odd_idx)
            k_e_in = tl.load(qkv_txt_ptr + base_k + even_idx)
            k_o_in = tl.load(qkv_txt_ptr + base_k + odd_idx)
            v_e_in = tl.load(qkv_txt_ptr + base_v + even_idx)
            v_o_in = tl.load(qkv_txt_ptr + base_v + odd_idx)

            q_ef = q_e_in.to(tl.float32)
            q_of = q_o_in.to(tl.float32)
            k_ef = k_e_in.to(tl.float32)
            k_of = k_o_in.to(tl.float32)
            q_rstd = tl.rsqrt((tl.sum(q_ef * q_ef, axis=0) + tl.sum(q_of * q_of, axis=0)) / HEAD_DIM + eps)
            k_rstd = tl.rsqrt((tl.sum(k_ef * k_ef, axis=0) + tl.sum(k_of * k_of, axis=0)) / HEAD_DIM + eps)

            qw_e = tl.load(qw_txt_ptr + even_idx).to(tl.float32)
            qw_o = tl.load(qw_txt_ptr + odd_idx).to(tl.float32)
            kw_e = tl.load(kw_txt_ptr + even_idx).to(tl.float32)
            kw_o = tl.load(kw_txt_ptr + odd_idx).to(tl.float32)

            q_ne = ((q_ef * q_rstd).to(in_dtype).to(tl.float32) * qw_e).to(in_dtype)
            q_no = ((q_of * q_rstd).to(in_dtype).to(tl.float32) * qw_o).to(in_dtype)
            k_ne = ((k_ef * k_rstd).to(in_dtype).to(tl.float32) * kw_e).to(in_dtype)
            k_no = ((k_of * k_rstd).to(in_dtype).to(tl.float32) * kw_o).to(in_dtype)

        tl.store(q_out_ptr + out_base + even_idx, q_ne)
        tl.store(q_out_ptr + out_base + odd_idx, q_no)
        tl.store(k_out_ptr + out_base + even_idx, k_ne)
        tl.store(k_out_ptr + out_base + odd_idx, k_no)
        tl.store(v_out_ptr + out_base + even_idx, v_e_in)
        tl.store(v_out_ptr + out_base + odd_idx, v_o_in)

    @triton.jit
    def _residual_gate_layernorm_scale_shift_kernel(
        x_ptr,
        upd_ptr,
        gate_ptr,
        scale_ptr,
        shift_ptr,
        x_out_ptr,
        norm_out_ptr,
        x_stride_b,
        x_stride_s,
        upd_stride_b,
        upd_stride_s,
        gate_stride_b,
        scale_stride_b,
        shift_stride_b,
        x_out_stride_b,
        x_out_stride_s,
        norm_out_stride_b,
        norm_out_stride_s,
        seq_len,
        eps,
        FEATURES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse ``x_out = x + upd * gate`` and ``norm_out = LN(x_out) * (1 + scale) + shift``."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        s = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < FEATURES

        x_in = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0)
        upd_in = tl.load(upd_ptr + b * upd_stride_b + s * upd_stride_s + cols, mask=mask, other=0.0)
        gate_in = tl.load(gate_ptr + b * gate_stride_b + cols, mask=mask, other=0.0)

        x_new = (
            x_in.to(tl.float32) + (upd_in.to(tl.float32) * gate_in.to(tl.float32)).to(x_in.dtype).to(tl.float32)
        ).to(x_in.dtype)
        tl.store(x_out_ptr + b * x_out_stride_b + s * x_out_stride_s + cols, x_new, mask=mask)

        xf = tl.where(mask, x_new.to(tl.float32), 0.0)
        mean = tl.sum(xf, axis=0) / FEATURES
        diff = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / FEATURES
        rstd = tl.rsqrt(var + eps)
        normed = (diff * rstd).to(x_in.dtype).to(tl.float32)

        sc = tl.load(scale_ptr + b * scale_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + b * shift_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
        mod_out = (normed * (1.0 + sc) + sh).to(x_in.dtype)
        tl.store(norm_out_ptr + b * norm_out_stride_b + s * norm_out_stride_s + cols, mod_out, mask=mask)

    @triton.jit
    def _dual_residual_gate_add_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        gate_ptr,
        out_ptr,
        x_stride_b,
        x_stride_s,
        a_stride_b,
        a_stride_s,
        b_stride_b,
        b_stride_s,
        gate_stride_b,
        out_stride_b,
        out_stride_s,
        seq_len,
        FEATURES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse ``x + a * gate + b * gate`` in one pass over HBM."""
        row = tl.program_id(0).to(tl.int64)
        batch = row // seq_len
        seq = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < FEATURES

        x_in = tl.load(x_ptr + batch * x_stride_b + seq * x_stride_s + cols, mask=mask, other=0.0)
        a_in = tl.load(a_ptr + batch * a_stride_b + seq * a_stride_s + cols, mask=mask, other=0.0)
        b_in = tl.load(b_ptr + batch * b_stride_b + seq * b_stride_s + cols, mask=mask, other=0.0)
        g_in = tl.load(gate_ptr + batch * gate_stride_b + cols, mask=mask, other=0.0).to(tl.float32)

        y = x_in.to(tl.float32) + (a_in.to(tl.float32) + b_in.to(tl.float32)) * g_in
        tl.store(out_ptr + batch * out_stride_b + seq * out_stride_s + cols, y.to(x_in.dtype), mask=mask)

    @triton.jit
    def _layernorm_scale_shift_fp8_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        codes_ptr,
        row_scale_ptr,
        x_stride_b,
        x_stride_s,
        scale_stride_b,
        shift_stride_b,
        codes_stride_r,
        seq_len,
        eps,
        fp8_max,
        tiny,
        FEATURES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Single-pass affine-free LayerNorm + AdaLN scale/shift + rowwise FP8 quant."""
        row = tl.program_id(0).to(tl.int64)
        b = row // seq_len
        s = row % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < FEATURES

        x_in = tl.load(x_ptr + b * x_stride_b + s * x_stride_s + cols, mask=mask, other=0.0)
        xf = tl.where(mask, x_in.to(tl.float32), 0.0)
        mean = tl.sum(xf, axis=0) / FEATURES
        diff = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / FEATURES
        normed = (diff * tl.rsqrt(var + eps)).to(x_in.dtype).to(tl.float32)

        sc = tl.load(scale_ptr + b * scale_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + b * shift_stride_b + cols, mask=mask, other=0.0).to(tl.float32)
        y = (normed * (1.0 + sc) + sh).to(x_in.dtype).to(tl.float32)
        y = tl.where(mask, y, 0.0)

        amax = tl.max(tl.abs(y), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        row_scale = tl.maximum(amax / fp8_max_f, tiny_f)
        q = tl.minimum(tl.maximum(y / row_scale, -fp8_max_f), fp8_max_f)
        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        tl.store(codes_ptr + row * codes_stride_r + cols, q.to(out_dtype), mask=mask)
        tl.store(row_scale_ptr + row, row_scale)

    @triton.jit
    def _gelu_tanh_fp8_1pass_kernel(
        in_ptr,
        out_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        in_stride,
        out_stride,
        BLOCK_SIZE: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_in = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        g = (x * tl.sigmoid(2.0 * inner)).to(x_in.dtype).to(tl.float32)
        g = tl.where(mask, g, 0.0)

        amax = tl.max(tl.abs(g), axis=0)
        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        q = tl.minimum(tl.maximum(g / scale, -fp8_max_f), fp8_max_f)
        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        tl.store(out_ptr + row * out_stride + cols, q.to(out_dtype), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _gelu_tanh_fp8_monotone_kernel(
        in_ptr,
        out_ptr,
        scale_ptr,
        n_cols,
        fp8_max,
        tiny,
        in_stride,
        out_stride,
        BLOCK_N: tl.constexpr,
        FP8_E5M2: tl.constexpr,
    ):
        """Tiled monotone GELU-tanh + rowwise FP8 quant supporting arbitrary strided rows."""
        row = tl.program_id(0).to(tl.int64)
        row_in = in_ptr + row * in_stride
        row_out = out_ptr + row * out_stride

        max_x = tl.full((), -1e30, tl.float32)
        for start in tl.range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            x = tl.load(row_in + cols, mask=mask, other=-1e30).to(tl.float32)
            max_x = tl.maximum(max_x, tl.max(x, axis=0))

        if (max_x >= 0.4) & (max_x < 1e30):
            inner_m = 0.7978845608028654 * (max_x + 0.044715 * max_x * max_x * max_x)
            g_m = (max_x * tl.sigmoid(2.0 * inner_m)).to(in_ptr.dtype.element_ty).to(tl.float32)
            amax = tl.abs(g_m)
        else:
            amax = tl.full((), 0.0, tl.float32)
            for start in tl.range(0, n_cols, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < n_cols
                x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
                inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
                g = (x * tl.sigmoid(2.0 * inner)).to(in_ptr.dtype.element_ty).to(tl.float32)
                g = tl.where(mask, g, 0.0)
                amax = tl.maximum(amax, tl.max(tl.abs(g), axis=0))

        fp8_max_f = tl.cast(fp8_max, tl.float32)
        tiny_f = tl.cast(tiny, tl.float32)
        scale = tl.maximum(amax / fp8_max_f, tiny_f)
        inv_scale = 1.0 / scale
        tl.store(scale_ptr + row, scale)

        out_dtype = tl.float8e5 if FP8_E5M2 else tl.float8e4nv
        for start in tl.range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
            inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            g = (x * tl.sigmoid(2.0 * inner)).to(in_ptr.dtype.element_ty).to(tl.float32)
            q = tl.minimum(tl.maximum(g * inv_scale, -fp8_max_f), fp8_max_f)
            tl.store(row_out + cols, q.to(out_dtype), mask=mask)

    return {
        "triton": triton,
        "qkv_rmsnorm_rope": _qkv_rmsnorm_interleaved_rope_kernel,
        "double_qkv_rmsnorm_rope": _double_qkv_rmsnorm_interleaved_rope_kernel,
        "res_gate_ln_adaln": _residual_gate_layernorm_scale_shift_kernel,
        "dual_res_gate": _dual_residual_gate_add_kernel,
        "ln_adaln_fp8": _layernorm_scale_shift_fp8_kernel,
        "gelu_fp8_1pass": _gelu_tanh_fp8_1pass_kernel,
        "gelu_fp8_monotone": _gelu_tanh_fp8_monotone_kernel,
    }


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_qkv_rmsnorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_qkv_rmsnorm_rope_op(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    rope_len: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernels = _load_hunyuan_triton_kernels()
    b, s, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    q_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty((b, s, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if b == 0 or s == 0:
        return q_out, k_out, v_out
    with torch.cuda.device(qkv.device):
        kernels["qkv_rmsnorm_rope"][(b * s * num_heads,)](
            qkv,
            q_weight,
            k_weight,
            cos,
            sin,
            q_out,
            k_out,
            v_out,
            qkv.stride(0),
            qkv.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            s,
            rope_len,
            float(eps),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=head_dim // 2,
            HAS_ROPE=has_rope,
            num_warps=4,
        )
    return q_out, k_out, v_out


@_hunyuan_qkv_rmsnorm_rope_op.register_fake
def _hunyuan_qkv_rmsnorm_rope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    rope_len: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, rope_len, has_rope, eps
    b, s, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    shape = (b, s, num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
        torch.empty(shape, dtype=qkv.dtype, device=qkv.device),
    )


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_double_qkv_rmsnorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_double_qkv_rmsnorm_rope_op(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernels = _load_hunyuan_triton_kernels()
    b, img_len, three_c = qkv_img.shape
    txt_len = qkv_txt.shape[1]
    total_len = img_len + txt_len
    head_dim = three_c // (3 * num_heads)
    q_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    k_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    v_out = torch.empty((b, total_len, num_heads, head_dim), dtype=qkv_img.dtype, device=qkv_img.device)
    if b == 0 or total_len == 0:
        return q_out, k_out, v_out
    with torch.cuda.device(qkv_img.device):
        kernels["double_qkv_rmsnorm_rope"][(b * total_len * num_heads,)](
            qkv_img,
            qkv_txt,
            qw_img,
            kw_img,
            qw_txt,
            kw_txt,
            cos,
            sin,
            q_out,
            k_out,
            v_out,
            qkv_img.stride(0),
            qkv_img.stride(1),
            qkv_txt.stride(0),
            qkv_txt.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            img_len,
            total_len,
            float(eps),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=head_dim // 2,
            HAS_ROPE=has_rope,
            num_warps=4,
        )
    return q_out, k_out, v_out


@_hunyuan_double_qkv_rmsnorm_rope_op.register_fake
def _hunyuan_double_qkv_rmsnorm_rope_fake(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    has_rope: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del qw_img, kw_img, qw_txt, kw_txt, cos, sin, has_rope, eps
    b, img_len, three_c = qkv_img.shape
    total_len = img_len + qkv_txt.shape[1]
    head_dim = three_c // (3 * num_heads)
    shape = (b, total_len, num_heads, head_dim)
    return (
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
        torch.empty(shape, dtype=qkv_img.dtype, device=qkv_img.device),
    )


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_res_gate_ln_adaln",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_res_gate_ln_adaln_op(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_hunyuan_triton_kernels()
    triton = kernels["triton"]
    b, s, features = x.shape
    x_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)
    if b == 0 or s == 0:
        return x_out, norm_out
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8
    with torch.cuda.device(x.device):
        kernels["res_gate_ln_adaln"][(b * s,)](
            x,
            update,
            gate,
            scale,
            shift,
            x_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            update.stride(0),
            update.stride(1),
            gate.stride(0),
            scale.stride(0),
            shift.stride(0),
            x_out.stride(0),
            x_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            s,
            float(eps),
            FEATURES=features,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return x_out, norm_out


@_hunyuan_res_gate_ln_adaln_op.register_fake
def _hunyuan_res_gate_ln_adaln_fake(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del update, gate, scale, shift, eps
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_dual_res_gate_add",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_dual_res_gate_add_op(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    kernels = _load_hunyuan_triton_kernels()
    triton = kernels["triton"]
    batch, seq_len, features = x.shape
    out = torch.empty_like(x)
    if batch == 0 or seq_len == 0:
        return out
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8
    with torch.cuda.device(x.device):
        kernels["dual_res_gate"][(batch * seq_len,)](
            x,
            a,
            b_term,
            gate,
            out,
            x.stride(0),
            x.stride(1),
            a.stride(0),
            a.stride(1),
            b_term.stride(0),
            b_term.stride(1),
            gate.stride(0),
            out.stride(0),
            out.stride(1),
            seq_len,
            FEATURES=features,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out


@_hunyuan_dual_res_gate_add_op.register_fake
def _hunyuan_dual_res_gate_add_fake(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    del a, b_term, gate
    return torch.empty_like(x)


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_ln_adaln_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_ln_adaln_fp8_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_hunyuan_triton_kernels()
    triton = kernels["triton"]
    b, s, features = x.shape
    rows = b * s
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    codes = torch.empty((rows, features), dtype=fp8_dtype, device=x.device)
    row_scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows == 0 or features == 0:
        return codes, row_scales
    block_size = triton.next_power_of_2(features)
    num_warps = 4 if block_size <= 2048 else 8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    with torch.cuda.device(x.device):
        kernels["ln_adaln_fp8"][(rows,)](
            x,
            scale,
            shift,
            codes,
            row_scales,
            x.stride(0),
            x.stride(1),
            scale.stride(0),
            shift.stride(0),
            codes.stride(0),
            s,
            float(eps),
            fp8_max,
            tiny,
            FEATURES=features,
            BLOCK_SIZE=block_size,
            FP8_E5M2=fp8_is_e5m2,
            num_warps=num_warps,
        )
    return codes, row_scales


@_hunyuan_ln_adaln_fp8_op.register_fake
def _hunyuan_ln_adaln_fp8_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del scale, shift, eps
    b, s, features = x.shape
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    return (
        torch.empty((b * s, features), dtype=fp8_dtype, device=x.device),
        torch.empty((b * s, 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "worldfoundry::hunyuan13b_gelu_tanh_rowwise_fp8",
    mutates_args=(),
    device_types="cuda",
)
def _hunyuan_gelu_tanh_rowwise_fp8_op(
    value: torch.Tensor,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = _load_hunyuan_triton_kernels()
    triton = kernels["triton"]
    source = value if value.stride(1) == 1 else value.contiguous()
    rows, cols = source.shape
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    codes = torch.empty((rows, cols), dtype=fp8_dtype, device=source.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=source.device)
    if rows == 0 or cols == 0:
        return codes, scales
    fp8_max = float(torch.finfo(fp8_dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    with torch.cuda.device(source.device):
        if cols <= 4096:
            block_size = triton.next_power_of_2(cols)
            num_warps = 4 if block_size <= 2048 else 8
            kernels["gelu_fp8_1pass"][(rows,)](
                source,
                codes,
                scales,
                cols,
                fp8_max,
                tiny,
                source.stride(0),
                codes.stride(0),
                BLOCK_SIZE=block_size,
                FP8_E5M2=fp8_is_e5m2,
                num_warps=num_warps,
            )
        else:
            kernels["gelu_fp8_monotone"][(rows,)](
                source,
                codes,
                scales,
                cols,
                fp8_max,
                tiny,
                source.stride(0),
                codes.stride(0),
                BLOCK_N=4096,
                FP8_E5M2=fp8_is_e5m2,
                num_warps=8,
            )
    return codes, scales


@_hunyuan_gelu_tanh_rowwise_fp8_op.register_fake
def _hunyuan_gelu_tanh_rowwise_fp8_fake(
    value: torch.Tensor,
    fp8_is_e5m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e5m2 if fp8_is_e5m2 else torch.float8_e4m3fn
    return (
        torch.empty(value.shape, dtype=fp8_dtype, device=value.device),
        torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device),
    )


# ──────────────────────────────────────────────────────────────────────────
# Public Python Wrappers with Exact Portable Fallbacks
# ──────────────────────────────────────────────────────────────────────────


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (x.float() * rstd).to(x.dtype) * weight


def _apply_interleaved_rope_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = cos[:rope_len, 0::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    s = sin[:rope_len, 0::2].to(device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    def _rot(t: torch.Tensor) -> torch.Tensor:
        prefix = t[:, :rope_len]
        pf = prefix.float()
        e, o = pf[..., 0::2], pf[..., 1::2]
        out_p = torch.stack((e * c - o * s, o * c + e * s), dim=-1).flatten(-2).to(t.dtype)
        if rope_len == t.shape[1]:
            return out_p
        return torch.cat((out_p, t[:, rope_len:]), dim=1)

    return _rot(q), _rot(k)


def fused_hunyuan_qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    num_heads: int,
    rope_len: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + per-head QK-RMSNorm + interleaved 3D-RoPE for HunyuanVideo."""
    return fused_hunyuan_qkv_rmsnorm_rope_hopper(
        qkv,
        q_weight,
        k_weight,
        freqs_cis,
        num_heads=num_heads,
        rope_len=rope_len,
        eps=eps,
    )


def fused_hunyuan_double_qkv_rmsnorm_rope(
    qkv_img: torch.Tensor,
    qkv_txt: torch.Tensor,
    qw_img: torch.Tensor,
    kw_img: torch.Tensor,
    qw_txt: torch.Tensor,
    kw_txt: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    num_heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused double-stream QKV split + QK-RMSNorm + video 3D-RoPE -> joint [B, L, H, D]."""
    return fused_hunyuan_double_qkv_rmsnorm_rope_hopper(
        qkv_img,
        qkv_txt,
        qw_img,
        kw_img,
        qw_txt,
        kw_txt,
        freqs_cis,
        num_heads=num_heads,
        eps=eps,
    )


def fused_hunyuan_residual_gate_layernorm_scale_shift(
    x: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``x_out = x + update * gate`` and ``LN(x_out) * (1 + scale) + shift``."""
    return fused_dit_res_gate_ln_adaln(x, update, gate, scale, shift, eps=eps)


def fused_hunyuan_dual_residual_gate_add(
    x: torch.Tensor,
    a: torch.Tensor,
    b_term: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse ``x + (a + b_term) * gate`` for ``MMSingleStreamBlock``."""
    return fused_dit_dual_res_gate_add(x, a, b_term, gate)


def fused_hunyuan_layernorm_scale_shift_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    eps: float = 1e-6,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse affine-free LayerNorm + AdaLN scale/shift + rowwise FP8 quant when eligible."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
    features = int(x.shape[-1])
    if (
        x.ndim == 3
        and scale.ndim == 2
        and shift.ndim == 2
        and scale.shape == shift.shape == (x.shape[0], features)
        and 0 < features <= 8192
        and _triton_cuda_ok(x, scale, shift)
    ):
        return _hunyuan_ln_adaln_fp8_op(x, scale, shift, float(eps), fp8_dtype == torch.float8_e5m2)
    return None


def fused_hunyuan_gelu_tanh_rowwise_fp8(
    value: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused GELU-tanh + rowwise FP8 quantization supporting strided 2D/3D slices."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
    if value.ndim == 3:
        b, s, c = value.shape
        # Preserve strided [B*S, C] view whenever dim 0 is contiguous with dim 1 (e.g. B=1 slice of linear1).
        if b == 1:
            mat = value[0]
        elif value.stride(0) == s * value.stride(1):
            mat = value.as_strided((b * s, c), (value.stride(1), value.stride(2)))
        else:
            mat = value.reshape(b * s, c).contiguous()
    elif value.ndim == 2:
        mat = value
    else:
        raise ValueError(f"fused_hunyuan_gelu_tanh_rowwise_fp8 expects 2D or 3D tensor, got {value.ndim}D")

    rows, cols = mat.shape
    if rows == 0 or cols == 0:
        return (
            torch.empty((rows, cols), dtype=fp8_dtype, device=mat.device),
            torch.empty((rows, 1), dtype=torch.float32, device=mat.device),
        )
    if (
        mat.device.type == "cuda"
        and not bool(getattr(torch.version, "hip", None))
        and mat.dtype in _FLOAT_DTYPES
        and mat.stride(1) == 1
        and not (torch.is_grad_enabled() and mat.requires_grad)
    ):
        return _hunyuan_gelu_tanh_rowwise_fp8_op(mat, fp8_dtype == torch.float8_e5m2)

    activated = F.gelu(mat, approximate="tanh")
    fp8_max = float(torch.finfo(fp8_dtype).max)
    scale = (activated.detach().float().abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (activated.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return quantized, scale.float()


def run_fused_fp8_gelu_down_proj(
    mlp_hidden: torch.Tensor,
    down_proj: nn.Module,
    *,
    runtime: Hunyuan13BAccelerationRuntime | None = None,
) -> torch.Tensor:
    """Apply GELU-tanh + down-projection, using fused Triton GELU+FP8 when eligible."""
    if (
        isinstance(down_proj, Float8Linear)
        and down_proj.low_precision_enabled
        and down_proj.scaling == "rowwise"
        and down_proj.input_scale is None
        and mlp_hidden.numel() > 0
        and _fp8_linear_eligible(
            mlp_hidden,
            down_proj.out_features,
            down_proj._hardware_eligible,
            min_gemm_work=0.0 if down_proj.weight is None else None,
        )
    ):
        codes, scales = fused_hunyuan_gelu_tanh_rowwise_fp8(mlp_hidden, down_proj.fp8_dtype)
        out = cast(Any, down_proj)._forward_quantized(codes, scales, mlp_hidden.shape, mlp_hidden.dtype)
        if runtime is not None and not _is_compiling():
            runtime.fused_fp8_ffn_calls += 1
        return out
    return down_proj(F.gelu(mlp_hidden, approximate="tanh"))


def run_adaln_linear(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    linear: nn.Module,
    *,
    eps: float = 1e-6,
    runtime: Hunyuan13BAccelerationRuntime | None = None,
) -> torch.Tensor:
    """Fuse ``LN(x)*(1+scale)+shift`` + rowwise FP8 quant directly into ``linear`` when eligible."""
    if (
        isinstance(linear, Float8Linear)
        and linear.low_precision_enabled
        and linear.scaling == "rowwise"
        and linear.input_scale is None
        and x.numel() > 0
        and _fp8_linear_eligible(
            x,
            linear.out_features,
            linear._hardware_eligible,
            min_gemm_work=0.0 if linear.weight is None else None,
        )
    ):
        quant = fused_hunyuan_layernorm_scale_shift_fp8(x, scale, shift, eps=eps, fp8_dtype=linear.fp8_dtype)
        if quant is not None:
            codes, scales = quant
            if runtime is not None and not _is_compiling():
                runtime.adaln_fp8_calls += 1
            return cast(Any, linear)._forward_quantized(codes, scales, x.shape, x.dtype)
    norm_x = fused_dit_layernorm_adaln(x, scale, shift, eps=eps)
    if runtime is not None and not _is_compiling():
        runtime.adaln_calls += 1
    return linear(norm_x)


# ──────────────────────────────────────────────────────────────────────────
# Projection Fusion & Calibrated Rowwise FP8 (use_fast_accum=False)
# ──────────────────────────────────────────────────────────────────────────


def _fuse_two_linears_vertical(first: nn.Linear, second: nn.Linear) -> nn.Linear:
    any_bias = first.bias is not None or second.bias is not None
    with torch.no_grad():
        weight = torch.cat((first.weight.data, second.weight.data), dim=0).contiguous()
        if any_bias:
            b1 = (
                first.bias.data
                if first.bias is not None
                else torch.zeros(first.out_features, device=weight.device, dtype=weight.dtype)
            )
            b2 = (
                second.bias.data
                if second.bias is not None
                else torch.zeros(second.out_features, device=weight.device, dtype=weight.dtype)
            )
            bias: torch.Tensor | None = torch.cat((b1, b2), dim=0).contiguous()
        else:
            bias = None
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    fused.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        fused.bias = nn.Parameter(bias, requires_grad=False)
    return fused


def fuse_hunyuan13b_projections(model: nn.Module, *, release_originals: bool = True) -> dict[str, int]:
    """Losslessly fuse ``MMSingleStreamBlock`` ``to_qkv + ff[0]`` into ``linear1``.

    Double-stream blocks already store fused ``component_a.to_qkv`` and
    ``component_b.to_qkv`` (``C -> 3C``) projections; this function validates
    them and records their count in the returned receipt.
    """
    double_qkv_verified = 0
    for block in getattr(model, "double_blocks", ()):
        for comp_name in ("component_a", "component_b"):
            comp = getattr(block, comp_name, None)
            qkv = getattr(comp, "to_qkv", None)
            if isinstance(qkv, (nn.Linear, Float8Linear)):
                double_qkv_verified += 1

    single_linear1_fused = 0
    for idx, block in enumerate(getattr(model, "single_blocks", ())):
        if getattr(block, "linear1", None) is not None:
            single_linear1_fused += 1
            continue
        to_qkv = getattr(block, "to_qkv", None)
        ff = getattr(block, "ff", None)
        if (
            isinstance(to_qkv, nn.Linear)
            and isinstance(ff, nn.Sequential)
            and len(ff) >= 3
            and isinstance(ff[0], nn.Linear)
        ):
            block.linear1 = _fuse_two_linears_vertical(to_qkv, ff[0])
            block._qkv_out_features = int(to_qkv.out_features)
            block._mlp_hidden_dim = int(ff[0].out_features)
            if release_originals:
                block.to_qkv = None
                ff[0] = nn.Identity()
            single_linear1_fused += 1
        if release_originals and (idx + 1) % 8 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if release_originals and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "double_qkv_fused_verified": double_qkv_verified,
        "single_linear1_fused": single_linear1_fused,
    }


def quantize_hunyuan13b_fp8(
    model: nn.Module,
    *,
    scaling: Literal["rowwise", "tensorwise"] = "rowwise",
    use_fast_accum: bool = False,
    keep_dense_fallback: bool = False,
    min_features: int = 1024,
    exclude_edge_blocks: bool = True,
    quant_txt_stream: bool = False,
    fuse_fp8_ffn: bool = True,
    exclude: Sequence[str] = DEFAULT_HUNYUAN13B_QUANT_EXCLUDES,
) -> dict[str, Any]:
    """Replace eligible HunyuanVideo-13B linear layers with calibrated FP8 W8A8.

    Defaults to Hopper rowwise scaling with ``use_fast_accum=False``, excluding
    sensitive embeddings, AdaLN modulations, text refiner, final layer, double-stream
    text pathway, and edge blocks (``double_blocks.0`` and final ``single_blocks``).
    """
    double_blocks = getattr(model, "double_blocks", ())
    single_blocks = getattr(model, "single_blocks", ())
    excludes = list(exclude)
    if not quant_txt_stream:
        excludes.append(".component_b.")
    if exclude_edge_blocks:
        if len(double_blocks) > 0:
            excludes.append("double_blocks.0.")
        if len(single_blocks) > 0:
            excludes.append(f"single_blocks.{len(single_blocks) - 1}.")
    exclude_tuple = tuple(dict.fromkeys(excludes))

    replaced = 0
    skipped = 0

    def _is_excluded(full_name: str) -> bool:
        return any(token in full_name for token in exclude_tuple)

    def _visit(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced, skipped
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Float8Linear):
                replaced += 1
                continue
            if isinstance(child, nn.Linear):
                if (
                    _is_excluded(full_name)
                    or min(child.in_features, child.out_features) < int(min_features)
                    or child.in_features % 16 != 0
                    or child.out_features % 16 != 0
                ):
                    skipped += 1
                    continue
                q_mod = Float8Linear(
                    child.weight,
                    child.bias,
                    keep_dense_fallback=keep_dense_fallback,
                    scaling=scaling,
                    use_fast_accum=use_fast_accum,
                )
                setattr(parent, name, q_mod)
                del child
                replaced += 1
                if not keep_dense_fallback and replaced % 32 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif child is not None:
                _visit(child, full_name)

    _visit(model, "")

    fused_ffn_enabled = 0
    if fuse_fp8_ffn and scaling == "rowwise":
        for block in double_blocks:
            for comp_name in ("component_a", "component_b"):
                comp = getattr(block, comp_name, None)
                ff = getattr(comp, "ff", None)
                if (
                    isinstance(ff, nn.Sequential)
                    and len(ff) >= 3
                    and isinstance(ff[2], Float8Linear)
                    and ff[2].scaling == "rowwise"
                ):
                    comp._fuse_fp8_ffn = True
                    fused_ffn_enabled += 1
        for block in single_blocks:
            ff = getattr(block, "ff", None)
            if (
                isinstance(ff, nn.Sequential)
                and len(ff) >= 3
                and isinstance(ff[2], Float8Linear)
                and ff[2].scaling == "rowwise"
            ):
                block._fuse_fp8_ffn = True
                fused_ffn_enabled += 1

    if not keep_dense_fallback and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "mode": "fp8",
        "scaling": scaling,
        "use_fast_accum": bool(use_fast_accum),
        "keep_dense_fallback": bool(keep_dense_fallback),
        "exclude_edge_blocks": bool(exclude_edge_blocks),
        "quant_txt_stream": bool(quant_txt_stream),
        "quantized_linears": replaced,
        "dense_bf16_linears": skipped,
        "fused_fp8_ffn_modules": fused_ffn_enabled,
    }


# ──────────────────────────────────────────────────────────────────────────
# Official HunyuanVideo-13B Polynomial TeaCache & Online EasyCache
# ──────────────────────────────────────────────────────────────────────────


def _eval_polynomial(x: float, coefficients: Sequence[float]) -> float:
    val = 0.0
    for coeff in coefficients:
        val = val * x + float(coeff)
    return max(0.0, float(val))


class Hunyuan13BTeaCache:
    """Official HunyuanVideo-13B polynomial TeaCache controller (tracking ``img_mod1``).

    Evaluates ``img_mod1 = modulate(double_blocks[0].component_a.norm1(img), mod1)``
    in canonical unpermuted video-token space before the 60-block transformer stack.
    On a hit, skips all 20 double blocks + 40 single blocks and returns
    ``img + cached_residual``.
    """

    algorithm = "teacache"

    def __init__(
        self,
        *,
        threshold: float = 0.15,
        total_steps: int = 50,
        warmup_steps: int = 6,
        dense_last: int = 2,
        max_consecutive_skips: int = 2,
        coefficients: Sequence[float] = OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS,
        eps: float = 1e-6,
    ) -> None:
        _validate_float(threshold, name="teacache threshold", minimum=0.0)
        _validate_int(total_steps, name="teacache total_steps", minimum=1)
        _validate_int(warmup_steps, name="teacache warmup_steps", minimum=0)
        _validate_int(dense_last, name="teacache dense_last", minimum=0)
        _validate_int(max_consecutive_skips, name="teacache max_consecutive_skips", minimum=0)
        if not coefficients:
            raise ValueError("TeaCache coefficients must not be empty")
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._prev_probe: torch.Tensor | None = None
        self._cached_residual: torch.Tensor | None = None
        self._accumulated: float = 0.0
        self._consecutive_skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device] | None = None
        self.events: list[BlockCacheEvent] = []

    def evaluate_and_run(
        self,
        *,
        step: int,
        total_steps: int | None,
        img_in: torch.Tensor,
        modulated_inp: torch.Tensor,
        num_blocks: int,
        run_stack: Callable[[], torch.Tensor],
        decision_sync: Callable[[bool], bool] | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Either reuse cached 60-block residual or execute ``run_stack`` and refresh cache."""
        count = self.total_steps if total_steps is None else int(total_steps)
        layout = (tuple(img_in.shape), img_in.dtype, img_in.device)
        probe = modulated_inp.detach()
        reason: str | None = None

        if torch.is_grad_enabled():
            self._cached_residual = None
            self._prev_probe = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._cached_residual = None
            self._prev_probe = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif count > 0 and step >= max(count - self.dense_last, 0):
            reason = "dense-last"
        elif self._cached_residual is None or self._prev_probe is None:
            reason = "seed"
        elif self._consecutive_skips >= self.max_consecutive_skips:
            reason = "max-skip"

        if (
            reason is None
            and self._prev_probe is not None
            and self._prev_probe.shape == probe.shape
            and self._prev_probe.device == probe.device
        ):
            # Compute relative L1 drift in float32 on device, scalar sync once outside CUDA Graph.
            diff_mean = (probe - self._prev_probe).abs().float().mean()
            ref_mean = self._prev_probe.abs().float().mean().clamp_min(self.eps)
            raw_rel_l1 = float((diff_mean / ref_mean).item())
            rescaled = _eval_polynomial(raw_rel_l1, self.coefficients)
            self._accumulated += rescaled
            if not math.isfinite(self._accumulated) or self._accumulated >= self.threshold:
                reason = "threshold"

        local_hit = reason is None
        if decision_sync is not None and not torch.is_grad_enabled():
            hit = bool(decision_sync(local_hit))
            if not hit and reason is None:
                reason = "sp-sync"
        else:
            hit = local_hit

        self._prev_probe = probe.clone() if not torch.is_grad_enabled() else None
        all_blocks = tuple(range(num_blocks))
        if hit:
            out = img_in + cast(torch.Tensor, self._cached_residual)
            self._consecutive_skips += 1
            dense_blocks: tuple[int, ...] = ()
            skipped_blocks: tuple[int, ...] = all_blocks
        else:
            out = run_stack()
            if not torch.is_grad_enabled():
                self._cached_residual = (out - img_in).detach().clone()
                self._layout = layout
            self._accumulated = 0.0
            self._consecutive_skips = 0
            dense_blocks = all_blocks
            skipped_blocks = ()

        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                dense_blocks,
                skipped_blocks,
                self._accumulated,
            )
        )
        return out, hit

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
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "coefficients": list(self.coefficients),
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_consecutive_skips": self.max_consecutive_skips,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


class Hunyuan13BEasyCache:
    """Online transform-factor EasyCache controller for HunyuanVideo-13B."""

    algorithm = "easycache"

    def __init__(
        self,
        *,
        threshold: float = 0.08,
        total_steps: int = 50,
        warmup_steps: int = 6,
        dense_last: int = 2,
        max_consecutive_skips: int = 2,
        eps: float = 1e-6,
    ) -> None:
        _validate_float(threshold, name="easycache threshold", minimum=0.0)
        _validate_int(total_steps, name="easycache total_steps", minimum=1)
        _validate_int(warmup_steps, name="easycache warmup_steps", minimum=0)
        _validate_int(dense_last, name="easycache dense_last", minimum=0)
        _validate_int(max_consecutive_skips, name="easycache max_consecutive_skips", minimum=0)
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._prev_input: torch.Tensor | None = None
        self._dense_input: torch.Tensor | None = None
        self._dense_output: torch.Tensor | None = None
        self._cached_residual: torch.Tensor | None = None
        self._transform_factor: float | None = None
        self._accumulated: float = 0.0
        self._consecutive_skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device] | None = None
        self.events: list[BlockCacheEvent] = []

    def evaluate_and_run(
        self,
        *,
        step: int,
        total_steps: int | None,
        img_in: torch.Tensor,
        modulated_inp: torch.Tensor,
        num_blocks: int,
        run_stack: Callable[[], torch.Tensor],
        decision_sync: Callable[[bool], bool] | None = None,
    ) -> tuple[torch.Tensor, bool]:
        count = self.total_steps if total_steps is None else int(total_steps)
        layout = (tuple(img_in.shape), img_in.dtype, img_in.device)
        probe = modulated_inp.detach()
        reason: str | None = None

        if torch.is_grad_enabled():
            self._cached_residual = None
            self._prev_input = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._cached_residual = None
            self._prev_input = None
            self._dense_input = None
            self._dense_output = None
            self._transform_factor = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif count > 0 and step >= max(count - self.dense_last, 0):
            reason = "dense-last"
        elif self._cached_residual is None or self._transform_factor is None or self._prev_input is None:
            reason = "seed"
        elif self._consecutive_skips >= self.max_consecutive_skips:
            reason = "max-skip"

        if (
            reason is None
            and self._prev_input is not None
            and self._dense_output is not None
            and self._prev_input.shape == probe.shape
            and self._prev_input.device == probe.device
        ):
            delta_in = float((probe - self._prev_input).abs().float().mean().item())
            out_norm = max(float(self._dense_output.abs().float().mean().item()), self.eps)
            est = float(cast(float, self._transform_factor) * delta_in / out_norm)
            self._accumulated += est
            if not math.isfinite(self._accumulated) or self._accumulated >= self.threshold:
                reason = "threshold"

        local_hit = reason is None
        if decision_sync is not None and not torch.is_grad_enabled():
            hit = bool(decision_sync(local_hit))
            if not hit and reason is None:
                reason = "sp-sync"
        else:
            hit = local_hit

        all_blocks = tuple(range(num_blocks))
        if hit:
            out = img_in + cast(torch.Tensor, self._cached_residual)
            self._consecutive_skips += 1
            dense_blocks: tuple[int, ...] = ()
            skipped_blocks: tuple[int, ...] = all_blocks
        else:
            out = run_stack()
            if not torch.is_grad_enabled():
                out_probe = out.detach()
                if (
                    self._dense_input is not None
                    and self._dense_output is not None
                    and self._dense_input.shape == probe.shape
                    and self._dense_input.device == probe.device
                ):
                    in_diff = max(float((probe - self._dense_input).abs().float().mean().item()), self.eps)
                    out_diff = float((out_probe - self._dense_output).abs().float().mean().item())
                    self._transform_factor = out_diff / in_diff
                self._dense_input = probe.clone()
                self._dense_output = out_probe.clone()
                self._cached_residual = (out - img_in).detach().clone()
                self._layout = layout
            self._accumulated = 0.0
            self._consecutive_skips = 0
            dense_blocks = all_blocks
            skipped_blocks = ()

        self._prev_input = probe.clone() if not torch.is_grad_enabled() else None
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                dense_blocks,
                skipped_blocks,
                self._accumulated,
            )
        )
        return out, hit

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
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_consecutive_skips": self.max_consecutive_skips,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


# ──────────────────────────────────────────────────────────────────────────
# Maskless Attention Dispatch (Hopper FA3 + SM90 Sol-Attn + Ulysses SP)
# ──────────────────────────────────────────────────────────────────────────


def _resolve_sp_context(enabled: bool) -> tuple[bool, Any, int, int]:
    if not enabled:
        return False, None, 1, 0
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return False, None, 1, 0
    from worldfoundry.core.distributed.sequence_parallel.runtime import get_sequence_parallel_group

    sp_group = get_sequence_parallel_group()
    if sp_group is not None:
        sp_size = int(dist.get_world_size(sp_group))
        if sp_size > 1:
            return True, sp_group, sp_size, int(dist.get_rank(sp_group))
    from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state

    pstate = get_parallel_state()
    if pstate.sp_enabled and int(pstate.sp) > 1:
        return True, pstate.sp_group, int(pstate.sp), int(pstate.sp_rank)
    return False, None, 1, 0


def run_maskless_joint_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_sol_attn: bool,
    dense_backend: Literal["torch", "flash_attention_3"],
    sol_cfg: Hunyuan13BSolAttnConfig | None,
    sink_tokens: int,
    sink_start: int | None,
    runtime: Hunyuan13BAccelerationRuntime | None = None,
) -> torch.Tensor:
    """Execute unmasked joint video+text attention on contiguous BSHD tensors."""
    if use_sol_attn:
        assert sol_cfg is not None
        if q.device.type != "cuda" or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
            raise ValueError(
                "HunyuanVideo-13B Sol-Attn requires CUDA BF16 tensors with head_dim=128; "
                f"got device={q.device}, dtype={q.dtype}, head_dim={q.shape[-1]}"
            )
        if torch.is_grad_enabled():
            raise RuntimeError("HunyuanVideo-13B Sol-Attn is inference-only")
        from sol_attn import sol_attn as _sol_attn_fn

        tokens = int(q.shape[1])
        arch = tuple(torch.cuda.get_device_capability(q.device))
        kv_splits = int(sol_cfg.kv_splits)
        if arch != (9, 0):
            kv_splits = 1
        elif kv_splits > 1:
            route_groups = ((tokens + 63) // 64 + 63) // 64
            while kv_splits > route_groups and kv_splits > 1:
                kv_splits //= 2
        resolved_sink = min(max(int(sink_tokens), 0), tokens) if sol_cfg.compress_kv else tokens
        resolved_start = (
            None
            if sink_start is None or not sol_cfg.compress_kv
            else min(max(int(sink_start), 0), max(tokens - resolved_sink, 0))
        )
        out = _sol_attn_fn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            tau=float(sol_cfg.tau),
            thresh_type=sol_cfg.thresh_type,
            kv_splits=kv_splits,
            sink_tokens=resolved_sink,
            sink_start=resolved_start,
        )
        if runtime is not None and not _is_compiling():
            runtime.sol_attn_calls += 1
        return out

    if dense_backend == "flash_attention_3":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"HunyuanVideo-13B FlashAttention-3 requires CUDA FP16/BF16, got device={q.device}, dtype={q.dtype}"
            )
        flash_attn_interface = __import__("flash_attn_interface")
        out = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        if runtime is not None and not _is_compiling():
            runtime.fa3_calls += 1
        return out

    # Exact PyTorch SDPA fallback (only when explicitly requested as dense_backend="torch").
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)
    if runtime is not None and not _is_compiling():
        runtime.torch_sdpa_calls += 1
    return out


def _even_all_gather(x: torch.Tensor, *, dim: int, group: Any, world_size: int) -> torch.Tensor:
    """Zero-host-sync contiguous all-gather for equal-shaped SP shards (CUDA-Graph safe)."""
    import torch.distributed as dist

    src = x.contiguous()
    gathered = torch.empty((world_size, *src.shape), dtype=src.dtype, device=src.device)
    dist.all_gather_into_tensor(gathered, src, group=group)
    if dim == 0:
        return gathered.reshape(world_size * src.shape[0], *src.shape[1:])
    perm = [*range(1, dim + 1), 0, *range(dim + 1, src.ndim + 1)]
    out_shape = list(src.shape)
    out_shape[dim] *= world_size
    return gathered.permute(*perm).reshape(*out_shape)


def run_sp_maskless_joint_attention(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    q_txt: torch.Tensor,
    k_txt: torch.Tensor,
    v_txt: torch.Tensor,
    *,
    sp_group: Any,
    sp_size: int,
    sp_rank: int,
    use_sol_attn: bool,
    dense_backend: Literal["torch", "flash_attention_3"],
    sol_cfg: Hunyuan13BSolAttnConfig | None,
    sink_tokens: int,
    sink_start: int | None,
    runtime: Hunyuan13BAccelerationRuntime | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ulysses SP maskless joint attention (all-to-all video, head-shard text)."""
    from worldfoundry.core.distributed.sequence_parallel.runtime import (
        all_to_all_4D,
        all_to_all_4d_many,
    )

    q_img_full, k_img_full, v_img_full = all_to_all_4d_many(
        (q_img, k_img, v_img),
        group=sp_group,
        scatter_dim=2,
        gather_dim=1,
        assume_even=True,
    )
    local_heads = int(q_txt.shape[2]) // int(sp_size)
    head_offset = int(sp_rank) * local_heads
    q_txt_h = q_txt.narrow(2, head_offset, local_heads)
    k_txt_h = k_txt.narrow(2, head_offset, local_heads)
    v_txt_h = v_txt.narrow(2, head_offset, local_heads)

    img_full_len = int(q_img_full.shape[1])
    q_joint = torch.cat((q_img_full, q_txt_h), dim=1)
    k_joint = torch.cat((k_img_full, k_txt_h), dim=1)
    v_joint = torch.cat((v_img_full, v_txt_h), dim=1)

    attn_joint = run_maskless_joint_attention(
        q_joint,
        k_joint,
        v_joint,
        use_sol_attn=use_sol_attn,
        dense_backend=dense_backend,
        sol_cfg=sol_cfg,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
        runtime=runtime,
    )
    attn_img_full = attn_joint[:, :img_full_len]
    attn_txt_local = attn_joint[:, img_full_len:]

    attn_img_local = all_to_all_4D(
        attn_img_full,
        group=sp_group,
        scatter_dim=1,
        gather_dim=2,
        assume_even=True,
    )
    attn_txt_full = _even_all_gather(attn_txt_local, dim=2, group=sp_group, world_size=int(sp_size))
    if runtime is not None and not _is_compiling():
        runtime.sp_all_to_all_calls += 2
    return attn_img_local, attn_txt_full


# ──────────────────────────────────────────────────────────────────────────
# HunyuanVideo 3D Causal VAE Acceleration
# ──────────────────────────────────────────────────────────────────────────


def accelerate_hunyuan_video_vae(
    vae: nn.Module,
    *,
    dtype: torch.dtype = torch.bfloat16,
    channels_last_3d: bool = True,
    decode_autocast: bool = True,
    compile_decoder: bool = False,
    tiling: bool = True,
) -> dict[str, Any]:
    """Accelerate ``HunyuanVideoCausal3DAutoencoder`` with BF16, optional ``channels_last_3d``, and compile.

    Empirical H200 note: on eager tiled CausalConv3d + GroupNorm decodes, BF16 NCDHW
    (``channels_last_3d=False, decode_autocast=False``) measured ~1.29x faster than FP32
    at 51.8 dB PSNR, whereas ``channels_last_3d=True`` incurred NCDHW<->NDHWC conversions
    around ``F.pad`` / ``GroupNorm``. Both layouts are supported and idempotently configurable.
    """
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported VAE dtype {dtype!r}")
    vae.to(dtype=dtype).eval()
    if hasattr(vae, "enable_tiling"):
        cast(Any, vae).enable_tiling(bool(tiling))

    conv3d_total = 0
    conv3d_converted = 0
    if channels_last_3d:
        layout_report = convert_convolution_weight_layouts(vae, conv3d=True)
        conv3d_total = int(layout_report.conv3d_total)
        conv3d_converted = int(layout_report.conv3d_converted)
    else:
        with torch.no_grad():
            for module in vae.modules():
                if isinstance(module, nn.Conv3d) and module.weight.ndim == 5:
                    conv3d_total += 1
                    if not module.weight.is_contiguous(memory_format=torch.contiguous_format):
                        module.weight.data = module.weight.data.contiguous(memory_format=torch.contiguous_format)
                        conv3d_converted += 1

    if hasattr(vae, "decoder"):
        orig_decoder = getattr(vae, "_sol_vae_orig_decoder", None)
        if not isinstance(orig_decoder, nn.Module):
            orig_decoder = cast(nn.Module, vae.decoder)
            cast(Any, vae)._sol_vae_orig_decoder = orig_decoder
        if compile_decoder:
            if not getattr(vae, "_sol_vae_decoder_compiled", False):
                cast(Any, vae).decoder = torch.compile(orig_decoder, mode="default")
                cast(Any, vae)._sol_vae_decoder_compiled = True
        else:
            cast(Any, vae).decoder = orig_decoder
            cast(Any, vae)._sol_vae_decoder_compiled = False

    base_decode_sample = getattr(vae, "_sol_vae_orig_decode_sample", None)
    if not callable(base_decode_sample):
        orig_decode_sample = getattr(vae, "_decode_sample", None)
        if callable(orig_decode_sample):
            base_decode_sample = orig_decode_sample
            cast(Any, vae)._sol_vae_orig_decode_sample = base_decode_sample

    if callable(base_decode_sample):
        decode_fn = base_decode_sample

        def _accelerated_decode_sample(value: torch.Tensor) -> torch.Tensor:
            inp = value
            if channels_last_3d and inp.ndim == 5:
                inp = inp.contiguous(memory_format=torch.channels_last_3d)
            if decode_autocast and inp.device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    return cast(torch.Tensor, decode_fn(inp))
            return cast(torch.Tensor, decode_fn(inp.to(dtype=dtype)))

        cast(Any, vae)._decode_sample = _accelerated_decode_sample
        cast(Any, vae)._sol_vae_wrapped = True

    receipt: dict[str, Any] = {
        "dtype": str(dtype),
        "channels_last_3d": bool(channels_last_3d),
        "conv3d_total": conv3d_total,
        "conv3d_converted": conv3d_converted,
        "decode_autocast": bool(decode_autocast),
        "compile_decoder": bool(compile_decoder),
        "tiling": bool(tiling),
        "fallbacks": [],
    }
    cast(Any, vae)._sol_vae_receipt = receipt
    return receipt


# ──────────────────────────────────────────────────────────────────────────
# Runtime Controller & Top-Level Integrator
# ──────────────────────────────────────────────────────────────────────────


def _try_compact_text_tokens(
    txt: torch.Tensor,
    text_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    """Bitwise-losslessly compact uniform right-padded prefix text tokens outside graph capture.

    Returns ``(compacted_txt, is_unmasked)``. When ``is_unmasked`` is True,
    every remaining text token is valid across all batch items and attention
    can execute with ZERO mask/varlen overhead.
    """
    if text_mask is None:
        return txt, True
    if text_mask.ndim != 2 or text_mask.shape[0] != txt.shape[0] or text_mask.shape[1] != txt.shape[1]:
        return txt, False

    mask_bool = text_mask.to(dtype=torch.bool) if text_mask.dtype != torch.bool else text_mask
    # Check lengths on host ONCE per forward (strictly outside CUDA Graph capture!).
    counts = mask_bool.sum(dim=1).tolist()
    first_count = int(counts[0])
    if first_count <= 0:
        return txt, False
    if any(int(c) != first_count for c in counts):
        return txt, False
    padded_len = int(txt.shape[1])
    if first_count == padded_len:
        return txt, True
    # Verify contiguous prefix: all leading ``first_count`` entries must be True.
    if not bool(mask_bool[:, :first_count].all().item()):
        return txt, False
    return txt[:, :first_count].contiguous(), True


class Hunyuan13BAccelerationRuntime:
    """Stateful SOL-Engine execution runtime bound to one ``HunyuanVideoDiT``."""

    def __init__(
        self,
        model: nn.Module,
        config: Hunyuan13BAccelerationConfig,
        *,
        projection_fusion_report: Mapping[str, Any],
        quantization_report: Mapping[str, Any] | None,
    ) -> None:
        self.model = model
        self.config = config
        self.projection_fusion_report = dict(projection_fusion_report)
        self.quantization_report = dict(quantization_report) if quantization_report is not None else None

        num_double = len(getattr(model, "double_blocks", ()))
        num_single = len(getattr(model, "single_blocks", ()))
        self.dense_double_indices = config.sol_attn.resolve_dense_double_indices(num_double)
        self.dense_single_indices = config.sol_attn.resolve_dense_single_indices(num_single)

        # Resolve dense attention provider used on warmup steps, edge blocks, or when Sol-Attn is off.
        if config.attention_backend == "torch":
            self.dense_backend: Literal["torch", "flash_attention_3"] = "torch"
        else:
            fa3_ok = bool(attention_backend_capability("flash_attention_3").usable)
            if not fa3_ok:
                raise RuntimeError(
                    "HunyuanVideo-13B SOL acceleration requires Hopper FlashAttention-3 when "
                    f"attention_backend={config.attention_backend!r}, but flash_attention_3 is unavailable"
                )
            self.dense_backend = "flash_attention_3"

        self.use_sol_attn = bool(config.sol_attn.enabled or config.attention_backend == "sol_attn")
        if self.use_sol_attn and not bool(attention_backend_capability("sol_attn").usable):
            raise RuntimeError("HunyuanVideo-13B Sol-Attn requested, but sol_attn package/GPU is unavailable")

        self.piecewise_graph: PiecewiseBlockGraphRunner | None = (
            PiecewiseBlockGraphRunner(
                warmup=config.cuda_graph_warmup,
                max_graphs_per_segment=config.cuda_graph_max_graphs,
                clone_outputs=config.cuda_graph_clone_outputs,
                enabled=True,
            )
            if config.cuda_graph == "piecewise"
            else None
        )
        self._tile_perm_cache: dict[
            tuple[int, int, int, int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        self._caches: dict[tuple[str, str], Hunyuan13BTeaCache | Hunyuan13BEasyCache] = {}
        self._step_counters: dict[tuple[str, str], int] = {}
        self._segment_kernel_profile: dict[str, dict[str, int]] = {}
        self._fallbacks: list[str] = []
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.forward_calls: int = 0
        self.text_compactions: int = 0
        self.fa3_calls: int = 0
        self.sol_attn_calls: int = 0
        self.torch_sdpa_calls: int = 0
        self.qkv_rmsnorm_rope_calls: int = 0
        self.adaln_calls: int = 0
        self.adaln_fp8_calls: int = 0
        self.dual_res_gate_calls: int = 0
        self.fused_fp8_ffn_calls: int = 0
        self.sp_all_to_all_calls: int = 0
        self.replayed_fp8_kernel_calls: int = 0
        self._fp8_capture_overcount: int = 0
        self.sol_sparse_steps: int = 0
        self.sol_dense_warmup_steps: int = 0
        self._fallbacks.clear()

    def _snapshot_kernel_counters(self) -> dict[str, int]:
        fp8_lp = sum(
            int(mod.low_precision_kernel_calls) for mod in self.model.modules() if isinstance(mod, Float8Linear)
        )
        return {
            "fa3_calls": self.fa3_calls,
            "sol_attn_calls": self.sol_attn_calls,
            "torch_sdpa_calls": self.torch_sdpa_calls,
            "qkv_rmsnorm_rope_calls": self.qkv_rmsnorm_rope_calls,
            "adaln_calls": self.adaln_calls,
            "adaln_fp8_calls": self.adaln_fp8_calls,
            "dual_res_gate_calls": self.dual_res_gate_calls,
            "fused_fp8_ffn_calls": self.fused_fp8_ffn_calls,
            "sp_all_to_all_calls": self.sp_all_to_all_calls,
            "fp8_lp_calls": fp8_lp,
        }

    def _run_graph_segment_tracked(
        self,
        graph_runner: PiecewiseBlockGraphRunner,
        seg_name: str,
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        runner = graph_runner.get_or_register_segment(seg_name, fn)
        before_rep = runner.report()
        before_captures = int(before_rep.get("capture", 0))
        before_replays = int(before_rep.get("replay", 0))
        before_counts = self._snapshot_kernel_counters()
        result = graph_runner.run_segment(seg_name, *args)
        after_rep = runner.report()
        after_captures = int(after_rep.get("capture", 0))
        after_replays = int(after_rep.get("replay", 0))

        if after_captures > before_captures:
            # _capture runs max(warmup, 1) warmup iterations + 1 capture invocation before replay.
            after_counts = self._snapshot_kernel_counters()
            exec_count = max(int(getattr(runner, "_warmup", 0)), 1) + 1
            per_exec = {k: (after_counts[k] - before_counts[k]) // exec_count for k in before_counts}
            self._segment_kernel_profile[seg_name] = per_exec
            # Normalize runtime counters to 1 logical forward execution for this step.
            extra_invocations = exec_count - 1
            if extra_invocations > 0:
                self.fa3_calls -= extra_invocations * per_exec["fa3_calls"]
                self.sol_attn_calls -= extra_invocations * per_exec["sol_attn_calls"]
                self.torch_sdpa_calls -= extra_invocations * per_exec["torch_sdpa_calls"]
                self.qkv_rmsnorm_rope_calls -= extra_invocations * per_exec["qkv_rmsnorm_rope_calls"]
                self.adaln_calls -= extra_invocations * per_exec["adaln_calls"]
                self.adaln_fp8_calls -= extra_invocations * per_exec["adaln_fp8_calls"]
                self.dual_res_gate_calls -= extra_invocations * per_exec["dual_res_gate_calls"]
                self.fused_fp8_ffn_calls -= extra_invocations * per_exec["fused_fp8_ffn_calls"]
                self.sp_all_to_all_calls -= extra_invocations * per_exec["sp_all_to_all_calls"]
                self._fp8_capture_overcount += extra_invocations * per_exec["fp8_lp_calls"]
        elif after_replays > before_replays:
            prof = self._segment_kernel_profile.get(seg_name)
            if prof is not None:
                self.fa3_calls += prof["fa3_calls"]
                self.sol_attn_calls += prof["sol_attn_calls"]
                self.torch_sdpa_calls += prof["torch_sdpa_calls"]
                self.qkv_rmsnorm_rope_calls += prof["qkv_rmsnorm_rope_calls"]
                self.adaln_calls += prof["adaln_calls"]
                self.adaln_fp8_calls += prof["adaln_fp8_calls"]
                self.dual_res_gate_calls += prof["dual_res_gate_calls"]
                self.fused_fp8_ffn_calls += prof["fused_fp8_ffn_calls"]
                self.sp_all_to_all_calls += prof["sp_all_to_all_calls"]
                self.replayed_fp8_kernel_calls += prof["fp8_lp_calls"]
        else:
            after_counts = self._snapshot_kernel_counters()
            self._segment_kernel_profile[seg_name] = {k: after_counts[k] - before_counts[k] for k in before_counts}
        return result

    def reset(self) -> None:
        """Clear step caches and telemetry counters for a new generation request."""
        self._caches.clear()
        self._step_counters.clear()
        self._reset_counters()
        if self.piecewise_graph is not None:
            self.piecewise_graph.begin_request_window()
        for mod in self.model.modules():
            if isinstance(mod, Float8Linear):
                mod.reset_request_window()

    def close(self) -> None:
        """Release captured CUDA Graphs and cached buffers safely (required before NCCL teardown)."""
        if self.piecewise_graph is not None:
            self.piecewise_graph.invalidate()
        self._caches.clear()
        self._step_counters.clear()
        self._tile_perm_cache.clear()
        self._segment_kernel_profile.clear()
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()

    def __enter__(self) -> Hunyuan13BAccelerationRuntime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get_cache(
        self, request_id: str, branch: str, total_steps: int
    ) -> Hunyuan13BTeaCache | Hunyuan13BEasyCache | None:
        if self.config.cache_algorithm == "none":
            return None
        key = (request_id, branch)
        cache = self._caches.get(key)
        if cache is None:
            if len(self._caches) >= 32:
                oldest = next(iter(self._caches))
                self._caches.pop(oldest, None)
                self._step_counters.pop(oldest, None)
            if self.config.cache_algorithm == "teacache":
                cache = Hunyuan13BTeaCache(
                    threshold=self.config.cache_threshold,
                    total_steps=max(int(total_steps), 1),
                    warmup_steps=self.config.cache_warmup_steps,
                    dense_last=self.config.cache_dense_last,
                    max_consecutive_skips=self.config.cache_max_consecutive_skips,
                    coefficients=self.config.cache_teacache_coefficients,
                )
            else:
                cache = Hunyuan13BEasyCache(
                    threshold=self.config.cache_threshold,
                    total_steps=max(int(total_steps), 1),
                    warmup_steps=self.config.cache_warmup_steps,
                    dense_last=self.config.cache_dense_last,
                    max_consecutive_skips=self.config.cache_max_consecutive_skips,
                )
            self._caches[key] = cache
        return cache

    def _resolve_tile_perm(
        self,
        t: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        tile_3d = self.config.sol_attn.tile_3d
        if tile_3d is None:
            return None
        tf, th, tw = tile_3d
        key = (t, h, w, tf, th, tw, device)
        cached = self._tile_perm_cache.get(key)
        if cached is None:
            if len(self._tile_perm_cache) >= 8:
                self._tile_perm_cache.clear()
            cached = build_hunyuan_sol_tile_permutation(t, h, w, tile_3d, device)
            self._tile_perm_cache[key] = cached
        return cached

    def _compute_img_mod1_probe(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Compute double-block-0 video ``img_mod1`` modulated input for TeaCache/EasyCache."""
        double_blocks = getattr(self.model, "double_blocks", ())
        if len(double_blocks) == 0:
            return img
        comp_a = double_blocks[0].component_a
        mod1_shift, mod1_scale, _, _, _, _ = comp_a.mod(vec).chunk(6, dim=-1)
        if self.config.fused_ops:
            return fused_dit_layernorm_adaln(
                img,
                mod1_scale,
                mod1_shift,
                eps=float(comp_a.norm1.eps),
            )
        return comp_a.norm1(img) * (1.0 + mod1_scale.unsqueeze(1)) + mod1_shift.unsqueeze(1)

    def run_transformer_stack(
        self,
        *,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cos: torch.Tensor | None,
        freqs_sin: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        grid_thw: tuple[int, int, int],
        step_index: int | None = None,
        total_steps: int | None = None,
        request_id: str | None = None,
        branch: str = "positive",
        external_piecewise_graph: PiecewiseBlockGraphRunner | None = None,
    ) -> torch.Tensor:
        """Execute the 20 double + 40 single blocks under the configured SOL-Engine recipe."""
        self.forward_calls += 1
        req_id = request_id or "__default__"
        key = (req_id, branch)
        if step_index is None:
            resolved_step = self._step_counters.get(key, 0)
            self._step_counters[key] = resolved_step + 1
        else:
            resolved_step = int(step_index)
            if resolved_step == 0 and key in self._caches:
                self._caches[key].reset()
            self._step_counters[key] = resolved_step + 1
        resolved_total = int(total_steps) if total_steps is not None else max(resolved_step + 1, 50)

        # 1. Bitwise-lossless text padding compaction (strictly outside CUDA Graph capture).
        unmasked = False
        if self.config.compact_text_tokens:
            orig_txt_len = int(txt.shape[1])
            txt, unmasked = _try_compact_text_tokens(txt, text_mask)
            if unmasked and int(txt.shape[1]) < orig_txt_len:
                self.text_compactions += 1
        elif text_mask is None:
            unmasked = True

        # Fail closed on Sol-Attn if heterogeneous multi-batch padding prevented compaction.
        if self.use_sol_attn and not unmasked:
            raise ValueError(
                "HunyuanVideo-13B Sol-Attn requires uniform prefix text_mask across the batch so "
                "padding tokens can be compacted losslessly"
            )

        sp_enabled, sp_group, sp_size, sp_rank = _resolve_sp_context(self.config.sequence_parallel)
        if sp_enabled and not unmasked:
            raise ValueError("HunyuanVideo-13B Ulysses SP requires uniform prefix text_mask for lossless compaction")

        img_seq_len = int(img.shape[1])
        txt_seq_len = int(txt.shape[1])
        double_blocks = getattr(self.model, "double_blocks", ())
        single_blocks = getattr(self.model, "single_blocks", ())
        total_blocks = len(double_blocks) + len(single_blocks)

        # Decide dense warmup vs sparse Sol-Attn for this step.
        step_uses_sol = bool(
            self.use_sol_attn
            and not self.model.training
            and not torch.is_grad_enabled()
            and resolved_step >= int(self.config.sol_attn.dense_steps)
        )
        if self.use_sol_attn:
            if step_uses_sol:
                self.sol_sparse_steps += 1
            else:
                self.sol_dense_warmup_steps += 1

        graph_runner = external_piecewise_graph or self.piecewise_graph
        if self.model.training or torch.is_grad_enabled():
            graph_runner = None

        # Precompute cu_seqlens ONLY when heterogeneous batch prevents maskless compaction.
        cu_seqlens: torch.Tensor | None = None
        cu_tuple: tuple[int, ...] | None = None
        max_seqlen = img_seq_len + txt_seq_len
        if not unmasked:
            from worldfoundry.core.attention import get_cu_seqlens

            assert text_mask is not None
            cu_seqlens = get_cu_seqlens(text_mask, img_seq_len)
            if self.dense_backend == "torch":
                cu_tuple = tuple(int(v) for v in cu_seqlens.tolist())

        def _execute_full_stack() -> torch.Tensor:
            img_work = img
            cos_work = freqs_cos
            sin_work = freqs_sin
            tile_inv_perm: torch.Tensor | None = None
            video_rem_sink = 0

            # 2. Optional 3D-tile permutation on video tokens when Sol-Attn is active on this step.
            t_grid, h_grid, w_grid = grid_thw
            if step_uses_sol and img_seq_len == t_grid * h_grid * w_grid:
                perm_info = self._resolve_tile_perm(t_grid, h_grid, w_grid, img_work.device)
                if perm_info is not None:
                    tile_perm, tile_inv_perm, video_rem_sink = perm_info
                    if video_rem_sink < img_seq_len:
                        img_work = img_work.index_select(1, tile_perm)
                        if cos_work is not None:
                            cos_work = cos_work.index_select(0, tile_perm)
                        if sin_work is not None:
                            sin_work = sin_work.index_select(0, tile_perm)
                    else:
                        # Grid smaller than one 3D tile: keep identity order AND preserve full
                        # video_rem_sink so un-tiled tokens remain safely in the dense suffix sink.
                        tile_inv_perm = None
                        video_rem_sink = img_seq_len

            # 3. Optional Ulysses SP chunking along video sequence dimension.
            if sp_enabled:
                if img_seq_len % sp_size != 0:
                    raise ValueError(
                        f"HunyuanVideo-13B Ulysses SP={sp_size} requires video token count ({img_seq_len}) "
                        f"to be divisible by sp_size={sp_size}"
                    )
                local_img_len = img_seq_len // sp_size
                start_tok = sp_rank * local_img_len
                img_work = img_work[:, start_tok : start_tok + local_img_len].contiguous()
                if cos_work is not None:
                    cos_work = cos_work[start_tok : start_tok + local_img_len].contiguous()
                if sin_work is not None:
                    sin_work = sin_work[start_tok : start_tok + local_img_len].contiguous()
            else:
                local_img_len = img_seq_len

            # Contiguous suffix sink covers leftover video tokens + ALL compacted text tokens!
            total_sink_tokens = video_rem_sink + txt_seq_len
            sink_start = img_seq_len - video_rem_sink

            mask_tag = "unmasked" if unmasked else "varlen"
            mode_tag = f"{'sol' if step_uses_sol else 'dense'}_{mask_tag}"
            if sp_enabled:
                mode_tag = f"{mode_tag}_sp{sp_size}"

            def _run_double_blocks(
                img_in: torch.Tensor,
                txt_in: torch.Tensor,
                vec_in: torch.Tensor,
                cos_in: torch.Tensor | None,
                sin_in: torch.Tensor | None,
                cu_dev: torch.Tensor | None,
                cu_host: tuple[int, ...] | None,
                max_sl: int,
                sink_tok_val: int,
                sink_start_val: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                cu_eff = torch.tensor(cu_host, dtype=torch.int32) if cu_host is not None else cu_dev
                freqs_pair = (cos_in, sin_in) if cos_in is not None and sin_in is not None else None
                for blk_idx, blk in enumerate(double_blocks):
                    blk_sol = bool(step_uses_sol and blk_idx not in self.dense_double_indices)
                    img_in, txt_in = blk._forward_accelerated(
                        img_in,
                        txt_in,
                        vec_in,
                        freqs_pair,
                        runtime=self,
                        unmasked=unmasked,
                        use_sol_attn=blk_sol,
                        sink_tokens=sink_tok_val,
                        sink_start=sink_start_val,
                        sp_enabled=sp_enabled,
                        sp_group=sp_group,
                        sp_size=sp_size,
                        sp_rank=sp_rank,
                        cu_seqlens=cu_eff,
                        max_seqlen=max_sl,
                    )
                return img_in, txt_in

            def _run_single_blocks(
                x_in: torch.Tensor,
                vec_in: torch.Tensor,
                cos_in: torch.Tensor | None,
                sin_in: torch.Tensor | None,
                cu_dev: torch.Tensor | None,
                cu_host: tuple[int, ...] | None,
                txt_len_val: int,
                max_sl: int,
                sink_tok_val: int,
                sink_start_val: int,
            ) -> torch.Tensor:
                cu_eff = torch.tensor(cu_host, dtype=torch.int32) if cu_host is not None else cu_dev
                freqs_pair = (cos_in, sin_in) if cos_in is not None and sin_in is not None else None
                for blk_idx, blk in enumerate(single_blocks):
                    blk_sol = bool(step_uses_sol and blk_idx not in self.dense_single_indices)
                    x_in = blk._forward_accelerated(
                        x_in,
                        vec_in,
                        freqs_pair,
                        txt_len=txt_len_val,
                        runtime=self,
                        unmasked=unmasked,
                        use_sol_attn=blk_sol,
                        sink_tokens=sink_tok_val,
                        sink_start=sink_start_val,
                        sp_enabled=sp_enabled,
                        sp_group=sp_group,
                        sp_size=sp_size,
                        sp_rank=sp_rank,
                        cu_seqlens=cu_eff,
                        max_seqlen=max_sl,
                    )
                return x_in

            if graph_runner is not None:
                double_seg = f"double_blocks_{mode_tag}"
                single_seg = f"single_blocks_{mode_tag}"
                img_out, txt_out = self._run_graph_segment_tracked(
                    graph_runner,
                    double_seg,
                    _run_double_blocks,
                    img_work,
                    txt,
                    vec,
                    cos_work,
                    sin_work,
                    cu_seqlens,
                    cu_tuple,
                    max_seqlen,
                    total_sink_tokens,
                    sink_start,
                )
                x_joint = torch.cat((img_out, txt_out), dim=1)
                if len(single_blocks) > 0:
                    x_joint = self._run_graph_segment_tracked(
                        graph_runner,
                        single_seg,
                        _run_single_blocks,
                        x_joint,
                        vec,
                        cos_work,
                        sin_work,
                        cu_seqlens,
                        cu_tuple,
                        txt_seq_len,
                        max_seqlen,
                        total_sink_tokens,
                        sink_start,
                    )
            else:
                img_out, txt_out = _run_double_blocks(
                    img_work,
                    txt,
                    vec,
                    cos_work,
                    sin_work,
                    cu_seqlens,
                    cu_tuple,
                    max_seqlen,
                    total_sink_tokens,
                    sink_start,
                )
                x_joint = torch.cat((img_out, txt_out), dim=1)
                if len(single_blocks) > 0:
                    x_joint = _run_single_blocks(
                        x_joint,
                        vec,
                        cos_work,
                        sin_work,
                        cu_seqlens,
                        cu_tuple,
                        txt_seq_len,
                        max_seqlen,
                        total_sink_tokens,
                        sink_start,
                    )

            img_final = x_joint[:, :local_img_len]

            # 4. Gather SP shards back to canonical full video token sequence.
            if sp_enabled:
                img_final = _even_all_gather(img_final, dim=1, group=sp_group, world_size=int(sp_size))

            # 5. Inverse 3D-tile permutation back to canonical raster order.
            if tile_inv_perm is not None:
                img_final = img_final.index_select(1, tile_inv_perm)

            return img_final

        # 6. Cross-step TeaCache / EasyCache wrapping canonical unpermuted full video tokens.
        cache = None if self.model.training else self._get_cache(req_id, branch, resolved_total)
        if cache is not None:
            modulated_inp = self._compute_img_mod1_probe(img, vec)
            decision_sync: Callable[[bool], bool] | None = None
            if sp_enabled and sp_group is not None:
                import torch.distributed as dist

                def _sp_sync_cache_decision(local_hit: bool) -> bool:
                    # Unanimous consensus (logical AND via MIN): every SP rank enters
                    # collective attention together if ANY rank requires a dense refresh.
                    flag = torch.tensor([1 if local_hit else 0], dtype=torch.int32, device=img.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=sp_group)
                    return bool(int(flag.item()) == 1)

                decision_sync = _sp_sync_cache_decision

            img_result, hit = cache.evaluate_and_run(
                step=resolved_step,
                total_steps=resolved_total,
                img_in=img,
                modulated_inp=modulated_inp,
                num_blocks=total_blocks,
                run_stack=_execute_full_stack,
                decision_sync=decision_sync,
            )
            if hit and graph_runner is not None:
                mask_tag = "unmasked" if unmasked else "varlen"
                mode_tag = f"{'sol' if step_uses_sol else 'dense'}_{mask_tag}"
                if sp_enabled:
                    mode_tag = f"{mode_tag}_sp{sp_size}"
                graph_runner.record_skip(f"double_blocks_{mode_tag}")
                if len(single_blocks) > 0:
                    graph_runner.record_skip(f"single_blocks_{mode_tag}")
            return img_result

        return _execute_full_stack()

    def receipt(self) -> dict[str, Any]:
        """Return complete JSON-serializable execution receipt."""
        fp8_lp_calls = 0
        fp8_dense_calls = 0
        fp8_modules = 0
        fp8_fast_accum_modules = 0
        for mod in self.model.modules():
            if isinstance(mod, Float8Linear):
                fp8_modules += 1
                if mod.use_fast_accum:
                    fp8_fast_accum_modules += 1
                rep = mod.runtime_report()
                fp8_lp_calls += int(cast(int, rep["low_precision_kernel_calls"]))
                fp8_dense_calls += int(cast(int, rep["dense_fallback_calls"]))
        logical_eager_fp8_lp = max(fp8_lp_calls - self._fp8_capture_overcount, 0)

        graph_rep: dict[str, Any] | None = None
        if self.piecewise_graph is not None:
            raw = self.piecewise_graph.report()
            segments = raw.get("segments", {})
            graph_rep = {
                **raw,
                "capture_failed": sum(int(seg.get("capture_failed", 0)) for seg in segments.values()),
            }

        cache_receipts = {f"{req}:{br}": c.receipt() for (req, br), c in self._caches.items()}
        total_cache_hits = sum(c.hits for c in self._caches.values())
        total_cache_events = sum(len(c.events) for c in self._caches.values())
        sp_enabled, _, sp_size, sp_rank = _resolve_sp_context(self.config.sequence_parallel)

        fallbacks = list(self._fallbacks)
        if fp8_dense_calls > 0:
            fallbacks.append(f"fp8_dense_fallback_calls={fp8_dense_calls}")
        if graph_rep is not None and int(graph_rep["capture_failed"]) > 0:
            fallbacks.append(f"cuda_graph_capture_failed={graph_rep['capture_failed']}")

        return {
            "engine": "worldfoundry-hunyuan13b-sol",
            "config": self.config.to_dict(),
            "forward_calls": self.forward_calls,
            "projection_fusion": dict(self.projection_fusion_report),
            "quantization": {
                **(self.quantization_report or {"mode": "none"}),
                "fp8_module_count": fp8_modules,
                "fp8_fast_accum_modules": fp8_fast_accum_modules,
                "low_precision_kernel_calls": logical_eager_fp8_lp + self.replayed_fp8_kernel_calls,
                "eager_low_precision_kernel_calls": logical_eager_fp8_lp,
                "replayed_low_precision_kernel_calls": self.replayed_fp8_kernel_calls,
                "dense_fallback_calls": fp8_dense_calls,
                "fused_fp8_ffn_calls": self.fused_fp8_ffn_calls,
                "adaln_fp8_calls": self.adaln_fp8_calls,
            },
            "attention": {
                "configured_backend": self.config.attention_backend,
                "dense_backend": self.dense_backend,
                "sol_attn_enabled": self.use_sol_attn,
                "fa3_calls": self.fa3_calls,
                "sol_attn_calls": self.sol_attn_calls,
                "torch_sdpa_calls": self.torch_sdpa_calls,
                "text_compactions": self.text_compactions,
                "sol_sparse_steps": self.sol_sparse_steps,
                "sol_dense_warmup_steps": self.sol_dense_warmup_steps,
                "dense_double_blocks": sorted(self.dense_double_indices),
                "dense_single_blocks": sorted(self.dense_single_indices),
            },
            "fused_ops": {
                "enabled": bool(self.config.fused_ops),
                "qkv_rmsnorm_rope_calls": self.qkv_rmsnorm_rope_calls,
                "adaln_calls": self.adaln_calls,
                "adaln_fp8_calls": self.adaln_fp8_calls,
                "dual_res_gate_calls": self.dual_res_gate_calls,
                "fused_fp8_ffn_calls": self.fused_fp8_ffn_calls,
            },
            "cache": {
                "algorithm": self.config.cache_algorithm,
                "events": total_cache_events,
                "hits": total_cache_hits,
                "misses": total_cache_events - total_cache_hits,
                "hit_rate": total_cache_hits / total_cache_events if total_cache_events else 0.0,
                "branches": cache_receipts,
            },
            "cuda_graph": graph_rep,
            "sequence_parallel": {
                "enabled": sp_enabled,
                "sp_size": sp_size,
                "sp_rank": sp_rank,
                "all_to_all_calls": self.sp_all_to_all_calls,
            },
            "fallbacks": fallbacks,
        }


def apply_hunyuan13b_acceleration(
    model: nn.Module,
    config: Hunyuan13BAccelerationConfig | str = "l4_sol_full",
) -> Hunyuan13BAccelerationRuntime:
    """Apply the full SOL-Engine acceleration stack to a ``HunyuanVideoDiT`` instance."""
    resolved_cfg = hunyuan13b_sol_profile(config) if isinstance(config, str) else config
    if not isinstance(resolved_cfg, Hunyuan13BAccelerationConfig):
        raise TypeError(f"expected Hunyuan13BAccelerationConfig or profile string, got {type(config).__name__}")

    existing_fp8 = [mod for mod in model.modules() if isinstance(mod, Float8Linear)]
    if resolved_cfg.quantization == "none" and existing_fp8:
        if any(mod.weight is None for mod in existing_fp8):
            raise RuntimeError(
                "cannot apply quantization='none' to a model already quantized with "
                "keep_dense_fallback=False; reload fresh BF16 weights first"
            )
        for mod in existing_fp8:
            mod.low_precision_enabled = False
    elif resolved_cfg.quantization == "fp8":
        first_param = next(model.parameters(), None)
        if (
            first_param is not None
            and first_param.device.type == "cuda"
            and not _fp8_hardware_eligible(first_param.device)
        ):
            raise RuntimeError(f"FP8 requested on ineligible CUDA device {first_param.device}")
        for mod in existing_fp8:
            mod.low_precision_enabled = True

    prev_runtime = getattr(model, "_sol_runtime", None)
    if prev_runtime is not None and callable(getattr(prev_runtime, "close", None)):
        prev_runtime.close()

    for block in getattr(model, "double_blocks", ()):
        block.component_a._fuse_fp8_ffn = False
        block.component_b._fuse_fp8_ffn = False
    for block in getattr(model, "single_blocks", ()):
        block._fuse_fp8_ffn = False

    fusion_report = (
        fuse_hunyuan13b_projections(model, release_originals=True)
        if resolved_cfg.fuse_projections
        else {"double_qkv_fused_verified": 0, "single_linear1_fused": 0}
    )

    quant_report: dict[str, Any] | None = None
    if resolved_cfg.quantization == "fp8":
        quant_report = quantize_hunyuan13b_fp8(
            model,
            scaling=resolved_cfg.fp8_scaling,
            use_fast_accum=resolved_cfg.fp8_use_fast_accum,
            keep_dense_fallback=resolved_cfg.keep_dense_fallback,
            min_features=resolved_cfg.quant_min_features,
            exclude_edge_blocks=resolved_cfg.quant_exclude_edge_blocks,
            quant_txt_stream=resolved_cfg.quant_txt_stream,
            fuse_fp8_ffn=resolved_cfg.fuse_fp8_ffn,
            exclude=resolved_cfg.quant_exclude,
        )

    runtime = Hunyuan13BAccelerationRuntime(
        model,
        resolved_cfg,
        projection_fusion_report=fusion_report,
        quantization_report=quant_report,
    )

    # Bind flags on blocks and set attention_backend attribute cleanly.
    if hasattr(model, "set_attention_backend"):
        cast(Any, model).set_attention_backend(resolved_cfg.attention_backend)
    for block in getattr(model, "double_blocks", ()):
        block._fused_ops = bool(resolved_cfg.fused_ops)
        block.component_a._fused_ops = bool(resolved_cfg.fused_ops)
        block.component_b._fused_ops = bool(resolved_cfg.fused_ops)
    for block in getattr(model, "single_blocks", ()):
        block._fused_ops = bool(resolved_cfg.fused_ops)

    cast(Any, model)._sol_runtime = runtime
    return runtime


__all__ = [
    "DEFAULT_HUNYUAN13B_QUANT_EXCLUDES",
    "OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS",
    "Hunyuan13BAccelerationConfig",
    "Hunyuan13BAccelerationRuntime",
    "Hunyuan13BEasyCache",
    "Hunyuan13BSolAttnConfig",
    "Hunyuan13BTeaCache",
    "accelerate_hunyuan_video_vae",
    "apply_hunyuan13b_acceleration",
    "build_hunyuan_sol_tile_permutation",
    "fuse_hunyuan13b_projections",
    "fused_hunyuan_double_qkv_rmsnorm_rope",
    "fused_hunyuan_dual_residual_gate_add",
    "fused_hunyuan_gelu_tanh_rowwise_fp8",
    "fused_hunyuan_layernorm_scale_shift_fp8",
    "fused_hunyuan_qkv_rmsnorm_rope",
    "fused_hunyuan_residual_gate_layernorm_scale_shift",
    "hunyuan13b_sol_profile",
    "quantize_hunyuan13b_fp8",
    "run_adaln_linear",
    "run_fused_fp8_gelu_down_proj",
    "run_maskless_joint_attention",
    "run_sp_maskless_joint_attention",
]
