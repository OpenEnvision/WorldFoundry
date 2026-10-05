"""MiniMax H3 fused-operator surface with portable PyTorch fallbacks.

Consumed by :class:`dit.MiniMaxH3DiTModel` for per-head QK RMSNorm, 3-D
RoPE, SwiGLU, and indexed AdaLN.  Triton kernels behind
``can_use_fused_*`` are optional; these bodies are the numerical reference.

Every function here mirrors the exact math of the SGLang MiniMax H3 DiT
(``sglang.multimodal_gen.runtime.models.dits.minimax_h3``) but uses only
plain PyTorch so the model runs on any device without the SGLang custom
CUDA/Triton kernels. Stage 8 of the integration wires the fused kernels behind
the ``can_use_fused_*`` probes; until then those probes return ``False`` and
the fallbacks below are always taken.

Numerics that must not drift from the reference:

* qk RMSNorm accumulates the variance in fp32, rounds back to the input dtype,
  then multiplies by the (input-dtype) learned weight — this is ``nn.RMSNorm``
  semantics, not ``F.rms_norm`` mixed-weight promotion.
* AdaLN modulation is ``x * (1 + scale[idx]) + shift[idx]`` and the gated
  residual is ``x + gate[idx] * other``, both selecting per-row modality
  vectors via ``index_select``.
* 3D RoPE concatenates the ``[t|h|w]`` half-frequencies twice to form a
  ``rot_dim``-wide table, rotates the first ``rot_dim`` head dims with
  ``rotate_half`` (NeoX layout), and passes the remaining dims through.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_BF16 = torch.bfloat16
_FP32 = torch.float32


# --------------------------------------------------------------------------- #
# Fused-kernel probes (Stage 8 replaces the bodies; fallbacks are exact).
# --------------------------------------------------------------------------- #
_FUSED_QKNORM_ROPE_HEAD_DIMS = frozenset({64, 128, 256})


def can_use_fused_qknorm_rope(q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor) -> bool:
    """Return whether the fused Triton qk-norm+RoPE kernel is eligible.

    Requires CUDA bf16 contiguous ``[T, num_heads, head_dim]`` q/k with a
    supported head_dim and an even rot_dim. Otherwise the split
    ``apply_qk_norm`` + ``apply_rope_qk`` reference path (exact) is used.
    """

    if not _fused_backend_enabled():
        return False
    if not (q.is_cuda and k.is_cuda and cos_sin_cache.is_cuda):
        return False
    if q.dtype != _BF16 or k.dtype != _BF16:
        return False
    if q.ndim != 3 or q.shape != k.shape:
        return False
    head_dim = int(q.shape[-1])
    if head_dim not in _FUSED_QKNORM_ROPE_HEAD_DIMS:
        return False
    rot_dim = int(cos_sin_cache.shape[-1])
    return rot_dim > 0 and rot_dim % 2 == 0 and rot_dim <= head_dim and q.is_contiguous() and k.is_contiguous()


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm then RoPE, using the fused kernel when eligible."""

    if can_use_fused_qknorm_rope(q, k, cos_sin_cache):
        from worldfoundry.core.kernels.triton_minimax_h3_qknorm_rope import fused_qknorm_rope

        return fused_qknorm_rope(
            q, k, q_norm.weight, k_norm.weight, cos_sin_cache, eps=float(q_norm.eps)
        )
    q, k = apply_qk_norm(q, k, q_norm, k_norm, q.shape[-1])
    return apply_rope_qk(q, k, cos_sin_cache, positions)


def _fused_backend_enabled() -> bool:
    """Whether fused kernels are permitted (env override + torch.compile guard)."""

    import os

    if torch.compiler.is_compiling():
        return False
    requested = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold() or "auto"
    return requested not in {"torch", "pytorch", "native", "off", "disabled"}


def can_use_fused_indexed_modulation(*tensors: torch.Tensor) -> bool:
    """Return whether the fused indexed AdaLN Triton kernels are eligible.

    Requires all operands on CUDA, bf16, and contiguous rows — the exact
    contract of the ported Triton kernels. Any other case falls back to the
    portable PyTorch math (which is the numerical reference).
    """

    if not tensors or not _fused_backend_enabled():
        return False
    first = tensors[0]
    if not first.is_cuda or first.dtype != _BF16:
        return False
    return all(
        t.is_cuda and t.dtype == _BF16 and t.is_contiguous() and t.stride(-1) == 1
        for t in tensors
    )


# --------------------------------------------------------------------------- #
# SwiGLU activation.
# --------------------------------------------------------------------------- #
def silu_mul(hidden: torch.Tensor) -> torch.Tensor:
    """Return ``silu(gate) * up`` for a fused ``[..., 2 * d]`` tensor.

    Delegates to the shared in-tree ``silu_and_mul`` operator, which already
    carries a Triton fast path plus the exact PyTorch fallback.
    """

    from worldfoundry.core.kernels import silu_and_mul

    return silu_and_mul(hidden)


# --------------------------------------------------------------------------- #
# Per-head Q/K RMSNorm.
# --------------------------------------------------------------------------- #
def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to ``q``/``k`` shaped ``[T, num_heads, head_dim]``."""

    del head_dim  # kept for signature parity with the fused kernel
    return q_norm(q), k_norm(k)


# --------------------------------------------------------------------------- #
# 3D RoPE (concat-doubled half frequencies, NeoX rotate_half).
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Build the ``[S, rot_dim]`` cos|sin cache from ``[S, rot_dim]`` freqs.

    ``freqs`` already contains the ``[t|h|w]`` half concatenated twice, so its
    first half is the unique set of angles; the cache stores
    ``cat(cos(half), sin(half))`` in the activation dtype.
    """

    half = freqs.shape[-1] // 2
    return (
        torch.cat((torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])), dim=-1)
        .to(dtype=dtype, copy=False)
        .contiguous()
    )


def _apply_rope_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading ``rot_dim`` dims of ``q``/``k`` ``[T, num_heads, D]``.

    ``cos_sin_cache`` is ``[T, rot_dim]`` = ``cat(cos_half, sin_half)``; it is
    re-doubled to full ``rot_dim`` cos/sin and broadcast over the head axis.
    ``positions`` is accepted for signature parity with the fused kernel and is
    unused on the portable path (the cache is already position-aligned).
    """

    del positions
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    return _apply_rope_cos_sin(q, cos, sin), _apply_rope_cos_sin(k, cos, sin)


# --------------------------------------------------------------------------- #
# Indexed AdaLN modulation.
# --------------------------------------------------------------------------- #
def indexed_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``x * (1 + scale[idx]) + shift[idx]`` cast to ``dtype``."""

    if dtype == _BF16 and can_use_fused_indexed_modulation(x, shift, scale):
        from worldfoundry.core.kernels.triton_minimax_h3_modulation import (
            indexed_scale_shift_bf16_,
        )

        # Kernel is in-place; clone to preserve the functional (non-mutating)
        # contract callers rely on for the pre-norm activation.
        return indexed_scale_shift_bf16_(x.clone(), shift, scale, indices.to(torch.int64))
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the gated residual ``x + gate[idx] * other`` cast to ``dtype``."""

    if dtype == _BF16 and can_use_fused_indexed_modulation(x, gate, other):
        from worldfoundry.core.kernels.triton_minimax_h3_modulation import (
            indexed_gate_bf16_,
        )

        return indexed_gate_bf16_(x.clone(), gate, other, indices.to(torch.int64))
    return (x + gate.index_select(0, indices) * other).to(dtype)


def make_rms_norm(size: int, *, eps: float, dtype: torch.dtype = _BF16) -> nn.RMSNorm:
    """Build an RMSNorm with fp32-accumulation semantics (reference contract)."""

    return nn.RMSNorm(size, eps=eps, dtype=dtype)


__all__ = [
    "apply_qk_norm",
    "apply_rope_qk",
    "can_use_fused_indexed_modulation",
    "can_use_fused_qknorm_rope",
    "indexed_gate",
    "indexed_scale_shift",
    "make_rms_norm",
    "qk_norm_rope",
    "rope_cos_sin_cache",
    "silu_mul",
]
