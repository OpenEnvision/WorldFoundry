"""MiniMax H3 fused-operator surface with portable PyTorch fallbacks.

Consumed by :class:`dit.MiniMaxH3DiTModel` for per-head QK RMSNorm, 3-D
RoPE, SwiGLU, and indexed AdaLN.  Triton kernels behind
``can_use_fused_*`` are optional; these bodies are the numerical reference.

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

import os

import torch
import torch.nn as nn

from worldfoundry.core.attention.rotary.embedding import rotate_half_chunked as _rotate_half

_BF16 = torch.bfloat16
_FP32 = torch.float32


# --------------------------------------------------------------------------- #
# Fused-kernel probes (fallbacks are exact).
# --------------------------------------------------------------------------- #
_FUSED_QKNORM_ROPE_HEAD_DIMS = frozenset({64, 128, 256})


def _fused_backend_enabled() -> bool:
    """Whether fused kernels are permitted (env override + torch.compile guard)."""

    if torch.compiler.is_compiling():
        return False
    requested = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold() or "auto"
    return requested not in {"torch", "pytorch", "native", "off", "disabled"}


def _resolve_rms_norm_eps(norm: nn.RMSNorm, dtype: torch.dtype) -> float:
    """Resolve ``nn.RMSNorm.eps`` matching PyTorch when ``eps is None``."""

    if norm.eps is not None:
        return float(norm.eps)
    return float(torch.finfo(dtype).eps)


def _autograd_active(*tensors: torch.Tensor | None) -> bool:
    """Return True when autograd must be preserved for any operand."""

    if not torch.is_grad_enabled():
        return False
    return any(t is not None and t.requires_grad for t in tensors)


def can_use_fused_qknorm_rope(q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor) -> bool:
    """Return whether the single-kernel strided Triton qk-norm+RoPE is eligible.

    Requires CUDA bf16 ``[T, num_heads, head_dim]`` q/k and bf16 ``[T, rot_dim]``
    ``cos_sin_cache`` with unit innermost stride (supporting direct strided slices
    from a fused QKV projection), a supported head_dim, and an even
    ``rot_dim <= head_dim`` matching ``T``.
    """

    if not _fused_backend_enabled():
        return False
    if not (q.is_cuda and k.is_cuda and cos_sin_cache.is_cuda):
        return False
    if q.device != k.device or q.device != cos_sin_cache.device:
        return False
    if q.dtype != _BF16 or k.dtype != _BF16 or cos_sin_cache.dtype != _BF16:
        return False
    if q.ndim != 3 or q.shape != k.shape or cos_sin_cache.ndim != 2:
        return False
    if int(cos_sin_cache.shape[0]) != int(q.shape[0]):
        return False
    head_dim = int(q.shape[-1])
    if head_dim not in _FUSED_QKNORM_ROPE_HEAD_DIMS:
        return False
    rot_dim = int(cos_sin_cache.shape[-1])
    return (
        rot_dim > 0
        and rot_dim % 2 == 0
        and rot_dim <= head_dim
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and cos_sin_cache.stride(-1) == 1
    )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm then RoPE, using the fused kernel when eligible."""

    if (
        can_use_fused_qknorm_rope(q, k, cos_sin_cache)
        and q_norm.weight is not None
        and k_norm.weight is not None
        and q_norm.weight.is_cuda
        and k_norm.weight.is_cuda
        and q_norm.weight.device == q.device
        and k_norm.weight.device == q.device
        and q_norm.weight.dtype == _BF16
        and k_norm.weight.dtype == _BF16
        and not _autograd_active(q, k, cos_sin_cache, q_norm.weight, k_norm.weight)
    ):
        q_eps = _resolve_rms_norm_eps(q_norm, q.dtype)
        k_eps = _resolve_rms_norm_eps(k_norm, k.dtype)
        if q_eps == k_eps:
            from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_qknorm_rope import (
                fused_qknorm_rope,
            )

            return fused_qknorm_rope(q, k, q_norm.weight, k_norm.weight, cos_sin_cache, eps=q_eps)
    q, k = apply_qk_norm(q, k, q_norm, k_norm, q.shape[-1])
    return apply_rope_qk(q, k, cos_sin_cache, positions)


def can_use_fused_indexed_modulation(*tensors: torch.Tensor) -> bool:
    """Return whether the fused indexed AdaLN Triton kernels are eligible.

    Requires all operands on the same CUDA device, bf16, with unit innermost
    stride (``stride(-1) == 1``) so row-strided slices from
    ``adaln_proj(...).chunk(...)`` dispatch directly without copying. Callers
    also enforce autograd safety and exact 2D/1D rank/shape compatibility
    before launching the Triton kernel.
    """

    if not tensors or not _fused_backend_enabled():
        return False
    first = tensors[0]
    if not first.is_cuda or first.dtype != _BF16:
        return False
    device = first.device
    return all(
        t.is_cuda and t.device == device and t.dtype == _BF16 and t.ndim in (1, 2) and t.stride(-1) == 1
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
# Indexed AdaLN modulation & fused RMSNorm + indexed AdaLN.
# --------------------------------------------------------------------------- #
def _as_index_tensor(indices: torch.Tensor) -> torch.Tensor:
    return indices if indices.dtype == torch.int64 else indices.to(torch.int64)


def indexed_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``x * (1 + scale[idx]) + shift[idx]`` cast to ``dtype``."""

    if (
        dtype == _BF16
        and x.ndim == 2
        and shift.ndim == 2
        and scale.ndim == 2
        and indices.ndim == 1
        and indices.is_cuda
        and indices.device == x.device
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == shift.shape[1] == scale.shape[1]
        and not _autograd_active(x, shift, scale)
        and can_use_fused_indexed_modulation(x, shift, scale)
    ):
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_modulation import (
            indexed_scale_shift_bf16,
        )

        return indexed_scale_shift_bf16(x, shift, scale, _as_index_tensor(indices))
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)


def rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    norm: nn.RMSNorm,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``norm(x) * (1 + scale[idx]) + shift[idx]`` using one fused kernel when eligible."""

    weight = norm.weight
    if (
        dtype == _BF16
        and isinstance(norm, nn.RMSNorm)
        and weight is not None
        and x.ndim == 2
        and weight.ndim == 1
        and shift.ndim == 2
        and scale.ndim == 2
        and indices.ndim == 1
        and indices.is_cuda
        and indices.device == x.device
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == weight.shape[0] == shift.shape[1] == scale.shape[1]
        and not _autograd_active(x, weight, shift, scale)
        and can_use_fused_indexed_modulation(x, weight, shift, scale)
    ):
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_modulation import (
            rms_norm_indexed_scale_shift_bf16,
        )

        return rms_norm_indexed_scale_shift_bf16(
            x,
            weight,
            shift,
            scale,
            _as_index_tensor(indices),
            eps=_resolve_rms_norm_eps(norm, x.dtype),
        )
    h = norm(x)
    return indexed_scale_shift(h, shift, scale, indices, dtype=dtype)


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the gated residual ``x + gate[idx] * other`` cast to ``dtype``."""

    if (
        dtype == _BF16
        and x.ndim == 2
        and gate.ndim == 2
        and other.ndim == 2
        and indices.ndim == 1
        and indices.is_cuda
        and indices.device == x.device
        and x.shape == other.shape
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == gate.shape[1]
        and not _autograd_active(x, gate, other)
        and can_use_fused_indexed_modulation(x, gate, other)
    ):
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_modulation import (
            indexed_gate_bf16,
        )

        return indexed_gate_bf16(x, gate, other, _as_index_tensor(indices))
    return (x + gate.index_select(0, indices) * other).to(dtype)


def indexed_gate_rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    norm: nn.RMSNorm,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_out, norm(x_out) * (1 + scale[idx]) + shift[idx])`` in one kernel when eligible."""

    weight = norm.weight
    if (
        dtype == _BF16
        and isinstance(norm, nn.RMSNorm)
        and weight is not None
        and x.ndim == 2
        and gate.ndim == 2
        and other.ndim == 2
        and weight.ndim == 1
        and shift.ndim == 2
        and scale.ndim == 2
        and indices.ndim == 1
        and indices.is_cuda
        and indices.device == x.device
        and x.shape == other.shape
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == gate.shape[1] == weight.shape[0] == shift.shape[1] == scale.shape[1]
        and not _autograd_active(x, gate, other, weight, shift, scale)
        and can_use_fused_indexed_modulation(x, gate, other, weight, shift, scale)
    ):
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_modulation import (
            indexed_gate_rms_norm_indexed_scale_shift_bf16,
        )

        return indexed_gate_rms_norm_indexed_scale_shift_bf16(
            x,
            gate,
            other,
            weight,
            shift,
            scale,
            _as_index_tensor(indices),
            eps=_resolve_rms_norm_eps(norm, x.dtype),
        )
    x_out = indexed_gate(x, gate, other, indices, dtype=dtype)
    h_out = rms_norm_indexed_scale_shift(x_out, norm, shift, scale, indices, dtype=dtype)
    return x_out, h_out


def make_rms_norm(size: int, *, eps: float, dtype: torch.dtype = _BF16) -> nn.RMSNorm:
    """Build an RMSNorm with fp32-accumulation semantics (reference contract)."""

    return nn.RMSNorm(size, eps=eps, dtype=dtype)


__all__ = [
    "apply_qk_norm",
    "apply_rope_qk",
    "can_use_fused_indexed_modulation",
    "can_use_fused_qknorm_rope",
    "indexed_gate",
    "indexed_gate_rms_norm_indexed_scale_shift",
    "indexed_scale_shift",
    "make_rms_norm",
    "qk_norm_rope",
    "rms_norm_indexed_scale_shift",
    "rope_cos_sin_cache",
    "silu_mul",
]
