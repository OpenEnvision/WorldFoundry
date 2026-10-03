"""Explicit Sol-Attn provider; optional package imports occur only on execution.

The supported contract is noncausal, unmasked BF16 self-attention, head_dim=128.
This is approximate attention, so callers must opt in and validate their own
model/trajectory. Dependency and shape errors never silently become dense SDPA.
"""

from __future__ import annotations

import math

import torch


def validate_sol_options(*, tau=1.0, thresh_type="diag", kv_splits=1, compress_kv=True) -> dict[str, object]:
    """Validate provider parameters before installing a model policy."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau < 0:
        raise ValueError("Sol-Attn tau must be finite and non-negative")
    if not isinstance(thresh_type, str) or thresh_type not in {"diag", "exact"}:
        raise ValueError("Sol-Attn thresh_type must be diag or exact")
    if isinstance(kv_splits, bool) or not isinstance(kv_splits, int) or kv_splits != 1:
        raise ValueError("WorldFoundry currently validates Sol-Attn with kv_splits=1 only")
    if type(compress_kv) is not bool:
        raise ValueError("Sol-Attn compress_kv must be a boolean")
    return {"tau": float(tau), "thresh_type": thresh_type, "kv_splits": kv_splits, "compress_kv": compress_kv}


@torch.library.custom_op("worldfoundry::sol_attention_forward", mutates_args=(), device_types="cuda")
def _sol_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    tau: float,
    thresh_type: str,
    kv_splits: int,
    compress_kv: bool,
) -> torch.Tensor:
    from sol_attn import sol_attn

    # The package's public sink range keeps overlapping KV blocks on its full
    # attention path. Covering the entire sequence disables mean compression;
    # it does not promise bitwise equivalence with another attention kernel.
    sink_tokens = 0 if compress_kv else q.shape[1]
    return sol_attn(
        q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, kv_splits=kv_splits, sink_tokens=sink_tokens
    )


@_sol_attention_forward.register_fake
def _sol_attention_fake(q, k, v, scale, tau, thresh_type, kv_splits, compress_kv):
    return torch.empty_like(q)


def sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    compress_kv: bool = True,
) -> torch.Tensor:
    """Run BTHD Sol-Attn; opt out of KV compression with ``compress_kv=False``.

    ``tau=0`` still applies mean-threshold routing when compression is enabled.
    The uncompressed BF16 path retains the alternate kernel's rounding.
    """
    from worldfoundry.core.attention.backends.probe import resolve_attention_backend

    resolve_attention_backend("sol_attn", q.device)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) <= 0 or q.shape[-1] != 128:
        raise ValueError("Sol-Attn requires equal nonempty [B,T,H,128] self-attention tensors")
    if q.device.type != "cuda" or any(item.device != q.device or item.dtype != torch.bfloat16 for item in (q, k, v)):
        raise ValueError("Sol-Attn requires BF16 Q/K/V on one NVIDIA CUDA device")
    if torch.is_grad_enabled():
        raise RuntimeError("Sol-Attn is inference-only; use no_grad or inference_mode")
    validate_sol_options(tau=tau, thresh_type=thresh_type, kv_splits=kv_splits, compress_kv=compress_kv)
    return _sol_attention_forward(
        q.contiguous(), k.contiguous(), v.contiguous(), scale, float(tau), thresh_type, kv_splits, compress_kv
    )
