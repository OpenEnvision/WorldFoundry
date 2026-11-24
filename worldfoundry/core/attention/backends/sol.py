"""Explicit Sol-Attn provider; optional package imports occur only on execution.

The supported contract is noncausal, unmasked BF16 self-attention, head_dim=128.
This is approximate attention, so callers must opt in and validate their own
model/trajectory. Dependency and shape errors never silently become dense SDPA.
"""

from __future__ import annotations

import math

import torch


def validate_sol_options(
    *,
    tau=1.0,
    thresh_type="diag",
    kv_splits=1,
    compress_kv=True,
    sink_tokens: int | None = None,
    sink_start: int | None = None,
    allow_multi_split: bool = False,
) -> dict[str, object]:
    """Validate provider parameters before installing a model policy."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau < 0:
        raise ValueError("Sol-Attn tau must be finite and non-negative")
    if not isinstance(thresh_type, str) or thresh_type not in {"diag", "exact"}:
        raise ValueError("Sol-Attn thresh_type must be diag or exact")
    allowed_splits = (1, 2, 4) if allow_multi_split else (1,)
    if isinstance(kv_splits, bool) or not isinstance(kv_splits, int) or kv_splits not in allowed_splits:
        if allow_multi_split:
            raise ValueError("Sol-Attn multi-split mode supports kv_splits in (1, 2, 4) on SM90")
        raise ValueError("WorldFoundry currently validates Sol-Attn with kv_splits=1 only")
    if type(compress_kv) is not bool:
        raise ValueError("Sol-Attn compress_kv must be a boolean")
    if sink_tokens is not None and (
        isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0
    ):
        raise ValueError("Sol-Attn sink_tokens must be None or a non-negative integer")
    if sink_start is not None and (isinstance(sink_start, bool) or not isinstance(sink_start, int) or sink_start < 0):
        raise ValueError("Sol-Attn sink_start must be None or a non-negative integer")
    out: dict[str, object] = {
        "tau": float(tau),
        "thresh_type": thresh_type,
        "kv_splits": kv_splits,
        "compress_kv": compress_kv,
    }
    if sink_tokens is not None:
        out["sink_tokens"] = int(sink_tokens)
    if sink_start is not None:
        out["sink_start"] = int(sink_start)
    if allow_multi_split:
        out["allow_multi_split"] = True
    return out


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
    sink_tokens: int,
    sink_start: int,
) -> torch.Tensor:
    sol_attn = __import__("sol_attn", fromlist=["sol_attn"]).sol_attn

    # The package's public sink range keeps overlapping KV blocks on its full
    # attention path. Covering the entire sequence disables mean compression;
    # it does not promise bitwise equivalence with another attention kernel.
    resolved_sink = int(sink_tokens) if sink_tokens >= 0 else (0 if compress_kv else int(q.shape[1]))
    resolved_sink = min(resolved_sink, int(q.shape[1]))
    resolved_start = None if sink_start < 0 else min(int(sink_start), max(int(q.shape[1]) - resolved_sink, 0))
    return sol_attn(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        sink_tokens=resolved_sink,
        sink_start=resolved_start,
    )


@_sol_attention_forward.register_fake
def _sol_attention_fake(q, k, v, scale, tau, thresh_type, kv_splits, compress_kv, sink_tokens, sink_start):
    del k, v, scale, tau, thresh_type, kv_splits, compress_kv, sink_tokens, sink_start
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
    sink_tokens: int | None = None,
    sink_start: int | None = None,
    allow_multi_split: bool = False,
) -> torch.Tensor:
    """Run BTHD Sol-Attn; opt out of KV compression with ``compress_kv=False``.

    ``tau=0`` still applies mean-threshold routing when compression is enabled.
    The uncompressed BF16 path retains the alternate kernel's rounding.
    Multi-split KV (``kv_splits in (2, 4)``) requires SM90 Hopper and
    ``allow_multi_split=True`` with ``kv_splits <= route_groups``.
    """
    from worldfoundry.core.attention.backends.probe import resolve_attention_backend

    resolve_attention_backend("sol_attn", q.device)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) <= 0 or q.shape[-1] != 128:
        raise ValueError("Sol-Attn requires equal nonempty [B,T,H,128] self-attention tensors")
    if q.device.type != "cuda" or any(item.device != q.device or item.dtype != torch.bfloat16 for item in (q, k, v)):
        raise ValueError("Sol-Attn requires BF16 Q/K/V on one NVIDIA CUDA device")
    if torch.is_grad_enabled():
        raise RuntimeError("Sol-Attn is inference-only; use no_grad or inference_mode")
    validate_sol_options(
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        compress_kv=compress_kv,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
        allow_multi_split=allow_multi_split,
    )
    if kv_splits > 1:
        arch = tuple(torch.cuda.get_device_capability(q.device))
        if arch != (9, 0):
            raise ValueError(f"Sol-Attn kv_splits={kv_splits} requires SM90 Hopper, got SM{arch[0]}{arch[1]}")
        route_groups = ((int(q.shape[1]) + 63) // 64 + 63) // 64
        if kv_splits > route_groups:
            raise ValueError(
                f"Sol-Attn kv_splits={kv_splits} exceeds route_groups={route_groups} for T={int(q.shape[1])}"
            )
    encoded_sink = -1 if sink_tokens is None else int(sink_tokens)
    encoded_start = -1 if sink_start is None else int(sink_start)
    return _sol_attention_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        scale,
        float(tau),
        thresh_type,
        kv_splits,
        compress_kv,
        encoded_sink,
        encoded_start,
    )
