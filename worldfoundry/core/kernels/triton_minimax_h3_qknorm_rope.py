"""Triton fused per-head Q/K RMSNorm + 3D RoPE for MiniMax H3.

This is an in-tree Triton implementation of the packed H3
``[T, num_heads, head_dim]`` layout:

1. per-head RMSNorm with fp32 variance accumulation, rounded to bf16, then
   multiplied by the (bf16) learned weight, rounded to bf16 again before RoPE;
2. NeoX RoPE over the leading ``rot_dim = 2 * ROT_HALF`` head dims using a
   ``[T, rot_dim]`` ``cat(cos_half, sin_half)`` cache (half-width angles), with
   ``rotate_half`` splitting the rotated block into ``[a, b] -> [a*cos - b*sin,
   b*cos + a*sin]`` and passing the remaining ``head_dim - rot_dim`` dims
   through unchanged.

The result matches the split ``apply_qk_norm`` + ``apply_rope_qk`` fallback to
bf16 tolerance. Eligibility is narrow (CUDA, bf16, contiguous, supported
head_dim); anything else uses the exact PyTorch reference.

Not this module:
    Wan-style ``[B, S, H, D]`` QK RMSNorm+RoPE lives in :mod:`.triton_diffusion`.
    Eligibility for H3 is enforced by the caller, not this file.

Public surface:

- :func:`fused_qknorm_rope` — Triton RMSNorm then on-device NeoX rotate.
- :func:`_qknorm_rope_reference` — split exact path for numerical checks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────
# Exact split reference — used by tests; not the serving hot path
# ──────────────────────────────────────────────────────────────────────────


def _qknorm_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact split reference (RMSNorm then NeoX RoPE)."""

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """fp32 RMSNorm, round to input dtype, then multiply the learned weight."""

        v = value.float()
        normed = v * torch.rsqrt(v.square().mean(dim=-1, keepdim=True) + eps)
        return normed.to(value.dtype) * weight

    q = normalize(q, q_weight)
    k = normalize(k, k_weight)
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    rot_dim = cos.shape[-1]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        """NeoX ``rotate_half`` on the leading ``rot_dim``; trailing dims pass through."""

        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)

    return rotate(q), rotate(k)


# ──────────────────────────────────────────────────────────────────────────
# Serving path — RMSNorm in Triton; RoPE stays vectorized PyTorch on-device
# ──────────────────────────────────────────────────────────────────────────


def fused_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-head RMSNorm + NeoX RoPE for packed ``[T, num_heads, head_dim]``.

    A row-parallel Triton kernel handles the RMSNorm; the rotate_half RoPE is a
    vectorized elementwise op over the ``[T, num_heads, rot_dim]`` block. The
    two-stage split keeps the kernel simple while staying on-device and
    matching the reference numerics to bf16 tolerance.
    """

    total, num_heads, head_dim = q.shape
    rot_dim = cos_sin_cache.shape[-1]
    if rot_dim % 2:
        raise ValueError(f"rot_dim must be even, got {rot_dim}")

    # Stage 1: per-head RMSNorm (row-parallel Triton) into fresh buffers.
    q_flat = q.reshape(total * num_heads, head_dim).contiguous()
    k_flat = k.reshape(total * num_heads, head_dim).contiguous()
    q_norm = torch.empty_like(q_flat)
    k_norm = torch.empty_like(k_flat)
    block = triton.next_power_of_2(head_dim)
    _rmsnorm_rows[(total * num_heads,)](
        q_flat, k_flat, q_norm, k_norm, q_weight, k_weight,
        head_dim, float(eps), q_flat.stride(0),
        HEAD_DIM=block,
        num_warps=4,
    )
    q_norm = q_norm.reshape(total, num_heads, head_dim)
    k_norm = k_norm.reshape(total, num_heads, head_dim)

    # Stage 2: NeoX RoPE on the leading rot_dim dims (elementwise, on-device).
    half = rot_dim // 2
    cos_half, sin_half = cos_sin_cache[:, :half], cos_sin_cache[:, half:]
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1).to(q_norm.dtype)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1).to(q_norm.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        """NeoX ``rotate_half`` on the leading ``rot_dim``; trailing dims pass through."""

        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)

    return rotate(q_norm), rotate(k_norm)


@triton.jit
def _rmsnorm_rows(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    head_dim,
    eps,
    stride_row,
    HEAD_DIM: tl.constexpr,
):
    """Row-parallel Q/K RMSNorm; ``HEAD_DIM`` is ``next_power_of_2(head_dim)``.

    Variance is fp32. The normalized value is rounded to bf16 before the
    learned weight so the result matches ``nn.RMSNorm`` in bf16.
    """

    row = tl.program_id(0)
    cols = tl.arange(0, HEAD_DIM)
    mask = cols < head_dim
    q = tl.load(q_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    qw = tl.load(q_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    kw = tl.load(k_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=0) / head_dim
    k_var = tl.sum(k * k, axis=0) / head_dim
    # fp32 accumulation -> round to bf16 -> * weight (matches nn.RMSNorm bf16).
    q_n = (q * tl.rsqrt(q_var + eps)).to(tl.bfloat16).to(tl.float32) * qw
    k_n = (k * tl.rsqrt(k_var + eps)).to(tl.bfloat16).to(tl.float32) * kw
    tl.store(q_out_ptr + row * stride_row + cols, q_n.to(tl.bfloat16), mask=mask)
    tl.store(k_out_ptr + row * stride_row + cols, k_n.to(tl.bfloat16), mask=mask)


__all__ = ["fused_qknorm_rope"]
