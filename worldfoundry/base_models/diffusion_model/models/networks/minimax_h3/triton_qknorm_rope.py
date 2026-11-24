"""Triton fused per-head Q/K RMSNorm + 3D RoPE for MiniMax H3.

This is a single-pass, stride-aware Triton implementation for the packed H3
``[T, num_heads, head_dim]`` layout (including strided views sliced directly
from a fused QKV projection without an intermediate ``.contiguous()`` copy):

1. per-head RMSNorm with fp32 variance accumulation, rounded to bf16, then
   multiplied by the (bf16) learned weight;
2. NeoX RoPE over the leading ``rot_dim = 2 * ROT_HALF`` head dims using a
   ``[T, rot_dim]`` ``cat(cos_half, sin_half)`` cache (half-width angles), with
   ``rotate_half`` splitting the rotated block into ``[a, b] -> [a*cos - b*sin,
   b*cos + a*sin]`` and passing the remaining ``head_dim - rot_dim`` dims
   through unchanged.

Both stages execute inside one register-resident Triton kernel, eliminating
all intermediate HBM round-trips and PyTorch RoPE ``cat``/``mul``/``add``
launches.

Not this module:
    Wan-style ``[B, S, H, D]`` QK RMSNorm+RoPE lives in :mod:`worldfoundry.core.kernels.triton_diffusion`.
    Eligibility for H3 is enforced by the caller, not this file.

Public surface:

- :func:`fused_qknorm_rope` — single-kernel strided Triton RMSNorm + NeoX RoPE.
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
# Serving path — single-kernel strided Triton QK RMSNorm + 3D NeoX RoPE
# ──────────────────────────────────────────────────────────────────────────


@triton.jit
def _fused_qknorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    eps,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    out_stride_t,
    out_stride_h,
    cs_stride_t,
    HEAD_DIM: tl.constexpr,
    ROT_HALF: tl.constexpr,
    PASS_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
):
    """Fused per-(token, head) strided Q/K RMSNorm + NeoX RoPE in one HBM pass."""

    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)

    q_base = q_ptr + token * q_stride_t + head * q_stride_h
    k_base = k_ptr + token * k_stride_t + head * k_stride_h
    q_out_base = q_out_ptr + token * out_stride_t + head * out_stride_h
    k_out_base = k_out_ptr + token * out_stride_t + head * out_stride_h
    cs_base = cos_sin_ptr + token * cs_stride_t

    offs_h = tl.arange(0, BLOCK_HALF)
    mask_h = offs_h < ROT_HALF

    q1 = tl.load(q_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + ROT_HALF + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    k1 = tl.load(k_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    k2 = tl.load(k_base + ROT_HALF + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    qw1 = tl.load(q_weight_ptr + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
    qw2 = tl.load(q_weight_ptr + ROT_HALF + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
    kw1 = tl.load(k_weight_ptr + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
    kw2 = tl.load(k_weight_ptr + ROT_HALF + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)

    q_sum_sq = tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)
    k_sum_sq = tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)

    if PASS_DIM > 0:
        offs_p = tl.arange(0, BLOCK_PASS)
        mask_p = offs_p < PASS_DIM
        rot_dim = 2 * ROT_HALF
        qp = tl.load(q_base + rot_dim + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        kp = tl.load(k_base + rot_dim + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        qwp = tl.load(q_weight_ptr + rot_dim + offs_p, mask=mask_p, other=0.0).to(tl.bfloat16)
        kwp = tl.load(k_weight_ptr + rot_dim + offs_p, mask=mask_p, other=0.0).to(tl.bfloat16)
        q_sum_sq += tl.sum(qp * qp, axis=0)
        k_sum_sq += tl.sum(kp * kp, axis=0)

    inv_head_dim = 1.0 / HEAD_DIM
    q_rstd = tl.rsqrt(q_sum_sq * inv_head_dim + eps)
    k_rstd = tl.rsqrt(k_sum_sq * inv_head_dim + eps)

    # Match nn.RMSNorm(bf16): fp32 rsqrt -> round to bf16 -> multiply bf16 weight.
    q1_n = ((q1 * q_rstd).to(tl.bfloat16) * qw1).to(tl.float32)
    q2_n = ((q2 * q_rstd).to(tl.bfloat16) * qw2).to(tl.float32)
    k1_n = ((k1 * k_rstd).to(tl.bfloat16) * kw1).to(tl.float32)
    k2_n = ((k2 * k_rstd).to(tl.bfloat16) * kw2).to(tl.float32)

    cos_h = tl.load(cs_base + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16).to(tl.float32)
    sin_h = tl.load(cs_base + ROT_HALF + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16).to(tl.float32)

    q_rot1 = (q1_n * cos_h - q2_n * sin_h).to(tl.bfloat16)
    q_rot2 = (q2_n * cos_h + q1_n * sin_h).to(tl.bfloat16)
    k_rot1 = (k1_n * cos_h - k2_n * sin_h).to(tl.bfloat16)
    k_rot2 = (k2_n * cos_h + k1_n * sin_h).to(tl.bfloat16)

    tl.store(q_out_base + offs_h, q_rot1, mask=mask_h)
    tl.store(q_out_base + ROT_HALF + offs_h, q_rot2, mask=mask_h)
    tl.store(k_out_base + offs_h, k_rot1, mask=mask_h)
    tl.store(k_out_base + ROT_HALF + offs_h, k_rot2, mask=mask_h)

    if PASS_DIM > 0:
        qp_n = (qp * q_rstd).to(tl.bfloat16) * qwp
        kp_n = (kp * k_rstd).to(tl.bfloat16) * kwp
        tl.store(q_out_base + rot_dim + offs_p, qp_n, mask=mask_p)
        tl.store(k_out_base + rot_dim + offs_p, kp_n, mask=mask_p)


def fused_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel strided per-head RMSNorm + NeoX RoPE for ``[T, H, D]``.

    Accepts contiguous or strided ``q``/``k`` (such as direct slices from a
    fused ``qkv_proj`` output) as long as ``stride(-1) == 1`` and writes
    contiguous ``[T, H, D]`` bf16 outputs in a single Triton launch.
    """

    total, num_heads, head_dim = q.shape
    rot_dim = int(cos_sin_cache.shape[-1])
    if rot_dim <= 0 or rot_dim % 2 != 0 or rot_dim > head_dim:
        raise ValueError(f"rot_dim must be even and in (0, {head_dim}], got {rot_dim}")
    q_out = torch.empty((total, num_heads, head_dim), device=q.device, dtype=q.dtype)
    k_out = torch.empty((total, num_heads, head_dim), device=k.device, dtype=k.dtype)
    if total == 0 or num_heads == 0:
        return q_out, k_out

    cs = cos_sin_cache if cos_sin_cache.stride(-1) == 1 else cos_sin_cache.contiguous()
    qw = q_weight if q_weight.is_contiguous() else q_weight.contiguous()
    kw = k_weight if k_weight.is_contiguous() else k_weight.contiguous()

    rot_half = rot_dim // 2
    pass_dim = head_dim - rot_dim
    block_half = triton.next_power_of_2(rot_half)
    block_pass = triton.next_power_of_2(max(pass_dim, 1))

    _fused_qknorm_rope_kernel[(total, num_heads)](
        q,
        k,
        q_out,
        k_out,
        qw,
        kw,
        cs,
        float(eps),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        cs.stride(0),
        HEAD_DIM=head_dim,
        ROT_HALF=rot_half,
        PASS_DIM=pass_dim,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
        num_warps=4,
    )
    return q_out, k_out


__all__ = ["fused_qknorm_rope"]
