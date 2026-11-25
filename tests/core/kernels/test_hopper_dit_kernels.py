# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Hopper SM90 register-safe parity and edge-case tests for wide-H DiT Triton kernels.

Covers all optimized paths across:
- ``worldfoundry.core.kernels.triton_diffusion`` (Wan2.1/2.2, LingBot, Hunyuan, SANA)
- ``worldfoundry.base_models.diffusion_model.models.networks.cosmos3.fused_ops`` (Cosmos3-Super 64B)
- ``worldfoundry.base_models.diffusion_model.models.networks.ltx.fused_ops`` (LTX-2.3/2.5 22B A+V)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.fused_ops import (
    fused_residual_add_rmsnorm,
    fused_rmsnorm_rope_gqa,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.fused_ops import (
    fused_ltx_dual_rmsnorm_scale_shift,
    fused_ltx_gated_attention_fp8,
    fused_ltx_post_sa_residual_rms_norm,
    fused_ltx_post_sa_residual_rmsnorm_adaln,
    fused_ltx_qk_rmsnorm,
    fused_ltx_qk_rmsnorm_rope,
    fused_ltx_rmsnorm_scale_shift_fp8,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import (
    LTXRopeType,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from worldfoundry.core.attention import rotate_half
from worldfoundry.core.kernels import (
    fused_layer_norm_scale_shift,
    hidden_qk_rmsnorm_rope_3d,
    hidden_qk_rmsnorm_rope_freqs,
    qk_rmsnorm_rope,
    residual_gate_add,
    rms_norm_scale_shift,
)
from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

_CUDA = torch.cuda.is_available()
_HOPPER_PLUS = _CUDA and torch.cuda.get_device_capability()[0] >= 9


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    denom = expected.float().norm().clamp_min(1e-8)
    return float((diff / denom).item())


# ──────────────────────────────────────────────────────────────────────────
# CPU fallback sanity tests (run under CUDA_VISIBLE_DEVICES="")
# ──────────────────────────────────────────────────────────────────────────


def test_cpu_fallback_parity_wide_dit_ops() -> None:
    torch.manual_seed(7)
    b, s, heads, d = 1, 16, 4, 32
    h = heads * d
    q = torch.randn(b, s, h, dtype=torch.float32)
    k = torch.randn(b, s, h, dtype=torch.float32)
    qw = torch.randn(h, dtype=torch.float32)
    kw = torch.randn(h, dtype=torch.float32)
    ang = torch.randn(s, 1, d // 2, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(ang), ang)

    with torch.inference_mode():
        qf, kf = hidden_qk_rmsnorm_rope_freqs(q, k, qw, kw, freqs, num_heads=heads, eps=1e-6)
        assert qf.shape == q.shape and kf.shape == k.shape
        assert torch.isfinite(qf).all() and torch.isfinite(kf).all()

        # Cosmos3 CPU fallbacks
        q3 = torch.randn(s, 8, d, dtype=torch.float32)
        k3 = torch.randn(s, 2, d, dtype=torch.float32)
        w3 = torch.randn(d, dtype=torch.float32)
        cos3 = torch.randn(s, d, dtype=torch.float32).cos()
        sin3 = torch.randn(s, d, dtype=torch.float32).sin()
        qo3, ko3 = fused_rmsnorm_rope_gqa(q3, k3, w3, w3, cos3, sin3, eps=1e-6)
        assert qo3.shape == q3.shape and ko3.shape == k3.shape

        r3 = torch.randn(s, h, dtype=torch.float32)
        u3 = torch.randn(s, h, dtype=torch.float32)
        res3, norm3 = fused_residual_add_rmsnorm(r3, u3, qw, eps=1e-6)
        torch.testing.assert_close(res3, r3 + u3, rtol=0, atol=0)
        torch.testing.assert_close(norm3, F.rms_norm(res3, (h,), weight=qw, eps=1e-6), rtol=1e-5, atol=1e-6)

        # LTX CPU fallbacks
        cos_i = torch.randn(b, s, h, dtype=torch.float32).cos()
        sin_i = torch.randn(b, s, h, dtype=torch.float32).sin()
        ql, kl = fused_ltx_qk_rmsnorm_rope(
            q, k, qw, kw, (cos_i, sin_i), None, eps=1e-6, heads=heads, dim_head=d, rope_type=LTXRopeType.INTERLEAVED
        )
        assert ql.shape == q.shape and kl.shape == k.shape


# ──────────────────────────────────────────────────────────────────────────
# 1. triton_diffusion.py GPU parity across H = 1536, 2048, 3072, 4096, 5120
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("hidden,heads", [(1536, 12), (2048, 16), (3072, 24), (4096, 32), (5120, 40), (96 * 4, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_gpu_hidden_qk_rmsnorm_rope_freqs_parity(hidden: int, heads: int, dtype: torch.dtype, precision: str) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    b, s = 2, 64
    d = hidden // heads
    # Use strided slices from a fused QKV tensor to exercise non-contiguous seq stride.
    qkv = torch.randn(b, s, 3 * hidden, device=device, dtype=dtype)
    q, k, _ = qkv.chunk(3, dim=-1)
    qw = torch.randn(hidden, device=device, dtype=dtype)
    kw = torch.randn(hidden, device=device, dtype=dtype)
    freq_dtype = torch.float64 if precision == "fp64" else torch.float32
    angles = torch.randn(s, 1, d // 2, device=device, dtype=freq_dtype)
    freqs = torch.polar(torch.ones_like(angles), angles)

    with torch.inference_mode():
        q_tri, k_tri = hidden_qk_rmsnorm_rope_freqs(q, k, qw, kw, freqs, num_heads=heads, eps=1e-6, precision=precision)

        # Reference: WanRMSNorm -> interleaved RoPE
        def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            xn = (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + 1e-6)).to(x.dtype) * w
            pairs = xn.view(b, s, heads, d // 2, 2)
            ev, od = pairs[..., 0].to(freq_dtype), pairs[..., 1].to(freq_dtype)
            c, sn = freqs.real.unsqueeze(0), freqs.imag.unsqueeze(0)
            return torch.stack((ev * c - od * sn, od * c + ev * sn), dim=-1).reshape(b, s, hidden).to(x.dtype)

        q_ref = _ref(q, qw)
        k_ref = _ref(k, kw)

    tol = 1e-5 if dtype == torch.float32 else 2e-4
    assert _rel_l2(q_tri, q_ref) < tol
    assert _rel_l2(k_tri, k_ref) < tol


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize(
    "hidden,heads,head_start,head_end,valid_tokens",
    [
        (1536, 12, 0, 12, 64),
        (3072, 24, 4, 20, 50),
        (5120, 40, 0, 40, 64),
        (5120, 40, 8, 32, 48),
        (384, 4, 1, 3, 40),  # non-pow2 head_dim=96 exercises general fallback path
    ],
)
@pytest.mark.parametrize("freq_dtype", [torch.float64, torch.float32])
def test_gpu_hidden_qk_rmsnorm_rope_3d_parity(
    hidden: int,
    heads: int,
    head_start: int,
    head_end: int,
    valid_tokens: int,
    freq_dtype: torch.dtype,
) -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    grid = (2, 4, 8)
    s = grid[0] * grid[1] * grid[2]
    d = hidden // heads
    qkv = torch.randn(2, s, 3 * hidden, device=device, dtype=dtype)
    q, k, _ = qkv.chunk(3, dim=-1)
    qw = torch.randn(hidden, device=device, dtype=dtype)
    kw = torch.randn(hidden, device=device, dtype=dtype)
    angles = torch.randn(32, d // 2, device=device, dtype=freq_dtype)
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1)

    with torch.inference_mode():
        q_tri, k_tri = hidden_qk_rmsnorm_rope_3d(
            q,
            k,
            qw,
            kw,
            freqs,
            num_heads=heads,
            grid_size=grid,
            eps=1e-6,
            sequence_offset=0,
            start_frame=1,
            valid_tokens=valid_tokens,
            head_start=head_start,
            head_end=head_end,
        )
        # Exact PyTorch reference via torch backend
        from worldfoundry.core.kernels.diffusion import _hidden_qk_rmsnorm_rope_3d_torch

        q_ref, k_ref = _hidden_qk_rmsnorm_rope_3d_torch(
            q, k, qw, kw, freqs, heads, grid, 1e-6, 0, 1, valid_tokens, head_start, head_end
        )

    assert _rel_l2(q_tri, q_ref) < 2e-4
    assert _rel_l2(k_tri, k_ref) < 2e-4


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [32, 600])
@pytest.mark.parametrize("heads,head_dim", [(32, 128), (40, 128), (12, 96)])
@pytest.mark.parametrize("freq_dtype", [torch.float64, torch.float32])
def test_gpu_apply_real_rope_bshd_parity(seq_len: int, heads: int, head_dim: int, freq_dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    x = torch.randn(1, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    ang = torch.randn(1, seq_len, 1, head_dim // 2, device=device, dtype=freq_dtype)
    cos, sin = ang.cos(), ang.sin()

    with torch.inference_mode():
        out_tri = apply_real_rope_bshd(x, cos, sin)
        xp = x.to(freq_dtype).reshape(1, seq_len, heads, head_dim // 2, 2)
        x0, x1 = xp.unbind(-1)
        out_ref = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2).to(x.dtype)

    assert _rel_l2(out_tri, out_ref) < 1e-4


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("heads,head_dim", [(12, 64), (32, 128), (40, 128), (8, 96)])
@pytest.mark.parametrize("rope_fp32", [False, True])
def test_gpu_qk_rmsnorm_rope_tiled_parity(heads: int, head_dim: int, rope_fp32: bool) -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, s = 2, 64
    q = torch.randn(b, s, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(b, s, heads, head_dim, device=device, dtype=dtype)
    qw = torch.randn(head_dim, device=device, dtype=dtype)
    kw = torch.randn(head_dim, device=device, dtype=dtype)
    freqs = torch.randn(s, 1, 1, head_dim, device=device, dtype=dtype)

    with torch.inference_mode():
        q_tri, k_tri = qk_rmsnorm_rope(q, k, qw, kw, freqs, eps=1e-6, rope_fp32=rope_fp32)
        from worldfoundry.core.kernels.diffusion import _qk_rmsnorm_rope_torch

        q_ref, k_ref = _qk_rmsnorm_rope_torch(q, k, qw, kw, freqs, 1e-6, rope_fp32)

    assert _rel_l2(q_tri, q_ref) < 3e-3
    assert _rel_l2(k_tri, k_ref) < 3e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("features", [1536, 2048, 3072, 4096, 5120])
def test_gpu_wide_adaln_and_residual_gate_parity(features: int) -> None:
    torch.manual_seed(51)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(1, 64, features, device=device, dtype=dtype)
    u = torch.randn_like(x)
    sc = torch.randn(1, 1, features, device=device, dtype=dtype)
    sh = torch.randn(1, 1, features, device=device, dtype=dtype)
    gt = torch.randn(1, 1, features, device=device, dtype=dtype)
    w = torch.randn(features, device=device, dtype=dtype)

    with torch.inference_mode():
        ln_out = fused_layer_norm_scale_shift(x, sc, sh, eps=1e-6, backend="triton")
        rms_out = rms_norm_scale_shift(x, sc, sh, eps=1e-6, weight=w, backend="triton")
        rg_out = residual_gate_add(x, u, gt, backend="triton")

    ln_ref = F.layer_norm(x, (features,), eps=1e-6) * (1 + sc) + sh
    rms_ref = F.rms_norm(x, (features,), weight=w, eps=1e-6) * (1 + sc) + sh
    rg_ref = x + u * gt
    assert _rel_l2(ln_out, ln_ref) < 1e-4
    assert _rel_l2(rms_out, rms_ref) < 3.5e-3
    torch.testing.assert_close(rg_out, rg_ref, rtol=0, atol=0)


# ──────────────────────────────────────────────────────────────────────────
# 2. cosmos3/fused_ops.py GPU parity across GQA & residual+RMSNorm
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [64, 600])  # exercises both head-tiled (<512) and 1cta (>=512) paths
@pytest.mark.parametrize("hq,hk,d", [(32, 8, 128), (64, 8, 128), (12, 4, 96)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gpu_cosmos3_rmsnorm_rope_gqa_parity(seq_len: int, hq: int, hk: int, d: int, dtype: torch.dtype) -> None:
    torch.manual_seed(61)
    device = torch.device("cuda")
    q = torch.randn(seq_len, hq, d, device=device, dtype=dtype)
    k = torch.randn(seq_len, hk, d, device=device, dtype=dtype)
    qw = torch.randn(d, device=device, dtype=dtype)
    kw = torch.randn(d, device=device, dtype=dtype)
    ang = torch.randn(seq_len, d, device=device, dtype=dtype)
    cos, sin = ang.cos(), ang.sin()

    with torch.inference_mode():
        q_tri, k_tri = fused_rmsnorm_rope_gqa(q, k, qw, kw, cos, sin, eps=1e-6)
        qn = F.rms_norm(q, (d,), weight=qw, eps=1e-6)
        kn = F.rms_norm(k, (d,), weight=kw, eps=1e-6)
        q_ref = qn * cos.unsqueeze(1) + rotate_half(qn) * sin.unsqueeze(1)
        k_ref = kn * cos.unsqueeze(1) + rotate_half(kn) * sin.unsqueeze(1)

    assert _rel_l2(q_tri, q_ref) < 5e-3
    assert _rel_l2(k_tri, k_ref) < 5e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("hidden", [1536, 2048, 3072, 4096, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gpu_cosmos3_residual_add_rmsnorm_single_pass_parity(hidden: int, dtype: torch.dtype) -> None:
    torch.manual_seed(71)
    device = torch.device("cuda")
    r = torch.randn(128, hidden, device=device, dtype=dtype)
    u = torch.randn(128, hidden, device=device, dtype=dtype)
    w = torch.randn(hidden, device=device, dtype=dtype)

    with torch.inference_mode():
        res_tri, norm_tri = fused_residual_add_rmsnorm(r, u, w, eps=1e-6)
        res_ref = r + u
        norm_ref = F.rms_norm(res_ref, (hidden,), weight=w, eps=1e-6)

    torch.testing.assert_close(res_tri, res_ref, rtol=0, atol=0)
    tol = 1e-5 if dtype == torch.float32 else 3e-3
    assert _rel_l2(norm_tri, norm_ref) < tol


# ──────────────────────────────────────────────────────────────────────────
# 3. ltx/fused_ops.py GPU parity across video (4096/5120) & audio (2048)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("heads,dim_head", [(32, 64), (32, 128), (40, 128)])
@pytest.mark.parametrize("rope_type", [LTXRopeType.SPLIT, LTXRopeType.INTERLEAVED])
@pytest.mark.parametrize("paired", [True, False])
def test_gpu_ltx_qk_rmsnorm_rope_wide_parity(heads: int, dim_head: int, rope_type: LTXRopeType, paired: bool) -> None:
    torch.manual_seed(81)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, t_q = 2, 48
    t_k = t_q if paired else 24
    inner = heads * dim_head

    q = torch.randn(b, t_q, inner, device=device, dtype=dtype)
    k = torch.randn(b, t_k, inner, device=device, dtype=dtype)
    qw = torch.empty(inner, device=device, dtype=dtype).uniform_(0.5, 1.5)
    kw = torch.empty(inner, device=device, dtype=dtype).uniform_(0.5, 1.5)

    pos_q = torch.rand(b, 3, t_q, 2, device=device, dtype=torch.float32) * 16.0
    q_pe = precompute_freqs_cis(
        indices_grid=pos_q,
        dim=inner,
        out_dtype=dtype,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=heads,
        rope_type=rope_type,
    )
    if paired:
        k_pe = None
        k_pe_ref = q_pe
    else:
        pos_k = torch.rand(b, 1, t_k, 2, device=device, dtype=torch.float32) * 16.0
        k_pe = precompute_freqs_cis(
            indices_grid=pos_k,
            dim=inner,
            out_dtype=dtype,
            theta=10000.0,
            max_pos=[20],
            num_attention_heads=heads,
            rope_type=rope_type,
        )
        k_pe_ref = k_pe

    with torch.inference_mode():
        q_tri, k_tri = fused_ltx_qk_rmsnorm_rope(
            q, k, qw, kw, q_pe, k_pe, eps=1e-6, heads=heads, dim_head=dim_head, rope_type=rope_type
        )
        q_ref = apply_rotary_emb(F.rms_norm(q, (inner,), weight=qw, eps=1e-6), q_pe, rope_type)
        k_ref = apply_rotary_emb(F.rms_norm(k, (inner,), weight=kw, eps=1e-6), k_pe_ref, rope_type)

    assert _rel_l2(q_tri, q_ref) < 5e-3
    assert _rel_l2(k_tri, k_ref) < 5e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("heads,dim_head", [(32, 64), (32, 128), (40, 128)])
def test_gpu_ltx_wide_adaln_dual_post_sa_and_fp8_parity(heads: int, dim_head: int) -> None:
    torch.manual_seed(91)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, t = 2, 32
    inner = heads * dim_head

    res = torch.randn(b, t, inner, device=device, dtype=dtype)
    val = torch.randn(b, t, inner, device=device, dtype=dtype)
    gate = torch.randn(b, 1, inner, device=device, dtype=dtype)
    s1 = torch.randn(b, 1, inner, device=device, dtype=dtype)
    sh1 = torch.randn(b, 1, inner, device=device, dtype=dtype)
    s2 = torch.randn(b, t, inner, device=device, dtype=dtype)
    sh2 = torch.randn(b, t, inner, device=device, dtype=dtype)
    qw = torch.empty(inner, device=device, dtype=dtype).uniform_(0.5, 1.5)
    kw = torch.empty(inner, device=device, dtype=dtype).uniform_(0.5, 1.5)
    gl = torch.randn(b, t, heads, device=device, dtype=dtype)

    with torch.inference_mode():
        # Post-SA residual + RMSNorm (bitwise exact residual)
        out_p, norm_p = fused_ltx_post_sa_residual_rms_norm(res, val, gate, eps=1e-6)
        out_ref = res + val * gate
        torch.testing.assert_close(out_p, out_ref, rtol=0, atol=0)
        assert _rel_l2(norm_p, F.rms_norm(out_ref, (inner,), None, eps=1e-6)) < 2e-3

        # Post-SA residual + RMSNorm + AdaLN
        out_pa, mod_pa = fused_ltx_post_sa_residual_rmsnorm_adaln(res, val, gate, s1, sh1, eps=1e-6)
        torch.testing.assert_close(out_pa, out_ref, rtol=0, atol=0)
        assert _rel_l2(mod_pa, F.rms_norm(out_ref, (inner,), None, eps=1e-6) * (1 + s1) + sh1) < 5e-3

        # Dual RMSNorm + AdaLN (including tiled >4096 path at 5120)
        d1, d2 = fused_ltx_dual_rmsnorm_scale_shift(res, s1, sh1, s2, sh2, eps=1e-6)
        rn = F.rms_norm(res, (inner,), None, eps=1e-6)
        assert _rel_l2(d1, rn * (1 + s1) + sh1) < 5e-3
        assert _rel_l2(d2, rn * (1 + s2) + sh2) < 5e-3

        # QK RMSNorm paired
        qn, kn = fused_ltx_qk_rmsnorm(res, val, qw, kw, eps=1e-6)
        assert _rel_l2(qn, F.rms_norm(res, (inner,), qw, eps=1e-6)) < 3e-3
        assert _rel_l2(kn, F.rms_norm(val, (inner,), kw, eps=1e-6)) < 3e-3

        # FP8 gated attention & RMSNorm+AdaLN+FP8 (including zero row safety)
        res_z = res.clone()
        res_z[0, 0].zero_()
        for fp8_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            ga_fp8 = fused_ltx_gated_attention_fp8(res_z, gl, heads=heads, dim_head=dim_head, fp8_dtype=fp8_dt)
            assert ga_fp8 is not None
            q_ga, s_ga = ga_fp8
            assert torch.isfinite(q_ga.float()).all() and torch.isfinite(s_ga).all()
            ga_ref = (res_z.view(b, t, heads, dim_head) * (2.0 * torch.sigmoid(gl)).unsqueeze(-1)).view_as(res_z)
            assert _rel_l2((q_ga.float() * s_ga).view_as(res_z), ga_ref) < 0.08

            rn_fp8 = fused_ltx_rmsnorm_scale_shift_fp8(res_z, s1, sh1, eps=1e-6, fp8_dtype=fp8_dt)
            assert rn_fp8 is not None
            q_rn, s_rn = rn_fp8
            assert torch.isfinite(q_rn.float()).all() and torch.isfinite(s_rn).all()
            rn_ref = F.rms_norm(res_z, (inner,), None, eps=1e-6) * (1 + s1) + sh1
            assert _rel_l2((q_rn.float() * s_rn).view_as(res_z), rn_ref) < 0.08


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="Hopper SM90+ required for register-spill check")
def test_hopper_sm90_wide_h5120_kernels_zero_register_spills() -> None:
    """Verify wide-H=5120 Triton kernels compile with zero local-memory register spills on SM90."""
    from worldfoundry.base_models.diffusion_model.models.networks.cosmos3 import fused_ops as c3
    from worldfoundry.base_models.diffusion_model.models.networks.ltx import fused_ops as ltx
    from worldfoundry.core.kernels import triton_diffusion as td

    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, s, h, heads = 1, 512, 5120, 40
    d = h // heads

    with torch.inference_mode():
        q = torch.randn(b, s, h, device=device, dtype=dtype)
        k = torch.randn(b, s, h, device=device, dtype=dtype)
        qw = torch.randn(h, device=device, dtype=dtype)
        kw = torch.randn(h, device=device, dtype=dtype)
        f64 = torch.polar(
            torch.ones(s, 1, d // 2, device=device, dtype=torch.float64),
            torch.randn(s, 1, d // 2, device=device, dtype=torch.float64),
        )
        f3d = torch.randn(64, d // 2, 2, device=device, dtype=torch.float64)
        td.hidden_qk_rmsnorm_rope_freqs(q, k, qw, kw, f64, num_heads=heads, eps=1e-6)
        td.hidden_qk_rmsnorm_rope_3d(
            q,
            k,
            qw,
            kw,
            f3d,
            num_heads=heads,
            grid_size=(8, 8, 8),
            eps=1e-6,
            sequence_offset=0,
            start_frame=0,
            valid_tokens=s,
            head_start=0,
            head_end=heads,
        )
        td.apply_real_rope_bshd(
            q.view(b, s, heads, d),
            torch.ones(1, s, 1, d // 2, device=device, dtype=torch.float64),
            torch.zeros(1, s, 1, d // 2, device=device, dtype=torch.float64),
        )
        td.qk_rmsnorm_rope(
            q.view(b, s, heads, d),
            k.view(b, s, heads, d),
            qw[:d].contiguous(),
            kw[:d].contiguous(),
            torch.zeros(s, 1, 1, d, device=device, dtype=dtype),
            1e-6,
            True,
        )
        c3.fused_rmsnorm_rope_gqa(
            torch.randn(s, 64, 128, device=device, dtype=dtype),
            torch.randn(s, 8, 128, device=device, dtype=dtype),
            qw[:128].contiguous(),
            kw[:128].contiguous(),
            torch.ones(s, 128, device=device, dtype=dtype),
            torch.zeros(s, 128, device=device, dtype=dtype),
        )
        c3.fused_residual_add_rmsnorm(q.view(s, h), k.view(s, h), qw)
        cos_s = torch.ones(b, heads, s, d // 2, device=device, dtype=torch.float32)
        sin_s = torch.zeros_like(cos_s)
        ltx.fused_ltx_qk_rmsnorm_rope(
            q, k, qw, kw, (cos_s, sin_s), None, eps=1e-6, heads=heads, dim_head=d, rope_type=LTXRopeType.SPLIT
        )
        cos_i = torch.ones(b, s, h, device=device, dtype=torch.float32)
        sin_i = torch.zeros_like(cos_i)
        ltx.fused_ltx_qk_rmsnorm_rope(
            q, k, qw, kw, (cos_i, sin_i), None, eps=1e-6, heads=heads, dim_head=d, rope_type=LTXRopeType.INTERLEAVED
        )
        cos_s4 = torch.ones(b, 32, s, 64, device=device, dtype=torch.float32)
        sin_s4 = torch.zeros_like(cos_s4)
        ltx.fused_ltx_qk_rmsnorm_rope(
            q[:, :, :4096].contiguous(),
            k[:, :, :4096].contiguous(),
            qw[:4096].contiguous(),
            kw[:4096].contiguous(),
            (cos_s4, sin_s4),
            None,
            eps=1e-6,
            heads=32,
            dim_head=128,
            rope_type=LTXRopeType.SPLIT,
        )
        sc = torch.zeros(b, 1, h, device=device, dtype=dtype)
        ltx.fused_ltx_dual_rmsnorm_scale_shift(q, sc, sc, sc, sc, eps=1e-6)
        ltx.fused_ltx_rmsnorm_scale_shift_fp8(q, sc, sc, eps=1e-6)
        ltx.fused_ltx_gated_attention_fp8(
            q, torch.zeros(b, s, heads, device=device, dtype=dtype), heads=heads, dim_head=d
        )

    def _assert_zero_spills(name: str, jit_fn: object, *, max_spills: int = 0, max_regs: int | None = None) -> None:
        dev_caches = getattr(jit_fn, "device_caches", None)
        assert dev_caches, f"{name} had no compiled device cache entries"
        checked = 0
        for entry in dev_caches.values():
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(cache_dict, dict):
                continue
            for compiled in cache_dict.values():
                checked += 1
                spills = int(getattr(compiled, "n_spills", 0))
                regs = int(getattr(compiled, "n_regs", 0))
                assert spills <= max_spills, f"{name} spilled {spills} bytes > {max_spills} (n_regs={regs})"
                if max_regs is not None:
                    assert regs <= max_regs, f"{name} used {regs} regs > {max_regs}"
        assert checked > 0, f"{name} had zero compiled kernels"

    for attr in (
        "_hidden_qk_rmsnorm_rope_freqs_kernel",
        "_hidden_qk_rmsnorm_rope_3d_kernel",
        "_real_rope_bshd_1cta_kernel",
    ):
        _assert_zero_spills(f"triton_diffusion.{attr}", getattr(td, attr))
    # _qk_rmsnorm_rope_kernel calls libdevice tl.cos/tl.sin whose Payne-Hanek range-reduction
    # slow path reserves an 8-byte local depot even at n_regs=32 (zero register-file pressure).
    _assert_zero_spills(
        "triton_diffusion._qk_rmsnorm_rope_kernel",
        td._qk_rmsnorm_rope_kernel,
        max_spills=8,
        max_regs=64,
    )
    for attr in ("_rmsnorm_rope_gqa_1cta_kernel", "_residual_add_rmsnorm_1pass_kernel"):
        _assert_zero_spills(f"cosmos3.{attr}", getattr(c3, attr))
    ltx_kernels = ltx._load_triton_kernels()
    for key in (
        "split_paired_2d_1pass",
        "split_paired_2d_tiled",
        "interleaved_paired",
        "dual_rmsnorm_adaln_tiled",
        "rmsnorm_adaln_fp8",
        "gated_attn_fp8",
    ):
        _assert_zero_spills(f"ltx.{key}", ltx_kernels[key])
