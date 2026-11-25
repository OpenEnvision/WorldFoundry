# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and parity tests for Hopper-coalesced HY-WorldPlay & HY1.5 Triton kernels."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.action_camera import (
    _apply_prope_mat,
    _fused_qkv_rmsnorm_rope_prope,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_modulation import (
    apply_gate,
    fused_layernorm_modulate,
    fused_residual_dual_gate,
    fused_residual_dual_gate_layernorm_modulate,
    fused_residual_gate,
    fused_residual_gate_layernorm_modulate,
    modulate,
)


def _ref_qkv_rmsnorm_rope_prope(
    qkv: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    p_mat: torch.Tensor,
    p_inv_t: torch.Tensor,
    *,
    heads_num: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    b, s, total = qkv.shape
    c = total // 3
    d = c // heads_num
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(b, s, heads_num, d)
    k = k.view(b, s, heads_num, d)
    v = v.view(b, s, heads_num, d).contiguous()

    def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps) * w.float()).to(v.dtype)

    qn = rms_norm(q, qw)
    kn = rms_norm(k, kw)

    # Interleaved RoPE
    def apply_rope(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x_even, x_odd = xf[..., 0::2], xf[..., 1::2]
        c_e = cos[:, 0::2].float().view(1, s, 1, d // 2)
        s_e = sin[:, 0::2].float().view(1, s, 1, d // 2)
        out = torch.empty_like(xf)
        out[..., 0::2] = x_even * c_e - x_odd * s_e
        out[..., 1::2] = x_even * s_e + x_odd * c_e
        return out.to(v.dtype)

    q_rope = apply_rope(qn)
    k_rope = apply_rope(kn)

    # 4x4 SE(3) PRoPE
    cams = p_mat.shape[1]

    def apply_mat(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x.float().reshape(b, cams, -1, 4), mat.float()).reshape(x.shape).to(v.dtype)

    q_prope = apply_mat(qn, p_mat)
    k_prope = apply_mat(kn, p_inv_t)
    v_prope = apply_mat(v, p_inv_t)
    return q_rope, k_rope, v, q_prope, k_prope, v_prope


def test_cpu_fallback_fused_residual_gate_layernorm_modulate() -> None:
    torch.manual_seed(0)
    b, t, hw, c = 2, 3, 4, 32
    s = t * hw
    x = torch.randn(b, s, c)
    y1 = torch.randn(b, s, c)
    y2 = torch.randn(b, s, c)
    gate = torch.randn(b * t, c)
    shift = torch.randn(b * t, c)
    scale = torch.randn(b * t, c)

    xo, xm = fused_residual_gate_layernorm_modulate(x, y1, gate, shift, scale)
    ref_xo = x + apply_gate(y1, gate=gate)
    ref_xm = modulate(F.layer_norm(ref_xo, (c,), eps=1e-6), shift=shift, scale=scale)
    torch.testing.assert_close(xo, ref_xo)
    torch.testing.assert_close(xm, ref_xm)

    xo2, xm2 = fused_residual_dual_gate_layernorm_modulate(x, y1, y2, gate, shift, scale)
    ref_xo2 = x + apply_gate(y1 + y2, gate=gate)
    ref_xm2 = modulate(F.layer_norm(ref_xo2, (c,), eps=1e-6), shift=shift, scale=scale)
    torch.testing.assert_close(xo2, ref_xo2)
    torch.testing.assert_close(xm2, ref_xm2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("heads", "head_dim"), [(1, 64), (12, 64), (16, 128), (24, 128), (7, 32)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_coalesced_qkv_rmsnorm_rope_prope_parity(heads: int, head_dim: int, dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    b, cams, tok_per_cam = 2, 3, 12
    s = cams * tok_per_cam
    c = heads * head_dim
    qkv = torch.randn(b, s, 3 * c, device="cuda", dtype=dtype)
    qw = torch.randn(head_dim, device="cuda", dtype=dtype)
    kw = torch.randn(head_dim, device="cuda", dtype=dtype)
    angles = torch.randn(s, head_dim // 2, device="cuda", dtype=torch.float32)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    p_mat = torch.randn(b, cams, 4, 4, device="cuda", dtype=torch.float32)
    p_inv_t = torch.randn(b, cams, 4, 4, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        got = _fused_qkv_rmsnorm_rope_prope(qkv, qw, kw, (cos, sin), p_mat, p_inv_t, heads_num=heads)
        ref = _ref_qkv_rmsnorm_rope_prope(qkv, qw, kw, cos, sin, p_mat, p_inv_t, heads_num=heads)

    assert got is not None
    for idx, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert g.shape == r.shape and g.dtype == r.dtype
        diff = (g.float() - r.float()).abs()
        mse = (diff**2).mean().item()
        peak = max(r.float().abs().max().item(), 1e-6)
        psnr = math.inf if mse == 0 else 10.0 * math.log10((peak * peak) / mse)
        assert psnr > 55.0, f"output[{idx}] psnr={psnr:.2f} dB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_coalesced_prope_mat4_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(3)
    b, cams, tok_per_cam, h, d = 2, 4, 37, 16, 128
    x = torch.randn(b, cams * tok_per_cam, h, d, device="cuda", dtype=dtype)
    mat = torch.randn(b, cams, 4, 4, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        got = _apply_prope_mat(x, mat)
        ref = torch.matmul(x.float().reshape(b, cams, -1, 4), mat).reshape(x.shape).to(dtype)

    if dtype == torch.float32:
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tanh", [False, True])
@pytest.mark.parametrize("hidden", [256, 2048, 3072])
def test_fused_residual_gate_layernorm_modulate_bit_exact(tanh: bool, hidden: int) -> None:
    torch.manual_seed(5)
    b, t, hw = 2, 3, 17
    s = t * hw
    x = torch.randn(b, s, hidden, device="cuda", dtype=torch.bfloat16)
    y1 = torch.randn(b, s, hidden, device="cuda", dtype=torch.bfloat16)
    y2 = torch.randn(b, s, hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(b * t, hidden, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(b * t, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(b * t, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        xo_ref = fused_residual_gate(x, y1, gate, tanh=tanh)
        xm_ref = fused_layernorm_modulate(xo_ref, shift=shift, scale=scale)
        xo_got, xm_got = fused_residual_gate_layernorm_modulate(x, y1, gate, shift, scale, tanh=tanh)
        assert torch.equal(xo_got, xo_ref)
        assert torch.equal(xm_got, xm_ref)

        if not tanh:
            xo2_ref = fused_residual_dual_gate(x, y1, y2, gate)
            xm2_ref = fused_layernorm_modulate(xo2_ref, shift=shift, scale=scale)
            xo2_got, xm2_got = fused_residual_dual_gate_layernorm_modulate(x, y1, y2, gate, shift, scale)
            assert torch.equal(xo2_got, xo2_ref)
            assert torch.equal(xm2_got, xm2_ref)
