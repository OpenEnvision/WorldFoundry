# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit, parity, CUDA Graph, and CPU-fallback tests for Wave 6 Cosmos3, LTX-2.5, and MiniMax-H3 kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.triton_vit_ops import (
    can_use_vit_qk_rmsnorm_rope,
    can_use_vit_scaled_residual_rmsnorm,
    vit_fused_qk_rmsnorm_rope,
    vit_rmsnorm,
    vit_scaled_residual_add_rmsnorm,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.vit_utils import (
    apply_rotary_pos_emb_qk,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.fused_ops import (
    fused_rmsnorm_rope_gqa,
    fused_rmsnorm_rope_gqa_cat_kv,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.fused_ops import (
    fused_ltx_dual_rmsnorm_scale_shift,
    fused_ltx_residual_gate_dual_rmsnorm_scale_shift,
)


def test_cpu_fallbacks() -> None:
    # Cosmos3 CPU fallback
    q = torch.randn(8, 4, 32)
    k = torch.randn(8, 2, 32)
    v = torch.randn(8, 2, 32)
    k_und = torch.randn(4, 2, 32)
    v_und = torch.randn(4, 2, 32)
    qw = torch.ones(32)
    kw = torch.ones(32)
    cos = torch.randn(8, 32)
    sin = torch.randn(8, 32)
    qo, ka, va = fused_rmsnorm_rope_gqa_cat_kv(q, k, v, k_und, v_und, qw, kw, cos, sin, eps=1e-6)
    qr, kr = fused_rmsnorm_rope_gqa(q, k, qw, kw, cos, sin, eps=1e-6)
    assert torch.allclose(qo, qr)
    assert torch.allclose(ka, torch.cat([k_und, kr], dim=0))
    assert torch.equal(va, torch.cat([v_und, v], dim=0))

    # LTX CPU fallback
    r = torch.randn(1, 4, 32)
    u = torch.randn(1, 4, 32)
    g = torch.randn(1, 1, 32)
    s1, sh1 = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
    s2, sh2 = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
    out, o1, o2 = fused_ltx_residual_gate_dual_rmsnorm_scale_shift(r, u, g, s1, sh1, s2, sh2, eps=1e-6)
    ref_out = r + u * g
    ref_o1, ref_o2 = fused_ltx_dual_rmsnorm_scale_shift(ref_out, s1, sh1, s2, sh2, eps=1e-6)
    assert torch.allclose(out, ref_out)
    assert torch.allclose(o1, ref_o1)
    assert torch.allclose(o2, ref_o2)

    # MiniMax-H3 ViT CPU fallback
    assert not can_use_vit_qk_rmsnorm_rope(
        torch.randn(1, 4, 4, 64),
        torch.randn(1, 4, 4, 64),
        torch.randn(1, 4, 1, 48),
        torch.randn(1, 4, 1, 48),
    )
    assert not can_use_vit_scaled_residual_rmsnorm(torch.randn(1, 4, 64), torch.randn(1, 4, 64), None, None)
    x_cpu = torch.randn(1, 4, 64)
    u_cpu = torch.randn(1, 4, 64)
    sc_cpu = torch.randn(64)
    w_cpu = torch.randn(64)
    ro, no = vit_scaled_residual_add_rmsnorm(x_cpu, u_cpu, sc_cpu, w_cpu, eps=1e-5)
    assert torch.allclose(ro, x_cpu + u_cpu * sc_cpu)
    assert torch.allclose(no, F.rms_norm(ro.float(), (64,), weight=w_cpu.float(), eps=1e-5))
    assert torch.allclose(vit_rmsnorm(x_cpu, w_cpu, eps=1e-5), F.rms_norm(x_cpu.float(), (64,), weight=w_cpu, eps=1e-5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("s_gen,s_und,hq,hk,d", [(600, 64, 64, 8, 128), (512, 32, 32, 8, 128)])
def test_cosmos3_rmsnorm_rope_gqa_cat_kv_bit_exact_and_cudagraph(
    s_gen: int,
    s_und: int,
    hq: int,
    hk: int,
    d: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qkv = torch.randn(s_gen, (hq + 2 * hk) * d, device=device, dtype=dtype)
    q_s, k_s, v_s = torch.split(qkv, [hq * d, hk * d, hk * d], dim=-1)
    q = q_s.view(s_gen, hq, d)
    k = k_s.view(s_gen, hk, d)
    v = v_s.view(s_gen, hk, d)
    k_und = torch.randn(s_und, hk, d, device=device, dtype=dtype)
    v_und = torch.randn(s_und, hk, d, device=device, dtype=dtype)
    qw = torch.randn(d, device=device, dtype=dtype)
    kw = torch.randn(d, device=device, dtype=dtype)
    cos = torch.randn(s_gen, d, device=device, dtype=dtype)
    sin = torch.randn(s_gen, d, device=device, dtype=dtype)

    with torch.inference_mode():
        q_ref, k_ref = fused_rmsnorm_rope_gqa(q, k, qw, kw, cos, sin, eps=1e-6)
        ka_ref = torch.cat([k_und, k_ref], dim=0)
        va_ref = torch.cat([v_und, v], dim=0)
        q_opt, ka_opt, va_opt = fused_rmsnorm_rope_gqa_cat_kv(q, k, v, k_und, v_und, qw, kw, cos, sin, eps=1e-6)

    assert torch.equal(q_opt, q_ref)
    assert torch.equal(ka_opt, ka_ref)
    assert torch.equal(va_opt, va_ref)

    # CUDA Graph capture safety
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            q_g, ka_g, va_g = fused_rmsnorm_rope_gqa_cat_kv(q, k, v, k_und, v_und, qw, kw, cos, sin, eps=1e-6)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(q_g, q_ref)
    assert torch.equal(ka_g, ka_ref)
    assert torch.equal(va_g, va_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("features", [2048, 4096, 5120])
def test_ltx_residual_gate_dual_rmsnorm_scale_shift_bit_exact_and_cudagraph(features: int) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, t = 1, 64
    r = torch.randn(b, t, features, device=device, dtype=dtype)
    v = torch.randn(b, t, features, device=device, dtype=dtype)
    gate = torch.randn(b, 1, features, device=device, dtype=dtype)
    s1 = torch.randn(b, 1, features, device=device, dtype=dtype)
    sh1 = torch.randn(b, 1, features, device=device, dtype=dtype)
    s2 = torch.randn(b, 1, features, device=device, dtype=dtype)
    sh2 = torch.randn(b, 1, features, device=device, dtype=dtype)

    with torch.inference_mode():
        ref_out = r + v * gate
        ref_o1, ref_o2 = fused_ltx_dual_rmsnorm_scale_shift(ref_out, s1, sh1, s2, sh2, eps=1e-6)
        out, o1, o2 = fused_ltx_residual_gate_dual_rmsnorm_scale_shift(r, v, gate, s1, sh1, s2, sh2, eps=1e-6)

    assert torch.equal(out, ref_out)
    assert torch.equal(o1, ref_o1)
    assert torch.equal(o2, ref_o2)

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out_g, o1_g, o2_g = fused_ltx_residual_gate_dual_rmsnorm_scale_shift(r, v, gate, s1, sh1, s2, sh2, eps=1e-6)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(out_g, ref_out)
    assert torch.equal(o1_g, ref_o1)
    assert torch.equal(o2_g, ref_o2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_minimax_h3_vit_qk_rmsnorm_rope_parity_and_cudagraph(affine: bool, dtype: torch.dtype) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    b, s, h, d, rot_dim = 2, 128, 32, 64, 48
    qkv = torch.randn(b, s, h, 3 * d, device=device, dtype=dtype)
    q, k, _ = torch.chunk(qkv, 3, dim=-1)
    angles = torch.randn(b, s, 1, rot_dim // 2, device=device, dtype=torch.float32)
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    norm_q = nn.RMSNorm(d, eps=1e-5, elementwise_affine=affine).to(device=device, dtype=torch.float32)
    norm_k = nn.RMSNorm(d, eps=1e-5, elementwise_affine=affine).to(device=device, dtype=torch.float32)
    if affine:
        nn.init.normal_(norm_q.weight, mean=1.0, std=0.1)
        nn.init.normal_(norm_k.weight, mean=1.0, std=0.1)

    with torch.inference_mode():
        q_n = norm_q(q.float()).to(dtype)
        k_n = norm_k(k.float()).to(dtype)
        q_ref, k_ref = apply_rotary_pos_emb_qk(q_n, k_n, (cos, sin))

        q_opt, k_opt = vit_fused_qk_rmsnorm_rope(
            q,
            k,
            cos,
            sin,
            q_weight=norm_q.weight if affine else None,
            k_weight=norm_k.weight if affine else None,
            eps=1e-5,
        )

    assert q_opt.is_contiguous() and k_opt.is_contiguous()
    assert F.cosine_similarity(q_ref.float().flatten(), q_opt.float().flatten(), dim=0).item() >= 0.99999
    assert F.cosine_similarity(k_ref.float().flatten(), k_opt.float().flatten(), dim=0).item() >= 0.99999
    tol = 3.5e-2 if dtype == torch.bfloat16 else 5e-3
    assert (q_ref.float() - q_opt.float()).abs().max().item() <= tol
    assert (k_ref.float() - k_opt.float()).abs().max().item() <= tol

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            qg, kg = vit_fused_qk_rmsnorm_rope(
                q,
                k,
                cos,
                sin,
                q_weight=norm_q.weight if affine else None,
                k_weight=norm_k.weight if affine else None,
                eps=1e-5,
            )
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(qg, q_opt)
    assert torch.equal(kg, k_opt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_scale", [True, False])
@pytest.mark.parametrize("mixed_fp32_res", [False, True])
def test_minimax_h3_vit_scaled_residual_rmsnorm_parity(use_scale: bool, mixed_fp32_res: bool) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    res_dtype = torch.float32 if mixed_fp32_res else torch.bfloat16
    upd_dtype = torch.bfloat16
    r = torch.randn(2, 128, 2048, device=device, dtype=res_dtype)
    u = torch.randn(2, 128, 2048, device=device, dtype=upd_dtype)
    sc = torch.randn(2048, device=device, dtype=torch.float32) * 0.2 if use_scale else None
    w = torch.randn(2048, device=device, dtype=torch.float32)

    with torch.inference_mode():
        ref_r = (r + u * sc).to(res_dtype) if sc is not None else (r + u).to(res_dtype)
        ref_n = F.rms_norm(ref_r.float(), (2048,), weight=w, eps=1e-5).to(upd_dtype)
        opt_r, opt_n = vit_scaled_residual_add_rmsnorm(r, u, sc, w, eps=1e-5, norm_out_dtype=upd_dtype)

    assert opt_r.dtype == res_dtype and opt_n.dtype == upd_dtype
    assert F.cosine_similarity(ref_r.float().flatten(), opt_r.float().flatten(), dim=0).item() >= 0.99999
    assert F.cosine_similarity(ref_n.float().flatten(), opt_n.float().flatten(), dim=0).item() >= 0.99999
    assert (ref_r.float() - opt_r.float()).abs().max().item() <= (1e-4 if mixed_fp32_res else 2e-2)
    assert (ref_n.float() - opt_n.float()).abs().max().item() <= 3e-2
