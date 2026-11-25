# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit, parity, spill, CUDA-Graph, compile, autograd, and CPU-fallback tests for Wan & HunyuanVideo Hopper DiT ops."""

from __future__ import annotations

import math

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original import (
    HunyuanVideoDiT,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original_acceleration import (
    hunyuan13b_sol_profile,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    WanModel,
)
from worldfoundry.core.attention import apply_nd_rotary_embedding
from worldfoundry.core.kernels import triton_dit_block_ops as ops
from worldfoundry.core.nn import RMSNorm

_CUDA = torch.cuda.is_available()
_HOPPER_PLUS = _CUDA and torch.cuda.get_device_capability()[0] >= 9


def _rel_l2(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.detach().float()
    r = ref.detach().float()
    return float(((a - r).norm() / r.norm().clamp_min(1e-8)).item())


def _psnr_cosine(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a = actual.detach().float()
    r = ref.detach().float()
    mse = float((a - r).square().mean().item())
    peak = float(r.abs().max().clamp_min(1e-8).item())
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)
    cos = float(F.cosine_similarity(a.reshape(1, -1), r.reshape(1, -1)).item())
    return psnr, cos


# ──────────────────────────────────────────────────────────────────────────
# 1. CPU Fallback, Autograd, Env Opt-out, and Hook Safety (always run)
# ──────────────────────────────────────────────────────────────────────────


def test_cpu_fallback_parity_all_dit_block_ops() -> None:
    torch.manual_seed(0)
    b, s, h, heads = 2, 16, 64, 4
    d = h // heads
    x = torch.randn(b, s, h)
    y = torch.randn_like(x)
    z = torch.randn_like(x)
    mod = torch.randn(1, 6, h)
    t_mod_3d = torch.randn(b, 6, h)
    t_mod_4d = torch.randn(b, s, 6, h)
    w = torch.randn(h)
    bi = torch.randn(h)

    with torch.inference_mode():
        # Wan modulation + LN + AdaLN (3D & 4D)
        for tm in (t_mod_3d, t_mod_4d):
            e = mod + tm if tm.ndim == 3 else mod.unsqueeze(1) + tm
            sh0, sc0, g0, sh1, sc1, g1 = e.chunk(6, dim=2 if tm.ndim == 4 else 1)
            if tm.ndim == 4:
                sh0, sc0, g0, sh1, sc1, g1 = (t.squeeze(2) for t in (sh0, sc0, g0, sh1, sc1, g1))
            ref_ln = F.layer_norm(x, (h,), eps=1e-6) * (1 + sc0) + sh0
            act_ln = ops.fused_wan_modulation_ln_adaln(x, mod, tm, eps=1e-6)
            torch.testing.assert_close(act_ln, ref_ln, rtol=1e-5, atol=1e-6)

            xr, nr = ops.fused_wan_res_gate_ln_affine(x, y, mod, tm, w, bi, eps=1e-6)
            ref_xr = x + y * g0
            torch.testing.assert_close(xr, ref_xr, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                nr, F.layer_norm(ref_xr, (h,), weight=w, bias=bi, eps=1e-6), rtol=1e-5, atol=1e-6
            )

            xa, na = ops.fused_wan_res_add_ln_adaln(x, y, mod, tm, eps=1e-6)
            ref_xa = x + y
            torch.testing.assert_close(xa, ref_xa, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(na, F.layer_norm(ref_xa, (h,), eps=1e-6) * (1 + sc1) + sh1, rtol=1e-5, atol=1e-6)

            xg = ops.fused_wan_res_gate_add(x, y, mod, tm)
            torch.testing.assert_close(xg, x + y * g1, rtol=1e-5, atol=1e-6)

        # Wan RMSNorm & Head
        rms_ref = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)) * w
        torch.testing.assert_close(ops.fused_wan_rmsnorm(x, w, eps=1e-6), rms_ref, rtol=1e-5, atol=1e-6)

        mod_h = torch.randn(1, 2, h)
        t_h = torch.randn(b, h)
        sh_h, sc_h = (mod_h + t_h.unsqueeze(1)).chunk(2, dim=1)
        head_ref = F.layer_norm(x, (h,), eps=1e-6) * (1 + sc_h) + sh_h
        torch.testing.assert_close(ops.fused_wan_head_ln_adaln(x, mod_h, t_h, eps=1e-6), head_ref, rtol=1e-5, atol=1e-6)

        # Hunyuan ops on CPU
        qkv = torch.randn(b, s, 3 * h)
        qkv_t = torch.randn(b, 4, 3 * h)
        qw = torch.randn(d)
        kw = torch.randn(d)
        ang = torch.randn(s, d // 2)
        cos = ang.cos().repeat_interleave(2, dim=1)
        sin = ang.sin().repeat_interleave(2, dim=1)
        qs, ks, vs = ops.fused_hunyuan_qkv_rmsnorm_rope_hopper(qkv, qw, kw, (cos, sin), num_heads=heads, rope_len=s - 4)
        assert qs.shape == ks.shape == vs.shape == (b, s, heads, d)
        qd, kd, vd = ops.fused_hunyuan_double_qkv_rmsnorm_rope_hopper(
            qkv, qkv_t, qw, kw, qw, kw, (cos, sin), num_heads=heads
        )
        assert qd.shape == kd.shape == vd.shape == (b, s + 4, heads, d)

        torch.testing.assert_close(ops.fused_strided_gelu_tanh(x), F.gelu(x, approximate="tanh"), rtol=1e-5, atol=1e-6)

        sc2 = torch.randn(b, h)
        sh2 = torch.randn(b, h)
        gt2 = torch.randn(b, h)
        torch.testing.assert_close(
            ops.fused_dit_layernorm_adaln(x, sc2, sh2, eps=1e-6),
            F.layer_norm(x, (h,), eps=1e-6) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1),
            rtol=1e-5,
            atol=1e-6,
        )
        xr2, nr2 = ops.fused_dit_res_gate_ln_adaln(x, y, gt2, sc2, sh2, eps=1e-6)
        torch.testing.assert_close(xr2, x + y * gt2.unsqueeze(1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            nr2, F.layer_norm(xr2, (h,), eps=1e-6) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            ops.fused_dit_dual_res_gate_add(x, y, z, gt2), x + (y + z) * gt2.unsqueeze(1), rtol=1e-5, atol=1e-6
        )


def test_env_opt_out_and_autograd_and_hooks_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    assert ops.dit_block_ops_enabled() is False
    monkeypatch.delenv("WORLDFOUNDRY_KERNEL_BACKEND", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_DIT_BLOCK_OPS", "0")
    assert ops.dit_block_ops_enabled() is False
    monkeypatch.delenv("WORLDFOUNDRY_DIT_BLOCK_OPS", raising=False)
    assert ops.dit_block_ops_enabled() is True

    # Autograd backward check on DiTBlock + CrossAttention + Head
    torch.manual_seed(3)
    block = DiTBlock(has_image_input=False, dim=32, num_heads=2, ffn_dim=64)
    x = torch.randn(1, 8, 32, requires_grad=True)
    ctx = torch.randn(1, 4, 32, requires_grad=True)
    t_mod = torch.randn(1, 6, 32, requires_grad=True)
    freqs = torch.ones(8, 1, 8, dtype=torch.complex128)
    out = block(x, ctx, t_mod, freqs)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert t_mod.grad is not None and torch.isfinite(t_mod.grad).all()
    assert block.modulation.grad is not None and torch.isfinite(block.modulation.grad).all()

    # Forward hook check on norm3/gate/cross_attn.norm_q
    fired: list[str] = []
    block.eval()
    h3 = block.norm3.register_forward_hook(lambda *_: fired.append("norm3"))
    hg = block.gate.register_forward_hook(lambda *_: fired.append("gate"))
    hq = block.cross_attn.norm_q.register_forward_hook(lambda *_: fired.append("cross_norm_q"))
    try:
        with torch.inference_mode():
            _ = block(x.detach(), ctx.detach(), t_mod.detach(), freqs)
    finally:
        h3.remove()
        hg.remove()
        hq.remove()
    assert "norm3" in fired and "gate" in fired and "cross_norm_q" in fired


# ──────────────────────────────────────────────────────────────────────────
# 2. GPU Numerical Parity Tests (H=3072, H=5120, 3D & 4D t_mod, BF16/FP16/FP32)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("hidden", [1536, 3072, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("per_token", [False, True])
def test_gpu_wan_block_ops_parity_and_inplace(hidden: int, dtype: torch.dtype, per_token: bool) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    b, s = 2, 64
    x = torch.randn(b, s, hidden, device=device, dtype=dtype)
    y = torch.randn_like(x)
    mod = torch.randn(1, 6, hidden, device=device, dtype=dtype) * 0.2
    t_mod = (
        torch.randn(b, s, 6, hidden, device=device, dtype=dtype) * 0.2
        if per_token
        else torch.randn(b, 6, hidden, device=device, dtype=dtype) * 0.2
    )
    w = torch.randn(hidden, device=device, dtype=dtype)
    bi = torch.randn(hidden, device=device, dtype=dtype)

    tol_l2 = 1e-5 if dtype == torch.float32 else 5e-4
    min_psnr = 90.0 if dtype == torch.float32 else 60.0

    with torch.inference_mode():
        e = (mod.unsqueeze(1) + t_mod) if per_token else (mod + t_mod)
        sh0, sc0, g0, sh1, sc1, g1 = e.chunk(6, dim=2 if per_token else 1)
        if per_token:
            sh0, sc0, g0, sh1, sc1, g1 = (t.squeeze(2) for t in (sh0, sc0, g0, sh1, sc1, g1))

        # 1. modulation + LN + AdaLN
        ref_ln = F.layer_norm(x, (hidden,), eps=1e-6) * (1 + sc0) + sh0
        act_ln = ops.fused_wan_modulation_ln_adaln(x, mod, t_mod, eps=1e-6)
        psnr, cos = _psnr_cosine(act_ln, ref_ln)
        assert _rel_l2(act_ln, ref_ln) < tol_l2 and psnr >= min_psnr and cos > 0.99999

        # 2. res_gate + affine LN (functional + inplace)
        ref_x1 = x + y * g0
        ref_n3 = F.layer_norm(ref_x1, (hidden,), weight=w, bias=bi, eps=1e-6)
        act_x1, act_n3 = ops.fused_wan_res_gate_ln_affine(x, y, mod, t_mod, w, bi, eps=1e-6, inplace=False)
        fma_tol = 1e-6 if dtype == torch.float32 else 0.0
        torch.testing.assert_close(act_x1, ref_x1, rtol=fma_tol, atol=fma_tol)
        assert _rel_l2(act_n3, ref_n3) < tol_l2

        x_ip = x.clone()
        act_x1_ip, act_n3_ip = ops.fused_wan_res_gate_ln_affine(x_ip, y, mod, t_mod, w, bi, eps=1e-6, inplace=True)
        assert act_x1_ip.data_ptr() == x_ip.data_ptr()
        torch.testing.assert_close(act_x1_ip, act_x1, rtol=0, atol=0)
        torch.testing.assert_close(act_n3_ip, act_n3, rtol=0, atol=0)

        # 3. res_add + LN + AdaLN (functional + inplace)
        ref_x2 = x + y
        ref_n2 = F.layer_norm(ref_x2, (hidden,), eps=1e-6) * (1 + sc1) + sh1
        act_x2, act_n2 = ops.fused_wan_res_add_ln_adaln(x, y, mod, t_mod, eps=1e-6, inplace=False)
        torch.testing.assert_close(act_x2, ref_x2, rtol=0, atol=0)
        assert _rel_l2(act_n2, ref_n2) < tol_l2

        x_ip2 = x.clone()
        act_x2_ip, act_n2_ip = ops.fused_wan_res_add_ln_adaln(x_ip2, y, mod, t_mod, eps=1e-6, inplace=True)
        assert act_x2_ip.data_ptr() == x_ip2.data_ptr()
        torch.testing.assert_close(act_x2_ip, act_x2, rtol=0, atol=0)
        torch.testing.assert_close(act_n2_ip, act_n2, rtol=0, atol=0)

        # 4. res_gate_add (functional + inplace)
        ref_x3 = x + y * g1
        act_x3 = ops.fused_wan_res_gate_add(x, y, mod, t_mod, slot_gate=5, inplace=False)
        torch.testing.assert_close(act_x3, ref_x3, rtol=fma_tol, atol=fma_tol)
        x_ip3 = x.clone()
        act_x3_ip = ops.fused_wan_res_gate_add(x_ip3, y, mod, t_mod, slot_gate=5, inplace=True)
        assert act_x3_ip.data_ptr() == x_ip3.data_ptr()
        torch.testing.assert_close(act_x3_ip, act_x3, rtol=0, atol=0)

        # 5. WanRMSNorm
        xf = x.float()
        ref_rms = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dtype) * w
        act_rms = ops.fused_wan_rmsnorm(x, w, eps=1e-6)
        assert _rel_l2(act_rms, ref_rms) < tol_l2

        # 6. Head FP32 LN + AdaLN -> dtype
        mod_h = torch.randn(1, 2, hidden, device=device, dtype=dtype) * 0.2
        t_h = (
            torch.randn(b, s, hidden, device=device, dtype=dtype) * 0.2
            if per_token
            else torch.randn(b, hidden, device=device, dtype=dtype) * 0.2
        )
        if per_token:
            sh_h, sc_h = (mod_h.unsqueeze(0).float() + t_h.float().unsqueeze(2)).chunk(2, dim=2)
            ref_h = (F.layer_norm(x.float(), (hidden,), eps=1e-6) * (1 + sc_h.squeeze(2)) + sh_h.squeeze(2)).to(dtype)
        else:
            sh_h, sc_h = (mod_h.float() + t_h.float().unsqueeze(1)).chunk(2, dim=1)
            ref_h = (F.layer_norm(x.float(), (hidden,), eps=1e-6) * (1 + sc_h) + sh_h).to(dtype)
        act_h = ops.fused_wan_head_ln_adaln(x, mod_h, t_h, eps=1e-6, out_dtype=dtype)
        assert _rel_l2(act_h, ref_h) < tol_l2


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gpu_hunyuan_hopper_ops_parity_and_strided_safety(dtype: torch.dtype) -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    b, s_img, s_txt, heads, d = 2, 64, 18, 24, 128
    h = heads * d
    s_tot = s_img + s_txt

    # Strided linear1 [B, S_tot, 7*H] -> qkv [B, S_tot, 3*H], mlp [B, S_tot, 4*H]
    proj = torch.randn(b, s_tot, 7 * h, device=device, dtype=dtype)
    qkv_s = proj[..., : 3 * h]
    mlp_s = proj[..., 3 * h :]
    assert not qkv_s.is_contiguous() and not mlp_s.is_contiguous()

    norm_q = RMSNorm(dim=d, eps=1e-6).to(device=device, dtype=dtype)
    norm_k = RMSNorm(dim=d, eps=1e-6).to(device=device, dtype=dtype)
    norm_qb = RMSNorm(dim=d, eps=1e-6).to(device=device, dtype=dtype)
    norm_kb = RMSNorm(dim=d, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        for n in (norm_q, norm_k, norm_qb, norm_kb):
            assert n.weight is not None
            n.weight.uniform_(0.5, 1.5)

    ang = torch.randn(s_img, d // 2, device=device, dtype=torch.float32)
    cos = ang.cos().repeat_interleave(2, dim=1)
    sin = ang.sin().repeat_interleave(2, dim=1)

    tol_l2 = 1e-5 if dtype == torch.float32 else 2e-3
    with torch.inference_mode():
        # 1. Single-stream QKV + RMSNorm + RoPE on strided linear1 slice
        q_ref, k_ref, v_ref = qkv_s.contiguous().view(b, s_tot, 3, heads, d).unbind(dim=2)
        q_ref, k_ref = norm_q(q_ref), norm_k(k_ref)
        qa, ka = apply_nd_rotary_embedding(q_ref[:, :s_img], k_ref[:, :s_img], (cos, sin), head_first=False)
        q_ref = torch.cat((qa, q_ref[:, s_img:]), dim=1)
        k_ref = torch.cat((ka, k_ref[:, s_img:]), dim=1)

        q_tri, k_tri, v_tri = ops.fused_hunyuan_qkv_rmsnorm_rope_hopper(
            qkv_s, norm_q.weight, norm_k.weight, (cos, sin), num_heads=heads, rope_len=s_img, eps=1e-6
        )
        assert q_tri.is_contiguous() and k_tri.is_contiguous() and v_tri.is_contiguous()
        assert _rel_l2(q_tri, q_ref) < tol_l2
        assert _rel_l2(k_tri, k_ref) < tol_l2
        torch.testing.assert_close(v_tri, v_ref, rtol=0, atol=0)

        # 2. Double-stream direct-write QKV + RMSNorm + RoPE (zero torch.cat)
        qkv_img = torch.randn(b, s_img, 3 * h, device=device, dtype=dtype)
        qkv_txt = torch.randn(b, s_txt, 3 * h, device=device, dtype=dtype)
        qa2, ka2, va2 = qkv_img.view(b, s_img, 3, heads, d).unbind(dim=2)
        qa2, ka2 = apply_nd_rotary_embedding(norm_q(qa2), norm_k(ka2), (cos, sin), head_first=False)
        qb2, kb2, vb2 = qkv_txt.view(b, s_txt, 3, heads, d).unbind(dim=2)
        qb2, kb2 = norm_qb(qb2), norm_kb(kb2)
        qd_ref = torch.cat((qa2, qb2), dim=1)
        kd_ref = torch.cat((ka2, kb2), dim=1)
        vd_ref = torch.cat((va2, vb2), dim=1)

        qd_tri, kd_tri, vd_tri = ops.fused_hunyuan_double_qkv_rmsnorm_rope_hopper(
            qkv_img, qkv_txt, norm_q.weight, norm_k.weight, norm_qb.weight, norm_kb.weight, (cos, sin), num_heads=heads
        )
        assert _rel_l2(qd_tri, qd_ref) < tol_l2
        assert _rel_l2(kd_tri, kd_ref) < tol_l2
        torch.testing.assert_close(vd_tri, vd_ref, rtol=0, atol=0)

        # 3. Strided GELU(tanh) on linear1 MLP slice
        gelu_ref = F.gelu(mlp_s, approximate="tanh")
        gelu_tri = ops.fused_strided_gelu_tanh(mlp_s)
        assert gelu_tri.is_contiguous()
        assert _rel_l2(gelu_tri, gelu_ref) < tol_l2

        # 4. Generic DiT LN+AdaLN, ResGate+LN+AdaLN, DualResGate
        x = torch.randn(b, s_img, h, device=device, dtype=dtype)
        u = torch.randn_like(x)
        u2 = torch.randn_like(x)
        g = torch.randn(b, h, device=device, dtype=dtype)
        sc = torch.randn(b, h, device=device, dtype=dtype)
        sh = torch.randn(b, h, device=device, dtype=dtype)
        ln_tri = ops.fused_dit_layernorm_adaln(x, sc, sh, eps=1e-6)
        ln_ref = F.layer_norm(x, (h,), eps=1e-6) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        assert _rel_l2(ln_tri, ln_ref) < tol_l2

        xr_tri, nr_tri = ops.fused_dit_res_gate_ln_adaln(x, u, g, sc, sh, eps=1e-6)
        xr_ref = x + u * g.unsqueeze(1)
        nr_ref = F.layer_norm(xr_ref, (h,), eps=1e-6) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        fma_tol = 1e-6 if dtype == torch.float32 else 0.0
        torch.testing.assert_close(xr_tri, xr_ref, rtol=fma_tol, atol=fma_tol)
        assert _rel_l2(nr_tri, nr_ref) < tol_l2

        dual_tri = ops.fused_dit_dual_res_gate_add(x, u, u2, g)
        dual_ref = x + (u + u2) * g.unsqueeze(1)
        dual_tol = 1e-5 if dtype == torch.float32 else 5e-3
        assert _rel_l2(dual_tri, dual_ref) < dual_tol


# ──────────────────────────────────────────────────────────────────────────
# 3. Hopper SM90 Zero Register Spills (n_spills == 0) at H=3072 & H=5120
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="Hopper SM90+ required for register-spill check")
def test_hopper_sm90_all_dit_block_kernels_zero_spills() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    with torch.inference_mode():
        for h in (3072, 5120):
            x = torch.randn(1, 128, h, device=device, dtype=dtype)
            y = torch.randn_like(x)
            mod = torch.randn(1, 6, h, device=device, dtype=dtype)
            tm3 = torch.randn(1, 6, h, device=device, dtype=dtype)
            tm4 = torch.randn(1, 128, 6, h, device=device, dtype=dtype)
            w = torch.randn(h, device=device, dtype=dtype)
            bi = torch.randn(h, device=device, dtype=dtype)
            for tm in (tm3, tm4):
                ops.fused_wan_modulation_ln_adaln(x, mod, tm, eps=1e-6)
                ops.fused_wan_res_gate_ln_affine(x, y, mod, tm, w, bi, eps=1e-6)
                ops.fused_wan_res_add_ln_adaln(x, y, mod, tm, eps=1e-6)
                ops.fused_wan_res_gate_add(x, y, mod, tm)
            ops.fused_wan_rmsnorm(x, w, eps=1e-6)
            ops.fused_wan_head_ln_adaln(
                x,
                torch.randn(1, 2, h, device=device, dtype=dtype),
                torch.randn(1, h, device=device, dtype=dtype),
                out_dtype=dtype,
            )
            sc = torch.randn(1, h, device=device, dtype=dtype)
            sh = torch.randn(1, h, device=device, dtype=dtype)
            gt = torch.randn(1, h, device=device, dtype=dtype)
            ops.fused_dit_layernorm_adaln(x, sc, sh, eps=1e-6)
            ops.fused_dit_res_gate_ln_adaln(x, y, gt, sc, sh, eps=1e-6)
            ops.fused_dit_dual_res_gate_add(x, y, y, gt)

        # HunyuanVideo H=3072, heads=24, D=128
        proj = torch.randn(1, 160, 7 * 3072, device=device, dtype=dtype)
        qkv_i = torch.randn(1, 128, 3 * 3072, device=device, dtype=dtype)
        qkv_t = torch.randn(1, 32, 3 * 3072, device=device, dtype=dtype)
        qw = torch.randn(128, device=device, dtype=dtype)
        kw = torch.randn(128, device=device, dtype=dtype)
        cos = torch.ones(128, 128, device=device, dtype=torch.float32)
        sin = torch.zeros(128, 128, device=device, dtype=torch.float32)
        ops.fused_hunyuan_qkv_rmsnorm_rope_hopper(proj[..., : 3 * 3072], qw, kw, (cos, sin), num_heads=24, rope_len=128)
        ops.fused_hunyuan_double_qkv_rmsnorm_rope_hopper(qkv_i, qkv_t, qw, kw, qw, kw, (cos, sin), num_heads=24)
        ops.fused_strided_gelu_tanh(proj[..., 3 * 3072 :])

    suite = ops._get_triton_suite()
    for name in (
        "wan_mod_ln_adaln",
        "wan_res_gate_ln_affine",
        "wan_res_add_ln_adaln",
        "wan_res_gate_add",
        "wan_rmsnorm",
        "wan_head_ln_adaln",
        "hy_qkv_rmsnorm_rope",
        "hy_double_qkv_rmsnorm_rope",
        "strided_gelu_tanh",
        "dit_ln_adaln",
        "dit_res_gate_ln_adaln",
        "dit_dual_res_gate_add",
    ):
        jit_fn = getattr(suite, name)
        dev_caches = getattr(jit_fn, "device_caches", None)
        assert dev_caches, f"{name} had no compiled device cache entries"
        checked = 0
        for entry in dev_caches.values():
            cd = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(cd, dict):
                continue
            for compiled in cd.values():
                checked += 1
                spills = int(getattr(compiled, "n_spills", 0))
                regs = int(getattr(compiled, "n_regs", 0))
                assert spills == 0, f"{name} spilled {spills} bytes (n_regs={regs})"
        assert checked > 0, f"{name} checked 0 compiled kernels"


# ──────────────────────────────────────────────────────────────────────────
# 4. CUDA Graph Capture/Replay, torch.compile(fullgraph=True), End-to-End Parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("per_token", [False, True])
def test_gpu_wan_model_parity_cuda_graph_and_fullgraph_compile(
    monkeypatch: pytest.MonkeyPatch, per_token: bool
) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = (
        WanModel(
            dim=3072,
            in_dim=16,
            ffn_dim=4096,
            out_dim=16,
            text_dim=64,
            freq_dim=64,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=24,
            num_layers=2,
            has_image_input=False,
            require_vae_embedding=False,
            per_token_timestep=per_token,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    setattr(model, "_worldfoundry_fused_adaln", True)

    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype)
    ctx = torch.randn(1, 16, 64, device=device, dtype=dtype)
    t = torch.full((1, 2 * 4 * 4), 500.0, device=device) if per_token else torch.tensor([500.0], device=device)

    with torch.inference_mode():
        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
        ref = model(x, t, ctx)
        monkeypatch.delenv("WORLDFOUNDRY_KERNEL_BACKEND", raising=False)
        fused_fn = model(x, t, ctx, _worldfoundry_inplace_residual=False)
        fused_ip = model(x, t, ctx, _worldfoundry_inplace_residual=True)

    psnr, cos = _psnr_cosine(fused_fn, ref)
    assert _rel_l2(fused_fn, ref) < 3e-3 and psnr > 50.0 and cos > 0.9999
    torch.testing.assert_close(fused_ip, fused_fn, rtol=0, atol=0)

    # CUDA Graph capture & replay on DiTBlock
    blk = model.blocks[0]
    static_h = torch.randn(1, 32, 3072, device=device, dtype=dtype)
    static_ctx = torch.randn(1, 16, 3072, device=device, dtype=dtype)
    static_tmod = (
        torch.randn(1, 32, 6, 3072, device=device, dtype=dtype)
        if per_token
        else torch.randn(1, 6, 3072, device=device, dtype=dtype)
    )
    static_freqs = model.rotary_frequencies((2, 4, 4), device=device, precision="fp64")

    with torch.inference_mode():
        eager_out = blk(static_h, static_ctx, static_tmod, static_freqs, _worldfoundry_fused_adaln=True)
        # Warmup on side stream before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = blk(static_h, static_ctx, static_tmod, static_freqs, _worldfoundry_fused_adaln=True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_out = blk(static_h, static_ctx, static_tmod, static_freqs, _worldfoundry_fused_adaln=True)
        g.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)

    # Regional torch.compile(fullgraph=True) -> zero graph breaks!
    torch._dynamo.reset()
    compiled_blk = torch.compile(blk.forward, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        comp_out = compiled_blk(static_h, static_ctx, static_tmod, static_freqs, _worldfoundry_fused_adaln=True)
    assert dict(torch._dynamo.utils.counters.get("graph_break", {})) == {}
    assert _rel_l2(comp_out, eager_out) < 2e-3

    # Autograd on CUDA falls back safely and produces valid gradients
    x_g = torch.randn(1, 16, 3072, device=device, dtype=dtype, requires_grad=True)
    c_g = torch.randn(1, 8, 3072, device=device, dtype=dtype)
    tm_g = torch.randn(1, 6, 3072, device=device, dtype=dtype, requires_grad=True)
    f_g = model.rotary_frequencies((1, 4, 4), device=device, precision="fp64")
    out_g = blk(x_g, c_g, tm_g, f_g, _worldfoundry_fused_adaln=True)
    out_g.float().sum().backward()
    assert x_g.grad is not None and torch.isfinite(x_g.grad).all()
    assert tm_g.grad is not None and torch.isfinite(tm_g.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_hunyuan_dit_default_and_sol_l1_exact_parity_and_contiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(77)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = (
        HunyuanVideoDiT(
            in_channels=16,
            hidden_size=3072,
            text_dim=64,
            num_double_blocks=2,
            num_single_blocks=2,
            guidance_embed=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.abs().max().item() == 0.0:
                param.normal_(0.0, 0.02)

    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device, dtype=dtype)
    txt = torch.randn(1, 16, 64, device=device, dtype=dtype)
    mask = torch.zeros(1, 16, device=device, dtype=torch.long)
    mask[0, :8] = 1
    pooled = torch.randn(1, 768, device=device, dtype=dtype)
    guidance = torch.tensor([6.0], device=device, dtype=dtype)
    cos, sin = model.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        # Verify PatchEmbed produces contiguous output so all 20 double blocks hit Triton!
        img_emb = model.img_in(x)
        assert img_emb.is_contiguous() and img_emb.stride(-1) == 1

        monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
        ref_out = model(x, t, txt, mask, pooled, cos, sin, guidance)
        monkeypatch.delenv("WORLDFOUNDRY_KERNEL_BACKEND", raising=False)
        fused_default = model(x, t, txt, mask, pooled, cos, sin, guidance)

    psnr_d, cos_d = _psnr_cosine(fused_default, ref_out)
    assert _rel_l2(fused_default, ref_out) < 1e-2 and psnr_d > 48.0 and cos_d > 0.9999

    if _HOPPER_PLUS:
        rt = model.enable_sol_acceleration(hunyuan13b_sol_profile("l1_exact", cuda_graph="piecewise"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for step in range(3):
                sol_out = model(x, t, txt, mask, pooled, cos, sin, guidance, step_index=step, total_steps=3)
        rec = rt.receipt()
        assert rec["fallbacks"] == []
        assert rec["cuda_graph"]["capture_failed"] == 0
        psnr_s, cos_s = _psnr_cosine(sol_out, ref_out)
        assert _rel_l2(sol_out, ref_out) < 1.5e-2 and cos_s > 0.999
        model.close_sol_acceleration()
