# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive verification suite for HunyuanVideo-13B SOL-Engine acceleration."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.causal3d import (
    HunyuanVideoCausal3DAutoencoder,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.hunyuan_video import HunyuanVideoDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original import (
    HunyuanVideoDiT,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original_acceleration import (
    OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS,
    Hunyuan13BAccelerationConfig,
    Hunyuan13BEasyCache,
    Hunyuan13BSolAttnConfig,
    Hunyuan13BTeaCache,
    accelerate_hunyuan_video_vae,
    apply_hunyuan13b_acceleration,
    build_hunyuan_sol_tile_permutation,
    fuse_hunyuan13b_projections,
    fused_hunyuan_double_qkv_rmsnorm_rope,
    fused_hunyuan_dual_residual_gate_add,
    fused_hunyuan_gelu_tanh_rowwise_fp8,
    fused_hunyuan_layernorm_scale_shift_fp8,
    fused_hunyuan_qkv_rmsnorm_rope,
    fused_hunyuan_residual_gate_layernorm_scale_shift,
    hunyuan13b_sol_profile,
)
from worldfoundry.core.acceleration.quantization.linear import Float8Linear
from worldfoundry.core.attention import apply_nd_rotary_embedding
from worldfoundry.core.nn import RMSNorm, modulate_sequence_with_prefix

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    denom = bf.norm().clamp_min(1e-8)
    return float(((af - bf).norm() / denom).item())


def _init_nonzero_dit(model: HunyuanVideoDiT) -> None:
    """Replace zero-initialized final/modulation weights so outputs carry strong signal."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.abs().max().item() == 0.0:
                if "bias" in name:
                    param.uniform_(-0.05, 0.05)
                else:
                    param.normal_(0.0, 0.02)


def _make_mini_hd128_dit(
    *,
    num_double_blocks: int = 3,
    num_single_blocks: int = 4,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> HunyuanVideoDiT:
    """Build a compact HunyuanVideoDiT with real 13B width (C=3072, H=24, D=128)."""
    torch.manual_seed(17)
    model = HunyuanVideoDiT(
        in_channels=16,
        hidden_size=3072,
        text_dim=64,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
        guidance_embed=True,
    )
    model = model.to(device=device, dtype=dtype).eval()
    _init_nonzero_dit(model)
    return model


# ──────────────────────────────────────────────────────────────────────────
# 1. Config Validation & 3D Tile Permutation Tests (CPU + CUDA)
# ──────────────────────────────────────────────────────────────────────────


def test_config_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="tau"):
        Hunyuan13BSolAttnConfig(enabled=True, tau=-0.1)
    with pytest.raises(ValueError, match="kv_splits"):
        Hunyuan13BSolAttnConfig(enabled=True, kv_splits=cast(Any, 3))
    with pytest.raises(ValueError, match="tile_3d"):
        Hunyuan13BSolAttnConfig(enabled=True, tile_3d=(0, 8, 8))
    with pytest.raises(ValueError, match="unsupported HunyuanVideo-13B SOL profile"):
        hunyuan13b_sol_profile("not_a_real_tier")
    with pytest.raises(ValueError, match="cache_threshold"):
        Hunyuan13BAccelerationConfig(cache_threshold=-1.0)

    balanced = hunyuan13b_sol_profile("l4_sol_balanced")
    assert balanced.sol_attn.dense_steps == 6
    assert balanced.cache_threshold == pytest.approx(0.15)
    assert balanced.cache_max_consecutive_skips == 2
    assert balanced.fp8_use_fast_accum is False

    full = hunyuan13b_sol_profile("l4_sol_full")
    assert full.sol_attn.dense_steps == 5
    assert full.cache_warmup_steps == 8
    assert full.cache_threshold == pytest.approx(0.22)
    assert full.cache_max_consecutive_skips == 3
    assert full.fp8_use_fast_accum is False


@pytest.mark.parametrize(
    ("t", "h", "w", "expected_max_sink_ratio"),
    [
        (17, 45, 80, 0.01),  # 720x1280x65f: 272 / 61200 = 0.44%
        (17, 34, 60, 0.05),  # 544x960x65f: 952 / 34680 = 2.75%
        (4, 16, 16, 0.0),  # Exact multiple -> 0 leftover video sink
        (2, 4, 4, 1.0),  # Smaller than 1x8x8 tile -> identity fallback
        (0, 4, 4, 1.0),  # Empty grid edge case
    ],
)
def test_build_hunyuan_sol_tile_permutation_bijective(
    t: int,
    h: int,
    w: int,
    expected_max_sink_ratio: float,
) -> None:
    total = t * h * w
    perm, inv_perm, video_rem_sink = build_hunyuan_sol_tile_permutation(t, h, w, (1, 8, 8), torch.device("cpu"))
    assert perm.shape == (total,)
    assert inv_perm.shape == (total,)
    if total > 0:
        arange = torch.arange(total, dtype=torch.long)
        assert torch.equal(perm[inv_perm], arange)
        assert torch.equal(inv_perm[perm], arange)
        ratio = video_rem_sink / total
        assert ratio <= expected_max_sink_ratio + 1e-9
    else:
        assert video_rem_sink == 0


def test_teacache_and_easycache_guards_cpu() -> None:
    # TeaCache warmup, dense-last, max-skip, and hit semantics
    tc = Hunyuan13BTeaCache(
        threshold=0.25,
        total_steps=10,
        warmup_steps=2,
        dense_last=2,
        max_consecutive_skips=2,
        coefficients=OFFICIAL_HUNYUAN13B_TEACACHE_COEFFICIENTS,
    )
    base_img = torch.randn(1, 8, 16)
    base_probe = torch.ones(1, 8, 16)
    dense_calls = 0

    def _run_stack() -> torch.Tensor:
        nonlocal dense_calls
        dense_calls += 1
        return base_img + 0.5

    # Autograd enabled MUST refuse caching (safety guard)
    _, hit_grad = tc.evaluate_and_run(
        step=0, total_steps=10, img_in=base_img, modulated_inp=base_probe, num_blocks=60, run_stack=_run_stack
    )
    assert not hit_grad
    assert tc.events[-1].reason == "autograd"
    tc.reset()
    dense_calls = 0

    hits_by_step: list[bool] = []
    with torch.no_grad():
        for step in range(10):
            probe = base_probe + 0.002 * step
            out, hit = tc.evaluate_and_run(
                step=step,
                total_steps=10,
                img_in=base_img,
                modulated_inp=probe,
                num_blocks=60,
                run_stack=_run_stack,
            )
            assert torch.allclose(out, base_img + 0.5)
            hits_by_step.append(hit)

    # First 2 warmup steps and last 2 dense_last steps MUST never skip.
    assert hits_by_step[:2] == [False, False]
    assert hits_by_step[-2:] == [False, False]
    # Interior steps must produce hits without ever exceeding 2 consecutive skips.
    assert any(hits_by_step[2:-2])
    consec = 0
    for h_flag in hits_by_step:
        consec = consec + 1 if h_flag else 0
        assert consec <= 2
    rec = tc.receipt()
    assert rec["fallbacks"] == []
    assert rec["hits"] == sum(hits_by_step)
    assert rec["skipped_block_calls"] == sum(hits_by_step) * 60
    assert rec["dense_block_calls"] == dense_calls * 60

    # EasyCache online factor estimation & guards
    ec = Hunyuan13BEasyCache(threshold=0.20, total_steps=10, warmup_steps=2, dense_last=2, max_consecutive_skips=2)
    ec_hits: list[bool] = []
    with torch.no_grad():
        for step in range(10):
            probe = base_probe + 0.002 * step
            out, hit = ec.evaluate_and_run(
                step=step,
                total_steps=10,
                img_in=base_img,
                modulated_inp=probe,
                num_blocks=60,
                run_stack=lambda: base_img + 0.5,
            )
            assert torch.allclose(out, base_img + 0.5)
            ec_hits.append(hit)
    assert ec_hits[:2] == [False, False]
    assert ec_hits[-2:] == [False, False]
    assert any(ec_hits[2:-2])
    assert ec.receipt()["fallbacks"] == []


# ──────────────────────────────────────────────────────────────────────────
# 2. Triton Fused Kernel Exact Parity & Edge Cases (CUDA)
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_triton_fused_qkv_rmsnorm_rope_parity_and_strided_linear1() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    B, L_img, L_txt, H, D = 2, 64, 18, 24, 128
    L = L_img + L_txt
    C = H * D

    # Simulate strided single-stream linear1 output [B, L, 7*C] -> slice qkv [B, L, 3*C].
    proj = torch.randn(B, L, 7 * C, device=device, dtype=dtype)
    qkv_strided = proj[..., : 3 * C]
    assert not qkv_strided.is_contiguous()

    norm_q = RMSNorm(dim=D, eps=1e-6).to(device=device, dtype=dtype)
    norm_k = RMSNorm(dim=D, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm_q.weight.uniform_(0.5, 1.5)
        norm_k.weight.uniform_(0.5, 1.5)

    # Interleaved cos/sin matching get_nd_rotary_pos_embed(..., use_real=True).
    angles = torch.randn(L_img, D // 2, device=device, dtype=torch.float32)
    cos = angles.cos().repeat_interleave(2, dim=1)
    sin = angles.sin().repeat_interleave(2, dim=1)

    with torch.no_grad():
        # Reference path from original.py
        q_ref, k_ref, v_ref = qkv_strided.contiguous().view(B, L, 3, H, D).unbind(dim=2)
        q_ref = norm_q(q_ref)
        k_ref = norm_k(k_ref)
        qa, qb = q_ref[:, :L_img], q_ref[:, L_img:]
        ka, kb = k_ref[:, :L_img], k_ref[:, L_img:]
        qa, ka = apply_nd_rotary_embedding(qa, ka, (cos, sin), head_first=False)
        q_ref = torch.cat((qa, qb), dim=1)
        k_ref = torch.cat((ka, kb), dim=1)

        # Fused Triton kernel directly on strided linear1 slice!
        q_tri, k_tri, v_tri = fused_hunyuan_qkv_rmsnorm_rope(
            qkv_strided,
            norm_q.weight,
            norm_k.weight,
            (cos, sin),
            num_heads=H,
            rope_len=L_img,
            eps=1e-6,
        )

    assert q_tri.is_contiguous() and k_tri.is_contiguous() and v_tri.is_contiguous()
    assert _rel_l2(q_tri, q_ref) < 2e-3
    assert _rel_l2(k_tri, k_ref) < 2e-3
    assert torch.equal(v_tri, v_ref)

    # Double-stream zero-cat kernel parity
    qkv_img = torch.randn(B, L_img, 3 * C, device=device, dtype=dtype)
    qkv_txt = torch.randn(B, L_txt, 3 * C, device=device, dtype=dtype)
    norm_q_b = RMSNorm(dim=D, eps=1e-6).to(device=device, dtype=dtype)
    norm_k_b = RMSNorm(dim=D, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm_q_b.weight.uniform_(0.5, 1.5)
        norm_k_b.weight.uniform_(0.5, 1.5)
        qa2, ka2, va2 = qkv_img.view(B, L_img, 3, H, D).unbind(dim=2)
        qa2, ka2 = apply_nd_rotary_embedding(norm_q(qa2), norm_k(ka2), (cos, sin), head_first=False)
        qb2, kb2, vb2 = qkv_txt.view(B, L_txt, 3, H, D).unbind(dim=2)
        qb2, kb2 = norm_q_b(qb2), norm_k_b(kb2)
        q_joint_ref = torch.cat((qa2, qb2), dim=1)
        k_joint_ref = torch.cat((ka2, kb2), dim=1)
        v_joint_ref = torch.cat((va2, vb2), dim=1)

        q_joint_tri, k_joint_tri, v_joint_tri = fused_hunyuan_double_qkv_rmsnorm_rope(
            qkv_img,
            qkv_txt,
            norm_q.weight,
            norm_k.weight,
            norm_q_b.weight,
            norm_k_b.weight,
            (cos, sin),
            num_heads=H,
            eps=1e-6,
        )

    assert _rel_l2(q_joint_tri, q_joint_ref) < 2e-3
    assert _rel_l2(k_joint_tri, k_joint_ref) < 2e-3
    assert torch.equal(v_joint_tri, v_joint_ref)


@requires_cuda
def test_triton_fused_adaln_dual_res_and_gelu_fp8_edge_cases() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7)

    B, S, C = 2, 37, 3072
    x = torch.randn(B, S, C, device=device, dtype=dtype)
    upd = torch.randn(B, S, C, device=device, dtype=dtype)
    upd2 = torch.randn(B, S, C, device=device, dtype=dtype)
    gate = torch.randn(B, C, device=device, dtype=torch.float32)  # mixed FP32 mod + BF16 act
    scale = torch.randn(B, C, device=device, dtype=dtype)
    shift = torch.randn(B, C, device=device, dtype=dtype)

    with torch.no_grad():
        # 1. Residual-gate + LayerNorm + AdaLN
        x_ref = x + (upd * gate.unsqueeze(1)).to(dtype)
        norm_ref = modulate_sequence_with_prefix(F.layer_norm(x_ref, (C,), eps=1e-6), shift=shift, scale=scale)
        x_tri, norm_tri = fused_hunyuan_residual_gate_layernorm_scale_shift(x, upd, gate, scale, shift, eps=1e-6)
        assert _rel_l2(x_tri, x_ref) < 5e-3
        assert _rel_l2(norm_tri, norm_ref) < 1e-2

        # 2. Dual residual-gate add
        dual_ref = x + upd * gate.unsqueeze(1) + upd2 * gate.unsqueeze(1)
        dual_tri = fused_hunyuan_dual_residual_gate_add(x, upd, upd2, gate)
        assert _rel_l2(dual_tri, dual_ref.to(dtype)) < 5e-3

        # 3. LayerNorm + AdaLN + rowwise FP8 quant
        ln_ref = modulate_sequence_with_prefix(F.layer_norm(x, (C,), eps=1e-6), shift=shift, scale=scale)
        quant_pair = fused_hunyuan_layernorm_scale_shift_fp8(x, scale, shift, eps=1e-6)
        assert quant_pair is not None
        codes, scales = quant_pair
        deq = (codes.float() * scales).view(B, S, C)
        assert _rel_l2(deq, ln_ref) < 0.04

        # 4. GELU-tanh + rowwise FP8 quant: 1-pass (1024), monotone tiled (12288),
        #    strided linear1 slice, all-negative row, all-zero row, and empty tensor!
        for cols in (1024, 12288):
            mat = torch.randn(16, cols, device=device, dtype=dtype)
            mat[0].fill_(-2.5)  # all-negative row exercises non-monotone fallback scan
            mat[1].zero_()  # all-zero row exercises tiny-clamp safety
            codes_g, scales_g = fused_hunyuan_gelu_tanh_rowwise_fp8(mat)
            ref_g = F.gelu(mat, approximate="tanh")
            deq_g = codes_g.float() * scales_g
            assert torch.isfinite(deq_g).all()
            assert _rel_l2(deq_g, ref_g) < 0.04
            assert _rel_l2(deq_g[0], ref_g[0]) < 0.05
            assert deq_g[1].abs().max().item() == 0.0

        # Strided 3D slice directly from linear1 [1, 32, 21504] -> [1, 32, 12288]
        big = torch.randn(1, 32, 21504, device=device, dtype=dtype)
        mlp_view = big[..., 9216:]
        assert not mlp_view.is_contiguous()
        codes_s, scales_s = fused_hunyuan_gelu_tanh_rowwise_fp8(mlp_view)
        ref_s = F.gelu(mlp_view, approximate="tanh").reshape(-1, 12288)
        assert _rel_l2(codes_s.float() * scales_s, ref_s) < 0.04

        # Empty input edge case
        empty = torch.empty(0, 12288, device=device, dtype=dtype)
        ec, es = fused_hunyuan_gelu_tanh_rowwise_fp8(empty)
        assert ec.shape == (0, 12288) and es.shape == (0, 1)


# ──────────────────────────────────────────────────────────────────────────
# 3. Projection Fusion, Text Compaction & L1 Exact Parity vs L0 Baseline
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_l1_exact_fusion_compaction_fa3_graph_parity_and_multibatch_safety() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    ref_model = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    acc_model = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    acc_model.load_state_dict(ref_model.state_dict())

    # Verify idempotent projection fusion
    f1 = fuse_hunyuan13b_projections(acc_model, release_originals=True)
    f2 = fuse_hunyuan13b_projections(acc_model, release_originals=True)
    assert f1["single_linear1_fused"] == 2
    assert f2["single_linear1_fused"] == 2
    assert f1["double_qkv_fused_verified"] == 4
    assert all(blk.linear1 is not None and blk.to_qkv is None for blk in acc_model.single_blocks)

    runtime = apply_hunyuan13b_acceleration(acc_model, "l1_exact")

    # 1. B=1 with padded text_mask (6 valid out of 16) -> bitwise-lossless compaction + maskless FA3 + Piecewise Graph
    torch.manual_seed(123)
    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device, dtype=dtype)
    txt = torch.randn(1, 16, 64, device=device, dtype=dtype)
    mask = torch.zeros(1, 16, device=device, dtype=torch.long)
    mask[0, :6] = 1
    pooled = torch.randn(1, 768, device=device, dtype=dtype)
    guidance = torch.tensor([6.0], device=device, dtype=dtype)
    cos, sin = ref_model.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_ref = ref_model(x, t, txt, mask, pooled, cos, sin, guidance)
        out_acc = None
        for step in range(4):
            out_acc = acc_model(x, t, txt, mask, pooled, cos, sin, guidance, step_index=step, total_steps=4)

    assert out_acc is not None
    assert _rel_l2(out_acc, out_ref) < 2e-2
    receipt = runtime.receipt()
    assert receipt["fallbacks"] == []
    assert receipt["attention"]["text_compactions"] == 4
    assert receipt["attention"]["fa3_calls"] == 4 * (2 + 2)
    assert receipt["fused_ops"]["qkv_rmsnorm_rope_calls"] == 4 * (2 + 2)
    assert receipt["cuda_graph"]["capture"] >= 2
    assert receipt["cuda_graph"]["replay"] >= 4
    assert receipt["cuda_graph"]["capture_failed"] == 0

    # Verify reset() + pure replay still tracks kernel executions accurately!
    runtime.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_replay = acc_model(x, t, txt, mask, pooled, cos, sin, guidance, step_index=0, total_steps=4)
    assert _rel_l2(out_replay, out_ref) < 2e-2
    replay_rec = runtime.receipt()
    assert replay_rec["fallbacks"] == []
    assert replay_rec["attention"]["fa3_calls"] == 4

    # 2. B=2 with heterogeneous masks -> varlen fallback (no silent corruption!)
    x2 = torch.randn(2, 16, 2, 8, 8, device=device, dtype=dtype)
    t2 = torch.tensor([500.0, 250.0], device=device, dtype=dtype)
    txt2 = torch.randn(2, 16, 64, device=device, dtype=dtype)
    mask2 = torch.zeros(2, 16, device=device, dtype=torch.long)
    mask2[0, :5] = 1
    mask2[1, :11] = 1
    pooled2 = torch.randn(2, 768, device=device, dtype=dtype)
    guidance2 = torch.tensor([6.0, 6.0], device=device, dtype=dtype)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_ref2 = ref_model(x2, t2, txt2, mask2, pooled2, cos, sin, guidance2)
        out_acc2 = acc_model(x2, t2, txt2, mask2, pooled2, cos, sin, guidance2, step_index=0, total_steps=1)
    assert _rel_l2(out_acc2, out_ref2) < 2e-2


# ──────────────────────────────────────────────────────────────────────────
# 4. Calibrated Rowwise FP8 (use_fast_accum=False) + Fused FP8 FFN
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_l2_calibrated_rowwise_fp8_exact_accum_and_fused_ffn() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    ref_model = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=3, device=device, dtype=dtype)
    fp8_model = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=3, device=device, dtype=dtype)
    fp8_model.load_state_dict(ref_model.state_dict())

    runtime = apply_hunyuan13b_acceleration(fp8_model, "l2_fp8")

    # Verify sensitive modules & edge blocks stayed native BF16 nn.Linear!
    db0: Any = fp8_model.double_blocks[0]
    db1: Any = fp8_model.double_blocks[1]
    sb0: Any = fp8_model.single_blocks[0]
    sb_last: Any = fp8_model.single_blocks[-1]
    assert isinstance(fp8_model.final_layer.linear, nn.Linear)
    assert isinstance(db0.component_a.to_qkv, nn.Linear)
    assert isinstance(sb_last.linear1, nn.Linear)
    assert isinstance(db1.component_b.to_qkv, nn.Linear)  # text stream kept BF16
    assert isinstance(db1.component_a.mod.linear, nn.Linear)

    # Verify interior video / single heavy projections are rowwise Float8Linear with use_fast_accum=False!
    assert isinstance(db1.component_a.to_qkv, Float8Linear)
    assert isinstance(sb0.linear1, Float8Linear)
    assert isinstance(sb0.ff[2], Float8Linear)
    for mod in fp8_model.modules():
        if isinstance(mod, Float8Linear):
            assert mod.scaling == "rowwise"
            assert mod.use_fast_accum is False
            assert mod.weight is None  # keep_dense_fallback=False

    x = torch.randn(1, 16, 2, 16, 16, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device, dtype=dtype)
    txt = torch.randn(1, 16, 64, device=device, dtype=dtype)
    mask = torch.ones(1, 16, device=device, dtype=torch.long)
    mask[0, 10:] = 0
    pooled = torch.randn(1, 768, device=device, dtype=dtype)
    guidance = torch.tensor([6.0], device=device, dtype=dtype)
    cos, sin = ref_model.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_ref = ref_model(x, t, txt, mask, pooled, cos, sin, guidance)
        out_fp8 = None
        for step in range(4):
            out_fp8 = fp8_model(x, t, txt, mask, pooled, cos, sin, guidance, step_index=step, total_steps=4)

    assert out_fp8 is not None and torch.isfinite(out_fp8).all()
    assert _rel_l2(out_fp8, out_ref) < 0.12
    rec = runtime.receipt()
    assert rec["fallbacks"] == []
    assert rec["quantization"]["fp8_fast_accum_modules"] == 0
    assert rec["quantization"]["low_precision_kernel_calls"] > 0
    assert rec["quantization"]["fused_fp8_ffn_calls"] > 0
    assert rec["quantization"]["adaln_fp8_calls"] > 0
    assert rec["quantization"]["dense_fallback_calls"] == 0
    assert rec["cuda_graph"]["capture_failed"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 5. 3D-Tiled SM90 Sol-Attn + TeaCache / EasyCache + Piecewise CUDA Graph
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_l3_sol_attn_schedule_and_l4_teacache_easycache_denoiser_integration() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 3 double blocks + 4 single blocks -> dense edge blocks: double (0,1), single (0,2,3)
    # Interior Sol-Attn blocks: double[2] + single[1] = 2 Sol-Attn blocks per sparse step!
    model = _make_mini_hd128_dit(num_double_blocks=3, num_single_blocks=4, device=device, dtype=dtype)
    cfg = hunyuan13b_sol_profile(
        "l4_sol_full",
        kv_splits=1,
        sol_tau=1.0,
        sol_dense_steps=2,
        teacache_threshold=0.22,
    )
    apply_hunyuan13b_acceleration(model, cfg)
    denoiser = HunyuanVideoDenoiser(model, enable_cuda_graph="off")

    # Latent (1, 16, 2, 16, 16) -> post-patch (T'=2, H'=8, W'=8) = 128 video tokens (2 exact 1x8x8 tiles!)
    torch.manual_seed(99)
    x = torch.randn(1, 16, 2, 16, 16, device=device, dtype=dtype)
    txt = torch.randn(1, 20, 64, device=device, dtype=dtype)
    mask = torch.zeros(1, 20, device=device, dtype=torch.long)
    mask[0, :8] = 1
    pooled = torch.randn(1, 768, device=device, dtype=dtype)

    total_steps = 10
    with torch.no_grad():
        for step in range(total_steps):
            t_val = 1000.0 * (1.0 - step / total_steps)
            inp = DenoiserInput(
                latents=x + 0.001 * step,
                timestep=torch.tensor([t_val], device=device, dtype=dtype),
                next_timestep=cast(Any, None),
                step_index=step,
                total_steps=total_steps,
                request_id="req-sol-1",
                branch="positive",
                conditioning={
                    "text_states": txt,
                    "text_states_2": pooled,
                    "text_mask": mask,
                    "embedded_guidance_scale": 6.0,
                },
            )
            out = denoiser(inp).sample
            assert out.shape == x.shape
            assert torch.isfinite(out).all()

    rec = denoiser.acceleration_receipt()
    assert rec["fallbacks"] == []
    assert rec["attention"]["sol_dense_warmup_steps"] == 2
    assert rec["attention"]["sol_sparse_steps"] == total_steps - 2
    assert rec["attention"]["sol_attn_calls"] > 0
    assert rec["attention"]["fa3_calls"] > 0
    assert rec["cache"]["algorithm"] == "teacache"
    assert rec["cache"]["events"] == total_steps
    assert rec["cache"]["hits"] > 0
    assert rec["cuda_graph"]["capture_failed"] == 0
    assert rec["cuda_graph"]["cache_skips"] > 0

    # Verify request isolation: second request starts fresh at step 0 without colliding with req-sol-1
    inp_req2 = DenoiserInput(
        latents=x,
        timestep=torch.tensor([1000.0], device=device, dtype=dtype),
        next_timestep=cast(Any, None),
        step_index=0,
        total_steps=total_steps,
        request_id="req-sol-2",
        branch="positive",
        conditioning={"text_states": txt, "text_states_2": pooled, "text_mask": mask},
    )
    with torch.no_grad():
        assert torch.isfinite(denoiser(inp_req2).sample).all()
    rec2 = denoiser.acceleration_receipt()
    assert "req-sol-1:positive" in rec2["cache"]["branches"]
    assert "req-sol-2:positive" in rec2["cache"]["branches"]

    # Verify Sol-Attn fails closed on heterogeneous B=2 masks (never silently ignores masks!)
    mask_het = torch.zeros(2, 20, device=device, dtype=torch.long)
    mask_het[0, :4] = 1
    mask_het[1, :9] = 1
    with pytest.raises(ValueError, match="uniform prefix text_mask"):
        with torch.no_grad():
            denoiser(
                DenoiserInput(
                    latents=x.repeat(2, 1, 1, 1, 1),
                    timestep=torch.tensor([500.0, 500.0], device=device, dtype=dtype),
                    next_timestep=cast(Any, None),
                    step_index=0,
                    total_steps=2,
                    request_id="req-het",
                    branch="positive",
                    conditioning={
                        "text_states": txt.repeat(2, 1, 1),
                        "text_states_2": pooled.repeat(2, 1),
                        "text_mask": mask_het,
                    },
                )
            )

    # Verify EasyCache profile end-to-end
    ec_model = _make_mini_hd128_dit(num_double_blocks=3, num_single_blocks=4, device=device, dtype=dtype)
    ec_cfg = hunyuan13b_sol_profile("l4_easycache", sol_dense_steps=2, teacache_threshold=0.20)
    ec_runtime = apply_hunyuan13b_acceleration(ec_model, ec_cfg)
    cos, sin = ec_model.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for step in range(8):
            out = ec_model(
                x + 0.001 * step,
                torch.tensor([1000.0 - 100.0 * step], device=device, dtype=dtype),
                txt,
                mask,
                pooled,
                cos,
                sin,
                torch.tensor([6.0], device=device, dtype=dtype),
                step_index=step,
                total_steps=8,
                request_id="req-ec",
            )
            assert torch.isfinite(out).all()
    ec_rec = ec_runtime.receipt()
    assert ec_rec["fallbacks"] == []
    assert ec_rec["cache"]["algorithm"] == "easycache"
    assert ec_rec["cache"]["hits"] > 0
    assert ec_rec["cuda_graph"]["capture_failed"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 6. HunyuanVideo 3D Causal VAE Acceleration
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_hunyuan_causal3d_vae_acceleration_channels_last_3d_bf16() -> None:
    device = torch.device("cuda")
    torch.manual_seed(11)
    vae_ref = (
        HunyuanVideoCausal3DAutoencoder(
            in_channels=3,
            out_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=8,
            sample_size=64,
            sample_tsize=16,
            spatial_compression_ratio=2,
            time_compression_ratio=2,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    vae_acc = (
        HunyuanVideoCausal3DAutoencoder(
            in_channels=3,
            out_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=8,
            sample_size=64,
            sample_tsize=16,
            spatial_compression_ratio=2,
            time_compression_ratio=2,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    vae_acc.load_state_dict(vae_ref.state_dict())

    receipt = accelerate_hunyuan_video_vae(
        vae_acc,
        dtype=torch.bfloat16,
        channels_last_3d=True,
        decode_autocast=True,
        compile_decoder=False,
        tiling=False,
    )
    assert receipt["fallbacks"] == []
    assert receipt["conv3d_total"] > 0
    assert receipt["conv3d_converted"] == receipt["conv3d_total"]

    latents = torch.randn(1, 4, 3, 8, 8, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_video = vae_ref.decode(latents).sample
    with torch.no_grad():
        acc_video = vae_acc.decode(latents).sample

    assert acc_video.shape == ref_video.shape
    assert torch.isfinite(acc_video).all()
    assert _rel_l2(acc_video, ref_video) < 1e-2

    # Re-applying with updated flags must cleanly reconfigure without stacking wrappers
    # AND revert 3x3x3 Conv3d weights back to contiguous NCDHW layout.
    conv3x3 = next(m for m in vae_acc.modules() if isinstance(m, nn.Conv3d) and tuple(m.kernel_size) == (3, 3, 3))
    assert conv3x3.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert not conv3x3.weight.is_contiguous(memory_format=torch.contiguous_format)

    # Also verify FP32 latent input works cleanly under BF16 decode_autocast=True.
    with torch.no_grad():
        acc_fp32_in = vae_acc.decode(latents.float()).sample
    assert acc_fp32_in.shape == ref_video.shape
    assert torch.isfinite(acc_fp32_in).all()

    receipt_ncdhw = accelerate_hunyuan_video_vae(
        vae_acc,
        dtype=torch.bfloat16,
        channels_last_3d=False,
        decode_autocast=False,
        compile_decoder=False,
        tiling=False,
    )
    assert receipt_ncdhw["fallbacks"] == []
    assert receipt_ncdhw["channels_last_3d"] is False
    assert conv3x3.weight.is_contiguous(memory_format=torch.contiguous_format)
    with torch.no_grad():
        acc_ncdhw = vae_acc.decode(latents).sample
    assert acc_ncdhw.shape == ref_video.shape
    assert torch.isfinite(acc_ncdhw).all()

    # Verify compile_decoder=True -> True (no double wrap) -> False (restores original module).
    orig_decoder_cls = type(vae_acc.decoder)
    rec_comp1 = accelerate_hunyuan_video_vae(
        vae_acc,
        dtype=torch.bfloat16,
        channels_last_3d=False,
        decode_autocast=False,
        compile_decoder=True,
        tiling=False,
    )
    assert rec_comp1["compile_decoder"] is True
    compiled_dec = vae_acc.decoder
    rec_comp2 = accelerate_hunyuan_video_vae(
        vae_acc,
        dtype=torch.bfloat16,
        channels_last_3d=False,
        decode_autocast=False,
        compile_decoder=True,
        tiling=False,
    )
    assert rec_comp2["compile_decoder"] is True
    assert vae_acc.decoder is compiled_dec
    with torch.no_grad():
        acc_comp = vae_acc.decode(latents).sample
    assert torch.isfinite(acc_comp).all()
    assert _rel_l2(acc_comp, acc_ncdhw) < 2.5e-2
    assert _rel_l2(acc_comp, ref_video) < 2.5e-2

    rec_uncomp = accelerate_hunyuan_video_vae(
        vae_acc,
        dtype=torch.bfloat16,
        channels_last_3d=False,
        decode_autocast=False,
        compile_decoder=False,
        tiling=False,
    )
    assert rec_uncomp["compile_decoder"] is False
    assert type(vae_acc.decoder) is orig_decoder_cls


# ──────────────────────────────────────────────────────────────────────────
# 7. Adversarial Regression Suite (Strided Kernels, Telemetry, Graphs, Lifecycle)
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
def test_strided_triton_kernels_never_corrupt_memory() -> None:
    """Verify non-contiguous dim-0 and dim-1 strided inputs write in-bounds with exact parity."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(2026)

    for big_shape, slc in (
        ((2, 64, 3072), (slice(None), slice(None, 32), slice(None))),  # strided dim 0
        ((2, 32, 6144), (slice(None), slice(None), slice(None, 3072))),  # strided dim 1
    ):
        big = torch.randn(*big_shape, device=device, dtype=dtype)
        x = big[slc]
        assert not x.is_contiguous() and x.stride(-1) == 1
        upd = torch.randn(2, 32, 3072, device=device, dtype=dtype)
        upd2 = torch.randn(2, 32, 3072, device=device, dtype=dtype)
        gate = torch.randn(2, 3072, device=device, dtype=dtype)
        scale = torch.randn(2, 3072, device=device, dtype=dtype)
        shift = torch.randn(2, 3072, device=device, dtype=dtype)

        with torch.no_grad():
            x_ref = x + (upd * gate.unsqueeze(1)).to(dtype)
            norm_ref = modulate_sequence_with_prefix(F.layer_norm(x_ref, (3072,), eps=1e-6), shift=shift, scale=scale)
            x_tri, norm_tri = fused_hunyuan_residual_gate_layernorm_scale_shift(x, upd, gate, scale, shift, eps=1e-6)
            assert x_tri.is_contiguous() and norm_tri.is_contiguous()
            assert _rel_l2(x_tri, x_ref) < 5e-3
            assert _rel_l2(norm_tri, norm_ref) < 1e-2

            dual_ref = x + (upd + upd2) * gate.unsqueeze(1)
            dual_tri = fused_hunyuan_dual_residual_gate_add(x, upd, upd2, gate)
            assert dual_tri.is_contiguous()
            assert _rel_l2(dual_tri, dual_ref.to(dtype)) < 5e-3

    # B=2 strided linear1 slice with non-zero storage_offset
    big_l1 = torch.randn(2, 32, 21504, device=device, dtype=dtype)
    mlp_view = big_l1[..., 9216:]
    assert not mlp_view.is_contiguous() and mlp_view.storage_offset() == 9216
    with torch.no_grad():
        codes, scales = fused_hunyuan_gelu_tanh_rowwise_fp8(mlp_view)
        ref_g = F.gelu(mlp_view, approximate="tanh").reshape(-1, 12288)
        assert _rel_l2(codes.float() * scales, ref_g) < 0.04

    # Portable fallback on transposed non-contiguous QKV (stride(-1) != 1)
    qkv_t = torch.randn(1, 3 * 256, 8).transpose(1, 2)
    q_fb, k_fb, v_fb = fused_hunyuan_qkv_rmsnorm_rope(qkv_t, torch.ones(64), torch.ones(64), None, num_heads=4)
    assert q_fb.shape == (1, 8, 4, 64) and k_fb.shape == (1, 8, 4, 64) and v_fb.shape == (1, 8, 4, 64)


@requires_cuda
def test_exact_telemetry_warmup0_fp8_capture_varlen_and_l0_ref() -> None:
    """Verify exact logical kernel counts across warmup=0, FP8 capture, B=2 varlen, and l0_ref."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device, dtype=dtype)
    txt = torch.randn(1, 16, 64, device=device, dtype=dtype)
    mask = torch.zeros(1, 16, device=device, dtype=torch.long)
    mask[0, :6] = 1
    pooled = torch.randn(1, 768, device=device, dtype=dtype)
    guidance = torch.tensor([6.0], device=device, dtype=dtype)

    # 1. cuda_graph_warmup=0 must NOT double-count kernel calls!
    m_w0 = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    cos, sin = m_w0.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)
    rt_w0 = apply_hunyuan13b_acceleration(
        m_w0,
        Hunyuan13BAccelerationConfig(
            attention_backend="flash_attention_3", cuda_graph="piecewise", cuda_graph_warmup=0
        ),
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for s in range(4):
            m_w0(x, t, txt, mask, pooled, cos, sin, guidance, step_index=s, total_steps=4)
    rec_w0 = rt_w0.receipt()
    assert rec_w0["attention"]["fa3_calls"] == 4 * 4
    assert rec_w0["fused_ops"]["qkv_rmsnorm_rope_calls"] == 4 * 4
    rt_w0.close()

    # 2. l2_fp8 capture step without reset() must report exact logical low_precision_kernel_calls!
    m_fp8 = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=3, device=device, dtype=dtype)
    rt_fp8 = apply_hunyuan13b_acceleration(m_fp8, "l2_fp8")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for s in range(4):
            m_fp8(x, t, txt, mask, pooled, cos, sin, guidance, step_index=s, total_steps=4)
    rec_fp8 = rt_fp8.receipt()
    n_fp8 = rec_fp8["quantization"]["fp8_module_count"]
    assert rec_fp8["quantization"]["low_precision_kernel_calls"] == 4 * n_fp8
    assert rec_fp8["fallbacks"] == []
    rt_fp8.close()

    # 3. B=2 heterogeneous varlen multi-step + dynamic mask change on CUDA Graph replay + fa3_calls counted!
    ref_m = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    acc_m = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    acc_m.load_state_dict(ref_m.state_dict())
    rt_v = apply_hunyuan13b_acceleration(acc_m, "l1_exact")
    x2 = x.repeat(2, 1, 1, 1, 1)
    t2 = torch.tensor([500.0, 250.0], device=device, dtype=dtype)
    txt2 = txt.repeat(2, 1, 1)
    pooled2 = pooled.repeat(2, 1)
    g2 = guidance.repeat(2)
    m1 = torch.zeros(2, 16, device=device, dtype=torch.long)
    m1[0, :5] = 1
    m1[1, :11] = 1
    m2 = torch.zeros(2, 16, device=device, dtype=torch.long)
    m2[0, :3] = 1
    m2[1, :9] = 1
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for s in range(3):
            acc_m(x2, t2, txt2, m1, pooled2, cos, sin, g2, step_index=s, total_steps=4)
        out2_ref = ref_m(x2, t2, txt2, m2, pooled2, cos, sin, g2)
        out2_acc = acc_m(x2, t2, txt2, m2, pooled2, cos, sin, g2, step_index=3, total_steps=4)
    assert _rel_l2(out2_acc, out2_ref) < 2e-2
    rec_v = rt_v.receipt()
    assert rec_v["attention"]["fa3_calls"] == 4 * 4
    assert rec_v["cuda_graph"]["capture_failed"] == 0
    assert rec_v["fallbacks"] == []
    rt_v.close()

    # 4. l0_ref applied via apply_hunyuan13b_acceleration must count torch_sdpa_calls!
    m_l0 = _make_mini_hd128_dit(num_double_blocks=2, num_single_blocks=2, device=device, dtype=dtype)
    rt_l0 = apply_hunyuan13b_acceleration(m_l0, "l0_ref")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        m_l0(x, t, txt, mask, pooled, cos, sin, guidance, step_index=0, total_steps=1)
    assert rt_l0.receipt()["attention"]["torch_sdpa_calls"] == 4
    rt_l0.close()


@requires_cuda
def test_denoiser_graph_downgrade_teacache_safety_reapply_and_lifecycle() -> None:
    """Verify Denoiser whole->piecewise guard, TeaCache device/train/sync guards, and lifecycle."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. HunyuanVideoDenoiser(model, enable_cuda_graph=True) on SOL model must downgrade to piecewise with 0 failures!
    m = _make_mini_hd128_dit(num_double_blocks=3, num_single_blocks=4, device=device, dtype=dtype)
    rt = apply_hunyuan13b_acceleration(
        m, hunyuan13b_sol_profile("l4_sol_full", sol_dense_steps=1, cache_warmup_steps=1)
    )
    denoiser = HunyuanVideoDenoiser(m, enable_cuda_graph=True)
    assert denoiser._graph_runner is None
    assert denoiser._piecewise_graph is None  # reuses rt.piecewise_graph without duplicate pool

    x = torch.randn(1, 16, 2, 16, 16, device=device, dtype=dtype)
    txt = torch.randn(1, 20, 64, device=device, dtype=dtype)
    mask = torch.zeros(1, 20, device=device, dtype=torch.long)
    mask[0, :8] = 1
    pooled = torch.randn(1, 768, device=device, dtype=dtype)
    with torch.no_grad():
        for s in range(4):
            inp = DenoiserInput(
                latents=x + 0.001 * s,
                timestep=torch.tensor([1000.0 - 100.0 * s], device=device, dtype=dtype),
                next_timestep=cast(Any, None),
                step_index=s,
                total_steps=4,
                request_id="req-dw",
                branch="positive",
                conditioning={"text_states": txt, "text_states_2": pooled, "text_mask": mask},
            )
            assert torch.isfinite(denoiser(inp).sample).all()
    rec = denoiser.acceleration_receipt()
    assert rec["fallbacks"] == []
    assert rec["cuda_graph"]["capture_failed"] == 0
    assert rec["cuda_graph"]["capture"] >= 2

    # 2. model.train() with torch.no_grad() MUST NOT skip blocks via TeaCache!
    m.train()
    rt.reset()
    cos, sin = m.prepare_freqs(x)
    cos, sin = cos.to(device), sin.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for s in range(4):
            m(
                x,
                torch.tensor([500.0], device=device, dtype=dtype),
                txt,
                mask,
                pooled,
                cos,
                sin,
                torch.tensor([6.0], device=device, dtype=dtype),
                step_index=s,
                total_steps=4,
            )
    assert rt.receipt()["cache"]["hits"] == 0
    m.eval()

    # 3. Re-applying quantization="none" after keep_dense_fallback=False FP8 MUST fail closed!
    with pytest.raises(RuntimeError, match="keep_dense_fallback=False"):
        apply_hunyuan13b_acceleration(m, "l1_exact")

    denoiser.close()
    assert rt.piecewise_graph is not None and rt.piecewise_graph.report()["segment_count"] >= 2

    # 4. TeaCache CPU autograd -> CUDA no_grad device transition + decision_sync SP consensus override!
    tc = Hunyuan13BTeaCache(threshold=0.5, total_steps=6, warmup_steps=1, dense_last=1, max_consecutive_skips=2)
    x_cpu = torch.randn(1, 8, 16, requires_grad=True)
    p_cpu = torch.ones(1, 8, 16, requires_grad=True)
    tc.evaluate_and_run(
        step=0, total_steps=6, img_in=x_cpu, modulated_inp=p_cpu, num_blocks=4, run_stack=lambda: x_cpu + 0.1
    )
    x_gpu = torch.randn(1, 8, 16, device=device)
    p_gpu = torch.ones(1, 8, 16, device=device)
    with torch.no_grad():
        out1, hit1 = tc.evaluate_and_run(
            step=1, total_steps=6, img_in=x_gpu, modulated_inp=p_gpu, num_blocks=4, run_stack=lambda: x_gpu + 0.1
        )
        assert not hit1 and out1.device == x_gpu.device
        # Next step would hit locally, but decision_sync=False forces unanimous dense refresh ("sp-sync")!
        _, hit2 = tc.evaluate_and_run(
            step=2,
            total_steps=6,
            img_in=x_gpu,
            modulated_inp=p_gpu,
            num_blocks=4,
            run_stack=lambda: x_gpu + 0.1,
            decision_sync=lambda _local_hit: False,
        )
        assert not hit2
        assert tc.events[-1].reason == "sp-sync"

    # 5. Oversized tile_3d preserves full video_rem_sink & single-block token_replace after linear1 fusion
    m_small = _make_mini_hd128_dit(num_double_blocks=1, num_single_blocks=1, device=device, dtype=dtype)
    m_small_ref = _make_mini_hd128_dit(num_double_blocks=1, num_single_blocks=1, device=device, dtype=dtype)
    m_small_ref.load_state_dict(m_small.state_dict())
    fuse_hunyuan13b_projections(m_small, release_originals=True)
    hx = torch.randn(1, 24, 3072, device=device, dtype=dtype)
    hvec = torch.randn(1, 3072, device=device, dtype=dtype)
    htr = torch.randn(1, 3072, device=device, dtype=dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        sb_ref = m_small_ref.single_blocks[0](hx, hvec, freqs_cis=None, txt_len=8, token_replace_vec=htr, tr_token=4)
        sb_fus = m_small.single_blocks[0](hx, hvec, freqs_cis=None, txt_len=8, token_replace_vec=htr, tr_token=4)
    assert _rel_l2(sb_fus, sb_ref) < 1e-2

    m_tile = _make_mini_hd128_dit(num_double_blocks=3, num_single_blocks=4, device=device, dtype=dtype)
    cfg_big_tile = Hunyuan13BAccelerationConfig(
        attention_backend="sol_attn",
        sol_attn=Hunyuan13BSolAttnConfig(
            enabled=True,
            tau=1.0,
            kv_splits=4,
            tile_3d=(1, 16, 16),  # larger than 8x8 post-patch grid -> full video sink preserved
            dense_steps=0,
        ),
        quantization="none",
        cuda_graph="off",
    )
    with apply_hunyuan13b_acceleration(m_tile, cfg_big_tile) as rt_tile:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            out_tile = m_tile(
                x,
                torch.tensor([500.0], device=device, dtype=dtype),
                txt,
                mask,
                pooled,
                cos,
                sin,
                torch.tensor([6.0], device=device, dtype=dtype),
                step_index=0,
                total_steps=2,
            )
            assert torch.isfinite(out_tile).all()
        assert rt_tile.receipt()["fallbacks"] == []
        assert rt_tile.receipt()["attention"]["sol_attn_calls"] > 0


def test_hunyuan13b_resolve_sp_context_supports_runtime_sp_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _resolve_sp_context recognizes groups initialized via sequence_parallel.runtime."""
    import torch.distributed as dist

    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original_acceleration import (
        _resolve_sp_context,
    )
    from worldfoundry.core.distributed.sequence_parallel import runtime as sp_runtime

    fake_group = object()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2 if group is fake_group else 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is fake_group else 0)
    monkeypatch.setattr(sp_runtime, "get_sequence_parallel_group", lambda: fake_group)

    enabled, grp, world_size, rank = _resolve_sp_context(True)
    assert enabled is True
    assert grp is fake_group
    assert world_size == 2
    assert rank == 1
    assert _resolve_sp_context(False) == (False, None, 1, 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert _resolve_sp_context(True) == (False, None, 1, 0)
