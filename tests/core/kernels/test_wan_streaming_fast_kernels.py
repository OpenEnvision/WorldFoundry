# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit, parity, sync-elimination, hook, autograd, CUDA Graph, and CPU tests for streaming Wan kernels."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@contextlib.contextmanager
def _patch_cpu_flash_attention(mod: Any, attr: str = "flash_attention", *, force: bool = False):
    """Patch CUDA-only / forward-only upstream flash_attention to PyTorch SDPA when needed."""
    if torch.cuda.is_available() and not force:
        yield
        return

    def _sdpa_fallback(q, k, v, q_lens=None, k_lens=None, **_kwargs):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    orig = getattr(mod, attr)
    setattr(mod, attr, _sdpa_fallback)
    try:
        yield
    finally:
        setattr(mod, attr, orig)


def _psnr_and_cos(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a = actual.detach().float().reshape(-1)
    b = ref.detach().float().reshape(-1)
    mse = float(torch.mean((a - b) ** 2).item())
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    if mse == 0.0:
        return float("inf"), 1.0
    data_range = float((b.max() - b.min()).clamp_min(1e-6).item())
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse), cos


def _count_cuda_syncs(fn):
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    counts = {"cuda_item": 0, "cuda_tolist": 0}

    def _item(self):
        if self.is_cuda:
            counts["cuda_item"] += 1
        return orig_item(self)

    def _tolist(self):
        if self.is_cuda:
            counts["cuda_tolist"] += 1
        return orig_tolist(self)

    torch.Tensor.item = _item  # type: ignore[method-assign]
    torch.Tensor.tolist = _tolist  # type: ignore[method-assign]
    try:
        out = fn()
    finally:
        torch.Tensor.item = orig_item  # type: ignore[method-assign]
        torch.Tensor.tolist = orig_tolist  # type: ignore[method-assign]
    return out, counts


# ──────────────────────────────────────────────────────────────────────────
# 1. LingBot-Video Dense/MoE fused Triton kernels + block parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("token_level_mod", [False, True])
def test_lingbot_video_fused_block_kernels_parity(dtype: torch.dtype, token_level_mod: bool) -> None:
    from worldfoundry.synthesis.visual_generation.lingbot_video.acceleration import (
        fused_lingbot_layernorm_adaln,
        fused_lingbot_post_norm_residual_gate_add,
        fused_lingbot_post_norm_residual_gate_rmsnorm_adaln,
        fused_lingbot_qk_rmsnorm_rope,
        fused_lingbot_rmsnorm_adaln,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)
    B, S, H, D = 2, 32, 4, 64
    C = H * D
    M = S if token_level_mod else 1

    x = torch.randn(B, S, C, device=device, dtype=dtype)
    attn_out = torch.randn(B, S, C, device=device, dtype=dtype)
    ffn_out = torch.randn(B, S, C, device=device, dtype=dtype)
    w1 = torch.randn(C, device=device, dtype=torch.float32)
    w_post_a = torch.randn(C, device=device, dtype=torch.float32)
    w2 = torch.randn(C, device=device, dtype=torch.float32)
    w_post_f = torch.randn(C, device=device, dtype=torch.float32)

    mod = torch.randn(B, M, 6 * C, device=device, dtype=torch.float32)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp
    gate_msa = gate_msa.tanh()
    gate_mlp = gate_mlp.tanh()

    def _ref_rms(val: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        vf = val.float()
        return (w * (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps))).to(val.dtype)

    with torch.inference_mode():
        # 1) RMSNorm + AdaLN
        ref_attn_in = (_ref_rms(x, w1) * scale_msa + shift_msa).to(dtype)
        got_attn_in = fused_lingbot_rmsnorm_adaln(x, w1, scale_msa, shift_msa, eps=1e-6, out_dtype=dtype)
        psnr, cos = _psnr_and_cos(got_attn_in, ref_attn_in)
        assert psnr >= 50.0 and cos >= 0.9999

        # 2) 2-stage post-attn RMSNorm + residual gate + norm2 RMSNorm + FFN AdaLN
        ref_x1 = x + (gate_msa * _ref_rms(attn_out, w_post_a)).to(x.dtype)
        ref_ffn_in = (_ref_rms(ref_x1, w2) * scale_mlp + shift_mlp).to(dtype)
        got_x1, got_ffn_in = fused_lingbot_post_norm_residual_gate_rmsnorm_adaln(
            x, attn_out, w_post_a, gate_msa, w2, scale_mlp, shift_mlp, eps_post=1e-6, eps_norm2=1e-6, out_dtype=dtype
        )
        p1, c1 = _psnr_and_cos(got_x1, ref_x1)
        p2, c2 = _psnr_and_cos(got_ffn_in, ref_ffn_in)
        assert p1 >= 50.0 and c1 >= 0.9999
        assert p2 >= 50.0 and c2 >= 0.9999

        # 3) post-FFN RMSNorm + residual gate add
        ref_x2 = ref_x1 + (gate_mlp * _ref_rms(ffn_out, w_post_f)).to(ref_x1.dtype)
        got_x2 = fused_lingbot_post_norm_residual_gate_add(ref_x1, ffn_out, w_post_f, gate_mlp, eps_post=1e-6)
        p3, c3 = _psnr_and_cos(got_x2, ref_x2)
        assert p3 >= 50.0 and c3 >= 0.9999

        # 4) Output LayerNorm + AdaLN
        sc_out = torch.randn(B, M, C, device=device, dtype=torch.float32)
        sh_out = torch.randn(B, M, C, device=device, dtype=torch.float32)
        ref_ln = (F.layer_norm(x.float(), (C,), eps=1e-6) * (1.0 + sc_out) + sh_out).to(dtype)
        got_ln = fused_lingbot_layernorm_adaln(x, sc_out, sh_out, eps=1e-6, out_dtype=dtype)
        p4, c4 = _psnr_and_cos(got_ln, ref_ln)
        assert p4 >= 50.0 and c4 >= 0.9999

        # 5) Coalesced QK-RMSNorm + RoPE (BSHD and BHSD)
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype)
        qw = torch.randn(D, device=device, dtype=torch.float32)
        kw = torch.randn(D, device=device, dtype=torch.float32)
        ang = torch.randn(B, S, D // 2, device=device, dtype=torch.float32)
        cos_t, sin_t = torch.cos(ang), torch.sin(ang)

        def _ref_rot(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            tn = _ref_rms(t, w)
            pairs = tn.float().reshape(B, S, H, D // 2, 2)
            e, o = pairs[..., 0], pairs[..., 1]
            c_u, s_u = cos_t.unsqueeze(2), sin_t.unsqueeze(2)
            return torch.stack((e * c_u - o * s_u, e * s_u + o * c_u), dim=-1).flatten(-2).to(dtype)

        rq_ref, rk_ref = _ref_rot(q, qw), _ref_rot(k, kw)
        rq_got, rk_got = fused_lingbot_qk_rmsnorm_rope(q, k, qw, kw, (cos_t, sin_t), eps=1e-6, output_bhsd=False)
        pq, cq = _psnr_and_cos(rq_got, rq_ref)
        pk, ck = _psnr_and_cos(rk_got, rk_ref)
        assert pq >= 50.0 and cq >= 0.9999
        assert pk >= 50.0 and ck >= 0.9999

        rq_bh, rk_bh = fused_lingbot_qk_rmsnorm_rope(q, k, qw, kw, (cos_t, sin_t), eps=1e-6, output_bhsd=True)
        assert rq_bh.shape == (B, H, S, D)
        assert _psnr_and_cos(rq_bh.transpose(1, 2), rq_ref)[0] >= 50.0
        assert _psnr_and_cos(rk_bh.transpose(1, 2), rk_ref)[0] >= 50.0


def test_lingbot_video_block_hooks_autograd_and_cpu() -> None:
    from worldfoundry.synthesis.visual_generation.lingbot_video.transformer_lingbot_video import (
        LingBotVideoBlock,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(12)
    blk = (
        LingBotVideoBlock(
            hidden_size=128,
            num_attention_heads=4,
            intermediate_size=256,
            norm_eps=1e-6,
            qkv_bias=False,
            out_bias=True,
            num_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            decoder_sparse_step=1,
            mlp_only_layers=(),
            n_shared_experts=None,
            score_func="sigmoid",
            norm_topk_prob=True,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=1.0,
            layer_idx=0,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    # Keep norm weights & modulation in fp32 just like LingBotVideoTransformer3DModel.to()
    for mod in (blk.norm1, blk.norm2, blk.norm_post_attn, blk.norm_post_ffn, blk.attn.norm_q, blk.attn.norm_k):
        mod.float()
    blk.scale_shift_table.data = blk.scale_shift_table.data.float()

    x = torch.randn(1, 16, 128, device=device, dtype=dtype)
    temb6 = torch.randn(1, 1, 6 * 128, device=device, dtype=torch.float32)
    ang = torch.randn(1, 16, 16, device=device, dtype=torch.float32)
    rot_c = torch.polar(torch.ones_like(ang), ang)

    with torch.inference_mode():
        blk._use_fused_kernels = False
        blk.attn._use_fused_kernels = False
        ref = blk(x, temb6, rot_c)

        blk._use_fused_kernels = True
        blk.attn._use_fused_kernels = True
        got = blk(x, temb6, (rot_c.real.contiguous(), rot_c.imag.contiguous()))

    psnr, cos = _psnr_and_cos(got, ref)
    assert psnr >= 50.0 and cos >= 0.9999

    # Forward hooks on norms must still fire even when _use_fused_kernels=True
    fired: list[str] = []
    handles = [
        blk.norm1.register_forward_hook(lambda *_a: fired.append("norm1")),
        blk.norm_post_attn.register_forward_hook(lambda *_a: fired.append("norm_post_attn")),
        blk.norm2.register_forward_hook(lambda *_a: fired.append("norm2")),
        blk.norm_post_ffn.register_forward_hook(lambda *_a: fired.append("norm_post_ffn")),
    ]
    try:
        with torch.inference_mode():
            _ = blk(x, temb6, (rot_c.real.contiguous(), rot_c.imag.contiguous()))
    finally:
        for h in handles:
            h.remove()
    assert fired == ["norm1", "norm_post_attn", "norm2", "norm_post_ffn"]

    # Autograd backward works when grad enabled
    blk.train()
    xg = x.detach().clone().requires_grad_(True)
    out_g = blk(xg, temb6, rot_c)
    out_g.float().sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()


@requires_cuda
def test_lingbot_video_custom_ops_cuda_graph_and_compile() -> None:
    from worldfoundry.synthesis.visual_generation.lingbot_video.acceleration import (
        fused_lingbot_layernorm_adaln,
        fused_lingbot_post_norm_residual_gate_add,
        fused_lingbot_post_norm_residual_gate_rmsnorm_adaln,
        fused_lingbot_rmsnorm_adaln,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, C = 1, 64, 256
    x = torch.randn(B, S, C, device=device, dtype=dtype)
    a = torch.randn(B, S, C, device=device, dtype=dtype)
    f = torch.randn(B, S, C, device=device, dtype=dtype)
    w = torch.randn(C, device=device, dtype=torch.float32)
    sc = torch.randn(B, 1, C, device=device, dtype=torch.float32)
    sh = torch.randn(B, 1, C, device=device, dtype=torch.float32)
    gt = torch.randn(B, 1, C, device=device, dtype=torch.float32).tanh()

    def _step(xi: torch.Tensor, ai: torch.Tensor, fi: torch.Tensor) -> torch.Tensor:
        h = fused_lingbot_rmsnorm_adaln(xi, w, sc, sh, out_dtype=dtype)
        x1, h2 = fused_lingbot_post_norm_residual_gate_rmsnorm_adaln(h, ai, w, gt, w, sc, sh, out_dtype=dtype)
        x2 = fused_lingbot_post_norm_residual_gate_add(x1, fi + h2 * 0.0, w, gt)
        return fused_lingbot_layernorm_adaln(x2, sc, sh, out_dtype=dtype)

    with torch.inference_mode():
        ref = _step(x, a, f)

        # CUDA Graph capture + replay
        g = torch.cuda.CUDAGraph()
        sx, sa, sf = x.clone(), a.clone(), f.clone()
        _ = _step(sx, sa, sf)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            sout = _step(sx, sa, sf)
        sx.copy_(x)
        sa.copy_(a)
        sf.copy_(f)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(sout, ref)

        # torch.compile fullgraph=True verifies fake kernels match real output shapes/dtypes
        compiled = torch.compile(_step, fullgraph=True)
        out_c = compiled(x, a, f)
        assert torch.equal(out_c, ref)


# ──────────────────────────────────────────────────────────────────────────
# 2. LingBot WanModelFast & WanModel: parity, 0-sync, external mutation, hooks, autograd
# ──────────────────────────────────────────────────────────────────────────


def _make_tiny_lingbot_fast(device: torch.device, dtype: torch.dtype):
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.fast import WanModelFast

    torch.manual_seed(21)
    model = (
        WanModelFast(
            model_type="t2v",
            control_type="cam",
            patch_size=(1, 2, 2),
            text_len=16,
            in_dim=8,
            dim=128,
            ffn_dim=256,
            freq_dim=32,
            text_dim=32,
            out_dim=8,
            num_heads=4,
            num_layers=2,
            local_attn_size=4,
            sink_size=1,
            qk_norm=True,
            cross_attn_norm=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        torch.nn.init.normal_(model.head.head.weight, std=0.05)
    for blk in model.blocks:
        blk_any: Any = blk
        blk_any.self_attn.attention_backend = "torch"
    return model


def _make_lingbot_caches(num_layers: int, kv_size: int, device: torch.device, dtype: torch.dtype):
    kv = [
        {
            "k": torch.zeros(1, kv_size, 4, 32, device=device, dtype=dtype),
            "v": torch.zeros(1, kv_size, 4, 32, device=device, dtype=dtype),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(num_layers)
    ]
    ca = [
        {
            "k": torch.zeros(1, 16, 4, 32, device=device, dtype=dtype),
            "v": torch.zeros(1, 16, 4, 32, device=device, dtype=dtype),
            "is_init": False,
        }
        for _ in range(num_layers)
    ]
    return kv, ca


def test_lingbot_wan_fast_parity_syncs_mutation_hooks_and_autograd() -> None:
    import worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.fast as lingbot_fast_mod
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
        set_sf_fast_kernels_enabled,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = _make_tiny_lingbot_fast(device, dtype)

    lat_h, lat_w, chunk_f = 4, 4, 2
    frame_seqlen = (lat_h // 2) * (lat_w // 2)  # 4
    kv_size = 4 * frame_seqlen  # 16
    x_chunk = torch.randn(8, chunk_f, lat_h, lat_w, device=device, dtype=dtype)
    ctx = torch.randn(8, 32, device=device, dtype=dtype)
    t_1d = torch.tensor([500.0], device=device, dtype=torch.float32)

    # Multi-chunk rollout including eviction (chunks 0, 1, 2 -> chunk 2 evicts!)
    kv_fast, ca_fast = _make_lingbot_caches(2, kv_size, device, dtype)
    kv_ref, ca_ref = _make_lingbot_caches(2, kv_size, device, dtype)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )
    outs_fast, outs_ref = [], []
    with _patch_cpu_flash_attention(lingbot_fast_mod, "flash_attention"):
        with torch.inference_mode(), amp_ctx:
            for c in range(3):
                out_f, counts = _count_cuda_syncs(
                    lambda c=c: model(
                        [x_chunk],
                        t_1d,
                        [ctx],
                        seq_len=chunk_f * frame_seqlen,
                        kv_cache=kv_fast,
                        crossattn_cache=ca_fast,
                        current_start=c * chunk_f * frame_seqlen,
                        max_attention_size=kv_size,
                        frame_seqlen=frame_seqlen,
                    )[0]
                )
                outs_fast.append(out_f)
                if device.type == "cuda" and c >= 1:
                    assert counts["cuda_item"] == 0, f"Unexpected CUDA .item() syncs on chunk {c}: {counts}"
                    assert counts["cuda_tolist"] == 0, f"Unexpected CUDA .tolist() syncs on chunk {c}: {counts}"

            # Also verify inference-mode-allocated KV cache tensors use <= 1 batched sync total!
            if device.type == "cuda":
                kv_inf, ca_inf = _make_lingbot_caches(2, kv_size, device, dtype)
                _, inf_counts = _count_cuda_syncs(
                    lambda: model(
                        [x_chunk],
                        t_1d,
                        [ctx],
                        seq_len=chunk_f * frame_seqlen,
                        kv_cache=kv_inf,
                        crossattn_cache=ca_inf,
                        current_start=0,
                        max_attention_size=kv_size,
                        frame_seqlen=frame_seqlen,
                    )[0]
                )
                assert inf_counts["cuda_item"] == 0 and inf_counts["cuda_tolist"] <= 1, f"Too many syncs: {inf_counts}"

            set_sf_fast_kernels_enabled(model, False)
            try:
                for c in range(3):
                    outs_ref.append(
                        model(
                            [x_chunk],
                            t_1d,
                            [ctx],
                            seq_len=chunk_f * frame_seqlen,
                            kv_cache=kv_ref,
                            crossattn_cache=ca_ref,
                            current_start=c * chunk_f * frame_seqlen,
                            max_attention_size=kv_size,
                            frame_seqlen=frame_seqlen,
                        )[0]
                    )
            finally:
                set_sf_fast_kernels_enabled(model, None)

        for c, (of, or_) in enumerate(zip(outs_fast, outs_ref, strict=True)):
            psnr, cos = _psnr_and_cos(of, or_)
            assert psnr >= 50.0 and cos >= 0.9999, f"chunk {c} parity failed: psnr={psnr}, cos={cos}"

        # External KV cursor reset via .zero_() must invalidate host mirror safely!
        for layer in kv_fast:
            layer["k"].zero_()
            layer["v"].zero_()
            layer["global_end_index"].zero_()
            layer["local_end_index"].zero_()
        with torch.inference_mode(), amp_ctx:
            out_after_reset = model(
                [x_chunk],
                t_1d,
                [ctx],
                seq_len=chunk_f * frame_seqlen,
                kv_cache=kv_fast,
                crossattn_cache=ca_fast,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )[0]
        assert torch.equal(out_after_reset, outs_fast[0])
        assert int(kv_fast[0]["local_end_index"].item()) == chunk_f * frame_seqlen

        # Forward hooks on norms & text_embedding must fire
        fired: list[str] = []
        b0: Any = model.blocks[0]
        handles = [
            model.text_embedding.register_forward_hook(lambda *_a: fired.append("text_embedding")),
            b0.norm1.register_forward_hook(lambda *_a: fired.append("norm1")),
            b0.norm3.register_forward_hook(lambda *_a: fired.append("norm3")),
            b0.norm2.register_forward_hook(lambda *_a: fired.append("norm2")),
            b0.self_attn.norm_q.register_forward_hook(lambda *_a: fired.append("sa_norm_q")),
            b0.cross_attn.norm_q.register_forward_hook(lambda *_a: fired.append("ca_norm_q")),
            model.head.norm.register_forward_hook(lambda *_a: fired.append("head_norm")),
        ]
        try:
            with torch.inference_mode(), amp_ctx:
                _ = model(
                    [x_chunk],
                    t_1d,
                    [ctx],
                    seq_len=chunk_f * frame_seqlen,
                    kv_cache=kv_fast,
                    crossattn_cache=ca_fast,
                    current_start=0,
                    max_attention_size=kv_size,
                    frame_seqlen=frame_seqlen,
                )
        finally:
            for h in handles:
                h.remove()
        assert fired == ["text_embedding", "norm1", "sa_norm_q", "norm3", "ca_norm_q", "norm2", "head_norm"]

        # Autograd backward works when grad enabled (patch cross-attn FA3 -> SDPA since H200 FA3 wheel is fwd-only)
        model.train()
        kv_g, ca_g = _make_lingbot_caches(2, kv_size, device, dtype)
        xg = x_chunk.detach().clone().requires_grad_(True)
        with _patch_cpu_flash_attention(lingbot_fast_mod, "flash_attention", force=True), amp_ctx:
            og = model(
                [xg],
                t_1d,
                [ctx],
                seq_len=chunk_f * frame_seqlen,
                kv_cache=kv_g,
                crossattn_cache=ca_g,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )[0]
        og.float().sum().backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all()


# ──────────────────────────────────────────────────────────────────────────
# 3. LongLive fused kernels (with dual-K + V write) & CausalForcing / LongVideo
# ──────────────────────────────────────────────────────────────────────────


@requires_cuda
@pytest.mark.parametrize("exact", [True, False])
def test_longlive_coalesced_qk_v_and_wrappers_parity(exact: bool) -> None:
    from worldfoundry.pipelines.longlive.fast_kernels import (
        _wan_exact_real_rope_bshd,
        wan_layernorm_adaln,
        wan_qk_rmsnorm_rope_write_kv,
        wan_residual_gate_add,
        wan_rmsnorm,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(31)
    B, F_cur, HW = 1, 3, 16
    S = F_cur * HW
    H, D = 4, 64
    C = H * D

    qkv = torch.randn(B, S, 3 * C, device=device, dtype=dtype)
    q_proj, k_proj, v_proj = qkv.chunk(3, dim=-1)
    qw = torch.randn(C, device=device, dtype=dtype)
    kw = torch.randn(C, device=device, dtype=dtype)
    theta = torch.randn(S, D // 2, device=device, dtype=torch.float64)
    cos, sin = torch.cos(theta), torch.sin(theta)
    k_rot_dst = torch.zeros(B, S + 8, H, D, device=device, dtype=dtype)
    k_raw_dst = torch.zeros_like(k_rot_dst)
    v_dst = torch.zeros_like(k_rot_dst)

    with torch.inference_mode():
        q_got = wan_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            qw,
            kw,
            cos,
            sin,
            k_rot_dst,
            k_raw_dst,
            dst_token_offset=8,
            num_heads=H,
            head_dim=D,
            eps=1e-6,
            exact=exact,
            v_proj=v_proj,
            v_dst=v_dst,
        )

        def _ref_rope(t: torch.Tensor) -> torch.Tensor:
            xc = torch.view_as_complex(t.to(torch.float64).reshape(B, S, H, D // 2, 2))
            fc = torch.complex(cos, sin).view(1, S, 1, D // 2)
            return torch.view_as_real(xc * fc).flatten(3).to(torch.float32).to(dtype)

        qn = (q_proj.float() * torch.rsqrt(q_proj.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(q_proj) * qw
        kn = (k_proj.float() * torch.rsqrt(k_proj.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(k_proj) * kw
        qn = qn.view(B, S, H, D)
        kn = kn.view(B, S, H, D)

        assert torch.equal(v_dst[:, 8 : 8 + S], v_proj.view(B, S, H, D))
        if exact:
            assert torch.equal(q_got, _ref_rope(qn))
            assert torch.equal(k_raw_dst[:, 8 : 8 + S], kn)
            assert torch.equal(k_rot_dst[:, 8 : 8 + S], _ref_rope(kn))
        else:
            assert _psnr_and_cos(q_got, _ref_rope(qn))[0] >= 55.0
            assert _psnr_and_cos(k_raw_dst[:, 8 : 8 + S], kn)[0] >= 55.0
            assert _psnr_and_cos(k_rot_dst[:, 8 : 8 + S], _ref_rope(kn))[0] >= 55.0

        # Standalone coalesced RoPE helper (with out=)
        rope_out = torch.empty_like(qn)
        _wan_exact_real_rope_bshd(qn, cos, sin, out=rope_out)
        assert torch.equal(rope_out, _ref_rope(qn))

        # Wrappers
        x = torch.randn(B, S, C, device=device, dtype=dtype)
        y = torch.randn_like(x)
        mod = torch.randn(B, F_cur, 6, C, device=device, dtype=dtype).chunk(6, dim=2)
        ln_got = wan_layernorm_adaln(x, mod[1], mod[0], num_frames=F_cur, frame_seq=HW, exact=exact)
        ln_ref = (F.layer_norm(x, (C,), eps=1e-6).type_as(x).unflatten(1, (F_cur, HW)) * (1 + mod[1]) + mod[0]).flatten(
            1, 2
        )
        if exact:
            assert torch.equal(ln_got, ln_ref)
        else:
            assert _psnr_and_cos(ln_got, ln_ref)[0] >= 55.0

        rg_got = wan_residual_gate_add(x, y, mod[2], num_frames=F_cur, frame_seq=HW)
        rg_ref = x + (y.unflatten(1, (F_cur, HW)) * mod[2]).flatten(1, 2)
        assert torch.equal(rg_got, rg_ref)

        rms_got = wan_rmsnorm(x, qw, exact=exact)
        rms_ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x) * qw
        if exact:
            assert torch.equal(rms_got, rms_ref)
        else:
            assert _psnr_and_cos(rms_got, rms_ref)[0] >= 55.0


def test_causal_forcing_and_long_video_parity_syncs_and_autograd() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.causal_forcing import (
        CausalWanModel as CausalForcingModel,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
        set_sf_fast_kernels_enabled,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.long_video import (
        CausalWanModel as LongVideoModel,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for cls in (CausalForcingModel, LongVideoModel):
        torch.manual_seed(41)
        m = (
            cls(
                model_type="t2v",
                patch_size=(1, 2, 2),
                text_len=8,
                in_dim=8,
                dim=128,
                ffn_dim=256,
                freq_dim=32,
                text_dim=32,
                out_dim=8,
                num_heads=4,
                num_layers=2,
                local_attn_size=6,
                sink_size=1,
                qk_norm=True,
                cross_attn_norm=True,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        with torch.no_grad():
            torch.nn.init.normal_(m.head.head.weight, std=0.05)

        chunk_f, lat_h, lat_w = 3, 4, 4
        frame_seqlen = (lat_h // 2) * (lat_w // 2)  # 4
        block_toks = chunk_f * frame_seqlen  # 12
        if cls is LongVideoModel:
            for blk in m.blocks:
                blk_any: Any = blk
                blk_any.self_attn.frame_length = frame_seqlen
                blk_any.self_attn.block_length = block_toks
                blk_any.self_attn.max_attention_size = 6 * frame_seqlen

        kv_cap = 6 * frame_seqlen
        x_in = torch.randn(1, 8, chunk_f, lat_h, lat_w, device=device, dtype=dtype)
        t_in = torch.full((1, chunk_f), 500.0, device=device, dtype=dtype)
        ctx = torch.randn(1, 4, 32, device=device, dtype=dtype)

        def _mk():
            kv = [
                {
                    "k": torch.zeros(1, kv_cap, 4, 32, device=device, dtype=dtype),
                    "v": torch.zeros(1, kv_cap, 4, 32, device=device, dtype=dtype),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
                for _ in range(2)
            ]
            ca = [{"is_init": False} for _ in range(2)]
            return kv, ca

        kv_f, ca_f = _mk()
        kv_r, ca_r = _mk()

        with torch.inference_mode():
            outs_f = []
            for c in range(3):
                out_c, counts = _count_cuda_syncs(
                    lambda c=c: m(
                        x_in,
                        t_in,
                        ctx,
                        seq_len=block_toks,
                        kv_cache=kv_f,
                        crossattn_cache=ca_f,
                        current_start=c * block_toks,
                    )
                )
                outs_f.append(out_c)
                if device.type == "cuda" and c >= 1:
                    assert counts["cuda_item"] == 0, f"{cls.__name__} chunk {c} had syncs: {counts}"

            set_sf_fast_kernels_enabled(m, False)
            try:
                outs_r = [
                    m(
                        x_in,
                        t_in,
                        ctx,
                        seq_len=block_toks,
                        kv_cache=kv_r,
                        crossattn_cache=ca_r,
                        current_start=c * block_toks,
                    )
                    for c in range(3)
                ]
            finally:
                set_sf_fast_kernels_enabled(m, None)

        for c, (of, or_) in enumerate(zip(outs_f, outs_r, strict=True)):
            psnr, cos = _psnr_and_cos(of, or_)
            assert psnr >= 50.0 and cos >= 0.9999, f"{cls.__name__} chunk {c}: psnr={psnr}, cos={cos}"

        # Autograd backward check
        m.train()
        kv_g, ca_g = _mk()
        xg = x_in.detach().clone().requires_grad_(True)
        og = m(xg, t_in, ctx, seq_len=block_toks, kv_cache=kv_g, crossattn_cache=ca_g, current_start=0)
        og.float().sum().backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all()


# ──────────────────────────────────────────────────────────────────────────
# 4. Matrix-Game-2.0 fast kernels + CausalWanModel (causal_action_21.py)
# ──────────────────────────────────────────────────────────────────────────


def test_matrix_game2_action21_parity_hooks_and_autograd() -> None:
    import worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_21 as action_21_mod
    import worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 as causal_action_mod
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
        CausalWanModel,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
        set_sf_fast_kernels_enabled,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(51)
    model = (
        CausalWanModel(
            model_type="i2v",
            patch_size=(1, 2, 2),
            in_dim=20,
            dim=128,
            ffn_dim=256,
            freq_dim=32,
            text_dim=32,
            out_dim=16,
            num_heads=4,
            num_layers=2,
            local_attn_size=6,
            sink_size=0,
            qk_norm=True,
            cross_attn_norm=True,
            action_config={},
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        torch.nn.init.normal_(model.head.head.weight, std=0.05)

    B, T, H, W = 1, 3, 4, 4
    S_f = (H // 2) * (W // 2)  # 4
    x_in = torch.randn(B, 16, T, H, W, device=device, dtype=dtype)
    cc_in = torch.randn(B, 4, T, H, W, device=device, dtype=dtype)
    vis = torch.randn(B, 8, 1280, device=device, dtype=dtype)
    t_in = torch.full((B, T), 500.0, device=device, dtype=dtype)

    def _mk():
        kv = [
            {
                "k": torch.zeros(B, 6 * S_f, 4, 32, device=device, dtype=dtype),
                "v": torch.zeros(B, 6 * S_f, 4, 32, device=device, dtype=dtype),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(2)
        ]
        ca = [{"is_init": False} for _ in range(2)]
        return kv, ca

    kv_f, ca_f = _mk()
    kv_r, ca_r = _mk()
    none2 = [None, None]
    with (
        _patch_cpu_flash_attention(action_21_mod, "flash_attention"),
        _patch_cpu_flash_attention(causal_action_mod, "attention"),
    ):
        with torch.inference_mode():
            out_f = model(
                x_in,
                t_in,
                vis,
                cc_in,
                kv_cache=kv_f,
                kv_cache_mouse=none2,
                kv_cache_keyboard=none2,
                crossattn_cache=ca_f,
                current_start=0,
            )
            set_sf_fast_kernels_enabled(model, False)
            try:
                out_r = model(
                    x_in,
                    t_in,
                    vis,
                    cc_in,
                    kv_cache=kv_r,
                    kv_cache_mouse=none2,
                    kv_cache_keyboard=none2,
                    crossattn_cache=ca_r,
                    current_start=0,
                )
            finally:
                set_sf_fast_kernels_enabled(model, None)

        psnr, cos = _psnr_and_cos(out_f, out_r)
        assert psnr >= 50.0 and cos >= 0.9999

        # Forward hooks fire
        fired: list[str] = []
        b0: Any = model.blocks[0]
        handles = [
            b0.norm1.register_forward_hook(lambda *_a: fired.append("norm1")),
            b0.norm3.register_forward_hook(lambda *_a: fired.append("norm3")),
            b0.norm2.register_forward_hook(lambda *_a: fired.append("norm2")),
            model.head.norm.register_forward_hook(lambda *_a: fired.append("head_norm")),
        ]
        try:
            with torch.inference_mode():
                _ = model(
                    x_in,
                    t_in,
                    vis,
                    cc_in,
                    kv_cache=kv_f,
                    kv_cache_mouse=none2,
                    kv_cache_keyboard=none2,
                    crossattn_cache=ca_f,
                    current_start=0,
                )
        finally:
            for h in handles:
                h.remove()
        assert fired == ["norm1", "norm3", "norm2", "head_norm"]

        # Autograd backward check
        model.train()
        kv_g, ca_g = _mk()
        xg = x_in.detach().clone().requires_grad_(True)
        og = model(
            xg,
            t_in,
            vis,
            cc_in,
            kv_cache=kv_g,
            kv_cache_mouse=none2,
            kv_cache_keyboard=none2,
            crossattn_cache=ca_g,
            current_start=0,
        )
        og.float().sum().backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all()


@requires_cuda
def test_matrix_game2_fast_steady_deferred_residual_and_cuda_graph() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
        CausalWanModel,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.fast_kernels import (
        mg2_fast_steady_forward,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(61)
    B, T, H, W = 1, 3, 8, 8
    S = (H // 2) * (W // 2)

    model = (
        CausalWanModel(
            model_type="i2v",
            patch_size=(1, 2, 2),
            in_dim=20,
            dim=128,
            ffn_dim=256,
            freq_dim=64,
            text_dim=32,
            out_dim=16,
            num_heads=2,
            num_layers=2,
            local_attn_size=6,
            sink_size=0,
            qk_norm=True,
            cross_attn_norm=True,
            action_config={
                "blocks": [0],
                "mouse_dim_in": 2,
                "keyboard_dim_in": 4,
                "hidden_size": 32,
                "img_hidden_size": 128,
                "keyboard_hidden_dim": 128,
                "mouse_hidden_dim": 128,
                "heads_num": 2,
                "enable_mouse": True,
                "enable_keyboard": True,
            },
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        am0: Any = model.blocks[0].action_model
        torch.nn.init.normal_(model.head.head.weight, std=0.05)
        torch.nn.init.normal_(am0.proj_mouse.weight, std=0.05)
        torch.nn.init.normal_(am0.proj_keyboard.weight, std=0.05)

    xv = torch.randn(B, 16, T, H, W, device=device, dtype=dtype)
    cc = torch.randn(B, 4, T, H, W, device=device, dtype=dtype)
    t_in = torch.full((B, T), 500.0, device=device, dtype=torch.float32)
    kb = torch.randn(B, 13, 4, device=device, dtype=dtype)
    mouse = torch.randn(B, 13, 2, device=device, dtype=dtype)
    s_freqs = torch.pow(10000.0, -torch.arange(32, device=device, dtype=torch.float64).div(32))

    def _mk():
        kv = [
            {
                "k": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
                "v": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
            }
            for _ in range(2)
        ]
        km = [
            {
                "k": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
                "v": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
            }
            for _ in range(2)
        ]
        kk = [
            {
                "k": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
                "v": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
            }
            for _ in range(2)
        ]
        ca = [
            {
                "k": torch.randn(B, 8, 2, 64, device=device, dtype=dtype),
                "v": torch.randn(B, 8, 2, 64, device=device, dtype=dtype),
            }
            for _ in range(2)
        ]
        return kv, km, kk, ca

    kv1, km1, kk1, ca1 = _mk()
    kv2 = [{"k": d["k"].clone(), "v": d["v"].clone()} for d in kv1]
    km2 = [{"k": d["k"].clone(), "v": d["v"].clone()} for d in km1]
    kk2 = [{"k": d["k"].clone(), "v": d["v"].clone()} for d in kk1]

    with torch.inference_mode():
        out_ex = mg2_fast_steady_forward(
            model,
            xv,
            t_in,
            cc,
            kb,
            mouse,
            sinusoid_freqs=s_freqs,
            kv_cache=kv1,
            kv_cache_mouse=km1,
            kv_cache_keyboard=kk1,
            crossattn_cache=ca1,
            local_start_index=0,
            local_end_index=T * S,
            action_local_start=0,
            action_local_end=T,
            start_frame=0,
            attention_backend="torch",
            exact_norms=True,
        )
        out_fa = mg2_fast_steady_forward(
            model,
            xv,
            t_in,
            cc,
            kb,
            mouse,
            sinusoid_freqs=s_freqs,
            kv_cache=kv2,
            kv_cache_mouse=km2,
            kv_cache_keyboard=kk2,
            crossattn_cache=ca1,
            local_start_index=0,
            local_end_index=T * S,
            action_local_start=0,
            action_local_end=T,
            start_frame=0,
            attention_backend="torch",
            exact_norms=False,
        )
        psnr, cos = _psnr_and_cos(out_fa, out_ex)
        assert psnr >= 50.0 and cos >= 0.9999

        # CUDA Graph capture + replay of full 2-block ActionModule + non-ActionModule steady forward
        sxv, scc, skb, sm = xv.clone(), cc.clone(), kb.clone(), mouse.clone()
        g = torch.cuda.CUDAGraph()
        _ = mg2_fast_steady_forward(
            model,
            sxv,
            t_in,
            scc,
            skb,
            sm,
            sinusoid_freqs=s_freqs,
            kv_cache=kv2,
            kv_cache_mouse=km2,
            kv_cache_keyboard=kk2,
            crossattn_cache=ca1,
            local_start_index=0,
            local_end_index=T * S,
            action_local_start=0,
            action_local_end=T,
            start_frame=0,
            attention_backend="torch",
            exact_norms=False,
        )
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            sout = mg2_fast_steady_forward(
                model,
                sxv,
                t_in,
                scc,
                skb,
                sm,
                sinusoid_freqs=s_freqs,
                kv_cache=kv2,
                kv_cache_mouse=km2,
                kv_cache_keyboard=kk2,
                crossattn_cache=ca1,
                local_start_index=0,
                local_end_index=T * S,
                action_local_start=0,
                action_local_end=T,
                start_frame=0,
                attention_backend="torch",
                exact_norms=False,
            )
        sxv.copy_(xv)
        scc.copy_(cc)
        skb.copy_(kb)
        sm.copy_(mouse)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(sout, out_fa)
