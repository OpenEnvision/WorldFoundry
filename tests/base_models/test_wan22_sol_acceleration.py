# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive CPU + GPU test suite for Wave 3 Wan2.2 SOL-Engine acceleration.

Covers:
- Wan2.2 TI2V-5B: fused QKV, Triton FP64 hidden_qk_rmsnorm_rope_3d, Triton AdaLN,
  Hopper FA3, static cross-KV, regional torch.compile (0 graph breaks / 0 recompiles),
  calibrated EasyCache, optional rowwise FP8 (use_fast_accum=False), and VideoVAE.
- Wan2.2 T2V/I2V-A14B: dual-expert kernel fusion + FA3, in-tree PISA sparse attention
  routing (dense edge blocks 0/1/-1, dense warmup/tail, strict vs audited fallback),
  3D-Sol-Attn comparison path, explicit high->low expert-boundary EasyCache reset,
  piecewise CUDA Graph, and zero-fallback auditable receipts.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch._dynamo
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    Wan22DualExpertDenoiser,
    WanDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.runners.base import (
    RunnerComponents,
    Wan22DualExpertGuidanceRunner,
)
from worldfoundry.core.acceleration.quantization.fused_ffn import FusedFP8GELUFeedForward
from worldfoundry.core.acceleration.quantization.linear import Float8Linear
from worldfoundry.pipelines.wan.pipeline_wan_2p2 import (
    Wan2p2I2VA14BPipeline,
    Wan2p2Pipeline,
    Wan2p2T2VA14BPipeline,
)
from worldfoundry.pipelines.wan.wan22_sol_acceleration import (
    Wan22SolConfig,
    Wan22SolDualExpertController,
    Wan22SolEasyCache,
    Wan22SolExpertController,
    accelerate_wan22_denoiser,
    accelerate_wan22_pipeline,
    accelerate_wan22_vae,
    wan22_sol_profile,
    wan22_sol_runtime_report,
)

HAS_CUDA = torch.cuda.is_available()
HAS_SM90 = HAS_CUDA and torch.cuda.get_device_capability(0)[0] >= 9


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    denom = bf.norm().item()
    if denom == 0.0:
        return float((af - bf).norm().item())
    return float((af - bf).norm().item() / denom)


def _tiny_wan_model(
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    dim: int = 256,
    num_heads: int = 2,
    ffn_dim: int = 512,
    num_layers: int = 4,
    in_dim: int = 16,
    per_token_timestep: bool = False,
    seed: int = 0,
) -> WanModel:
    torch.manual_seed(seed)
    model = WanModel(
        dim=dim,
        in_dim=in_dim,
        ffn_dim=ffn_dim,
        out_dim=in_dim,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=num_heads,
        num_layers=num_layers,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
        per_token_timestep=per_token_timestep,
    )
    with torch.no_grad():
        return model.to(device=device, dtype=dtype).eval()


def _schedule(times: tuple[float, ...], device: torch.device) -> tuple[SchedulerStep, ...]:
    return tuple(
        SchedulerStep(
            index,
            torch.tensor(float(t), device=device),
            torch.tensor(float(nt), device=device),
        )
        for index, (t, nt) in enumerate(zip(times, (*times[1:], 0.0), strict=True))
    )


# =============================================================================
# 1. Config & Profile Contract Tests (CPU)
# =============================================================================


@pytest.mark.parametrize(
    "preset",
    [
        "wan22_5b_exact",
        "wan22_5b_sol",
        "wan22_5b_sol_fp8",
        "wan22_a14b_exact",
        "wan22_a14b_sol",
        "wan22_a14b_sol_attn",
        "wan22_a14b_sol_fp8",
        "5b_exact",
        "5b_sol",
        "a14b_sol",
    ],
)
def test_wan22_sol_profiles_enforce_quality_mandates(preset: str) -> None:
    cfg = wan22_sol_profile(preset, sm_major=9)
    assert cfg.rope_precision == "fp64"
    assert cfg.fp8_use_fast_accum is False
    payload = cfg.to_dict()
    assert payload["rope_precision"] == "fp64"
    assert payload["fp8_use_fast_accum"] is False


def test_wan22_sol_config_validation_rejects_unsafe_combinations() -> None:
    with pytest.raises(ValueError, match="use_fast_accum=False"):
        Wan22SolConfig(fp8=True, fp8_use_fast_accum=True)
    with pytest.raises(ValueError, match="sol_attn cannot be combined with torch.compile"):
        Wan22SolConfig(sparse_attention="sol_attn", compile=True)
    with pytest.raises(ValueError, match="pisa TMA kernels"):
        Wan22SolConfig(sparse_attention="pisa", compile=True)
    with pytest.raises(ValueError, match="pisa_density"):
        Wan22SolConfig(pisa_density=0.0)
    with pytest.raises(ValueError, match="pisa_block_size"):
        Wan22SolConfig(pisa_block_size=48)
    with pytest.raises(ValueError, match="unsupported Wan2.2 SOL preset"):
        wan22_sol_profile("nonexistent_preset")
    with pytest.raises(ValueError, match="easycache_low_noise_warmup_steps"):
        Wan22SolConfig(easycache_low_noise_warmup_steps=-1)
    with pytest.raises(ValueError, match="easycache_low_noise_dense_last"):
        Wan22SolConfig(easycache_low_noise_dense_last=-1)
    with pytest.raises(ValueError, match="easycache_low_noise_max_skip_steps"):
        Wan22SolConfig(easycache_low_noise_max_skip_steps=0)
    with pytest.raises(TypeError, match="sol_overrides must be a mapping"):
        Wan2p2Pipeline.from_pretrained("unused", sol_profile="wan22_5b_sol", sol_overrides=["bad"])


# =============================================================================
# 2. Calibrated EasyCache Boundary & Edge-Case Tests (CPU)
# =============================================================================


def test_wan22_easycache_guards_boundaries_and_edge_cases() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(1, 32, 64)
    dense_calls = 0

    def run_block(block_id: int, hidden: torch.Tensor) -> torch.Tensor:
        nonlocal dense_calls
        if block_id == 0:
            dense_calls += 1
        return hidden * 1.01 + 0.02

    cache = Wan22SolEasyCache(
        threshold=0.35,
        total_steps=12,
        warmup_steps=3,
        dense_last=2,
        max_skip_steps=2,
        subsample_stride=8,
    )
    with torch.inference_mode():
        for step in range(12):
            out = cache.run_blocks(step, x0 + 0.001 * step, run_block, block_count=4, total_steps=12)
            assert out.shape == x0.shape

    receipt = cache.receipt()
    assert receipt["fallbacks"] == []
    assert receipt["events"] == 12
    assert receipt["hits"] > 0
    assert dense_calls == receipt["misses"]
    for ev in cache.events:
        if ev.step < 3:
            assert not ev.hit and ev.reason == "warmup"
        if ev.step >= 10:
            assert not ev.hit and ev.reason == "dense-last"
    consec = 0
    for ev in cache.events:
        consec = consec + 1 if ev.hit else 0
        assert consec <= 2

    # Threshold=0.0 keeps every step dense and bitwise identical.
    zero_cache = Wan22SolEasyCache(threshold=0.0, total_steps=6, warmup_steps=1, dense_last=1)
    with torch.inference_mode():
        for step in range(6):
            val = x0 + 0.001 * step
            out = zero_cache.run_blocks(step, val, run_block, block_count=4, total_steps=6)
            expected = val
            for bid in range(4):
                expected = run_block(bid, expected)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert zero_cache.hits == 0

    # Layout change invalidates cleanly.
    with torch.inference_mode():
        _ = cache.run_blocks(5, torch.randn(1, 48, 64), run_block, block_count=4, total_steps=12)
    assert cache.events[-1].reason == "layout-change" and not cache.events[-1].hit

    # Non-finite input invalidates cleanly.
    bad = x0.clone()
    bad[0, 0, 0] = float("nan")
    with torch.inference_mode():
        _ = cache.run_blocks(5, bad, run_block, block_count=4, total_steps=12)
    assert cache.events[-1].reason == "nonfinite-input" and not cache.events[-1].hit

    # Autograd disables caching cleanly.
    x_grad = torch.randn(1, 32, 64, requires_grad=True)
    _ = cache.run_blocks(5, x_grad, run_block, block_count=4, total_steps=12)
    assert cache.events[-1].reason == "autograd" and not cache.events[-1].hit

    # Explicit clear_residuals drops tensors while preserving audit events.
    event_count_before = len(cache.events)
    cache.clear_residuals()
    assert cache._residual is None
    assert cache._dense_input is None
    assert cache._dense_output is None
    assert cache._factor is None
    assert len(cache.events) == event_count_before
    assert cache.receipt()["residuals_cleared"] is True


# =============================================================================
# 3. Wan2.2-5B Single-Expert Controller Tests (CPU)
# =============================================================================


@pytest.mark.parametrize("all_ones_mask", [True, False])
def test_wan22_5b_fusion_parity_and_uninstall_cpu(all_ones_mask: bool) -> None:
    device = torch.device("cpu")
    ref_model = _tiny_wan_model(device, dtype=torch.float32, per_token_timestep=True, seed=7)
    opt_model = copy.deepcopy(ref_model)
    d_ref = WanDenoiser(ref_model, compute_dtype=torch.float32)
    d_opt = WanDenoiser(opt_model, compute_dtype=torch.float32)

    cfg = Wan22SolConfig(
        fuse_qkv=True,
        qkv_strategy="packed",
        fused_rope=False,
        rope_precision="fp64",
        fused_adaln=True,
        static_cross_kv=True,
        attention_backend="torch",
        sparse_attention="none",
        easycache=False,
        compile=False,
        cuda_graph="off",
    )
    ctrl, install_rec = accelerate_wan22_denoiser(d_opt, cfg)
    assert isinstance(ctrl, Wan22SolExpertController)
    assert install_rec["fallbacks"] == []
    assert install_rec["fusion"]["fuse_qkv_blocks"] == 4

    lat = torch.randn(1, 16, 2, 8, 8)
    ctx = torch.randn(1, 12, 64)
    mask = torch.ones(1, 1, 2, 8, 8)
    if not all_ones_mask:
        mask[:, :, 0] = 0.0

    with torch.inference_mode():
        for step in (0, 1):
            inp = DenoiserInput(
                latents=lat,
                timestep=torch.tensor(900.0 - step * 100.0),
                next_timestep=torch.tensor(800.0 - step * 100.0),
                conditioning={
                    "context": ctx,
                    "denoise_mask": mask,
                    "wan_denoise_mask_is_all_ones": all_ones_mask,
                },
                step_index=step,
                total_steps=2,
                branch="positive",
                request_id="req-5b-cpu",
            )
            y_ref = d_ref(inp).sample
            y_opt = d_opt(inp).sample
            assert _rel_l2(y_opt, y_ref) < 1e-4

    d_opt.end_request("req-5b-cpu")
    rep = cast(dict[str, Any], d_opt.runtime_optimization_report("req-5b-cpu"))
    assert rep["fallbacks"] == []
    assert rep["runtime"]["static_cross_kv"]["hits"] >= 4

    # Uninstall must restore a working forward even though QKV weights were merged.
    ctrl.uninstall()
    with torch.inference_mode():
        inp_after = DenoiserInput(
            latents=lat,
            timestep=torch.tensor(900.0),
            next_timestep=torch.tensor(800.0),
            conditioning={
                "context": ctx,
                "denoise_mask": mask,
                "wan_denoise_mask_is_all_ones": all_ones_mask,
            },
            step_index=0,
            total_steps=1,
            branch="positive",
            request_id="req-after-uninstall",
        )
        y_after = d_opt(inp_after).sample
        assert torch.isfinite(y_after).all()
    d_opt.end_request("req-after-uninstall")


# =============================================================================
# 4. Wan2.2-A14B Dual-Expert Boundary Cache Reset & PISA Fallback Tests (CPU)
# =============================================================================


def test_wan22_a14b_dual_expert_boundary_reset_and_branch_isolation_cpu() -> None:
    device = torch.device("cpu")
    high_d = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=11), compute_dtype=torch.float32)
    low_d = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=22), compute_dtype=torch.float32)
    dual = Wan22DualExpertDenoiser(high_d, low_d, boundary_ratio=0.875)

    cfg = Wan22SolConfig(
        fuse_qkv=True,
        fused_rope=False,
        rope_precision="fp64",
        fused_adaln=True,
        static_cross_kv=True,
        attention_backend="torch",
        sparse_attention="none",
        easycache=True,
        easycache_threshold=0.8,
        easycache_warmup_steps=2,
        easycache_dense_last=1,
        easycache_max_skip_steps=2,
        easycache_low_noise_threshold=0.8,
        easycache_low_noise_warmup_steps=2,
        easycache_low_noise_dense_last=1,
        easycache_low_noise_max_skip_steps=2,
        compile=False,
        cuda_graph="off",
    )
    ctrl, install_rec = accelerate_wan22_denoiser(dual, cfg)
    assert isinstance(ctrl, Wan22SolDualExpertController)
    assert install_rec["fallbacks"] == []

    # 6 high-noise steps (>= 875) followed by 6 low-noise steps (< 875).
    times = (990.0, 960.0, 930.0, 910.0, 895.0, 880.0, 860.0, 720.0, 560.0, 400.0, 240.0, 100.0)
    sched = _schedule(times, device)
    lat = torch.randn(1, 16, 2, 8, 8)
    pos_ctx = torch.randn(1, 8, 64)
    neg_ctx = torch.randn(1, 8, 64)

    runner = Wan22DualExpertGuidanceRunner(
        model_id="wan2.2-t2v-a14b-test",
        boundary_ratio=0.875,
        high_noise_guidance_scale=4.0,
        low_noise_guidance_scale=3.0,
        components=RunnerComponents(
            denoiser=dual,
            conditioner=cast(
                Any,
                SimpleNamespace(
                    encode=lambda *args, **kwargs: Conditioning(
                        positive={"context": pos_ctx},
                        negative={"context": neg_ctx},
                    )
                ),
            ),
            latent_initializer=cast(Any, SimpleNamespace(initialize=lambda *args, **kwargs: lat.clone())),
            scheduler=cast(
                Any,
                SimpleNamespace(
                    schedule=lambda sampling, **kwargs: sched,
                    scale_model_input=lambda x, step: x,
                    step=lambda pred, step, x, **kwargs: x + 0.001 * pred,
                ),
            ),
            decoder=cast(Any, SimpleNamespace(decode=lambda x, req: x)),
        ),
    )

    with torch.inference_mode():
        output = runner.run(DiffusionRequest(prompt="test", sampling=SamplingConfig(num_inference_steps=len(sched))))
    assert torch.isfinite(output.sample).all()

    report = cast(dict[str, Any], dual.runtime_optimization_report())
    assert report["fallbacks"] == []
    dual_sol = report["runtime"]["wan22_sol_dual_expert"]
    assert dual_sol["expert_boundary_resets"] == 1
    ev0 = dual_sol["boundary_reset_events"][0]
    assert ev0["from_expert"] == "high-noise"
    assert ev0["to_expert"] == "low-noise"
    assert ev0["global_step"] == 6
    assert ev0["cleared_branches"] == ["negative", "positive"]

    ec = dual_sol["easycache"]
    assert ec["enabled"] is True
    assert ec["high-noise"]["hits"] > 0
    assert ec["low-noise"]["hits"] > 0
    assert set(ec["high-noise"]["branches"]) == {"positive", "negative"}
    assert set(ec["low-noise"]["branches"]) == {"positive", "negative"}
    # Live caches must be released after end_request()
    assert len(ctrl.high_controller._request_caches) == 0
    assert len(ctrl.low_controller._request_caches) == 0
    ctrl.uninstall()


def test_wan22_pisa_cpu_audited_fallback_and_strict_rejection() -> None:
    device = torch.device("cpu")
    lat = torch.randn(1, 16, 2, 8, 8)
    ctx = torch.randn(1, 8, 64)

    # Non-strict mode on CPU falls back to dense SDPA and explicitly records the fallback!
    d_fb = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=1), compute_dtype=torch.float32)
    ctrl_fb, _ = accelerate_wan22_denoiser(
        d_fb,
        Wan22SolConfig(
            attention_backend="torch",
            sparse_attention="pisa",
            pisa_density=0.15,
            pisa_dense_steps=0,
            pisa_dense_last_steps=0,
            pisa_dense_blocks=(0, -1),
            pisa_tile_3d=None,
            pisa_strict=False,
            easycache=False,
            compile=False,
            cuda_graph="off",
        ),
    )
    with torch.inference_mode():
        inp = DenoiserInput(
            latents=lat,
            timestep=torch.tensor(500.0),
            next_timestep=torch.tensor(400.0),
            conditioning={"context": ctx},
            step_index=0,
            total_steps=2,
            branch="positive",
            request_id="req-pisa-fb",
        )
        out = d_fb(inp).sample
        assert torch.isfinite(out).all()
    d_fb.end_request("req-pisa-fb")
    rep = cast(dict[str, Any], d_fb.runtime_optimization_report("req-pisa-fb"))
    attn = rep["runtime"]["wan22_sol"]["attention"]
    assert attn["pisa_calls"] == 0
    assert attn["pisa_fallback_calls"] == 2  # blocks 1 and 2 (0 and 3 are dense edge blocks)
    assert any("pisa:ineligible" in fb for fb in rep["fallbacks"])
    assert any("fused_rope:" in fb for fb in rep["fallbacks"])
    ctrl_fb.uninstall()

    # Strict mode raises fail-closed on ineligible device/shape.
    d_strict = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=2), compute_dtype=torch.float32)
    ctrl_st, _ = accelerate_wan22_denoiser(
        d_strict,
        Wan22SolConfig(
            attention_backend="torch",
            sparse_attention="pisa",
            pisa_density=0.15,
            pisa_dense_steps=0,
            pisa_dense_last_steps=0,
            pisa_dense_blocks=(0, -1),
            pisa_tile_3d=None,
            pisa_strict=True,
            easycache=False,
            compile=False,
            cuda_graph="off",
        ),
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="PISA sparse attention is ineligible"):
        d_strict(inp)
    ctrl_st.uninstall()


def test_wan22_vae_and_pipeline_surface_cpu() -> None:
    class _MiniVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(16, 3, kernel_size=3, padding=1)
            self.decode_autocast_dtype: torch.dtype | None = None

    mini_dec = SimpleNamespace(
        vae=_MiniVAE(),
        channels_last_3d_requested=False,
        channels_last_3d_effective=0,
        conv3d_count=0,
        weight_dtype_requested=torch.float32,
    )
    vae_rec = accelerate_wan22_vae(mini_dec, channels_last_3d=True, decode_autocast=True, cast_bf16=False)
    assert vae_rec["fallbacks"] == []
    assert vae_rec["conv3d_layers_converted"] == 2
    assert mini_dec.channels_last_3d_requested is True
    assert mini_dec.channels_last_3d_effective == 2

    # Pipeline preset bindings
    assert Wan2p2Pipeline.DEFAULT_SOL_PRESET == "wan22_5b_sol"
    assert Wan2p2T2VA14BPipeline.DEFAULT_SOL_PRESET == "wan22_a14b_sol"
    assert Wan2p2I2VA14BPipeline.DEFAULT_SOL_PRESET == "wan22_a14b_sol"

    # accelerate_wan22_pipeline + wan22_sol_runtime_report duck-typed pipeline check
    d = WanDenoiser(_tiny_wan_model(torch.device("cpu"), dtype=torch.float32, seed=5), compute_dtype=torch.float32)
    fake_pipe = SimpleNamespace(
        model_id="wan2.2-ti2v-5b",
        native_pipeline=SimpleNamespace(components=SimpleNamespace(denoiser=d, decoder=mini_dec)),
    )
    rec = accelerate_wan22_pipeline(
        fake_pipe,
        Wan22SolConfig(
            attention_backend="torch",
            sparse_attention="none",
            easycache=True,
            compile=False,
            cuda_graph="off",
        ),
    )
    assert rec["fallbacks"] == []
    assert rec["vae"]["conv3d_layers_converted"] == 2
    live = wan22_sol_runtime_report(fake_pipe)
    assert live["engine"] == "worldfoundry-sol-wan22"


# =============================================================================
# 5. Hopper GPU Verification Suite (FA3, Triton FP64 RoPE, Compile, PISA, Sol-Attn, FP8)
# =============================================================================


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
def test_wan22_5b_gpu_fusion_fa3_compile_easycache_zero_fallback() -> None:
    device = torch.device("cuda:0")
    ref_model = _tiny_wan_model(
        device,
        dtype=torch.bfloat16,
        dim=512,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        per_token_timestep=True,
        seed=101,
    )
    opt_model = copy.deepcopy(ref_model)
    ec_model = copy.deepcopy(ref_model)
    d_ref = WanDenoiser(ref_model, compute_dtype=torch.bfloat16)
    d_opt = WanDenoiser(opt_model, compute_dtype=torch.bfloat16)
    d_ec = WanDenoiser(ec_model, compute_dtype=torch.bfloat16)

    torch._dynamo.reset()
    ctrl1, rec1 = accelerate_wan22_denoiser(d_opt, wan22_sol_profile("wan22_5b_exact", compile_fullgraph=True))
    assert rec1["fallbacks"] == []

    lat = torch.randn(1, 16, 4, 16, 16, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 4, 16, 16, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        inp0 = DenoiserInput(
            latents=lat,
            timestep=torch.tensor(900.0, device=device),
            next_timestep=torch.tensor(800.0, device=device),
            conditioning={"context": ctx, "denoise_mask": mask, "wan_denoise_mask_is_all_ones": True},
            step_index=0,
            total_steps=3,
            branch="positive",
            request_id="gpu-5b-l1",
        )
        y_ref = d_ref(inp0).sample
        y_opt0 = d_opt(inp0).sample
        _ = d_opt(inp0.with_updates(step_index=1, timestep=torch.tensor(800.0, device=device))).sample
        _ = d_opt(inp0.with_updates(step_index=2, timestep=torch.tensor(700.0, device=device))).sample

    d_opt.end_request("gpu-5b-l1")
    rep1 = cast(dict[str, Any], d_opt.runtime_optimization_report("gpu-5b-l1"))
    assert _rel_l2(y_opt0, y_ref) < 0.02
    assert rep1["fallbacks"] == []
    assert dict(torch._dynamo.utils.counters.get("graph_break", {})) == {}
    assert rep1["runtime"]["wan22_sol"]["compile"]["request_calls"] == 12
    assert rep1["runtime"]["wan22_sol"]["compile"]["request_failures"] == 0
    assert rep1["runtime"]["wan22_sol"]["attention"]["fa3_total_calls"] >= 24
    assert rep1["runtime"]["wan22_sol"]["attention"]["triton_rope_calls"] >= 12
    ctrl1.uninstall()

    # L2 EasyCache + Fusion + FA3 + Regional Compile
    torch._dynamo.reset()
    ctrl2, rec2 = accelerate_wan22_denoiser(
        d_ec,
        wan22_sol_profile(
            "wan22_5b_sol",
            easycache_threshold=0.5,
            easycache_warmup_steps=2,
            easycache_dense_last=1,
            compile_fullgraph=True,
        ),
    )
    assert rec2["fallbacks"] == []
    with torch.inference_mode():
        for step in range(8):
            inp = DenoiserInput(
                latents=lat + 0.001 * step,
                timestep=torch.tensor(900.0 - 50.0 * step, device=device),
                next_timestep=torch.tensor(max(850.0 - 50.0 * step, 0.0), device=device),
                conditioning={"context": ctx, "denoise_mask": mask, "wan_denoise_mask_is_all_ones": True},
                step_index=step,
                total_steps=8,
                branch="positive",
                request_id="gpu-5b-l2",
            )
            out = d_ec(inp).sample
            assert torch.isfinite(out).all()
    d_ec.end_request("gpu-5b-l2")
    rep2 = cast(dict[str, Any], d_ec.runtime_optimization_report("gpu-5b-l2"))
    assert rep2["fallbacks"] == []
    assert rep2["runtime"]["wan22_sol"]["easycache"]["hits"] > 0
    ctrl2.uninstall()


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
def test_wan22_5b_gpu_rowwise_fp8_use_fast_accum_false_and_edge_exclusion() -> None:
    device = torch.device("cuda:0")
    model = _tiny_wan_model(
        device,
        dtype=torch.bfloat16,
        dim=1024,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        per_token_timestep=True,
        seed=55,
    )
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)
    ctrl, rec = accelerate_wan22_denoiser(
        denoiser,
        wan22_sol_profile(
            "wan22_5b_sol_fp8",
            compile=False,
            easycache=False,
            fp8_min_features=1024,
            fp8_exclude_edge_blocks=True,
        ),
    )
    assert rec["fallbacks"] == []
    assert rec["quantization"]["fp8_enabled"] is True
    assert rec["quantization"]["use_fast_accum"] is False
    assert rec["quantization"]["fp8_layers"] > 0
    assert rec["quantization"]["fp8_fused_ffns"] == 2  # interior blocks 1 and 2 only!

    # Edge blocks 0 and 3 must stay BF16, interior blocks 1 and 2 use FP8 with use_fast_accum=False.
    assert not isinstance(model.blocks[0].ffn, FusedFP8GELUFeedForward)
    assert not isinstance(model.blocks[3].ffn, FusedFP8GELUFeedForward)
    assert isinstance(model.blocks[1].ffn, FusedFP8GELUFeedForward)
    assert isinstance(model.blocks[2].ffn, FusedFP8GELUFeedForward)
    for mod in model.modules():
        if isinstance(mod, Float8Linear):
            assert mod.use_fast_accum is False
            assert mod.scaling == "rowwise"
            assert mod.weight is None  # keep_dense_fallback=False

    lat = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 2, 16, 16, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        inp = DenoiserInput(
            latents=lat,
            timestep=torch.tensor(800.0, device=device),
            next_timestep=torch.tensor(700.0, device=device),
            conditioning={"context": ctx, "denoise_mask": mask, "wan_denoise_mask_is_all_ones": True},
            step_index=0,
            total_steps=2,
            branch="positive",
            request_id="gpu-fp8",
        )
        out0 = denoiser(inp).sample
        out1 = denoiser(inp.with_updates(step_index=1, timestep=torch.tensor(700.0, device=device))).sample
        assert torch.isfinite(out0).all() and torch.isfinite(out1).all()
    denoiser.end_request("gpu-fp8")
    rep = denoiser.runtime_optimization_report("gpu-fp8")
    assert rep["fallbacks"] == []
    ctrl.uninstall()


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
@pytest.mark.parametrize("sparse_preset", ["wan22_a14b_sol", "wan22_a14b_sol_attn"])
def test_wan22_a14b_gpu_dual_expert_pisa_and_sol_attn_with_boundary_reset(sparse_preset: str) -> None:
    device = torch.device("cuda:0")
    high_m = _tiny_wan_model(device, dtype=torch.bfloat16, dim=512, num_heads=4, ffn_dim=1024, num_layers=5, seed=31)
    low_m = _tiny_wan_model(device, dtype=torch.bfloat16, dim=512, num_heads=4, ffn_dim=1024, num_layers=5, seed=32)
    dual = Wan22DualExpertDenoiser(
        WanDenoiser(high_m, compute_dtype=torch.bfloat16),
        WanDenoiser(low_m, compute_dtype=torch.bfloat16),
        boundary_ratio=0.875,
    )

    extra: dict[str, Any] = {
        "easycache_threshold": 0.45,
        "easycache_warmup_steps": 2,
        "easycache_dense_last": 1,
        "easycache_low_noise_threshold": 0.45,
        "easycache_low_noise_warmup_steps": 2,
        "easycache_low_noise_dense_last": 1,
        "cuda_graph": "piecewise",
    }
    if sparse_preset == "wan22_a14b_sol":
        extra.update(
            pisa_density=0.20,
            pisa_block_size=64,
            pisa_min_sequence_length=256,
            pisa_dense_steps=2,
            pisa_dense_last_steps=1,
            pisa_dense_blocks=(0, 1, -1),
            pisa_strict=True,
        )
    else:
        extra.update(
            sol_tau=1.0,
            sol_dense_steps=2,
            sol_dense_last_steps=1,
            sol_dense_blocks=(0, 1, -1),
        )

    ctrl, rec = accelerate_wan22_denoiser(dual, wan22_sol_profile(sparse_preset, **extra))
    assert rec["fallbacks"] == []

    # 12 steps: 6 high-noise + 6 low-noise, seq len = 4*16*16 = 1024 tokens
    times = (990.0, 960.0, 930.0, 910.0, 895.0, 880.0, 860.0, 750.0, 600.0, 450.0, 300.0, 150.0)
    sched = _schedule(times, device)
    lat = torch.randn(1, 16, 4, 32, 32, device=device, dtype=torch.bfloat16)
    ctx_p = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    ctx_n = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)

    dual.prepare_request_schedule("gpu-a14b", sched)
    with torch.inference_mode():
        for i, st in enumerate(sched):
            for branch, c_val in (("positive", ctx_p), ("negative", ctx_n)):
                inp = DenoiserInput(
                    latents=lat + 0.001 * i,
                    timestep=st.timestep,
                    next_timestep=st.next_timestep,
                    conditioning={"context": c_val},
                    step_index=i,
                    total_steps=len(sched),
                    branch=branch,
                    request_id="gpu-a14b",
                )
                out = dual(inp)
                assert torch.isfinite(out.sample).all()
    dual.end_request("gpu-a14b")

    rep = cast(dict[str, Any], dual.runtime_optimization_report())
    assert rep["fallbacks"] == []
    dual_sol = rep["runtime"]["wan22_sol_dual_expert"]
    assert dual_sol["expert_boundary_resets"] == 1
    assert dual_sol["easycache"]["total_hits"] > 0
    assert dual_sol["attention"]["total_fa3_calls"] > 0
    if sparse_preset == "wan22_a14b_sol":
        assert dual_sol["attention"]["total_pisa_calls"] > 0
        assert dual_sol["attention"]["total_pisa_fallback_calls"] == 0
    else:
        assert dual_sol["attention"]["total_sol_attn_calls"] > 0
    ctrl.uninstall()


# =============================================================================
# 6. Regression Suite: State Restoration, Honest Graph Receipts, Edge Cases
# =============================================================================


def test_wan22_uninstall_state_restoration_and_request_id_reuse_cpu() -> None:
    device = torch.device("cpu")
    model = _tiny_wan_model(device, dtype=torch.float32, per_token_timestep=True, seed=19)
    denoiser = WanDenoiser(model, compute_dtype=torch.float32)
    sentinel_graph = object()
    denoiser._graph_runner = cast(Any, sentinel_graph)

    ctrl, _ = accelerate_wan22_denoiser(
        denoiser,
        Wan22SolConfig(
            attention_backend="torch",
            fused_rope=False,
            fused_adaln=True,
            compile=False,
            cuda_graph="off",
            easycache=False,
        ),
    )
    assert denoiser._graph_runner is None

    lat = torch.randn(1, 16, 2, 8, 8)
    ctx = torch.randn(1, 8, 64)
    mask = torch.ones(1, 1, 2, 8, 8)
    with torch.inference_mode():
        # Reuse the exact same request_id across two finalized requests: counters must reset!
        for _ in range(2):
            for step in range(2):
                denoiser(
                    DenoiserInput(
                        latents=lat,
                        timestep=torch.tensor(800.0 - 100.0 * step),
                        next_timestep=torch.tensor(700.0 - 100.0 * step),
                        conditioning={
                            "context": ctx,
                            "denoise_mask": mask,
                            "wan_denoise_mask_is_all_ones": True,
                        },
                        step_index=step,
                        total_steps=2,
                        branch="positive",
                        request_id="reused-req-id",
                    )
                )
            denoiser.end_request("reused-req-id")
    rep = cast(dict[str, Any], denoiser.runtime_optimization_report("reused-req-id"))
    assert rep["runtime"]["wan22_sol"]["attention"]["sdpa_self_calls"] == 8
    assert rep["effective"]["attention"] == "torch"
    assert rep["effective"]["rope_precision"] == "complex-fp64"

    ctrl.uninstall()
    assert denoiser._graph_runner is sentinel_graph
    assert not hasattr(model, "_worldfoundry_compile_runtime")
    assert not hasattr(model, "_worldfoundry_compile_config")
    assert not hasattr(model, "_worldfoundry_fused_rope")
    assert not hasattr(model, "_worldfoundry_fused_adaln")


def test_wan22_dual_expert_direct_caller_multi_run_and_single_expert_window_cpu() -> None:
    device = torch.device("cpu")
    hi = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=21), compute_dtype=torch.float32)
    lo = WanDenoiser(_tiny_wan_model(device, dtype=torch.float32, seed=22), compute_dtype=torch.float32)
    dual = Wan22DualExpertDenoiser(hi, lo, boundary_ratio=0.875)
    ctrl, _ = accelerate_wan22_denoiser(
        dual,
        Wan22SolConfig(
            attention_backend="torch",
            fused_rope=False,
            sparse_attention="none",
            easycache=True,
            easycache_threshold=0.9,
            easycache_warmup_steps=2,
            easycache_dense_last=1,
            easycache_low_noise_threshold=0.9,
            easycache_low_noise_warmup_steps=2,
            easycache_low_noise_dense_last=1,
            compile=False,
            cuda_graph="off",
        ),
    )
    assert isinstance(ctrl, Wan22SolDualExpertController)

    lat = torch.randn(1, 16, 2, 8, 8)
    ctx = torch.randn(1, 8, 64)
    times = (990.0, 950.0, 910.0, 880.0, 800.0, 600.0, 400.0, 200.0)
    with torch.inference_mode():
        # Two consecutive direct rollouts (request_id=None, no prepare_request_schedule).
        for _ in range(2):
            for step, t in enumerate(times):
                dual(
                    DenoiserInput(
                        latents=lat + 0.0001 * step,
                        timestep=torch.tensor(t),
                        next_timestep=torch.tensor(max(t - 50.0, 0.0)),
                        conditioning={"context": ctx},
                        step_index=step,
                        total_steps=len(times),
                        branch="positive",
                        request_id=None,
                    )
                )
    rep_direct = cast(dict[str, Any], dual.runtime_optimization_report())
    dual_sol = rep_direct["runtime"]["wan22_sol_dual_expert"]
    assert dual_sol["expert_boundary_resets"] == 1
    assert dual_sol["boundary_reset_events"][0]["from_expert"] == "high-noise"
    assert dual_sol["boundary_reset_events"][0]["to_expert"] == "low-noise"
    lo_events = ctrl.low_controller._request_caches["__direct__"]["positive"].events
    assert lo_events[0].step == 0 and lo_events[0].reason == "warmup"
    assert lo_events[1].step == 1 and lo_events[1].reason == "warmup"

    # High-only explicit request must NOT inherit stale low-noise attention counts.
    with torch.inference_mode():
        dual(
            DenoiserInput(
                latents=lat,
                timestep=torch.tensor(950.0),
                next_timestep=torch.tensor(900.0),
                conditioning={"context": ctx},
                step_index=0,
                total_steps=1,
                branch="positive",
                request_id="req-high-only",
            )
        )
    dual.end_request("req-high-only")
    rep_hi_only = wan22_sol_runtime_report(dual, request_id="req-high-only")
    lo_attn = rep_hi_only["runtime"]["wan22_sol_dual_expert"]["attention"]["low-noise"]
    assert lo_attn["sdpa_self_calls"] == 0 and lo_attn["fa3_total_calls"] == 0
    ctrl.uninstall()


def test_wan22_vae_conv2d_channels_last_and_idempotent_fast_decode_cpu() -> None:
    class _ToyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d = nn.Conv3d(4, 4, kernel_size=1)
            self.conv2d = nn.Conv2d(4, 4, kernel_size=1)

        def forward(
            self,
            x: torch.Tensor,
            feat_cache: list[Any] | None = None,
            feat_idx: list[int] | None = None,
        ) -> tuple[torch.Tensor, list[Any], list[int]]:
            y = self.conv3d(x)
            b, c, t, h, w = y.shape
            y2 = self.conv2d(y.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
            out = y2.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            return out, (feat_cache or []), (feat_idx or [0])

    class _ToyVideoVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.z_dim = 4
            self.conv2 = nn.Conv3d(4, 4, kernel_size=1)
            self.decoder = _ToyDecoder()
            self._feat_map: list[Any] = []
            self._conv_idx: list[int] = [0]

        def clear_cache(self) -> None:
            self._feat_map = []
            self._conv_idx = [0]

        def decode(self, z: torch.Tensor, scale: list[torch.Tensor]) -> torch.Tensor:
            self.clear_cache()
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
            x = self.conv2(z)
            out = None
            for i in range(x.shape[2]):
                self._conv_idx = [0]
                part, self._feat_map, self._conv_idx = self.decoder(
                    x[:, :, i : i + 1], feat_cache=self._feat_map, feat_idx=self._conv_idx
                )
                out = part if out is None else torch.cat([out, part], dim=2)
            assert out is not None
            return out

    class _ToyOuterVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _ToyVideoVAE()
            self.scale = [torch.zeros(4), torch.ones(4)]
            self.decode_autocast_dtype: torch.dtype | None = None

    outer = _ToyOuterVAE().eval()
    dec = SimpleNamespace(
        vae=outer,
        channels_last_3d_requested=False,
        channels_last_3d_effective=0,
        conv3d_count=0,
        channels_last_requested=False,
        channels_last_effective=0,
        conv2d_count=0,
    )
    z = torch.randn(1, 4, 3, 4, 4)
    with torch.no_grad():
        ref = outer.model.decode(z, outer.scale)
    rec1 = accelerate_wan22_vae(
        dec, channels_last_3d=True, channels_last_2d=True, cast_bf16=False, compile_decoder=False
    )
    rec2 = accelerate_wan22_vae(
        dec, channels_last_3d=True, channels_last_2d=True, cast_bf16=False, compile_decoder=False
    )
    assert rec1["fallbacks"] == [] and rec2["fallbacks"] == []
    assert rec1["conv3d_layers_converted"] == 2 and rec1["conv2d_layers_converted"] == 1
    assert rec1["fast_decode_installed"] is True
    assert dec.channels_last_requested is True and dec.channels_last_effective == 1
    with torch.no_grad():
        got = outer.model.decode(z, outer.scale)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
def test_wan22_pisa_piecewise_graph_honest_fallback_accounting_gpu() -> None:
    device = torch.device("cuda:0")
    model = _tiny_wan_model(device, dtype=torch.bfloat16, dim=512, num_heads=4, ffn_dim=1024, num_layers=4, seed=71)
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)
    # seq_len = 2*8*8 = 128 < pisa_min_sequence_length=1024 -> PISA falls back to FA3 inside captured graph!
    ctrl, _ = accelerate_wan22_denoiser(
        denoiser,
        Wan22SolConfig(
            sparse_attention="pisa",
            pisa_min_sequence_length=1024,
            pisa_dense_steps=0,
            pisa_dense_last_steps=0,
            pisa_dense_blocks=(0, -1),
            pisa_tile_3d=None,
            pisa_strict=False,
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            easycache=False,
            compile=False,
        ),
    )
    lat = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        for req_id in ("pisa-graph-cold", "pisa-graph-warm"):
            for step in range(3):
                denoiser(
                    DenoiserInput(
                        latents=lat,
                        timestep=torch.tensor(800.0 - 50.0 * step, device=device),
                        next_timestep=torch.tensor(700.0 - 50.0 * step, device=device),
                        conditioning={"context": ctx},
                        step_index=step,
                        total_steps=3,
                        branch="positive",
                        request_id=req_id,
                    )
                )
            denoiser.end_request(req_id)
            rep = cast(dict[str, Any], denoiser.runtime_optimization_report(req_id))
            attn = rep["runtime"]["wan22_sol"]["attention"]
            assert attn["pisa_calls"] == 0
            assert attn["pisa_fallback_calls"] == 6
            assert attn["fa3_self_calls"] == 12
            assert any("pisa:ineligible" in fb for fb in rep["fallbacks"])
    ctrl.uninstall()


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
def test_wan22_fp8_reapply_fail_closed_and_compiled_fullgraph_receipt_gpu() -> None:
    device = torch.device("cuda:0")
    model = _tiny_wan_model(
        device,
        dtype=torch.bfloat16,
        dim=1024,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        per_token_timestep=True,
        seed=88,
    )
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)

    torch._dynamo.reset()
    ctrl1, rec1 = accelerate_wan22_denoiser(
        denoiser,
        wan22_sol_profile("wan22_5b_sol_fp8", compile=True, compile_fullgraph=True, easycache=False),
    )
    assert rec1["quantization"]["fp8_layers"] == 8
    assert rec1["quantization"]["fp8_fused_ffns"] == 2

    lat = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 2, 16, 16, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        inp = DenoiserInput(
            latents=lat,
            timestep=torch.tensor(800.0, device=device),
            next_timestep=torch.tensor(700.0, device=device),
            conditioning={"context": ctx, "denoise_mask": mask, "wan_denoise_mask_is_all_ones": True},
            step_index=0,
            total_steps=2,
            branch="positive",
            request_id="gpu-fp8-compile",
        )
        assert torch.isfinite(denoiser(inp).sample).all()
        assert torch.isfinite(denoiser(inp.with_updates(step_index=1)).sample).all()
    denoiser.end_request("gpu-fp8-compile")
    rep = cast(dict[str, Any], denoiser.runtime_optimization_report("gpu-fp8-compile"))
    assert rep["fallbacks"] == []
    assert dict(torch._dynamo.utils.counters.get("graph_break", {})) == {}
    assert rep["effective"]["quantization"] == "compiled-fp8-executed"
    assert rep["effective"]["fuse_qkv_runtime"] == "compiled-graph-executed"
    assert rep["runtime"]["wan22_sol"]["quantization"]["compiled_fp8_traces"] > 0

    # Re-applying fp8=True must preserve accurate layer counts; fp8=False must fail closed.
    ctrl2, rec2 = accelerate_wan22_denoiser(
        denoiser,
        wan22_sol_profile("wan22_5b_sol_fp8", compile=False, easycache=False),
    )
    assert rec2["quantization"]["fp8_layers"] == 8
    assert rec2["quantization"]["fp8_fused_ffns"] == 2
    with pytest.raises(RuntimeError, match="Float8Linear\\(keep_dense_fallback=False\\)"):
        accelerate_wan22_denoiser(denoiser, wan22_sol_profile("wan22_5b_exact", compile=False))
    ctrl2.uninstall()
    del ctrl1


@pytest.mark.skipif(not HAS_SM90, reason="Requires SM90+ Hopper/Blackwell GPU")
def test_wan22_5b_gpu_prefix_rope_i2v_per_token_and_dynamic_resolution() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
        SelfAttention,
        apply_wan_qk_norm_rope,
    )

    device = torch.device("cuda:0")
    model = _tiny_wan_model(
        device,
        dtype=torch.bfloat16,
        dim=512,
        num_heads=4,
        ffn_dim=1024,
        num_layers=2,
        per_token_timestep=True,
        seed=91,
    )
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)

    # 1. Verify prefix/extra tokens (q.shape[1] != f*h*w) fall through to exact indexed FP64 RoPE.
    ctrl_rope, _ = accelerate_wan22_denoiser(denoiser, Wan22SolConfig(compile=False, easycache=False))
    sa = model.blocks[0].self_attn
    assert isinstance(sa, SelfAttention)
    proc = sa.get_processor()
    f, h, w = 2, 8, 8
    seq = f * h * w
    q_pref = torch.randn(1, seq + 16, 512, device=device, dtype=torch.bfloat16)
    k_pref = torch.randn(1, seq + 16, 512, device=device, dtype=torch.bfloat16)
    freqs_64 = model.rotary_frequencies((f, h, w), device=device, precision="fp64")
    freqs_pref = torch.cat([freqs_64, freqs_64[:16]], dim=0)
    table_64 = model.fused_rope_table(device=device, precision="fp64")
    with torch.inference_mode():
        q_ref, k_ref = apply_wan_qk_norm_rope(sa, q_pref, k_pref, freqs_pref, precision="fp64")
        q_got, k_got = proc._apply_qk_norm_rope(
            sa,
            q_pref,
            k_pref,
            freqs_pref,
            fused_table=table_64,
            fused_grid=(f, h, w),
            fused_indexed=False,
            precision="fp64",
        )
    assert (q_got - q_ref).abs().max().item() < 0.01
    assert (k_got - k_ref).abs().max().item() < 0.01
    ctrl_rope.uninstall()

    # 2. Verify compiled Wan2.2-5B handles T2V -> I2V (per-token t_mod.ndim==4) -> dynamic resolution.
    torch._dynamo.reset()
    ctrl, _ = accelerate_wan22_denoiser(
        denoiser,
        wan22_sol_profile("wan22_5b_sol", easycache_threshold=0.5, easycache_warmup_steps=2, easycache_dense_last=1),
    )
    lat1 = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    mask1 = torch.ones(1, 1, 2, 16, 16, device=device, dtype=torch.bfloat16)
    mask_i2v = mask1.clone()
    mask_i2v[:, :, 0] = 0.0
    lat2 = torch.randn(1, 16, 2, 24, 24, device=device, dtype=torch.bfloat16)
    mask2 = torch.ones(1, 1, 2, 24, 24, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        for req_id, cur_lat, cur_mask, all_ones in (
            ("r-t2v", lat1, mask1, True),
            ("r-i2v", lat1, mask_i2v, False),
            ("r-res2", lat2, mask2, True),
        ):
            for step in range(4):
                out = denoiser(
                    DenoiserInput(
                        latents=cur_lat + 0.001 * step,
                        timestep=torch.tensor(900.0 - 50.0 * step, device=device),
                        next_timestep=torch.tensor(850.0 - 50.0 * step, device=device),
                        conditioning={
                            "context": ctx,
                            "denoise_mask": cur_mask,
                            "wan_denoise_mask_is_all_ones": all_ones,
                        },
                        step_index=step,
                        total_steps=4,
                        branch="positive",
                        request_id=req_id,
                    )
                ).sample
                assert torch.isfinite(out).all()
            denoiser.end_request(req_id)
            rep = cast(dict[str, Any], denoiser.runtime_optimization_report(req_id))
            assert rep["fallbacks"] == []
            if not all_ones:
                assert rep["effective"]["wan_timestep_mode"] == "per-token"
    ctrl.uninstall()
