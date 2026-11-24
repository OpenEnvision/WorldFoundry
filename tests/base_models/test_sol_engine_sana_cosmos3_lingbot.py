# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive CPU + GPU verification suite for Wave 3 SOL-Engine models:

- SANA-Video: GraphWrappedDenoiserMixin (whole & piecewise), Triton block fusion,
  fast LiteLA ReLU-RoPE, NHWC GLUMBConvTemp, Hopper FA3 cross-attn + static KV,
  calibrated EasyCache (online + zero-sync + boundary guards), regional compile
  (zero recompiles), and VideoVAE acceleration.
- Cosmos3-Super: bitwise-exact und_seq static KV cache (rel_l2 == 0.0 & torch.equal),
  TeaCache / FirstBlock gen-stream caching, step-selective rowwise FP8 (use_fast_accum=False,
  keep_dense_fallback=False, zero weight duplication) + SM100 NVFP4 hook, lossless
  projection fusion, Triton GQA RMSNorm+RoPE & residual-add+RMSNorm, Hopper FA3 GQA
  (32Q/8KV & 64Q/8KV), and Cosmos3JointDenoiser piecewise CUDA Graph.
- LingBot-Video: lossless QKV & SwiGLU fusion, compact broadcast AdaLN, Triton fused
  QK-RMSNorm+3D-RoPE / AdaLN / residual-gate, Hopper FA3 + in-tree PISA sparse attention
  with dense warmup/tail & edge-block (0, 1, -1) guards, branch-isolated EasyCache,
  piecewise CUDA Graph, and runtime CLI sol_profile forwarding.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    DenoiserInput,
    ModalityState,
    MultiModalDenoiserInput,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos3 import Cosmos3JointDenoiser
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.acceleration import (
    Cosmos3AccelerationConfig,
    Cosmos3GenStreamCache,
    StepSelectiveFloat8Linear,
    StepSelectiveNVFP4Linear,
    accelerate_cosmos3,
    cosmos3_runtime_receipt,
    cosmos3_sol_profile,
    fuse_cosmos3_projections,
    quantize_cosmos3_step_selective,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.fused_ops import (
    cosmos3_fa3_attention,
    fused_residual_add_rmsnorm,
    fused_rmsnorm_rope_gqa,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.model import Cosmos3OmniTransformer
from worldfoundry.base_models.diffusion_model.models.networks.sana.acceleration import (
    SanaAccelerationConfig,
    SanaEasyCache,
    accelerate_sana_vae,
    accelerate_sana_video,
    sana_runtime_receipt,
    sana_video_sol_profile,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.basic_modules import GLUMBConvTemp
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import (
    LiteLAReLURope,
    WanRotaryPosEmbed,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale_video import SanaMSVideo
from worldfoundry.base_models.diffusion_model.models.representations.cosmos3 import build_cosmos3_sequence_layout
from worldfoundry.core.attention import rotate_half, scaled_dot_product_attention
from worldfoundry.synthesis.visual_generation.lingbot_video.acceleration import (
    LingBotAccelerationConfig,
    LingBotEasyCache,
    accelerate_lingbot_video,
    lingbot_runtime_receipt,
    lingbot_sol_profile,
)
from worldfoundry.synthesis.visual_generation.lingbot_video.runtime import LingBotVideoRuntime
from worldfoundry.synthesis.visual_generation.lingbot_video.transformer_lingbot_video import (
    LingBotVideoTransformer3DModel,
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


def _tiny_sana_video(device: torch.device, dtype: torch.dtype = torch.float32) -> SanaMSVideo:
    model = SanaMSVideo(
        input_size=16,
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        caption_channels=64,
        model_max_length=16,
        qk_norm=True,
        y_norm=True,
        attn_type="LiteLAReLURope",
        ffn_type="GLUMBConvTemp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=True,
        cross_attn_type="flash",
        rope_fhw_dim=(8, 12, 12),
    )
    with torch.no_grad():
        return model.to(device=device, dtype=dtype).eval()


def _tiny_cosmos3(
    device: torch.device,
    *,
    num_layers: int = 4,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    intermediate_size: int = 256,
    dtype: torch.dtype = torch.float32,
) -> Cosmos3OmniTransformer:
    model = Cosmos3OmniTransformer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        latent_channel=16,
        latent_patch_size=2,
        patch_latent_dim=64,
        vocab_size=256,
        rope_scaling={"mrope_section": [4, 6, 6]},
        action_gen=False,
        sound_gen=False,
    )
    with torch.no_grad():
        return model.to(device=device, dtype=dtype).eval()


def _cosmos3_forward_kwargs(
    model: Cosmos3OmniTransformer,
    input_ids: torch.Tensor,
    latent: torch.Tensor,
    timestep: float,
) -> dict[str, Any]:
    device = latent.device
    t = latent.shape[2]
    mask = torch.ones(1, 1, t, 1, 1, device=device, dtype=torch.float32)
    dummy_pos = torch.zeros(1, device=device, dtype=torch.long)
    layout = build_cosmos3_sequence_layout(
        input_ids,
        {"video": ModalityState(latent=latent, denoise_mask=mask, positions=dummy_pos, clean_latent=latent)},
        model.config,
        fps=24.0,
    ).values
    return {
        "input_ids": layout["input_ids"],
        "text_indexes": layout["text_indexes"],
        "position_ids": layout["position_ids"],
        "und_len": layout["und_len"],
        "sequence_length": layout["sequence_length"],
        "vision_tokens": [latent],
        "vision_token_shapes": layout["vision_token_shapes"],
        "vision_sequence_indexes": layout["vision_sequence_indexes"],
        "vision_mse_loss_indexes": layout["vision_mse_loss_indexes"],
        "vision_timesteps": torch.full(
            (int(cast(int, layout["num_noisy_vision_tokens"])),),
            float(timestep),
            device=device,
            dtype=torch.float32,
        ),
        "vision_noisy_frame_indexes": layout["vision_noisy_frame_indexes"],
    }


def _cosmos3_sample_inputs(
    device: torch.device,
    dtype: torch.dtype,
    *,
    und_len: int = 12,
    t: int = 2,
    h: int = 4,
    w: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(1, 200, (und_len,), generator=g).to(device)
    latent = torch.randn(1, 16, t, h, w, generator=g).to(device=device, dtype=dtype)
    return input_ids, latent


def _tiny_lingbot(
    device: torch.device,
    *,
    depth: int = 5,
    hidden_size: int = 128,
    num_heads: int = 4,
    intermediate_size: int = 256,
    dtype: torch.dtype = torch.float32,
) -> LingBotVideoTransformer3DModel:
    model = LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=16,
        out_channels=16,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        depth=depth,
        text_dim=64,
        freq_dim=32,
        axes_dims=(8, 12, 12),
        axes_lens=(64, 64, 64),
        rope_theta=256.0,
        num_experts=0,
    )
    with torch.no_grad():
        return model.to(device=device, dtype=dtype).eval()


# =============================================================================
# 1. SANA-Video Tests (CPU + GPU)
# =============================================================================


def test_sana_easycache_online_and_zero_sync_boundary_guards() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 16, 32)
    calls: list[int] = []

    def stack_fn(x: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return x * 1.02 + 0.05

    # Online strided mode: warmup=3, dense_last=2, max_skip_steps=2 over 12 steps.
    cache = SanaEasyCache(
        threshold=0.25,
        total_steps=12,
        warmup_steps=3,
        dense_last=2,
        max_skip_steps=2,
        subsample_stride=4,
    )
    with torch.inference_mode():
        for step in range(12):
            x_step = x0 + 0.001 * step
            out, _ = cache.run_stack(step, x_step, stack_fn, block_count=4, total_steps=12)
            assert out.shape == x0.shape

    receipt = cache.receipt()
    assert receipt["mode"] == "online-strided"
    assert receipt["fallbacks"] == []
    assert receipt["events"] == 12
    assert receipt["hits"] > 0
    # Warmup (0, 1, 2) and dense_last (10, 11) must NEVER hit.
    for ev in cache.events:
        if ev.step < 3:
            assert not ev.hit and ev.reason == "warmup"
        if ev.step >= 10:
            assert not ev.hit and ev.reason == "dense-last"
    # Verify max consecutive skip guard <= 2.
    consec = 0
    for ev in cache.events:
        consec = consec + 1 if ev.hit else 0
        assert consec <= 2

    # Zero-sync schedule mode: explicitly respects warmup, dense_last, and max_skip.
    zs = SanaEasyCache(
        total_steps=10,
        warmup_steps=2,
        dense_last=2,
        max_skip_steps=1,
        zero_sync_skip_steps=(0, 1, 3, 4, 5, 8, 9),
    )
    with torch.inference_mode():
        for step in range(10):
            zs.run_stack(step, x0, stack_fn, block_count=4, total_steps=10)
    zs_receipt = zs.receipt()
    assert zs_receipt["mode"] == "zero-sync-schedule"
    assert zs_receipt["fallbacks"] == []
    hit_steps = [ev.step for ev in zs.events if ev.hit]
    assert hit_steps == [3, 5]  # 0/1 blocked by warmup, 4 blocked by max_skip=1, 8/9 blocked by dense_last

    # Layout change invalidates residual cleanly without crashing.
    x_other = torch.randn(2, 24, 32)
    with torch.inference_mode():
        _, hit_layout = cache.run_stack(5, x_other, stack_fn, block_count=4, total_steps=12)
    assert not hit_layout
    assert cache.events[-1].reason == "layout-change"

    # Autograd disables caching cleanly.
    x_grad = torch.randn(2, 16, 32, requires_grad=True)
    _, hit_grad = cache.run_stack(5, x_grad, stack_fn, block_count=4, total_steps=12)
    assert not hit_grad
    assert cache.events[-1].reason == "autograd"


def test_sana_fused_modules_and_denoiser_cpu_parity() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    ref = _tiny_sana_video(device, dtype=torch.float32)
    opt = copy.deepcopy(ref)

    cfg = SanaAccelerationConfig(
        block_fusion=True,
        block_fusion_backend="auto",
        fast_litela_rope=True,
        rope_precision="fp64",
        fast_rms_norm=True,
        fuse_glumbconv=True,
        cross_attn_backend="torch",
        static_cross_kv=True,
        easycache=False,
        compile=False,
        cuda_graph="off",
    )
    receipt = accelerate_sana_video(opt, cfg)
    assert receipt["fallbacks"] == []
    assert receipt["fusion"]["fused_blocks"] == 4
    assert receipt["fusion"]["fused_glumbconv_layers"] == 4
    assert receipt["fusion"]["fast_litela_rope_layers"] == 4

    x = torch.randn(1, 16, 2, 8, 8)
    t = torch.tensor([500.0])
    y = torch.randn(1, 1, 10, 64)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    with torch.inference_mode():
        out_ref = ref(x, t, y, mask=mask)
        out_opt_1 = opt(x, t, y, mask=mask)
        out_opt_2 = opt(x, t, y, mask=mask)
    assert _rel_l2(out_opt_1, out_ref) < 1e-4
    assert torch.allclose(out_opt_1, out_opt_2, atol=1e-6)
    rt = sana_runtime_receipt(opt)
    assert rt["runtime"]["static_cross_kv_hits"] >= 4

    # Validate SanaDenoiser end-to-end EasyCache + GraphWrappedDenoiserMixin + report.
    denoiser = SanaDenoiser(opt, cuda_graph="piecewise")
    denoiser._feature_cache_config = None
    opt_ec_receipt = accelerate_sana_video(
        opt,
        SanaAccelerationConfig(
            cross_attn_backend="torch",
            easycache=True,
            easycache_threshold=0.5,
            easycache_warmup_steps=2,
            easycache_dense_last=1,
            easycache_max_skip_steps=2,
            cuda_graph="piecewise",
        ),
    )
    assert opt_ec_receipt["easycache"]["enabled"] is True
    denoiser = SanaDenoiser(opt, cuda_graph="piecewise")
    with torch.inference_mode():
        for step in range(6):
            inp = DenoiserInput(
                latents=x + 0.001 * step,
                timestep=torch.tensor([500.0 - 10.0 * step]),
                next_timestep=torch.tensor([max(500.0 - 10.0 * (step + 1), 0.0)]),
                conditioning={"context": y.squeeze(1), "context_mask": mask},
                step_index=step,
                total_steps=6,
                request_id="req-sana-1",
                branch="positive",
            )
            out = denoiser(inp)
            assert out.sample.shape == x.shape
    report = denoiser.runtime_optimization_report()
    assert report["fallbacks"] == []
    fc_report = cast(dict[str, Any], report["feature_cache"])
    assert fc_report["hits"] > 0
    denoiser.end_request("req-sana-1")


def test_sana_vae_acceleration_helper_cpu() -> None:
    class _FakeDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(4, 4, kernel_size=3, padding=1)

    class _FakeWrapper:
        def __init__(self) -> None:
            self.vae = _FakeDecoder()
            self.tiled = True
            self.parallel_degree = 1
            self.channels_last_3d_requested = False
            self.channels_last_3d_effective = 0

    wrapper = _FakeWrapper()
    receipt = accelerate_sana_vae(wrapper, channels_last_3d=True, decode_autocast_bf16=False, compile_decoder=False)
    assert receipt["conv3d_layers_converted"] == 1
    assert wrapper.vae.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert wrapper.channels_last_3d_effective == 1
    assert receipt["fallbacks"] == []


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for SANA GPU tests")
def test_sana_gpu_fa3_triton_compile_zero_recompiles_and_piecewise_graph() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    ref = _tiny_sana_video(device, dtype=torch.bfloat16)
    opt_graph = copy.deepcopy(ref)
    opt_compile = copy.deepcopy(ref)

    x = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    t = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    y = torch.randn(1, 1, 16, 64, device=device, dtype=torch.bfloat16)
    mask = torch.tensor([[1] * 10 + [0] * 6], device=device)

    # 1. Exact GPU parity with Triton fusion + Hopper FA3 + Piecewise CUDA Graph.
    profile = sana_video_sol_profile("exact", sm_major=9 if HAS_SM90 else 8, cuda_graph="piecewise")
    receipt = accelerate_sana_video(opt_graph, profile)
    assert receipt["fallbacks"] == []

    with torch.inference_mode():
        ref_out = ref(x, t, y, mask=mask)
        for _ in range(4):
            g_out = opt_graph(x, t, y, mask=mask)
    assert _rel_l2(g_out, ref_out) < 0.02
    rt_g = sana_runtime_receipt(opt_graph)
    assert rt_g["fallbacks"] == []
    if HAS_SM90:
        assert rt_g["runtime"]["cross_attn_fa3_calls"] > 0
        assert rt_g["runtime"]["cross_attn_sdpa_calls"] == 0
    pg = rt_g["runtime"]["piecewise_graph"]
    assert pg is not None
    assert pg["capture"] >= 1
    assert pg["replay"] >= 2

    # 2. Regional torch.compile on SANA blocks: verify ZERO recompiles across steps.
    torch._dynamo.reset()
    compile_cfg = SanaAccelerationConfig(
        cross_attn_backend="flash_attention_3" if HAS_SM90 else "torch",
        easycache=False,
        compile=True,
        compile_mode="default",
        cuda_graph="off",
    )
    c_receipt = accelerate_sana_video(opt_compile, compile_cfg)
    assert c_receipt["compile"]["compiled_blocks"] == 4
    with torch.inference_mode():
        # Warmup step compiles once.
        _ = opt_compile(x, t, y, mask=mask)
        frames_after_warmup = int(torch._dynamo.utils.counters["frames"]["total"])
        for step in range(4):
            c_out = opt_compile(x + 0.01 * step, t - 10.0 * step, y, mask=mask)
        frames_after_steps = int(torch._dynamo.utils.counters["frames"]["total"])
    assert frames_after_steps == frames_after_warmup, (
        f"Expected 0 recompiles across denoising steps, saw {frames_after_steps - frames_after_warmup}"
    )
    assert torch.isfinite(c_out).all()


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for GLUMBConvTemp / LiteLA GPU parity")
def test_sana_glumbconv_and_litela_rope_gpu_parity() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    glumb = (
        GLUMBConvTemp(128, 256, 128, norm=(None, None, None), act=("silu", "silu", None))
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    x_tok = torch.randn(1, 2 * 8 * 8, 128, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        glumb._fused_glumbconv = False
        ref_g = glumb(x_tok, HW=(2, 8, 8))
        glumb._fused_glumbconv = True
        opt_g = glumb(x_tok, HW=(2, 8, 8))
    assert _rel_l2(opt_g, ref_g) < 0.015

    rope = WanRotaryPosEmbed(attention_head_dim=32, patch_size=(1, 2, 2), max_seq_len=64, fhw_dim=(8, 12, 12)).to(
        device
    )
    attn = LiteLAReLURope(128, 128, dim=32, qk_norm=True).to(device=device, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        freqs_c = rope((2, 8, 8), device=device)
        cos_sin = rope.forward_real((2, 8, 8), device=device, dtype=torch.float64)
        attn._fast_litela_rope = False
        ref_a = attn(x_tok, rotary_emb=freqs_c)
        attn._fast_litela_rope = True
        opt_a = attn(x_tok, rotary_emb=cos_sin)
    assert _rel_l2(opt_a, ref_a) < 0.015


# =============================================================================
# 2. Cosmos3-Super Tests (CPU + GPU)
# =============================================================================


def test_cosmos3_und_static_kv_cache_bitwise_exact_cpu() -> None:
    torch.manual_seed(123)
    device = torch.device("cpu")
    ref_model = _tiny_cosmos3(device, num_layers=4, dtype=torch.float32)
    opt_model = copy.deepcopy(ref_model)

    cfg = Cosmos3AccelerationConfig(
        und_static_kv_cache=True,
        attention_backend="torch",
        fused_ops=False,
        fuse_projections=False,
        quantization="none",
        cache_algorithm="none",
        cuda_graph="off",
    )
    accelerate_cosmos3(opt_model, cfg)

    und_pos, latent0 = _cosmos3_sample_inputs(device, torch.float32, seed=1)
    und_neg, _ = _cosmos3_sample_inputs(device, torch.float32, seed=2)
    und_new, _ = _cosmos3_sample_inputs(device, torch.float32, seed=3)

    with torch.inference_mode():
        # Verify bitwise-exact parity across 5 denoising steps AND across positive/negative CFG branches!
        for step in range(5):
            lat_step = latent0 + 0.05 * step
            ts_val = 1000.0 - 150.0 * step
            for branch, und_toks in (("positive", und_pos), ("negative", und_neg)):
                fw = _cosmos3_forward_kwargs(ref_model, und_toks, lat_step, ts_val)
                ref_vis, _, _ = ref_model(**fw)
                opt_vis, _, _ = opt_model(
                    **fw,
                    step_index=step,
                    total_steps=5,
                    cache_branch=branch,
                )
                assert _rel_l2(opt_vis[0], ref_vis[0]) == 0.0
                assert torch.equal(opt_vis[0], ref_vis[0])

        # Prompt invalidation: changing input_ids on branch "positive" must MISS and stay bitwise exact!
        fw_new = _cosmos3_forward_kwargs(ref_model, und_new, latent0, 500.0)
        ref_new, _, _ = ref_model(**fw_new)
        opt_new, _, _ = opt_model(
            **fw_new,
            step_index=0,
            total_steps=5,
            cache_branch="positive",
        )
        assert torch.equal(opt_new[0], ref_new[0])

    rt = cosmos3_runtime_receipt(opt_model)
    kv = rt["runtime"]["und_static_kv_cache"]
    assert kv["hits"] == 8  # 4 steps * 2 branches
    assert kv["misses"] == 3  # step 0 pos, step 0 neg, prompt-change miss
    assert kv["und_layers_skipped"] == 8 * 4
    assert rt["fallbacks"] == []


def test_cosmos3_projection_fusion_and_teacache_firstblock_cpu() -> None:
    torch.manual_seed(17)
    device = torch.device("cpu")
    ref_model = _tiny_cosmos3(device, num_layers=4, dtype=torch.float32)
    fused_model = copy.deepcopy(ref_model)
    fusion_receipt = fuse_cosmos3_projections(fused_model, release_originals=True)
    assert fusion_receipt == {"gen_qkv_fused": 4, "swiglu_gate_up_fused": 8}
    layer0_fused: Any = fused_model.layers[0]
    assert layer0_fused.self_attn.add_q_proj is None
    assert layer0_fused.mlp_moe_gen.gate_proj is None

    und_toks, lat = _cosmos3_sample_inputs(device, torch.float32, seed=9)
    fw = _cosmos3_forward_kwargs(ref_model, und_toks, lat, 600.0)
    with torch.inference_mode():
        ref_vis, _, _ = ref_model(**fw)
        fused_vis, _, _ = fused_model(**fw)
    assert _rel_l2(fused_vis[0], ref_vis[0]) < 1e-5

    # Test TeaCache & FirstBlock controllers with boundary guards.
    for alg in ("teacache", "firstblock"):
        cache = Cosmos3GenStreamCache(
            algorithm=alg,
            threshold=0.25,
            total_steps=10,
            warmup_steps=3,
            dense_last=2,
            max_consecutive_skips=2,
            coefficients=(1.0, 0.0),
        )
        g0 = torch.randn(8, 128)
        with torch.inference_mode():
            for step in range(10):
                r0 = torch.ones_like(g0) + 0.005 * step
                out, _ = cache.decide_and_run_tail(
                    step=step,
                    total_steps=10,
                    gen_after_0=g0 + 0.001 * step,
                    first_block_residual=r0,
                    num_layers=4,
                    run_tail=lambda h: h * 1.05 + 0.1,
                )
                assert out.shape == g0.shape
        rc = cache.receipt()
        assert rc["algorithm"] == alg
        assert rc["hits"] > 0
        assert rc["fallbacks"] == []
        for ev in cache.events:
            if ev.step < 3:
                assert not ev.hit and ev.reason == "warmup"
            if ev.step >= 8:
                assert not ev.hit and ev.reason == "dense-last"


def test_cosmos3_step_selective_fp8_and_nvfp4_zero_duplication_cpu() -> None:
    torch.manual_seed(33)
    w = torch.randn(64, 64, dtype=torch.bfloat16)
    b = torch.randn(64, dtype=torch.bfloat16)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    ref = F.linear(x, w, b)

    # FP8 zero weight duplication: weight is None, use_fast_accum=False.
    fp8 = StepSelectiveFloat8Linear(w, b, keep_dense_fallback=False, scaling="rowwise", use_fast_accum=False)
    assert fp8.weight is None
    assert fp8.use_fast_accum is False
    with torch.inference_mode():
        y_fp8_cpu = fp8(x)
        # Empty input edge case.
        y_empty = fp8(torch.empty(0, 64, dtype=torch.bfloat16))
    assert y_empty.shape == (0, 64)
    assert _rel_l2(y_fp8_cpu, ref) < 0.05
    assert fp8.runtime_report()["keep_dense_fallback"] is False

    # NVFP4 zero weight duplication hook (SM100 ready).
    nv4 = StepSelectiveNVFP4Linear(w, b, keep_dense_fallback=False)
    assert nv4.weight is None
    with torch.inference_mode():
        y_nv4 = nv4(x)
        y_nv4_empty = nv4(torch.empty(0, 64, dtype=torch.bfloat16))
    assert y_nv4_empty.shape == (0, 64)
    assert _rel_l2(y_nv4, ref) < 0.15
    assert nv4.runtime_report()["keep_dense_fallback"] is False

    # Check sol_nvfp4 preset selects nvfp4 on SM100 and fp8 on SM90.
    assert cosmos3_sol_profile("sol_nvfp4", sm_major=10).quantization == "nvfp4"
    assert cosmos3_sol_profile("sol_nvfp4", sm_major=9).quantization == "fp8"
    assert cosmos3_sol_profile("sol_balanced", sm_major=9).fp8_use_fast_accum is False
    assert cosmos3_sol_profile("sol_balanced", sm_major=9).keep_dense_fallback is False


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for Cosmos3 GPU tests")
def test_cosmos3_gpu_kernels_fa3_gqa_and_bitwise_exact_kv_cache() -> None:
    torch.manual_seed(55)
    device = torch.device("cuda")

    # 1. Verify Triton fused_rmsnorm_rope_gqa for BOTH 32Q/8KV (Nano) and 64Q/8KV (Super) + FA3 GQA.
    with torch.inference_mode():
        for h_q, h_k in ((32, 8), (64, 8)):
            s, d = 128, 128
            q = torch.randn(s, h_q, d, device=device, dtype=torch.bfloat16)
            k = torch.randn(s, h_k, d, device=device, dtype=torch.bfloat16)
            v = torch.randn(s, h_k, d, device=device, dtype=torch.bfloat16)
            qw = torch.randn(d, device=device, dtype=torch.bfloat16)
            kw = torch.randn(d, device=device, dtype=torch.bfloat16)
            ang = torch.randn(s, d, device=device, dtype=torch.bfloat16)
            cos, sin = ang.cos(), ang.sin()

            q_tri, k_tri = fused_rmsnorm_rope_gqa(q, k, qw, kw, cos, sin, eps=1e-6)
            qn = F.rms_norm(q, (d,), weight=qw, eps=1e-6)
            kn = F.rms_norm(k, (d,), weight=kw, eps=1e-6)
            q_ref = qn * cos.unsqueeze(1) + rotate_half(qn) * sin.unsqueeze(1)
            k_ref = kn * cos.unsqueeze(1) + rotate_half(kn) * sin.unsqueeze(1)
            assert _rel_l2(q_tri, q_ref) < 0.01
            assert _rel_l2(k_tri, k_ref) < 0.01

            if HAS_SM90:
                for causal in (True, False):
                    fa3_out = cosmos3_fa3_attention(q_tri, k_tri, v, causal=causal)
                    sdpa_out = (
                        scaled_dot_product_attention(
                            q_tri.transpose(0, 1).unsqueeze(0),
                            k_tri.transpose(0, 1).unsqueeze(0),
                            v.transpose(0, 1).unsqueeze(0),
                            is_causal=causal,
                            enable_gqa=True,
                        )
                        .squeeze(0)
                        .transpose(0, 1)
                        .flatten(-2, -1)
                    )
                    assert _rel_l2(fa3_out, sdpa_out) < 0.01

        # 2. Verify fused_residual_add_rmsnorm at Cosmos3-Super width 5120.
        r = torch.randn(64, 5120, device=device, dtype=torch.bfloat16)
        u = torch.randn(64, 5120, device=device, dtype=torch.bfloat16)
        w = torch.randn(5120, device=device, dtype=torch.bfloat16)
        res_t, norm_t = fused_residual_add_rmsnorm(r, u, w, eps=1e-6)
        assert torch.equal(res_t, r + u)
        assert _rel_l2(norm_t, F.rms_norm(r + u, (5120,), weight=w, eps=1e-6)) < 0.005

    # 3. Bitwise-exact und static KV cache on GPU (with identical attention/projection paths).
    ref_gpu = _tiny_cosmos3(
        device,
        num_layers=4,
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        dtype=torch.bfloat16,
    )
    opt_gpu = copy.deepcopy(ref_gpu)
    accelerate_cosmos3(
        opt_gpu,
        Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend="torch",
            fused_ops=False,
            fuse_projections=False,
            quantization="none",
            cache_algorithm="none",
            cuda_graph="off",
        ),
    )
    und_toks, lat = _cosmos3_sample_inputs(device, torch.bfloat16, seed=4)
    with torch.inference_mode():
        for step in range(4):
            lat_s = lat + 0.02 * step
            fw = _cosmos3_forward_kwargs(ref_gpu, und_toks, lat_s, 800.0 - 100.0 * step)
            r_vis, _, _ = ref_gpu(**fw)
            o_vis, _, _ = opt_gpu(
                **fw,
                step_index=step,
                total_steps=4,
            )
            assert _rel_l2(o_vis[0], r_vis[0]) == 0.0
            assert torch.equal(o_vis[0], r_vis[0])


@pytest.mark.skipif(not HAS_SM90, reason="Hopper SM90+ required for Cosmos3 FP8 + FA3 + Piecewise Graph test")
def test_cosmos3_gpu_step_selective_fp8_teacache_piecewise_graph_denoiser() -> None:
    torch.manual_seed(77)
    device = torch.device("cuda")
    model = _tiny_cosmos3(
        device,
        num_layers=4,
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        dtype=torch.bfloat16,
    )
    denoiser = Cosmos3JointDenoiser(model, cuda_graph="piecewise")
    cfg = Cosmos3AccelerationConfig(
        und_static_kv_cache=True,
        attention_backend="flash_attention_3",
        fused_ops=True,
        fuse_projections=True,
        quantization="fp8",
        fp8_scaling="rowwise",
        fp8_use_fast_accum=False,
        keep_dense_fallback=False,
        quant_min_features=128,
        quant_exclude_edge_blocks=True,
        quant_start_step=1,
        quant_dense_last_steps=1,
        cache_algorithm="teacache",
        cache_threshold=0.35,
        cache_warmup_steps=3,
        cache_dense_last=1,
        cache_max_consecutive_skips=2,
        cuda_graph="piecewise",
        cuda_graph_warmup=1,
    )
    receipt = denoiser.apply_sol_engine(cfg)
    assert receipt["fallbacks"] == []
    assert receipt["quantization"]["quantized_linears"] > 0
    assert receipt["quantization"]["use_fast_accum"] is False
    assert receipt["quantization"]["keep_dense_fallback"] is False
    # Verify edge blocks (0 and 3) stayed BF16 nn.Linear, interior blocks (1 and 2) got StepSelectiveFloat8Linear.
    layers_any: Any = model.layers
    assert isinstance(layers_any[0].mlp_moe_gen.gate_up_proj, nn.Linear)
    assert isinstance(layers_any[1].mlp_moe_gen.gate_up_proj, StepSelectiveFloat8Linear)
    assert layers_any[1].mlp_moe_gen.gate_up_proj.weight is None
    assert isinstance(layers_any[3].mlp_moe_gen.gate_up_proj, nn.Linear)

    input_ids = torch.randint(1, 200, (12,), device=device)
    lat = torch.randn(1, 16, 2, 4, 4, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 2, 1, 1, device=device, dtype=torch.float32)
    dummy_pos = torch.zeros(1, device=device, dtype=torch.long)

    with torch.inference_mode():
        for step in range(7):
            m_in = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(
                        latent=lat + 0.002 * step,
                        denoise_mask=mask,
                        positions=dummy_pos,
                        clean_latent=lat,
                    )
                },
                timestep=torch.tensor(800.0 - 80.0 * step, device=device),
                conditioning={"input_ids": input_ids, "fps": 24.0, "request_id": "req-cosmos3-gpu"},
                step_index=step,
                total_steps=7,
                branch="positive",
            )
            m_out = denoiser(m_in)
            assert m_out.samples["video"].shape == lat.shape
            assert torch.isfinite(m_out.samples["video"]).all()

    report = denoiser.runtime_optimization_report()
    rt = report["cosmos3_sol_engine"]["runtime"]
    assert report["fallbacks"] == []
    assert rt["attention_fa3_calls"] > 0
    assert rt["attention_sdpa_calls"] == 0
    assert rt["und_static_kv_cache"]["hits"] == 6
    assert rt["quant_step_bf16_boundary_calls"] > 0  # steps 0 and 6 ran in BF16 mode
    assert rt["quant_step_fp8_calls"] > 0  # interior steps 1..5 ran in native FP8 mode
    assert rt["gen_cache_branches"]["positive"]["hits"] > 0
    pg = rt["piecewise_graph"]
    assert pg is not None
    assert pg["capture"] >= 1
    assert pg["replay"] >= 1
    denoiser.end_request("req-cosmos3-gpu")


# =============================================================================
# 3. LingBot-Video Tests (CPU + GPU)
# =============================================================================


def test_lingbot_fusion_compact_adaln_and_easycache_cpu() -> None:
    torch.manual_seed(21)
    device = torch.device("cpu")
    ref = _tiny_lingbot(device, depth=5, dtype=torch.float32)
    opt = copy.deepcopy(ref)

    cfg = LingBotAccelerationConfig(
        fuse_qkv=True,
        fuse_swiglu=True,
        compact_adaln=True,
        fused_kernels=True,
        attention_backend="torch",
        pisa_enabled=False,
        easycache=False,
        cuda_graph="off",
    )
    receipt = accelerate_lingbot_video(opt, cfg)
    assert receipt["fallbacks"] == []
    assert receipt["projection_fusion"] == {"qkv_fused": 5, "swiglu_fused": 5}

    lat = torch.randn(1, 16, 2, 8, 8)
    ts = torch.tensor([500.0])
    enc = torch.randn(1, 12, 64)
    mask = torch.ones(1, 12, dtype=torch.long)

    with torch.inference_mode():
        ref_out = ref(lat, ts, enc, encoder_attention_mask=mask, return_dict=False)[0]
        opt_out = opt(lat, ts, enc, encoder_attention_mask=mask, return_dict=False)[0]
    assert _rel_l2(opt_out, ref_out) < 1e-4

    # Verify EasyCache branch isolation + boundary guards + receipts.
    ec = LingBotEasyCache(threshold=0.3, total_steps=10, warmup_steps=3, dense_last=2, max_skip_steps=2)
    h0 = torch.randn(1, 32, 128)
    with torch.inference_mode():
        for step in range(10):
            out, _ = ec.run_stack(
                step=step,
                total_steps=10,
                value=h0 + 0.001 * step,
                block_count=5,
                run_stack_fn=lambda v: v * 1.01 + 0.02,
            )
            assert out.shape == h0.shape
    rc = ec.receipt()
    assert rc["hits"] > 0
    assert rc["fallbacks"] == []
    for ev in ec.events:
        if ev.step < 3:
            assert not ev.hit and ev.reason == "warmup"
        if ev.step >= 8:
            assert not ev.hit and ev.reason == "dense-last"


def test_lingbot_runtime_forwards_sol_profile_flags(tmp_path: Path) -> None:
    runtime = LingBotVideoRuntime(tmp_path / "ckpt", device="cuda:6")
    plan = runtime.build_plan(
        request={
            "prompt": "a glowing waterfall in a forest",
            "mode": "t2v",
            "num_inference_steps": 20,
            "sol_profile": "sol_balanced",
            "run_refiner": True,
            "refiner_sol_profile": "refiner_sol",
        },
        output_path=tmp_path / "out.mp4",
    )
    assert plan.command[plan.command.index("--sol_profile") + 1] == "sol_balanced"
    assert plan.command[plan.command.index("--refiner_sol_profile") + 1] == "refiner_sol"


@pytest.mark.skipif(not HAS_SM90, reason="Hopper SM90+ required for LingBot FA3 + PISA TMA + Piecewise Graph tests")
def test_lingbot_gpu_fusion_pisa_guards_easycache_and_piecewise_graph() -> None:
    torch.manual_seed(91)
    device = torch.device("cuda")
    # Use head_dim=128 (hidden_size=512, num_heads=4) to exercise real TMA PISA & FA3 kernels.
    ref = (
        LingBotVideoTransformer3DModel(
            patch_size=(1, 2, 2),
            in_channels=16,
            out_channels=16,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            depth=5,
            text_dim=64,
            freq_dim=64,
            axes_dims=(32, 48, 48),
            axes_lens=(128, 128, 128),
            rope_theta=256.0,
            num_experts=0,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    # Create sequence length > 1024 so PISA triggers: T=4, H=32, W=32 -> 4 * 16 * 16 = 1024 video + 16 text = 1040.
    lat = torch.randn(1, 16, 4, 32, 32, device=device, dtype=torch.bfloat16)
    ts = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    enc = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 16, dtype=torch.long, device=device)

    # 1. Exact lossless fusion + Hopper FA3 + Piecewise Graph parity vs baseline.
    opt_exact = copy.deepcopy(ref)
    exact_receipt = accelerate_lingbot_video(opt_exact, lingbot_sol_profile("exact", sm_major=9))
    assert exact_receipt["fallbacks"] == []
    with torch.inference_mode():
        ref_out = ref(lat, ts, enc, encoder_attention_mask=mask, return_dict=False)[0]
        for _ in range(4):
            exact_out = opt_exact(lat, ts, enc, encoder_attention_mask=mask, return_dict=False)[0]
    assert _rel_l2(exact_out, ref_out) < 0.015
    rt_exact = lingbot_runtime_receipt(opt_exact)
    assert rt_exact["fallbacks"] == []
    assert rt_exact["runtime"]["fa3_calls"] > 0
    assert rt_exact["runtime"]["sdpa_calls"] == 0
    assert rt_exact["runtime"]["piecewise_graph"]["capture"] >= 1
    assert rt_exact["runtime"]["piecewise_graph"]["replay"] >= 2

    # 2. Full SOL-Engine (Fusion + PISA + EasyCache + Piecewise Graph):
    #    Verify dense warmup/tail steps, dense edge blocks (0, 1, 4), PISA interior blocks (2, 3),
    #    branch-isolated EasyCache, and zero capture failures / zero fallbacks!
    opt_sol = copy.deepcopy(ref)
    sol_cfg = LingBotAccelerationConfig(
        attention_backend="flash_attention_3",
        pisa_enabled=True,
        pisa_density=0.25,
        pisa_block_size=64,
        pisa_min_sequence_length=1024,
        pisa_dense_steps=2,
        pisa_dense_last_steps=1,
        pisa_dense_blocks=(0, 1, -1),
        pisa_strict=True,
        easycache=True,
        easycache_threshold=0.35,
        easycache_warmup_steps=3,
        easycache_dense_last=1,
        easycache_max_skip_steps=2,
        cuda_graph="piecewise",
        cuda_graph_warmup=1,
    )
    sol_receipt = accelerate_lingbot_video(opt_sol, sol_cfg)
    assert sol_receipt["fallbacks"] == []
    assert sol_receipt["attention"]["pisa"]["dense_blocks"] == [0, 1, 4]
    assert opt_sol.blocks[0].attn._pisa_block_eligible is False
    assert opt_sol.blocks[1].attn._pisa_block_eligible is False
    assert opt_sol.blocks[2].attn._pisa_block_eligible is True
    assert opt_sol.blocks[3].attn._pisa_block_eligible is True
    assert opt_sol.blocks[4].attn._pisa_block_eligible is False

    with torch.inference_mode():
        for step in range(9):
            for branch in ("positive", "negative"):
                out = opt_sol(
                    lat + 0.002 * step,
                    torch.tensor([900.0 - 80.0 * step], device=device, dtype=torch.bfloat16),
                    enc if branch == "positive" else enc * 0.5,
                    encoder_attention_mask=mask,
                    return_dict=False,
                    step_index=step,
                    total_steps=9,
                    cache_branch=branch,
                )[0]
                assert out.shape == lat.shape
                assert torch.isfinite(out).all()

    rt_sol = lingbot_runtime_receipt(opt_sol)
    assert rt_sol["fallbacks"] == []
    assert rt_sol["runtime"]["pisa_calls"] > 0
    assert rt_sol["runtime"]["fa3_calls"] > 0
    assert rt_sol["runtime"]["sdpa_calls"] == 0
    # Edge blocks MUST have 0 PISA calls; interior blocks MUST have > 0 PISA calls!
    assert opt_sol.blocks[0].attn._pisa_calls == 0
    assert opt_sol.blocks[1].attn._pisa_calls == 0
    assert opt_sol.blocks[4].attn._pisa_calls == 0
    assert opt_sol.blocks[2].attn._pisa_calls > 0
    assert opt_sol.blocks[3].attn._pisa_calls > 0
    # Both CFG branches have independent EasyCache hits!
    ec_branches = rt_sol["runtime"]["easycache_branches"]
    assert ec_branches["positive"]["hits"] > 0
    assert ec_branches["negative"]["hits"] > 0
    # Both dense and PISA graph segments captured with 0 failures!
    pg_sol = rt_sol["runtime"]["piecewise_graph"]
    assert pg_sol is not None
    assert "lingbot_blocks_dense" in pg_sol["segments"]
    assert "lingbot_blocks_pisa" in pg_sol["segments"]
    assert pg_sol["capture"] >= 2
    assert pg_sol["replay"] >= 2


# =============================================================================
# 4. Adversarial Edge-Case & Regression Tests (CPU + GPU)
# =============================================================================


def test_sana_regression_inference_mode_caches_zero_sync_and_reapply_cpu() -> None:
    torch.manual_seed(101)
    device = torch.device("cpu")
    model = _tiny_sana_video(device, dtype=torch.float32)

    class _FakeVae(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(4, 4, kernel_size=3, padding=1)

    vae = _FakeVae()
    rec = accelerate_sana_video(
        model,
        SanaAccelerationConfig(
            cross_attn_backend="torch",
            static_cross_kv=True,
            vae_channels_last_3d=True,
            vae_decode_autocast=False,
        ),
        vae=vae,
    )
    assert rec["vae"] is not None
    assert rec["vae"]["decode_bf16"] is False
    assert vae.conv.weight.dtype == torch.float32

    x = torch.randn(1, 16, 2, 8, 8)
    t = torch.tensor([500.0])
    y = torch.randn(1, 1, 10, 64)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    with torch.inference_mode():
        y_mut = y.clone()
        _ = model(x, t, y_mut, mask=mask)
        y_mut.fill_(3.25)
        out_mut = model(x, t, y_mut, mask=mask)
        out_fresh = model(x, t, torch.full_like(y_mut, 3.25), mask=mask)
        assert torch.equal(out_mut, out_fresh)

    # Zero-sync EasyCache schedule runs end-to-end through SanaDenoiser.
    denoiser = SanaDenoiser(model)
    denoiser.apply_sol_engine(
        SanaAccelerationConfig(
            cross_attn_backend="torch",
            easycache=True,
            easycache_warmup_steps=2,
            easycache_dense_last=1,
            easycache_max_skip_steps=1,
            easycache_zero_sync_skip_steps=(0, 2, 3, 4, 5),
            cuda_graph="off",
        )
    )
    with torch.inference_mode():
        for step in range(6):
            inp = DenoiserInput(
                latents=x + 0.001 * step,
                timestep=torch.tensor([500.0 - 10.0 * step]),
                next_timestep=torch.tensor([max(500.0 - 10.0 * (step + 1), 0.0)]),
                conditioning={"context": y.squeeze(1), "context_mask": mask},
                step_index=step,
                total_steps=6,
                request_id="req-sana-zs-cpu",
                branch="positive",
            )
            denoiser(inp)
    zs_rep = denoiser.runtime_optimization_report("req-sana-zs-cpu")
    fc_zs = cast(dict[str, Any], zs_rep["feature_cache"])
    assert fc_zs["hits"] == 2  # steps 2 and 4 (0/1 warmup, 3 blocked by max_skip=1, 5 dense_last)
    denoiser.end_request("req-sana-zs-cpu")

    # Re-applying with optimizations disabled clears previous fusion flags cleanly.
    off_rec = denoiser.apply_sol_engine(
        SanaAccelerationConfig(
            block_fusion=False,
            fast_litela_rope=False,
            fast_rms_norm=False,
            fuse_glumbconv=False,
            cross_attn_backend="torch",
            easycache=False,
            compile=False,
            cuda_graph="off",
        )
    )
    assert off_rec["fusion"]["fused_blocks"] == 0
    assert off_rec["fusion"]["fast_litela_rope_layers"] == 0
    assert off_rec["fusion"]["fused_glumbconv_layers"] == 0
    assert denoiser.graph_report() is None


@pytest.mark.skipif(not HAS_SM90, reason="Hopper SM90+ required for SANA GPU regression tests")
def test_sana_regression_gpu_piecewise_skips_fa3_exact_and_whole_graph_nonprefix_mask() -> None:
    torch.manual_seed(102)
    device = torch.device("cuda")
    ref = _tiny_sana_video(device, dtype=torch.bfloat16)
    opt_pw = copy.deepcopy(ref)
    opt_whole = copy.deepcopy(ref)

    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=torch.bfloat16)
    t = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    y = torch.randn(1, 1, 10, 64, device=device, dtype=torch.bfloat16)
    right_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], device=device)
    left_mask = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], device=device)

    # 1. Piecewise graph cache_skips must match EasyCache hits, and FA3 calls must exclude skipped steps.
    d_pw = SanaDenoiser(opt_pw)
    d_pw.apply_sol_engine(
        SanaAccelerationConfig(
            cross_attn_backend="flash_attention_3",
            easycache=True,
            easycache_threshold=0.5,
            easycache_warmup_steps=2,
            easycache_dense_last=1,
            easycache_max_skip_steps=2,
            cuda_graph="piecewise",
        )
    )
    with torch.inference_mode():
        for step in range(6):
            d_pw(
                DenoiserInput(
                    latents=x + 0.001 * step,
                    timestep=torch.tensor([500.0 - 10.0 * step], device=device),
                    next_timestep=torch.tensor([max(500.0 - 10.0 * (step + 1), 0.0)], device=device),
                    conditioning={"context": y.squeeze(1), "context_mask": right_mask},
                    step_index=step,
                    total_steps=6,
                    request_id="req-sana-pw-gpu",
                    branch="positive",
                )
            )
    rep = d_pw.runtime_optimization_report("req-sana-pw-gpu")
    hits = int(cast(dict[str, Any], rep["feature_cache"])["hits"])
    assert hits > 0
    assert int(cast(dict[str, Any], rep["cuda_graph"])["cache_skips"]) == hits
    fa3_calls = int(cast(dict[str, Any], rep["sana_sol_engine"])["runtime"]["cross_attn_fa3_calls"])
    assert fa3_calls == len(opt_pw.blocks) * (6 - hits)
    d_pw.end_request("req-sana-pw-gpu")

    # 2. Whole-model CUDA Graph with left-padded (non-prefix) mask must NOT corrupt cross-attention.
    d_whole = SanaDenoiser(opt_whole)
    d_whole.apply_sol_engine(
        SanaAccelerationConfig(
            cross_attn_backend="flash_attention_3",
            easycache=False,
            compile=False,
            cuda_graph="whole",
        )
    )
    with torch.inference_mode():
        ref_left = ref(x, t, y, mask=left_mask)[:, :16]
        inp_w = DenoiserInput(
            latents=x,
            timestep=t,
            next_timestep=t,
            conditioning={"context": y.squeeze(1), "context_mask": left_mask},
            step_index=0,
            total_steps=4,
            request_id="req-sana-whole-left",
            branch="positive",
        )
        out_whole = d_whole(inp_w).sample
        assert _rel_l2(out_whole, ref_left) < 0.02
    g_rep = d_whole.graph_report()
    assert g_rep is not None and g_rep["capture_failed"] == 0


def test_cosmos3_regression_multimodal_layout_rotary_teacache_and_neg_first_cpu() -> None:
    torch.manual_seed(103)
    device = torch.device("cpu")
    model = Cosmos3OmniTransformer(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        latent_channel=16,
        latent_patch_size=2,
        patch_latent_dim=64,
        vocab_size=256,
        rope_scaling={"mrope_section": [4, 6, 6]},
        action_gen=True,
        sound_gen=True,
        sound_dim=8,
        action_dim=4,
    ).eval()

    # Whole-model graph must be rejected cleanly on variable-length Cosmos3.
    with pytest.raises(ValueError, match="whole-model"):
        Cosmos3JointDenoiser(model, cuda_graph="whole")

    ids, lat = _cosmos3_sample_inputs(device, torch.float32, seed=11)
    vmask = torch.ones(1, 1, 2, 1, 1)
    dpos = torch.zeros(1, dtype=torch.long)
    snd_lat = torch.randn(8, 4)
    smask1 = torch.ones(1, 4)
    smask2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    # 1. Multimodal sound.denoise_mask change must invalidate _layout_cache cleanly.
    denoiser = Cosmos3JointDenoiser(model)
    with torch.inference_mode():
        _ = denoiser(
            MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(latent=lat, denoise_mask=vmask, positions=dpos, clean_latent=lat),
                    "sound": ModalityState(latent=snd_lat, denoise_mask=smask1, positions=dpos, clean_latent=snd_lat),
                },
                timestep=torch.tensor(500.0),
                conditioning={"input_ids": ids, "fps": 24.0},
                step_index=0,
                total_steps=4,
                branch="positive",
            )
        )
        out_s2 = denoiser(
            MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(latent=lat, denoise_mask=vmask, positions=dpos, clean_latent=lat),
                    "sound": ModalityState(latent=snd_lat, denoise_mask=smask2, positions=dpos, clean_latent=snd_lat),
                },
                timestep=torch.tensor(500.0),
                conditioning={"input_ids": ids, "fps": 24.0},
                step_index=0,
                total_steps=4,
                branch="positive",
            )
        )
        out_s2_true = Cosmos3JointDenoiser(model)(
            MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(latent=lat, denoise_mask=vmask, positions=dpos, clean_latent=lat),
                    "sound": ModalityState(latent=snd_lat, denoise_mask=smask2, positions=dpos, clean_latent=snd_lat),
                },
                timestep=torch.tensor(500.0),
                conditioning={"input_ids": ids, "fps": 24.0},
                step_index=0,
                total_steps=4,
                branch="positive",
            )
        )
        assert torch.equal(out_s2.samples["sound"], out_s2_true.samples["sound"])

    # 2. Negative-first CFG at step 0 must not wipe negative's und_kv cache.
    m_neg = copy.deepcopy(model)
    d_neg = Cosmos3JointDenoiser(m_neg)
    d_neg.apply_sol_engine(
        Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend="torch",
            fused_ops=False,
            fuse_projections=False,
            cuda_graph="off",
        )
    )
    ids_neg, _ = _cosmos3_sample_inputs(device, torch.float32, seed=12)
    with torch.inference_mode():
        for step in range(2):
            for branch, tok in (("negative", ids_neg), ("positive", ids)):
                d_neg(
                    MultiModalDenoiserInput(
                        modalities={
                            "video": ModalityState(
                                latent=lat + 0.01 * step, denoise_mask=vmask, positions=dpos, clean_latent=lat
                            )
                        },
                        timestep=torch.tensor(800.0 - 100.0 * step),
                        conditioning={"input_ids": tok, "fps": 24.0},
                        step_index=step,
                        total_steps=4,
                        branch=branch,
                    )
                )
    kv_neg = cosmos3_runtime_receipt(m_neg)["runtime"]["und_static_kv_cache"]
    assert kv_neg["hits"] == 2 and kv_neg["misses"] == 2

    # 3. und_static_kv_cache=False + TeaCache must still run TeaCache and record hits.
    m_nokv = copy.deepcopy(model)
    d_nokv = Cosmos3JointDenoiser(m_nokv)
    d_nokv.apply_sol_engine(
        Cosmos3AccelerationConfig(
            und_static_kv_cache=False,
            attention_backend="torch",
            fused_ops=False,
            fuse_projections=False,
            cache_algorithm="teacache",
            cache_threshold=0.5,
            cache_warmup_steps=2,
            cache_dense_last=1,
            cuda_graph="off",
        )
    )
    with torch.inference_mode():
        for step in range(6):
            d_nokv(
                MultiModalDenoiserInput(
                    modalities={
                        "video": ModalityState(
                            latent=lat + 0.001 * step, denoise_mask=vmask, positions=dpos, clean_latent=lat
                        )
                    },
                    timestep=torch.tensor(800.0 - 50.0 * step),
                    conditioning={"input_ids": ids, "fps": 24.0, "request_id": "req-nokv"},
                    step_index=step,
                    total_steps=6,
                    branch="positive",
                )
            )
    tc_nokv = cosmos3_runtime_receipt(m_nokv)["runtime"]["gen_cache_branches"]["positive"]
    assert tc_nokv["events"] == 6 and tc_nokv["hits"] > 0

    # 4. In-place inference_mode position_ids mutation & mid-run und_cache miss with warm TeaCache.
    m_rot = copy.deepcopy(model)
    accelerate_cosmos3(
        m_rot,
        Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend="torch",
            fused_ops=False,
            fuse_projections=False,
            cache_algorithm="teacache",
            cache_threshold=10.0,
            cache_warmup_steps=1,
            cache_dense_last=0,
            cuda_graph="off",
        ),
    )
    fw = _cosmos3_forward_kwargs(model, ids, lat, 500.0)
    with torch.inference_mode():
        pos = fw["position_ids"].clone()
        fw_mut = dict(fw, position_ids=pos)
        _ = m_rot(**fw_mut, step_index=0, total_steps=6, cache_branch="pos")
        _ = m_rot(**fw_mut, step_index=1, total_steps=6, cache_branch="pos")
        pos.add_(25)
        out_mut = m_rot(**fw_mut, step_index=2, total_steps=6, cache_branch="pos")[0][0]
        out_ref = model(**fw_mut)[0][0]
        assert torch.equal(out_mut, out_ref)
        last_ev = m_rot._cosmos3_gen_caches["pos"].events[-1]
        assert not last_ev.hit and last_ev.reason == "und-cache-miss"


@pytest.mark.skipif(not HAS_SM90, reason="Hopper SM90+ required for Cosmos3 FP8 autocast & replay telemetry tests")
def test_cosmos3_regression_gpu_fp8_small_gemm_autocast_and_replay_telemetry() -> None:
    torch.manual_seed(104)
    device = torch.device("cuda")
    w = torch.randn(128, 128, device=device, dtype=torch.bfloat16)

    # Small GEMM with keep_dense_fallback=True must not trigger spurious fallback or false FP8 count.
    lin_kd = StepSelectiveFloat8Linear(w, keep_dense_fallback=True, scaling="rowwise", use_fast_accum=False)
    with torch.inference_mode():
        _ = lin_kd(torch.randn(8, 128, device=device, dtype=torch.bfloat16))
    assert lin_kd.step_fp8_calls == 0
    assert lin_kd.dense_fallback_calls == 0
    assert lin_kd.step_bf16_calls == 1

    # FP32 input inside BF16 autocast must run native FP8 instead of BF16 dequant.
    lin_nd = StepSelectiveFloat8Linear(w, keep_dense_fallback=False, scaling="rowwise", use_fast_accum=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_ac = lin_nd(torch.randn(8, 128, device=device, dtype=torch.float32))
    assert lin_nd.step_fp8_calls == 1
    assert lin_nd.step_bf16_calls == 0
    assert y_ac.dtype == torch.bfloat16

    # Replay telemetry: quant_step_fp8_calls & attention_fa3_calls must grow across replayed steps!
    model = _tiny_cosmos3(
        device,
        num_layers=4,
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        dtype=torch.bfloat16,
    )
    denoiser = Cosmos3JointDenoiser(model)
    denoiser.apply_sol_engine(
        Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend="flash_attention_3",
            fused_ops=True,
            fuse_projections=True,
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            keep_dense_fallback=False,
            quant_min_features=128,
            quant_exclude_edge_blocks=True,
            quant_start_step=1,
            quant_dense_last_steps=0,
            cache_algorithm="none",
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
        )
    )
    ids, lat = _cosmos3_sample_inputs(device, torch.bfloat16, seed=14)
    vmask = torch.ones(1, 1, 2, 1, 1, device=device)
    dpos = torch.zeros(1, device=device, dtype=torch.long)
    with torch.inference_mode():
        for step in range(4):
            denoiser(
                MultiModalDenoiserInput(
                    modalities={
                        "video": ModalityState(latent=lat, denoise_mask=vmask, positions=dpos, clean_latent=lat)
                    },
                    timestep=torch.tensor(800.0 - 50.0 * step, device=device),
                    conditioning={"input_ids": ids, "fps": 24.0, "request_id": "req-replay-1"},
                    step_index=step,
                    total_steps=8,
                    branch="positive",
                )
            )
        rt_mid = cosmos3_runtime_receipt(model)["runtime"]
        for step in range(4, 8):
            denoiser(
                MultiModalDenoiserInput(
                    modalities={
                        "video": ModalityState(latent=lat, denoise_mask=vmask, positions=dpos, clean_latent=lat)
                    },
                    timestep=torch.tensor(800.0 - 50.0 * step, device=device),
                    conditioning={"input_ids": ids, "fps": 24.0, "request_id": "req-replay-1"},
                    step_index=step,
                    total_steps=8,
                    branch="positive",
                )
            )
        rt_end = cosmos3_runtime_receipt(model)["runtime"]
    assert rt_end["quant_step_fp8_calls"] > rt_mid["quant_step_fp8_calls"]
    assert rt_end["attention_fa3_calls"] > rt_mid["attention_fa3_calls"]
    assert rt_end["piecewise_graph"]["replay"] >= 4
    # Turning cuda_graph off clears denoiser._piecewise_graph cleanly.
    denoiser.apply_sol_engine(Cosmos3AccelerationConfig(cuda_graph="off", quantization="none"))
    assert denoiser.graph_report() is None


def test_lingbot_regression_training_grad_and_cpu_pisa_guard() -> None:
    torch.manual_seed(105)
    device = torch.device("cpu")
    model = _tiny_lingbot(device, depth=4, dtype=torch.float32)
    rec = accelerate_lingbot_video(
        model,
        LingBotAccelerationConfig(
            fuse_qkv=True,
            fuse_swiglu=True,
            attention_backend="torch",
            pisa_enabled=True,
            easycache=False,
            cuda_graph="off",
        ),
    )
    # On CPU, pisa_ready=False so NO block may be marked _pisa_block_eligible.
    assert rec["attention"]["pisa"]["enabled"] is False
    for block in model.blocks:
        block_any: Any = block
        assert block_any.attn._pisa_block_eligible is False

    # Training mode must bypass requires_grad=False fused linears so gradients reach base weights.
    model.train()
    lat = torch.randn(1, 16, 2, 8, 8, requires_grad=True)
    ts = torch.tensor([500.0])
    enc = torch.randn(1, 8, 64)
    out = model(lat, ts, enc, return_dict=False)[0]
    out.sum().backward()
    block0: Any = model.blocks[0]
    assert block0.attn.to_q.weight.grad is not None
    assert block0.ffn.gate_proj.weight.grad is not None


@pytest.mark.skipif(not HAS_SM90, reason="Hopper SM90+ required for LingBot MoE & B=2 GPU regression tests")
def test_lingbot_regression_gpu_moe_safety_batched_cfg_and_nonbinary_mask() -> None:
    torch.manual_seed(106)
    device = torch.device("cuda")

    # 1. MoE model with sol_balanced disables piecewise graph cleanly (0 capture_failed, 0 NaN, shared_experts fused).
    with torch.no_grad():
        lb_moe = (
            LingBotVideoTransformer3DModel(
                patch_size=(1, 2, 2),
                in_channels=16,
                out_channels=16,
                hidden_size=512,
                num_attention_heads=4,
                intermediate_size=1024,
                depth=4,
                text_dim=64,
                freq_dim=64,
                axes_dims=(32, 48, 48),
                axes_lens=(128, 128, 128),
                rope_theta=256.0,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=256,
                decoder_sparse_step=1,
                n_shared_experts=1,
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        # Initialize router/grouped-expert parameters (constructed via torch.empty upstream).
        for name, param in lb_moe.named_parameters():
            if any(tok in name for tok in (".router.weight", ".experts.w1", ".experts.w2", ".experts.w3")):
                nn.init.normal_(param, std=0.02)
    lb_moe_ref = copy.deepcopy(lb_moe)
    rec_moe = accelerate_lingbot_video(lb_moe, "sol_balanced")
    assert rec_moe["fallbacks"] == []
    assert rec_moe["cuda_graph"]["mode"] == "off"
    assert rec_moe["projection_fusion"] == {"qkv_fused": 4, "swiglu_fused": 4}

    lat1 = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    ts1 = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    enc1 = torch.randn(1, 16, 64, device=device, dtype=torch.bfloat16)
    mask1 = torch.ones(1, 16, dtype=torch.long, device=device)
    with torch.inference_mode():
        ref_m = lb_moe_ref(lat1, ts1, enc1, encoder_attention_mask=mask1, return_dict=False)[0]
        sol_m0 = lb_moe(
            lat1,
            ts1,
            enc1,
            encoder_attention_mask=mask1,
            return_dict=False,
            step_index=0,
            total_steps=8,
            cache_branch="pos",
        )[0]
        assert torch.isfinite(sol_m0).all()
        assert _rel_l2(sol_m0, ref_m) < 0.02
        for step in range(1, 6):
            sol_ms = lb_moe(
                lat1 + 0.0005 * step,
                ts1 - 10.0 * step,
                enc1,
                encoder_attention_mask=mask1,
                return_dict=False,
                step_index=step,
                total_steps=8,
                cache_branch="pos",
            )[0]
            assert torch.isfinite(sol_ms).all()
    rt_moe = lingbot_runtime_receipt(lb_moe)
    assert rt_moe["fallbacks"] == []
    assert rt_moe["runtime"]["easycache_branches"]["pos"]["hits"] > 0

    # 2. Batched CFG (B=2) with padded masks runs cleanly with FA3 varlen telemetry and EasyCache hits.
    with torch.no_grad():
        lb_b2 = (
            LingBotVideoTransformer3DModel(
                patch_size=(1, 2, 2),
                in_channels=16,
                out_channels=16,
                hidden_size=512,
                num_attention_heads=4,
                intermediate_size=1024,
                depth=5,
                text_dim=64,
                freq_dim=64,
                axes_dims=(32, 48, 48),
                axes_lens=(128, 128, 128),
                rope_theta=256.0,
                num_experts=0,
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
    accelerate_lingbot_video(lb_b2, "sol_balanced")
    lat2 = torch.randn(2, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
    ts2 = torch.tensor([500.0, 500.0], device=device, dtype=torch.bfloat16)
    enc2 = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
    mask2 = torch.tensor([[1] * 10 + [0] * 6, [1] * 6 + [0] * 10], device=device)
    with torch.inference_mode():
        for step in range(10):
            out2 = lb_b2(
                lat2 + 0.001 * step,
                ts2 - 20.0 * step,
                enc2,
                encoder_attention_mask=mask2,
                return_dict=False,
                step_index=step,
                total_steps=10,
                cache_branch="batched_cfg",
            )[0]
            assert torch.isfinite(out2).all()
    rt_b2 = lingbot_runtime_receipt(lb_b2)
    assert rt_b2["fallbacks"] == []
    assert rt_b2["runtime"]["fa3_calls"] > 0
    assert rt_b2["runtime"]["easycache_branches"]["batched_cfg"]["hits"] > 0


def test_cosmos3_idempotent_reapply_counts_and_sana_data_info_piecewise_seg_key() -> None:
    # 1. Re-applying Cosmos3 projection fusion and step-selective quantization preserves exact counts.
    torch.manual_seed(29)
    model = _tiny_cosmos3(torch.device("cpu"), num_layers=4, dtype=torch.float32)
    first_fusion = fuse_cosmos3_projections(model, release_originals=True)
    second_fusion = fuse_cosmos3_projections(model, release_originals=True)
    assert first_fusion == {"gen_qkv_fused": 4, "swiglu_gate_up_fused": 8}
    assert second_fusion == first_fusion

    first_q = quantize_cosmos3_step_selective(model, mode="fp8", min_features=64, exclude_edge_blocks=False)
    second_q = quantize_cosmos3_step_selective(model, mode="fp8", min_features=64, exclude_edge_blocks=False)
    assert first_q["quantized_linears"] > 0
    assert second_q["quantized_linears"] == first_q["quantized_linears"]

    # 2. SANA Video forward pops data_info so piecewise CUDA Graph segment keys never stringify CUDA tensors.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        sana = _tiny_sana_video(device, dtype=torch.bfloat16)
        accelerate_sana_video(sana, "sol_graph")
        x = torch.randn(1, 16, 2, 16, 16, device=device, dtype=torch.bfloat16)
        t = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
        y = torch.randn(1, 1, 16, 64, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 16, dtype=torch.long, device=device)
        data_info = {
            "img_hw": torch.tensor([[256.0, 256.0]], device=device),
            "aspect_ratio": torch.tensor([1.0], device=device),
        }
        with torch.inference_mode():
            for step in range(4):
                out = sana(
                    x + 0.01 * step,
                    t - 10.0 * step,
                    y,
                    mask=mask,
                    data_info=data_info,
                    feature_cache_step=step,
                    feature_cache_total_steps=4,
                )
                assert torch.isfinite(out).all()
        rt = sana_runtime_receipt(sana)
        pg = rt["runtime"]["piecewise_graph"]
        segs = pg["segments"]
        assert len(segs) == 1
        seg_key = next(iter(segs.keys()))
        assert "data_info" not in seg_key and "tensor(" not in seg_key
        assert pg["capture"] >= 1
        assert pg["request_window"]["replay"] >= 1
