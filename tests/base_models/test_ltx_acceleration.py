# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Unit and CUDA verification tests for native LTX / LTX-2 inference acceleration."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import ModalityState, MultiModalDenoiserInput
from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
    LTXAudioDecoderModule,
    LTXMediaDecoder,
    LTXVideoDecoderModule,
    LTXVocoderModule,
    _resolve_vae_compile_mode,
    _resolve_vae_tiling,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import (
    NormLayerType,
    PaddingModeType,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoDecoder
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import (
    LTXAVTransformerModule,
    LTXJointDenoiser,
    LTXVideoDenoiser,
    LTXVideoTransformerModule,
    _apply_ltx_post_load_accelerations,
    build_ltx_joint_denoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.base_models.diffusion_model.models.networks.ltx.feed_forward import FeedForward
from worldfoundry.base_models.diffusion_model.models.networks.ltx.fused_ops import (
    enable_ltx_fp8_quantization,
    fused_ltx_dual_rmsnorm_scale_shift,
    fused_ltx_gated_attention,
    fused_ltx_gated_attention_fp8,
    fused_ltx_gelu_tanh_rowwise_fp8,
    fused_ltx_post_sa_residual_rms_norm,
    fused_ltx_post_sa_residual_rmsnorm_adaln,
    fused_ltx_qk_rmsnorm,
    fused_ltx_qk_rmsnorm_rope,
    fused_ltx_rmsnorm_scale_shift_fp8,
    fused_ltx_scale_shift,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import (
    LTXModel,
    LTXModelType,
    infer_video_grid_shape,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import (
    LTXRopeType,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import BasicAVTransformerBlock
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgsPreprocessor
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import SpatioTemporalScaleFactors
from worldfoundry.base_models.diffusion_model.optimizations import (
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)
from worldfoundry.base_models.diffusion_model.optimizations.plugins import validate_acceleration_installation
from worldfoundry.core.acceleration.quantization.linear import Float8Linear
from worldfoundry.core.attention.backends.dispatch import (
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.core.attention.backends.probe import probe_attention_backends
from worldfoundry.core.model_loading.optimize import apply_attention_policy
from worldfoundry.core.nn.transformer.contracts import AttentionBackendConfigurable, TransformerBlockStack


class _VelocityWrapper(torch.nn.Module):
    def __init__(self, velocity_model: LTXModel) -> None:
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.velocity_model(*args, **kwargs)


def _as_video_wrapper(model: LTXModel) -> LTXVideoTransformerModule:
    return cast(LTXVideoTransformerModule, _VelocityWrapper(model))


def _as_av_wrapper(model: LTXModel) -> LTXAVTransformerModule:
    return cast(LTXAVTransformerModule, _VelocityWrapper(model))


def _init_deterministic_weights(model: torch.nn.Module, seed: int = 42) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and param.ndim == 1:
                param.fill_(1.0)
            elif "bias" in name or "scale_shift_table" in name:
                param.zero_()
            else:
                fan_in = param.shape[1] if param.ndim >= 2 else param.shape[0]
                std = min(0.05, 1.0 / math.sqrt(max(fan_in, 1)))
                vals = torch.empty(param.shape, dtype=torch.float32).normal_(0.0, std, generator=g)
                param.copy_(vals.to(device=param.device, dtype=param.dtype))


def _build_small_video_model(
    *,
    num_layers: int = 2,
    heads: int = 8,
    dim_head: int = 128,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LTXModel:
    inner = heads * dim_head
    model = (
        LTXModel(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=heads,
            attention_head_dim=dim_head,
            in_channels=32,
            out_channels=32,
            num_layers=num_layers,
            cross_attention_dim=inner,
            rope_type=rope_type,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            use_prompt_adaln_single=False,
            ff_bias=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    _init_deterministic_weights(model, seed=101)
    return model


def _build_small_av_model(
    *,
    num_layers: int = 2,
    video_heads: int = 4,
    video_dim_head: int = 128,
    audio_heads: int = 4,
    audio_dim_head: int = 64,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LTXModel:
    v_inner = video_heads * video_dim_head
    a_inner = audio_heads * audio_dim_head
    model = (
        LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=video_heads,
            attention_head_dim=video_dim_head,
            in_channels=32,
            out_channels=32,
            num_layers=num_layers,
            cross_attention_dim=v_inner,
            audio_num_attention_heads=audio_heads,
            audio_attention_head_dim=audio_dim_head,
            audio_in_channels=32,
            audio_out_channels=32,
            audio_cross_attention_dim=a_inner,
            rope_type=LTXRopeType.SPLIT,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            use_prompt_adaln_single=False,
            ff_bias=False,
            audio_ff_bias=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    _init_deterministic_weights(model, seed=202)
    return model


def _make_video_input(
    *,
    t_v: int = 64,
    t_txt: int = 16,
    in_channels: int = 32,
    ctx_dim: int = 1024,
    sigma: float = 0.5,
    step_index: int = 0,
    total_steps: int = 6,
    request_id: str | None = None,
    branch: str = "positive",
    seed: int = 7,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MultiModalDenoiserInput:
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn((1, t_v, in_channels), generator=g).to(device=device, dtype=dtype)
    denoise_mask = torch.ones((1, t_v, 1), device=device, dtype=dtype)
    f_p = 1
    h_p = 8 if t_v % 8 == 0 else 1
    w_p = t_v // (f_p * h_p)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(f_p, dtype=torch.float32),
            torch.arange(h_p, dtype=torch.float32),
            torch.arange(w_p, dtype=torch.float32),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, -1)
    positions = torch.stack((grid, grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
    context = torch.randn((1, t_txt, ctx_dim), generator=g).to(device=device, dtype=dtype)
    state = ModalityState(latent=latent, clean_latent=latent, denoise_mask=denoise_mask, positions=positions)
    conditioning: dict[str, object] = {"video_context": context}
    if request_id is not None:
        conditioning["request_id"] = request_id
    return MultiModalDenoiserInput(
        modalities={"video": state},
        timestep=torch.tensor([sigma], device=device, dtype=dtype),
        conditioning=conditioning,
        branch=branch,
        step_index=step_index,
        total_steps=total_steps,
    )


def _make_av_input(
    *,
    t_v: int = 64,
    t_a: int = 32,
    t_txt: int = 16,
    v_ctx_dim: int = 512,
    a_ctx_dim: int = 256,
    sigma: float = 0.5,
    step_index: int = 0,
    total_steps: int = 6,
    request_id: str | None = None,
    seed: int = 11,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MultiModalDenoiserInput:
    v_in = _make_video_input(
        t_v=t_v,
        t_txt=t_txt,
        ctx_dim=v_ctx_dim,
        sigma=sigma,
        step_index=step_index,
        total_steps=total_steps,
        request_id=request_id,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    a_lat = torch.randn((1, t_a, 32), generator=g).to(device=device, dtype=dtype)
    a_mask = torch.ones((1, t_a, 1), device=device, dtype=dtype)
    a_grid = torch.arange(t_a, dtype=torch.float32).reshape(1, -1)
    a_pos = torch.stack((a_grid, a_grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
    a_ctx = torch.randn((1, t_txt, a_ctx_dim), generator=g).to(device=device, dtype=dtype)
    a_state = ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos)
    conditioning = dict(v_in.conditioning)
    conditioning["audio_context"] = a_ctx
    return MultiModalDenoiserInput(
        modalities={"video": v_in.modalities["video"], "audio": a_state},
        timestep=v_in.timestep,
        conditioning=conditioning,
        step_index=step_index,
        total_steps=total_steps,
    )


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a_f, b_f = actual.float(), expected.float()
    denom = b_f.norm().item()
    return float((a_f - b_f).norm().item() / max(denom, 1e-12))


def _release_denoiser(denoiser: Any) -> None:
    graph_runner = getattr(denoiser, "_graph_runner", None)
    if graph_runner is not None:
        graph_runner.invalidate()
    piecewise_graph = getattr(denoiser, "_piecewise_graph", None)
    if piecewise_graph is not None:
        piecewise_graph.invalidate()
    gc.collect()


# =====================================================================
# UNIT TESTS (CPU, weight-free)
# =====================================================================


@pytest.mark.unit
def test_ltx_protocol_conformance_and_attention_policy() -> None:
    video_model = _build_small_video_model(num_layers=2, heads=2, dim_head=64)
    av_model = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)

    assert isinstance(video_model, TransformerBlockStack)
    assert video_model.blocks is video_model.transformer_blocks
    block0 = cast(BasicAVTransformerBlock, video_model.transformer_blocks[0])
    assert isinstance(block0.attn1, AttentionBackendConfigurable)

    video_report = apply_attention_policy(video_model, "torch")
    assert video_report.configured_modules == 4
    av_report = apply_attention_policy(av_model, "torch")
    assert av_report.configured_modules == 12


@pytest.mark.unit
def test_ltx_attention_backend_validation_and_scoping() -> None:
    cross_attn = Attention(query_dim=256, context_dim=256, heads=2, dim_head=128)
    with pytest.raises(ValueError, match="self-attention"):
        cross_attn.set_attention_backend("sol_attn")

    small_head_self = Attention(query_dim=128, context_dim=None, heads=2, dim_head=64)
    with pytest.raises(ValueError, match="dim_head=128"):
        small_head_self.set_attention_backend("sol_attn")

    invalid_fp8_head = Attention(query_dim=64, context_dim=None, heads=2, dim_head=32)
    with pytest.raises(ValueError, match="dim_head in"):
        invalid_fp8_head.set_attention_backend("cudnn_fp8")

    valid_self = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128)
    with pytest.raises(ValueError, match="backend_options require"):
        valid_self.set_attention_backend(None, backend_options={"tau": 1.0})
    with pytest.raises(ValueError, match="unknown cudnn_fp8"):
        valid_self.set_attention_backend("cudnn_fp8", backend_options={"bogus": 1.0})
    with pytest.raises(ValueError, match="dynamic_scale must be a bool"):
        valid_self.set_attention_backend("cudnn_fp8", backend_options={"dynamic_scale": "yes"})
    with pytest.raises(ValueError, match="positive finite float"):
        valid_self.set_attention_backend("cudnn_fp8", backend_options={"descale_q": True})
    with pytest.raises(ValueError, match="positive and finite"):
        valid_self.set_attention_backend("cudnn_fp8", backend_options={"descale_q": 0.0})
    with pytest.raises(ValueError, match="positive and finite"):
        valid_self.set_attention_backend("cudnn_fp8", backend_options={"descale_k": float("nan")})
    with pytest.raises(ValueError, match="cannot be combined"):
        valid_self.set_attention_backend(
            "cudnn_fp8",
            backend_options={"dynamic_scale": True, "descale_q": 0.1},
        )

    valid_self.set_attention_backend("sol_attn", backend_options={"tau": 1.0})
    with pytest.raises(ValueError, match="does not support attention masks"):
        valid_self(torch.randn(1, 8, 256), mask=torch.zeros(1, 1, 8, 8))

    av_model = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    with pytest.raises(ValueError, match="self_only and cross_only"):
        av_model.configure_attention_backend("torch", self_only=True, cross_only=True)
    with pytest.raises(ValueError, match="self-attention"):
        av_model.configure_attention_backend("sol_attn", cross_only=True)

    n_sol = av_model.configure_attention_backend("sol", self_only=True, backend_options={"tau": 1.0})
    assert n_sol == 2
    for raw_block in av_model.transformer_blocks:
        block = cast(BasicAVTransformerBlock, raw_block)
        assert block.attn1.attention_backend == "sol_attn"
        assert block.audio_attn1.attention_backend is None
        assert block.attn2.attention_backend is None


@pytest.mark.unit
def test_ltx_qkv_kv_projection_fusion_parity_and_edge_cases() -> None:
    torch.manual_seed(0)
    self_ref = Attention(query_dim=128, context_dim=None, heads=2, dim_head=64).eval()
    self_fused = Attention(query_dim=128, context_dim=None, heads=2, dim_head=64).eval()
    self_fused.load_state_dict(self_ref.state_dict())
    assert self_fused.fuse_qkv() is True
    assert self_fused.fuse_qkv() is False

    x = torch.randn(2, 12, 128)
    ctx = torch.randn(2, 8, 128)
    torch.testing.assert_close(self_fused(x), self_ref(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(self_fused(x, context=ctx), self_ref(x, context=ctx), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        self_fused(x, all_perturbed=True),
        self_ref(x, all_perturbed=True),
        rtol=1e-5,
        atol=1e-5,
    )

    cross_ref = Attention(query_dim=128, context_dim=96, heads=2, dim_head=64).eval()
    cross_fused = Attention(query_dim=128, context_dim=96, heads=2, dim_head=64).eval()
    cross_fused.load_state_dict(cross_ref.state_dict())
    assert cross_fused.fuse_kv() is True
    assert cross_fused.fuse_kv() is False
    c_in = torch.randn(2, 10, 96)
    torch.testing.assert_close(cross_fused(x, context=c_in), cross_ref(x, context=c_in), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        cross_fused(x, context=c_in, all_perturbed=True),
        cross_ref(x, context=c_in, all_perturbed=True),
        rtol=1e-5,
        atol=1e-5,
    )

    av_ref = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    av_fused = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    av_fused.load_state_dict(av_ref.state_dict())
    stats = av_fused.fuse_projections(fuse_self_qkv=True, fuse_cross_kv=True)
    assert stats == {"fused_qkv": 4, "fused_kv": 8}

    inp = _make_av_input(t_v=16, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128)
    den_ref = LTXJointDenoiser(_as_av_wrapper(av_ref), compute_dtype=torch.float32)
    den_fused = LTXJointDenoiser(_as_av_wrapper(av_fused), compute_dtype=torch.float32)
    out_ref = den_ref(inp).samples
    out_fused = den_fused(inp).samples
    torch.testing.assert_close(out_fused["video"], out_ref["video"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_fused["audio"], out_ref["audio"], rtol=1e-4, atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "mask_values",
    [
        [[[0.0, 0.0], [0.0, 0.0]]],
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[1.0, 0.0, 0.25], [0.5, 1.0, 0.0]]],
    ],
)
def test_ltx_branchless_self_attention_mask_parity(dtype: torch.dtype, mask_values: list[Any]) -> None:
    model = _build_small_video_model(num_layers=1, heads=2, dim_head=64)
    prep = cast(TransformerArgsPreprocessor, model.video_args_preprocessor)
    mask = torch.tensor(mask_values, dtype=torch.float32)
    actual = prep._prepare_self_attention_mask(mask, dtype)
    assert actual is not None
    finfo = torch.finfo(dtype)
    expected = torch.full((mask.shape[0], 1, mask.shape[1], mask.shape[2]), finfo.min, dtype=dtype)
    positive = mask > 0
    if positive.any():
        expected[:, 0][positive] = torch.log(mask[positive].clamp(min=finfo.tiny)).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert prep._prepare_self_attention_mask(None, dtype) is None


@pytest.mark.unit
def test_ltx_acceleration_policy_manifest_and_validation() -> None:
    v_model = _build_small_video_model(num_layers=2, heads=2, dim_head=128)
    wrapper = _as_video_wrapper(v_model)
    policy = RuntimePolicy(
        device=torch.device("cpu"),
        dtype=torch.float32,
        options={"accelerations": {"attention_policy": {"self": "torch", "cross": "torch"}}},
    )
    context = ComponentBuildContext(
        model_id="ltx-test",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=policy,
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper, context)
    denoiser = LTXVideoDenoiser(wrapper, compute_dtype=torch.float32)

    # Must satisfy NativeDiffusionAssembler's post-build validation contract.
    validate_acceleration_installation(denoiser, context)
    report = denoiser.runtime_optimization_report()
    assert report["accelerations"]["installed"][0]["name"] == "attention_policy"
    assert report["accelerations"]["installed"][0]["approximate"] is False
    assert report["accelerations"]["installed"][0]["scopes"]["self"]["configured_modules"] == 2
    assert report["accelerations"]["installed"][0]["scopes"]["cross"]["configured_modules"] == 2

    # Unknown plugin must fail closed both directly and in validate_acceleration_installation.
    bad_ctx = ComponentBuildContext(
        model_id="ltx-test",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={"accelerations": {"unsupported_plugin": True}},
        ),
        checkpoints={},
    )
    with pytest.raises(ValueError, match="does not support requested acceleration plugins"):
        _apply_ltx_post_load_accelerations(wrapper, bad_ctx)
    with pytest.raises(ValueError, match="did not install requested acceleration plugins"):
        validate_acceleration_installation(denoiser, bad_ctx)

    for bad_policy in ({}, {"bad_scope": "torch"}, {"self": {}}):
        ctx_bad_scope = ComponentBuildContext(
            model_id="ltx-test",
            key=ComponentKey(ComponentKind.DENOISER, "main"),
            policy=RuntimePolicy(
                device=torch.device("cpu"),
                dtype=torch.float32,
                options={"accelerations": {"attention_policy": bad_policy}},
            ),
            checkpoints={},
        )
        with pytest.raises(ValueError):
            _apply_ltx_post_load_accelerations(wrapper, ctx_bad_scope)


@pytest.mark.unit
@pytest.mark.parametrize("disabled_mode", [False, "off", "OFF", "false", "none", "disabled"])
def test_ltx_cuda_graph_off_strings_leave_graph_runner_disabled(disabled_mode: bool | str) -> None:
    v_model = _build_small_video_model(num_layers=1, heads=2, dim_head=64)
    v_den = LTXVideoDenoiser(_as_video_wrapper(v_model), compute_dtype=torch.bfloat16, enable_cuda_graph=disabled_mode)
    assert v_den.graph_report() is None
    _release_denoiser(v_den)

    av_model = _build_small_av_model(num_layers=1, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    av_den = LTXJointDenoiser(_as_av_wrapper(av_model), compute_dtype=torch.bfloat16, enable_cuda_graph=disabled_mode)
    assert av_den.graph_report() is None
    _release_denoiser(av_den)


@pytest.mark.unit
@pytest.mark.parametrize("graph_mode", ["whole", "piecewise", True])
def test_ltx_cuda_graph_rejects_dynamic_cudnn_fp8_and_scaled_int8(graph_mode: bool | str) -> None:
    model = _build_small_video_model(num_layers=1, heads=2, dim_head=128)
    model.configure_attention_backend("cudnn_fp8", backend_options={"dynamic_scale": True})
    with pytest.raises(ValueError, match="dynamic_scale=True.*CUDA Graph"):
        LTXVideoDenoiser(_as_video_wrapper(model), compute_dtype=torch.bfloat16, enable_cuda_graph=graph_mode)

    model.configure_attention_backend("scaled_int8_fp8")
    with pytest.raises(ValueError, match="scaled_int8_fp8.*CUDA Graph"):
        LTXVideoDenoiser(_as_video_wrapper(model), compute_dtype=torch.bfloat16, enable_cuda_graph=graph_mode)

    # Static scales and default unit-scale cudnn_fp8 must be accepted.
    model.configure_attention_backend("cudnn_fp8", backend_options={"descale_q": 0.05})
    den = LTXVideoDenoiser(_as_video_wrapper(model), compute_dtype=torch.bfloat16, enable_cuda_graph=graph_mode)
    _release_denoiser(den)


@pytest.mark.unit
def test_ltx_feature_cache_request_isolation_and_contract() -> None:
    model = _build_small_video_model(num_layers=2, heads=2, dim_head=64)
    den = LTXVideoDenoiser(_as_video_wrapper(model), compute_dtype=torch.float32, teacache_threshold=0.25)

    missing_req = _make_video_input(t_v=16, t_txt=4, ctx_dim=128, request_id=None)
    with pytest.raises(ValueError, match="request_id"):
        den(missing_req)

    with torch.inference_mode():
        for step, sigma in enumerate((0.5000, 0.4998, 0.4996, 0.4994, 0.4992)):
            inp_c = _make_video_input(
                t_v=16,
                t_txt=4,
                ctx_dim=128,
                sigma=sigma,
                step_index=step,
                total_steps=5,
                request_id="req-a",
                branch="cond",
                seed=7,
            )
            inp_u = _make_video_input(
                t_v=16,
                t_txt=4,
                ctx_dim=128,
                sigma=sigma,
                step_index=step,
                total_steps=5,
                request_id="req-a",
                branch="uncond",
                seed=19,
            )
            den(inp_c)
            den(inp_u)
    report = den.feature_cache_report("req-a")
    assert report["hits"] > 0
    assert report["events"] == 10
    assert set(report["branches"]) == {"cond", "uncond"}
    den.end_feature_cache_request("req-a")


@pytest.mark.unit
def test_ltx_zero_sync_perturbation_fast_path_parity() -> None:
    av_model = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    inp = _make_av_input(t_v=16, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128)
    video = LTXJointDenoiser._modality(
        inp.modalities["video"],
        sigma=inp.timestep,
        context=cast(torch.Tensor, inp.conditioning["video_context"]),
        context_mask=None,
    )
    audio = LTXJointDenoiser._modality(
        inp.modalities["audio"],
        sigma=inp.timestep,
        context=cast(torch.Tensor, inp.conditioning["audio_context"]),
        context_mask=None,
    )
    with torch.inference_mode():
        v_fast, a_fast = av_model(video=video, audio=audio, perturbations=None)
        v_mat, a_mat = av_model(video=video, audio=audio, perturbations=BatchedPerturbationConfig.empty(1, 2))
    assert v_fast is not None and v_mat is not None
    assert a_fast is not None and a_mat is not None
    torch.testing.assert_close(v_fast, v_mat, rtol=0, atol=0)
    torch.testing.assert_close(a_fast, a_mat, rtol=0, atol=0)


@pytest.mark.unit
def test_ltx_video_vae_channels_last_3d_parity_and_causal_expand() -> None:
    torch.manual_seed(0)
    blocks = [
        ("res_x", {"num_layers": 1, "inject_noise": False}),
        ("compress_all", {"multiplier": 2, "residual": True}),
    ]
    dec_ref = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=list(reversed(blocks)),
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=16,
    ).eval()
    dec_cl = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=list(reversed(blocks)),
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=16,
    ).eval()
    dec_cl.load_state_dict(dec_ref.state_dict())
    dec_ref.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    dec_ref.per_channel_statistics.get_buffer("mean-of-means").zero_()
    dec_cl.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    dec_cl.per_channel_statistics.get_buffer("mean-of-means").zero_()

    converted = dec_cl.enable_channels_last_3d()
    assert converted > 0
    for mod in dec_cl.modules():
        if isinstance(mod, torch.nn.Conv3d):
            assert mod.weight.is_contiguous(memory_format=torch.channels_last_3d)

    lat = torch.randn(1, 16, 2, 4, 4)
    with torch.inference_mode():
        y_ref = dec_ref(lat)
        y_cl = dec_cl(lat)
    torch.testing.assert_close(y_cl, y_ref, rtol=1e-4, atol=1e-4)


# =====================================================================
# GPU TESTS (CUDA_VISIBLE_DEVICES=0)
# =====================================================================


_CUDA_AVAILABLE = torch.cuda.is_available()
_HOPPER_PLUS = _CUDA_AVAILABLE and torch.cuda.get_device_capability()[0] >= 9
_CUDNN_FP8_USABLE = _CUDA_AVAILABLE and probe_attention_backends("cuda")["cudnn_fp8"].usable
_SOL_ATTN_USABLE = _CUDA_AVAILABLE and probe_attention_backends("cuda")["sol_attn"].usable
_FA3_USABLE = _CUDA_AVAILABLE and probe_attention_backends("cuda")["flash_attention_3"].usable


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
@pytest.mark.parametrize("rope_type", [LTXRopeType.SPLIT, LTXRopeType.INTERLEAVED])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,dim_head", [(8, 128), (12, 64)])
def test_gpu_fused_qk_rmsnorm_rope_parity(
    rope_type: LTXRopeType,
    dtype: torch.dtype,
    heads: int,
    dim_head: int,
) -> None:
    device = torch.device("cuda")
    inner_dim = heads * dim_head
    torch.manual_seed(123)

    # Self-attention (paired launch, q_pe == k_pe)
    q = torch.randn((2, 64, inner_dim), device=device, dtype=dtype)
    k = torch.randn((2, 64, inner_dim), device=device, dtype=dtype)
    q_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6).to(device=device, dtype=dtype)
    k_norm = torch.nn.RMSNorm(inner_dim, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.uniform_(0.5, 1.5)
        k_norm.weight.uniform_(0.5, 1.5)

    pos = torch.rand((2, 3, 64, 2), device=device, dtype=torch.float32) * 16.0
    pe = precompute_freqs_cis(
        indices_grid=pos,
        dim=inner_dim,
        out_dtype=dtype,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=heads,
        rope_type=rope_type,
    )
    q_ref = apply_rotary_emb(q_norm(q), pe, rope_type)
    k_ref = apply_rotary_emb(k_norm(k), pe, rope_type)
    q_got, k_got = fused_ltx_qk_rmsnorm_rope(
        q,
        k,
        q_norm.weight,
        k_norm.weight,
        pe,
        None,
        eps=1e-6,
        heads=heads,
        dim_head=dim_head,
        rope_type=rope_type,
    )
    assert _rel_l2(q_got, q_ref) < 5e-3
    assert _rel_l2(k_got, k_ref) < 5e-3

    # Cross-attention (separate k_pe and distinct K sequence length)
    k_short = torch.randn((2, 24, inner_dim), device=device, dtype=dtype)
    k_pos = torch.rand((2, 1, 24, 2), device=device, dtype=torch.float32) * 16.0
    k_pe = precompute_freqs_cis(
        indices_grid=k_pos,
        dim=inner_dim,
        out_dtype=dtype,
        theta=10000.0,
        max_pos=[20],
        num_attention_heads=heads,
        rope_type=rope_type,
    )
    k_short_ref = apply_rotary_emb(k_norm(k_short), k_pe, rope_type)
    q_got2, k_got2 = fused_ltx_qk_rmsnorm_rope(
        q,
        k_short,
        q_norm.weight,
        k_norm.weight,
        pe,
        k_pe,
        eps=1e-6,
        heads=heads,
        dim_head=dim_head,
        rope_type=rope_type,
    )
    assert _rel_l2(q_got2, q_ref) < 5e-3
    assert _rel_l2(k_got2, k_short_ref) < 5e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
@pytest.mark.parametrize("hidden_dim", [768, 4096])
@pytest.mark.parametrize("gate_seq", [1, 32])
def test_gpu_fused_post_sa_residual_rms_norm_exact_parity(hidden_dim: int, gate_seq: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(7)
    residual = torch.randn((2, 32, hidden_dim), device=device, dtype=torch.bfloat16)
    value = torch.randn((2, 32, hidden_dim), device=device, dtype=torch.bfloat16)
    gate = torch.randn((2, gate_seq, hidden_dim), device=device, dtype=torch.bfloat16)

    out_ref = residual + value * gate
    norm_ref = torch.nn.functional.rms_norm(out_ref, (hidden_dim,), None, eps=1e-6)
    out_got, norm_got = fused_ltx_post_sa_residual_rms_norm(residual, value, gate, eps=1e-6)

    torch.testing.assert_close(out_got, out_ref, rtol=0, atol=0)
    assert _rel_l2(norm_got, norm_ref) < 2e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="FP8 W8A8 requires Hopper+ CUDA GPU")
def test_gpu_fp8_w8a8_and_fused_fp8_ffn_execution() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    m_ref = _build_small_video_model(num_layers=2, heads=8, dim_head=128, device=device, dtype=dtype)
    m_fp8 = _build_small_video_model(num_layers=2, heads=8, dim_head=128, device=device, dtype=dtype)
    m_fp8.load_state_dict(m_ref.state_dict())
    m_fp8.fuse_projections(fuse_self_qkv=True, fuse_cross_kv=True)
    m_fp8.enable_fused_ops()
    stats = enable_ltx_fp8_quantization(
        m_fp8,
        min_features=512,
        scaling="rowwise",
        use_fast_accum=True,
        keep_dense_fallback=False,
        fuse_fp8_ffn=True,
    )
    assert stats["replaced_linears"] > 0
    assert stats["fused_fp8_ffn"] == 2

    inp = _make_video_input(t_v=64, t_txt=16, ctx_dim=1024, device=device, dtype=dtype)
    den_ref = LTXVideoDenoiser(_as_video_wrapper(m_ref), compute_dtype=dtype)
    den_fp8 = LTXVideoDenoiser(_as_video_wrapper(m_fp8), compute_dtype=dtype)
    with torch.inference_mode():
        y_ref = den_ref(inp).samples["video"]
        y_fp8 = den_fp8(inp).samples["video"]

    fp8_calls = sum(
        int(getattr(mod, "low_precision_kernel_calls", 0)) for mod in m_fp8.modules() if isinstance(mod, Float8Linear)
    )
    dense_fallbacks = sum(
        int(getattr(mod, "dense_fallback_calls", 0)) for mod in m_fp8.modules() if isinstance(mod, Float8Linear)
    )
    fused_ffn_calls = sum(
        int(getattr(mod, "fused_fp8_calls", 0)) for mod in m_fp8.modules() if isinstance(mod, FeedForward)
    )
    fallback_ffn_calls = sum(
        int(getattr(mod, "fallback_fp8_calls", 0)) for mod in m_fp8.modules() if isinstance(mod, FeedForward)
    )
    assert fp8_calls > 0
    assert dense_fallbacks == 0
    assert fused_ffn_calls == 2
    assert fallback_ffn_calls == 0
    assert _rel_l2(y_fp8, y_ref) < 0.08
    _release_denoiser(den_ref)
    _release_denoiser(den_fp8)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_whole_cuda_graph_capture_replay_and_output_independence() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # VideoOnly whole CUDA Graph
    m_v = _build_small_video_model(num_layers=2, heads=4, dim_head=128, device=device, dtype=dtype)
    m_v.fuse_projections()
    m_v.enable_fused_ops()
    w_v = _as_video_wrapper(m_v)
    inp1 = _make_video_input(t_v=64, t_txt=16, ctx_dim=512, seed=1, device=device, dtype=dtype)
    inp2 = _make_video_input(t_v=64, t_txt=16, ctx_dim=512, seed=2, device=device, dtype=dtype)

    den_eager = LTXVideoDenoiser(w_v, compute_dtype=dtype, enable_cuda_graph=False)
    with torch.inference_mode():
        ref1 = den_eager(inp1).samples["video"].clone()
        ref2 = den_eager(inp2).samples["video"].clone()
    _release_denoiser(den_eager)
    del den_eager

    den_cg = LTXVideoDenoiser(w_v, compute_dtype=dtype, enable_cuda_graph="whole")
    with torch.inference_mode():
        out1 = den_cg(inp1).samples["video"]
        out2 = den_cg(inp2).samples["video"]
        out3 = den_cg(inp1).samples["video"]
    rep = den_cg.graph_report()
    assert rep is not None
    assert rep["lifetime"]["capture"] == 1
    assert rep["lifetime"]["replay"] == 3
    assert rep["lifetime"]["capture_failed"] == 0
    assert _rel_l2(out1, ref1) < 1e-4
    assert _rel_l2(out2, ref2) < 1e-4
    assert _rel_l2(out3, ref1) < 1e-4
    # Calling replay with inp2 must not alias/overwrite out1.
    assert not torch.allclose(out1, out2)
    _release_denoiser(den_cg)

    # Joint AudioVideo whole CUDA Graph
    m_av = _build_small_av_model(num_layers=2, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    m_av.fuse_projections()
    m_av.enable_fused_ops()
    w_av = _as_av_wrapper(m_av)
    av_inp = _make_av_input(t_v=64, t_a=32, t_txt=16, v_ctx_dim=512, a_ctx_dim=256, device=device, dtype=dtype)
    den_av_e = LTXJointDenoiser(w_av, compute_dtype=dtype, enable_cuda_graph=False)
    with torch.inference_mode():
        ref_av = den_av_e(av_inp).samples
        ref_v, ref_a = ref_av["video"].clone(), ref_av["audio"].clone()
    _release_denoiser(den_av_e)
    del den_av_e

    den_av_cg = LTXJointDenoiser(w_av, compute_dtype=dtype, enable_cuda_graph="whole")
    with torch.inference_mode():
        for _ in range(3):
            out_av = den_av_cg(av_inp).samples
    av_rep = den_av_cg.graph_report()
    assert av_rep is not None
    assert av_rep["lifetime"]["capture"] == 1
    assert av_rep["lifetime"]["replay"] == 3
    assert av_rep["lifetime"]["capture_failed"] == 0
    assert _rel_l2(out_av["video"], ref_v) < 1e-4
    assert _rel_l2(out_av["audio"], ref_a) < 1e-4
    _release_denoiser(den_av_cg)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_piecewise_cuda_graph_and_teacache() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = _build_small_video_model(num_layers=2, heads=4, dim_head=128, device=device, dtype=dtype)
    model.fuse_projections()
    model.enable_fused_ops()
    den = LTXVideoDenoiser(
        _as_video_wrapper(model),
        compute_dtype=dtype,
        enable_cuda_graph="piecewise",
        teacache_threshold=0.25,
    )
    with torch.inference_mode():
        for step, sigma in enumerate((0.5000, 0.4998, 0.4996, 0.4994, 0.4992, 0.4990)):
            inp = _make_video_input(
                t_v=64,
                t_txt=16,
                ctx_dim=512,
                sigma=sigma,
                step_index=step,
                total_steps=6,
                request_id="pw-tc-req",
                device=device,
                dtype=dtype,
            )
            out = den(inp).samples["video"]
            assert torch.isfinite(out).all()
    pw_rep = den.graph_report()
    assert pw_rep is not None
    tc_rep = den.feature_cache_report("pw-tc-req")
    assert pw_rep["capture"] >= 1
    assert pw_rep["capture_failed"] == 0
    assert pw_rep["cache_skips"] > 0
    assert tc_rep["hits"] > 0
    den.end_feature_cache_request("pw-tc-req")
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDNN_FP8_USABLE, reason="cudnn_fp8 is unavailable on this GPU")
def test_gpu_cudnn_fp8_graph_safe_and_dynamic_scale_fail_closed() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = _build_small_video_model(num_layers=2, heads=4, dim_head=128, device=device, dtype=dtype)
    wrapper = _as_video_wrapper(model)
    inp = _make_video_input(t_v=64, t_txt=16, ctx_dim=512, device=device, dtype=dtype)

    den_ref = LTXVideoDenoiser(wrapper, compute_dtype=dtype, enable_cuda_graph=False)
    with torch.inference_mode():
        ref = den_ref(inp).samples["video"].clone()
    _release_denoiser(den_ref)
    del den_ref

    # 1. Default unit-scale cudnn_fp8 + Whole CUDA Graph
    reset_attention_provider_runtime()
    model.configure_attention_backend("cudnn_fp8")
    den_unit = LTXVideoDenoiser(wrapper, compute_dtype=dtype, enable_cuda_graph="whole")
    with torch.inference_mode():
        for _ in range(3):
            out_unit = den_unit(inp).samples["video"]
    rep = den_unit.graph_report()
    assert rep is not None
    assert rep["lifetime"]["capture"] == 1
    assert rep["lifetime"]["replay"] == 3
    assert rep["lifetime"]["capture_failed"] == 0
    assert attention_provider_runtime_report().get("cudnn_fp8", {}).get("successes", 0) > 0
    assert _rel_l2(out_unit, ref) < 0.08
    _release_denoiser(den_unit)
    del den_unit

    # 2. Static calibrated cudnn_fp8 + Whole CUDA Graph (zero host sync + audited provider receipts)
    reset_attention_provider_runtime()
    model.configure_attention_backend(
        "cudnn_fp8",
        backend_options={"descale_q": 0.05, "descale_k": 0.05, "descale_v": 0.05},
    )
    den_static = LTXVideoDenoiser(wrapper, compute_dtype=dtype, enable_cuda_graph="whole")
    with torch.inference_mode():
        for _ in range(3):
            out_static = den_static(inp).samples["video"]
    rep_static = den_static.graph_report()
    assert rep_static is not None
    assert rep_static["lifetime"]["capture"] == 1
    assert rep_static["lifetime"]["replay"] == 3
    assert rep_static["lifetime"]["capture_failed"] == 0
    assert attention_provider_runtime_report().get("cudnn_fp8", {}).get("successes", 0) > 0
    assert _rel_l2(out_static, ref) < 0.08
    _release_denoiser(den_static)
    del den_static

    # 3. dynamic_scale=True works in eager mode with audited receipts and FAILS CLOSED inside CUDA Graph capture.
    reset_attention_provider_runtime()
    attn = Attention(query_dim=512, context_dim=None, heads=4, dim_head=128).to(device=device, dtype=dtype).eval()
    attn.set_attention_backend("cudnn_fp8", backend_options={"dynamic_scale": True})
    x = torch.randn((1, 64, 512), device=device, dtype=dtype)
    with torch.inference_mode():
        eager_out = attn(x)
    assert torch.isfinite(eager_out).all()
    assert attention_provider_runtime_report().get("cudnn_fp8", {}).get("successes", 0) == 1

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="dynamic_scale=True.*CUDA Graph capture"):
        with torch.cuda.graph(graph):
            attn(x)

    # 4. min_tokens keeps short sequences on exact SDPA and routes long sequences to cudnn_fp8.
    reset_attention_provider_runtime()
    attn_thresh = (
        Attention(query_dim=512, context_dim=None, heads=4, dim_head=128).to(device=device, dtype=dtype).eval()
    )
    attn_thresh.set_attention_backend("cudnn_fp8", backend_options={"min_tokens": 96})
    with torch.inference_mode():
        _ = attn_thresh(torch.randn((1, 64, 512), device=device, dtype=dtype))
        assert attention_provider_runtime_report().get("cudnn_fp8", {}).get("successes", 0) == 0
        _ = attn_thresh(torch.randn((1, 128, 512), device=device, dtype=dtype))
        assert attention_provider_runtime_report().get("cudnn_fp8", {}).get("successes", 0) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _SOL_ATTN_USABLE, reason="sol_attn is unavailable on this GPU")
def test_gpu_sol_attn_video_self_with_cuda_graph() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    m_av = _build_small_av_model(num_layers=2, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    n_sol = m_av.configure_attention_backend("sol_attn", self_only=True, backend_options={"tau": 1.0})
    assert n_sol == 2

    reset_attention_provider_runtime()
    inp = _make_av_input(t_v=128, t_a=32, t_txt=16, v_ctx_dim=512, a_ctx_dim=256, device=device, dtype=dtype)
    den = LTXJointDenoiser(_as_av_wrapper(m_av), compute_dtype=dtype, enable_cuda_graph="whole")
    with torch.inference_mode():
        for _ in range(3):
            out = den(inp).samples
    rep = den.graph_report()
    assert rep is not None
    assert rep["lifetime"]["capture"] == 1
    assert rep["lifetime"]["replay"] == 3
    assert rep["lifetime"]["capture_failed"] == 0
    assert attention_provider_runtime_report().get("sol_attn", {}).get("successes", 0) > 0
    assert torch.isfinite(out["video"]).all()
    assert torch.isfinite(out["audio"]).all()
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_regional_compile_with_fused_ops() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    m_ref = _build_small_video_model(num_layers=2, heads=4, dim_head=128, device=device, dtype=dtype)
    m_comp = _build_small_video_model(num_layers=2, heads=4, dim_head=128, device=device, dtype=dtype)
    m_comp.load_state_dict(m_ref.state_dict())
    m_comp.fuse_projections()
    m_comp.enable_fused_ops()
    for idx in range(len(m_comp.transformer_blocks)):
        m_comp.transformer_blocks[idx] = cast(
            torch.nn.Module,
            torch.compile(m_comp.transformer_blocks[idx], mode="default"),
        )

    inp = _make_video_input(t_v=64, t_txt=16, ctx_dim=512, device=device, dtype=dtype)
    den_ref = LTXVideoDenoiser(_as_video_wrapper(m_ref), compute_dtype=dtype)
    den_comp = LTXVideoDenoiser(_as_video_wrapper(m_comp), compute_dtype=dtype)
    with torch.inference_mode():
        y_ref = den_ref(inp).samples["video"]
        y_comp = den_comp(inp).samples["video"]
    assert _rel_l2(y_comp, y_ref) < 2e-2
    _release_denoiser(den_ref)
    _release_denoiser(den_comp)


def _write_tiny_ltx_av_checkpoint(tmp_path: Path) -> CheckpointSpec:
    ckpt_path = tmp_path / "tiny_ltx2_av.safetensors"
    cfg = {
        "transformer": {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
            "num_attention_heads": 8,
            "attention_head_dim": 128,
            "in_channels": 32,
            "out_channels": 32,
            "num_layers": 3,
            "cross_attention_dim": 1024,
            "norm_eps": 1e-6,
            "audio_num_attention_heads": 8,
            "audio_attention_head_dim": 64,
            "audio_in_channels": 32,
            "audio_out_channels": 32,
            "audio_cross_attention_dim": 512,
            "rope_type": "split",
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "use_prompt_adaln_single": False,
            "caption_proj_before_connector": True,
            "ff_bias": False,
            "audio_ff_bias": False,
        }
    }
    ref_mod = LTXAVTransformerModule(cfg).to(dtype=torch.bfloat16)
    _init_deterministic_weights(ref_mod, seed=303)
    prefix = "velocity_model."
    state_dict = {
        f"model.diffusion_model.{k.removeprefix(prefix)}": v.contiguous() for k, v in ref_mod.state_dict().items()
    }
    save_file(state_dict, str(ckpt_path), metadata={"config": json.dumps(cfg)})
    return CheckpointSpec(source=(str(ckpt_path),))


@pytest.mark.unit
def test_ltx_infer_video_grid_shape_and_sol_schedule_validation() -> None:
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(2, dtype=torch.float32),
            torch.arange(4, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, -1)
    pos = torch.stack((grid, grid + 1.0), dim=-1).unsqueeze(0)
    assert infer_video_grid_shape(pos) == (2, 4, 8)
    # Non-raster / appended reference tokens (e.g. IC-LoRA) must safely return None.
    appended = torch.cat((pos, pos[:, :, :4, :]), dim=2)
    assert infer_video_grid_shape(appended) is None

    # Validate dense_blocks + tile_3d + min_tokens option parsing & CPU fail-closed guard.
    v_model = _build_small_video_model(num_layers=3, heads=2, dim_head=128)
    wrapper = _as_video_wrapper(v_model)
    cpu_sol_ctx = ComponentBuildContext(
        model_id="ltx-test",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={"accelerations": {"attention_policy": {"self": {"backend": "sol_attn", "tile_3d": (1, 8, 8)}}}},
        ),
        checkpoints={},
    )
    with pytest.raises((ValueError, RuntimeError), match="BF16|unavailable|Ampere"):
        _apply_ltx_post_load_accelerations(wrapper, cpu_sol_ctx)

    # Validate min_tokens rejection on unsupported backends and invalid values.
    attn = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128)
    with pytest.raises(ValueError, match="min_tokens is only supported"):
        attn.set_attention_backend("torch", backend_options={"min_tokens": 128})
    with pytest.raises(ValueError, match="non-negative integer"):
        attn.set_attention_backend("cudnn_fp8", backend_options={"min_tokens": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        attn.set_attention_backend("cudnn_fp8", backend_options={"min_tokens": True})


@pytest.mark.unit
def test_ltx_build_cross_kv_only_and_honest_fusion_receipts(tmp_path: Path) -> None:
    ckpt = _write_tiny_ltx_av_checkpoint(tmp_path)
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={"fuse_qkv": False, "fuse_cross_kv": True},
        ),
        checkpoints={"weights": ckpt},
    )
    den = build_ltx_joint_denoiser(ctx)
    blk0 = cast(BasicAVTransformerBlock, den.model.velocity_model.transformer_blocks[0])
    assert not hasattr(blk0.attn1, "to_qkv")
    assert hasattr(blk0.attn2, "to_kv")
    snap = den.runtime_optimization_report()["applied_optimizations"]
    assert snap is not None
    assert snap.effective["fuse_cross_kv_blocks"] == 12
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="FP8 W8A8 requires Hopper+ CUDA GPU")
def test_gpu_build_ltx_default_fp8_excludes_and_regional_compile(tmp_path: Path) -> None:
    device = torch.device("cuda")
    ckpt = _write_tiny_ltx_av_checkpoint(tmp_path)
    # No explicit exclude= passed — build_ltx_joint_denoiser must auto-exclude AdaLN/stem/head!
    policy = RuntimePolicy(
        device=device,
        dtype=torch.bfloat16,
        compile=True,
        quantization=QuantizationPolicy(
            mode=QuantizationMode.FP8,
            options={"scaling": "rowwise", "min_features": 512, "fuse_fp8_ffn": True, "keep_dense_fallback": False},
        ),
        options={
            "compile_scope": "regional",
            "fuse_qkv": True,
            "ltx_fused_ops": True,
            "accelerations": {"attention_policy": {"self": "torch"}},
        },
    )
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=policy,
        checkpoints={"weights": ckpt},
    )
    den = build_ltx_joint_denoiser(ctx)
    validate_acceleration_installation(den, ctx)
    # Timestep AdaLN / stem / head must never be quantized.
    fp8_names = [name for name, mod in den.model.named_modules() if isinstance(mod, Float8Linear)]
    assert len(fp8_names) > 0
    assert all("adaln" not in name and "patchify_proj" not in name and "proj_out" not in name for name in fp8_names)
    snap = den.runtime_optimization_report()["applied_optimizations"]
    assert snap is not None
    assert snap.effective["fuse_qkv_blocks"] == 6
    assert snap.effective["fuse_cross_kv_blocks"] == 12
    assert snap.effective["fused_fp8_ffn_blocks"] == 6

    inp = _make_av_input(t_v=64, t_a=32, t_txt=16, v_ctx_dim=1024, a_ctx_dim=512, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = den(inp).samples
    assert torch.isfinite(out["video"]).all()
    assert torch.isfinite(out["audio"]).all()
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not _SOL_ATTN_USABLE, reason="sol_attn is unavailable on this GPU")
def test_gpu_sol_attn_fp32_autocast_and_tile_3d_schedule_with_cuda_graph() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Regression: FP32 latent under BF16 autocast WITHOUT ltx_fused_ops must not crash Sol-Attn.
    m_av = _build_small_av_model(num_layers=3, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    wrapper = _as_av_wrapper(m_av)
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            options={
                "accelerations": {
                    "attention_policy": {
                        "self": {
                            "backend": "sol_attn",
                            "tau": 0.5,
                            "tile_3d": (2, 4, 4),
                            "dense_blocks": [0, -1],
                            "dense_steps": 1,
                            "min_tokens": 96,
                        }
                    }
                }
            },
        ),
        checkpoints={},
    )
    # Reject non-BF16 Sol-Attn upfront on CUDA as well as invalid tile_3d shapes.
    bad_dtype_ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=torch.float16,
            options={"accelerations": {"attention_policy": {"self": {"backend": "sol_attn"}}}},
        ),
        checkpoints={},
    )
    with pytest.raises(ValueError, match="BF16"):
        _apply_ltx_post_load_accelerations(wrapper, bad_dtype_ctx)

    bad_tile_ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            options={"accelerations": {"attention_policy": {"self": {"backend": "sol_attn", "tile_3d": (0, 8, 8)}}}},
        ),
        checkpoints={},
    )
    with pytest.raises(ValueError, match="three positive integers"):
        _apply_ltx_post_load_accelerations(wrapper, bad_tile_ctx)

    _apply_ltx_post_load_accelerations(wrapper, ctx)
    den = LTXJointDenoiser(wrapper, compute_dtype=dtype, precision="autocast", enable_cuda_graph="whole")
    validate_acceleration_installation(den, ctx)

    # Confirm dense_blocks=[0, -1] left block 0 and block 2 on torch, and block 1 on sol_attn.
    blocks = [cast(BasicAVTransformerBlock, b) for b in m_av.transformer_blocks]
    assert blocks[0].attn1.attention_backend == "torch"
    assert blocks[1].attn1.attention_backend == "sol_attn"
    assert blocks[2].attn1.attention_backend == "torch"

    # Build FP32 latent input INSIDE inference_mode (inference tensors lack _version)
    # with 2x8x8 = 128 tokens >= min_tokens=96 and mixed I2V denoise_mask.
    g = torch.Generator(device="cpu").manual_seed(17)
    with torch.inference_mode():
        v_lat = torch.randn((1, 128, 32), generator=g).to(device=device, dtype=torch.float32)
        v_mask = torch.ones((1, 128, 1), device=device, dtype=torch.float32)
        v_mask[:, :16] = 0.0
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(2, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
                indexing="ij",
            ),
            dim=0,
        ).reshape(3, -1)
        v_pos = torch.stack((grid, grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
        assert v_pos.is_inference()
        v_ctx = torch.randn((1, 16, 512), generator=g).to(device=device, dtype=torch.float32)
        a_lat = torch.randn((1, 32, 32), generator=g).to(device=device, dtype=torch.float32)
        a_mask = torch.ones((1, 32, 1), device=device, dtype=torch.float32)
        a_grid = torch.arange(32, dtype=torch.float32).reshape(1, -1)
        a_pos = torch.stack((a_grid, a_grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
        a_ctx = torch.randn((1, 16, 256), generator=g).to(device=device, dtype=torch.float32)

    def _step_input(step: int) -> MultiModalDenoiserInput:
        return MultiModalDenoiserInput(
            modalities={
                "video": ModalityState(latent=v_lat, clean_latent=v_lat, denoise_mask=v_mask, positions=v_pos),
                "audio": ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos),
            },
            timestep=torch.tensor([0.5], device=device, dtype=torch.float32),
            conditioning={"video_context": v_ctx, "audio_context": a_ctx},
            step_index=step,
            total_steps=4,
        )

    reset_attention_provider_runtime()
    with torch.inference_mode():
        # Step 0: dense_steps=1 -> forces dense (0 sol_attn calls)
        out0 = den(_step_input(0)).samples
        assert attention_provider_runtime_report().get("sol_attn", {}).get("successes", 0) == 0
        # Steps 1..3: tiled Sol-Attn captured & replayed in CUDA Graph!
        for s in (1, 2, 3):
            out_s = den(_step_input(s)).samples
    assert torch.isfinite(out0["video"]).all()
    assert torch.isfinite(out_s["video"]).all()
    assert attention_provider_runtime_report().get("sol_attn", {}).get("successes", 0) > 0
    rep = den.graph_report()
    assert rep is not None
    assert rep["lifetime"]["capture"] == 2  # 1 dense graph + 1 tiled Sol-Attn graph
    assert rep["lifetime"]["capture_failed"] == 0
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_fused_post_sa_preserves_nonfinite_specials() -> None:
    device = torch.device("cuda")
    res = torch.zeros((1, 1, 16), device=device, dtype=torch.bfloat16)
    val = torch.ones((1, 1, 16), device=device, dtype=torch.bfloat16)
    gate = torch.ones((1, 1, 16), device=device, dtype=torch.bfloat16)
    val[0, 0, 0] = float("inf")
    val[0, 0, 1] = float("-inf")
    out, _ = fused_ltx_post_sa_residual_rms_norm(res, val, gate, eps=1e-6)
    assert torch.isposinf(out[0, 0, 0])
    assert torch.isneginf(out[0, 0, 1])


# =====================================================================
# WAVE 2 UNIT + GPU TESTS
# =====================================================================


@pytest.mark.unit
def test_ltx_uniform_denoise_mask_and_trivial_context_mask_parity() -> None:
    """Uniform denoise masks and no-op context masks optimize away; non-uniform / padded masks stay exact."""
    av_model = _build_small_av_model(num_layers=2, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    den = LTXJointDenoiser(_as_av_wrapper(av_model), compute_dtype=torch.float32)

    # 1. Uniform all-1s denoise_mask + all-1s int context_mask must match explicit uncollapsed forward exactly.
    base_inp = _make_av_input(t_v=16, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128, sigma=0.75)
    v_state = base_inp.modalities["video"]
    a_state = base_inp.modalities["audio"]
    v_ctx = cast(torch.Tensor, base_inp.conditioning["video_context"])
    a_ctx = cast(torch.Tensor, base_inp.conditioning["audio_context"])
    all_ones_mask = torch.ones((1, 6), dtype=torch.int64)
    all_zeros_float_mask = torch.zeros((1, 1, 16, 6), dtype=torch.float32)

    with torch.inference_mode():
        # Reference bypassing denoiser mask collapse directly on LTXModel
        v_mod_ref = LTXJointDenoiser._modality(
            v_state, sigma=base_inp.timestep, context=v_ctx, context_mask=None, uniform_denoise_mask=False
        )
        a_mod_ref = LTXJointDenoiser._modality(
            a_state, sigma=base_inp.timestep, context=a_ctx, context_mask=None, uniform_denoise_mask=False
        )
        v_vel_ref, a_vel_ref = av_model(v_mod_ref, a_mod_ref, None)
        assert v_vel_ref is not None and a_vel_ref is not None
        ref_v = v_state.latent - v_mod_ref.timesteps * v_vel_ref
        ref_a = a_state.latent - a_mod_ref.timesteps * a_vel_ref

        inp_triv = MultiModalDenoiserInput(
            modalities=base_inp.modalities,
            timestep=base_inp.timestep,
            conditioning={
                "video_context": v_ctx,
                "audio_context": a_ctx,
                "video_context_mask": all_ones_mask,
                "audio_context_mask": all_zeros_float_mask[:, :, :8, :],
            },
            step_index=0,
            total_steps=4,
        )
        got = den(inp_triv).samples
    torch.testing.assert_close(got["video"], ref_v, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got["audio"], ref_a, rtol=1e-5, atol=1e-5)

    # 2. Non-uniform I2V denoise_mask must NOT collapse and must match full per-token reference.
    i2v_mask = torch.ones((1, 16, 1), dtype=torch.float32)
    i2v_mask[:, :4, :] = 0.0
    i2v_state = ModalityState(
        latent=v_state.latent, clean_latent=v_state.clean_latent, denoise_mask=i2v_mask, positions=v_state.positions
    )
    padded_ctx_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int64)
    with torch.inference_mode():
        v_mod_i2v = LTXJointDenoiser._modality(
            i2v_state, sigma=base_inp.timestep, context=v_ctx, context_mask=padded_ctx_mask, uniform_denoise_mask=False
        )
        v_vel_i2v, a_vel_i2v = av_model(v_mod_i2v, a_mod_ref, None)
        assert v_vel_i2v is not None and a_vel_i2v is not None
        ref_v_i2v = i2v_state.latent - v_mod_i2v.timesteps * v_vel_i2v
        ref_a_i2v = a_state.latent - a_mod_ref.timesteps * a_vel_i2v

        inp_i2v = MultiModalDenoiserInput(
            modalities={"video": i2v_state, "audio": a_state},
            timestep=base_inp.timestep,
            conditioning={
                "video_context": v_ctx,
                "audio_context": a_ctx,
                "video_context_mask": padded_ctx_mask,
            },
            step_index=0,
            total_steps=4,
        )
        got_i2v = den(inp_i2v).samples
    torch.testing.assert_close(got_i2v["video"], ref_v_i2v, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_i2v["audio"], ref_a_i2v, rtol=1e-5, atol=1e-5)
    # Verify non-trivial mask genuinely changed output vs unmasked.
    assert not torch.allclose(got_i2v["video"], ref_v, atol=1e-3)


@pytest.mark.unit
def test_ltx_dense_backend_preservation_and_teacache_max_tokens() -> None:
    # 1. dense_backend option on configure_attention_backend preserves edge-block & fallback backend.
    v_model = _build_small_video_model(num_layers=3, heads=2, dim_head=128)
    n_cfg = v_model.configure_attention_backend(
        "sol_attn",
        self_only=True,
        dense_blocks=(0, -1),
        dense_backend="torch",
        backend_options={"tau": 0.5, "min_tokens": 64},
    )
    assert n_cfg == 1
    blocks = [cast(BasicAVTransformerBlock, b) for b in v_model.transformer_blocks]
    assert blocks[0].attn1.attention_backend == "torch"
    assert blocks[1].attn1.attention_backend == "sol_attn"
    assert blocks[1].attn1._dense_attention_backend == "torch"
    assert blocks[2].attn1.attention_backend == "torch"

    # 2. teacache_max_tokens activates TeaCache on Stage-1 (<= max_tokens) and bypasses Stage-2 (> max_tokens).
    small_v = _build_small_video_model(num_layers=2, heads=2, dim_head=64)
    den = LTXVideoDenoiser(
        _as_video_wrapper(small_v),
        compute_dtype=torch.float32,
        teacache_threshold=0.25,
        teacache_max_tokens=32,
    )
    with torch.inference_mode():
        # Stage 1: t_v=16 <= 32 -> TeaCache runs & records hits
        for step, sigma in enumerate((0.5000, 0.4998, 0.4996, 0.4994)):
            den(
                _make_video_input(
                    t_v=16,
                    t_txt=4,
                    ctx_dim=128,
                    sigma=sigma,
                    step_index=step,
                    total_steps=4,
                    request_id="stage-gate-req",
                )
            )
        rep_after_s1 = den.feature_cache_report("stage-gate-req")
        assert rep_after_s1["events"] == 4
        assert rep_after_s1["hits"] > 0

        # Stage 2: t_v=64 > 32 -> TeaCache bypassed cleanly (no extra events, does not even need request_id)
        for step, sigma in enumerate((0.4, 0.2)):
            out_s2 = den(
                _make_video_input(
                    t_v=64,
                    t_txt=4,
                    ctx_dim=128,
                    sigma=sigma,
                    step_index=step,
                    total_steps=2,
                    request_id=None,
                )
            ).samples["video"]
            assert torch.isfinite(out_s2).all()
        rep_after_s2 = den.feature_cache_report("stage-gate-req")
        assert rep_after_s2["events"] == 4
    den.end_feature_cache_request("stage-gate-req")


@pytest.mark.unit
def test_ltx_video_vae_enable_compile_guards_and_tiling_policy() -> None:
    blocks: list[tuple[str, int | dict[str, Any]]] = [("res_x", {"num_layers": 1, "inject_noise": False})]
    stochastic_ts = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=True,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=16,
    )
    assert stochastic_ts.enable_compile(mode="default") is False

    noise_blocks: list[tuple[str, int | dict[str, Any]]] = [("res_x", {"num_layers": 1, "inject_noise": True})]
    stochastic_noise = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=noise_blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=16,
    )
    assert stochastic_noise.enable_compile(mode="default") is False

    det_dec = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=16,
    )
    with pytest.raises(ValueError, match="unsupported VideoDecoder compile mode"):
        det_dec.enable_compile(mode="invalid-mode")

    # Policy vae_tiled=False overrides recipe tiled=True; compile mode resolves strings/bools cleanly.
    ctx_untiled = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DECODER, "media"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            options={"vae_tiled": False, "vae_compile": "default"},
        ),
        checkpoints={},
        component_options={"tiled": True, "spatial_tile_size": 768},
    )
    assert _resolve_vae_tiling(ctx_untiled) is None
    assert _resolve_vae_compile_mode(ctx_untiled) == "default"

    ctx_tiled = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DECODER, "media"),
        policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.bfloat16, options={"vae_compile": "off"}),
        checkpoints={},
        component_options={"tiled": True, "spatial_tile_size": 768},
    )
    assert _resolve_vae_tiling(ctx_tiled) is not None
    assert _resolve_vae_compile_mode(ctx_tiled) is None
    assert LTXMediaDecoder is not None
    assert LTXVideoDecoderModule is not None
    assert LTXAudioDecoderModule is not None
    assert LTXVocoderModule is not None


@pytest.mark.unit
def test_ltx_wave2_fused_ops_cpu_and_grad_fallbacks() -> None:
    torch.manual_seed(0)
    res = torch.randn((2, 8, 64), requires_grad=True)
    val = torch.randn((2, 8, 64))
    gate = torch.randn((2, 1, 64))
    scale = torch.randn((2, 1, 64))
    shift = torch.randn((2, 1, 64))
    out, mod = fused_ltx_post_sa_residual_rmsnorm_adaln(res, val, gate, scale, shift, eps=1e-6)
    expected_out = res + val * gate
    expected_mod = torch.nn.functional.rms_norm(expected_out, (64,), None, eps=1e-6) * (1 + scale) + shift
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mod, expected_mod, rtol=1e-5, atol=1e-5)

    attn_out = torch.randn((2, 8, 64))
    gate_logits = torch.randn((2, 8, 4))
    gated = fused_ltx_gated_attention(attn_out, gate_logits, heads=4, dim_head=16)
    expected_gated = (attn_out.view(2, 8, 4, 16) * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)).view(2, 8, 64)
    torch.testing.assert_close(gated, expected_gated, rtol=1e-5, atol=1e-5)
    assert fused_ltx_gated_attention_fp8(attn_out, gate_logits, heads=4, dim_head=16) is None
    assert fused_ltx_rmsnorm_scale_shift_fp8(attn_out, scale, shift, eps=1e-6) is None

    scale2 = torch.randn((2, 1, 64))
    shift2 = torch.randn((2, 1, 64))
    d1, d2 = fused_ltx_dual_rmsnorm_scale_shift(attn_out, scale, shift, scale2, shift2, eps=1e-6)
    normed = torch.nn.functional.rms_norm(attn_out, (64,), None, eps=1e-6)
    torch.testing.assert_close(d1, normed * (1 + scale) + shift, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(d2, normed * (1 + scale2) + shift2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_ltx_scale_shift(attn_out, scale, shift), attn_out * (1 + scale) + shift)

    qw = torch.ones(64)
    kw = torch.full((64,), 1.25)
    k_in = torch.randn((2, 5, 64))
    qn, kn = fused_ltx_qk_rmsnorm(attn_out, k_in, qw, kw, eps=1e-6)
    torch.testing.assert_close(qn, torch.nn.functional.rms_norm(attn_out, (64,), qw, eps=1e-6))
    torch.testing.assert_close(kn, torch.nn.functional.rms_norm(k_in, (64,), kw, eps=1e-6))


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_dim", [768, 2048, 4096])
def test_gpu_wave2_fused_triton_ops_parity_and_edge_cases(dtype: torch.dtype, hidden_dim: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(42)

    # 1. fused_ltx_post_sa_residual_rmsnorm_adaln (broadcast seq=1 and per-token seq=16)
    for mod_seq in (1, 16):
        res = torch.randn((2, 16, hidden_dim), device=device, dtype=dtype)
        val = torch.randn((2, 16, hidden_dim), device=device, dtype=dtype)
        gate = torch.randn((2, mod_seq, hidden_dim), device=device, dtype=dtype)
        scale_q = torch.randn((2, mod_seq, hidden_dim), device=device, dtype=dtype)
        shift_q = torch.randn((2, mod_seq, hidden_dim), device=device, dtype=dtype)
        out_got, mod_got = fused_ltx_post_sa_residual_rmsnorm_adaln(res, val, gate, scale_q, shift_q, eps=1e-6)
        out_ref = res + val * gate
        mod_ref = torch.nn.functional.rms_norm(out_ref, (hidden_dim,), None, eps=1e-6) * (1 + scale_q) + shift_q
        if dtype == torch.bfloat16:
            torch.testing.assert_close(out_got, out_ref, rtol=0, atol=0)
        else:
            assert _rel_l2(out_got, out_ref) < 2e-3
        assert _rel_l2(mod_got, mod_ref) < 5e-3

    # 2. fused_ltx_gated_attention & fused_ltx_gated_attention_fp8
    dim_head = 64
    heads = hidden_dim // dim_head
    x = torch.randn((2, 16, hidden_dim), device=device, dtype=dtype)
    gate_logits = torch.randn((2, 16, heads), device=device, dtype=dtype)
    gated_got = fused_ltx_gated_attention(x, gate_logits, heads=heads, dim_head=dim_head)
    gated_ref = (x.view(2, 16, heads, dim_head) * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)).view(
        2, 16, hidden_dim
    )
    assert _rel_l2(gated_got, gated_ref) < 3e-3

    # Include an all-zero token row to verify zero-scale clamping safety.
    x_zero = x.clone()
    x_zero[0, 0].zero_()
    fp8_gated = fused_ltx_gated_attention_fp8(x_zero, gate_logits, heads=heads, dim_head=dim_head)
    assert fp8_gated is not None
    q_fp8, s_fp8 = fp8_gated
    assert torch.isfinite(q_fp8.float()).all() and torch.isfinite(s_fp8).all()
    deq_gated = (q_fp8.float() * s_fp8).view_as(x_zero)
    gated_zero_ref = (x_zero.view(2, 16, heads, dim_head) * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)).view(
        2, 16, hidden_dim
    )
    assert _rel_l2(deq_gated, gated_zero_ref) < 0.05

    # 3. fused_ltx_rmsnorm_scale_shift_fp8 (seq=1 and seq=T + zero row)
    scale = torch.randn((2, 1, hidden_dim), device=device, dtype=dtype)
    shift = torch.randn((2, 1, hidden_dim), device=device, dtype=dtype)
    fp8_norm = fused_ltx_rmsnorm_scale_shift_fp8(x_zero, scale, shift, eps=1e-6)
    assert fp8_norm is not None
    qn_fp8, sn_fp8 = fp8_norm
    assert torch.isfinite(qn_fp8.float()).all() and torch.isfinite(sn_fp8).all()
    norm_ref = torch.nn.functional.rms_norm(x_zero, (hidden_dim,), None, eps=1e-6) * (1 + scale) + shift
    assert _rel_l2((qn_fp8.float() * sn_fp8).view_as(x_zero), norm_ref) < 0.05

    # 4. fused_ltx_dual_rmsnorm_scale_shift & fused_ltx_scale_shift & fused_ltx_qk_rmsnorm
    scale2 = torch.randn((2, 16, hidden_dim), device=device, dtype=dtype)
    shift2 = torch.randn((2, 16, hidden_dim), device=device, dtype=dtype)
    d1, d2 = fused_ltx_dual_rmsnorm_scale_shift(x, scale, shift, scale2, shift2, eps=1e-6)
    xn = torch.nn.functional.rms_norm(x, (hidden_dim,), None, eps=1e-6)
    assert _rel_l2(d1, xn * (1 + scale) + shift) < 5e-3
    assert _rel_l2(d2, xn * (1 + scale2) + shift2) < 5e-3
    assert _rel_l2(fused_ltx_scale_shift(x, scale2, shift2), x * (1 + scale2) + shift2) < 5e-3

    qw = torch.empty((hidden_dim,), device=device, dtype=dtype).uniform_(0.5, 1.5)
    kw = torch.empty((hidden_dim,), device=device, dtype=dtype).uniform_(0.5, 1.5)
    k_txt = torch.randn((2, 12, hidden_dim), device=device, dtype=dtype)
    q_n, k_n = fused_ltx_qk_rmsnorm(x, k_txt, qw, kw, eps=1e-6)
    assert _rel_l2(q_n, torch.nn.functional.rms_norm(x, (hidden_dim,), qw, eps=1e-6)) < 3e-3
    assert _rel_l2(k_n, torch.nn.functional.rms_norm(k_txt, (hidden_dim,), kw, eps=1e-6)) < 3e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="FP8 GELU-tanh quantization test requires Hopper+ CUDA GPU")
@pytest.mark.parametrize("cols", [1024, 8192, 16384])
def test_gpu_fused_gelu_tanh_rowwise_fp8_parity_and_edge_rows(cols: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(9)
    # Row 0: normal positive/negative (hits monotone fast path for cols > 4096)
    # Row 1: strictly negative (non-monotone GELU minimum around -0.75 -> hits exact fallback pass)
    # Row 2: all zeros (scale clamp safety)
    # Row 3: extreme positive near bf16 max (overflow guard -> hits exact fallback pass)
    x = torch.randn((4, cols), device=device, dtype=torch.bfloat16)
    x[1] = -torch.rand((cols,), device=device, dtype=torch.bfloat16) * 2.0 - 0.1
    x[2].zero_()
    x[3] = torch.randn((cols,), device=device, dtype=torch.bfloat16)
    x[3, 0] = 1e35

    q_got, s_got = fused_ltx_gelu_tanh_rowwise_fp8(x, torch.float8_e4m3fn)
    gelu_ref = torch.nn.functional.gelu(x.float(), approximate="tanh")
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    expected_scales = (gelu_ref.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(
        float(torch.finfo(torch.float32).tiny)
    )

    assert torch.isfinite(s_got[:3]).all()
    torch.testing.assert_close(s_got[:3], expected_scales[:3], rtol=1e-3, atol=1e-6)
    assert _rel_l2((q_got[:2].float() * s_got[:2]), gelu_ref[:2]) < 0.05
    assert torch.all(q_got[2].float() == 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not (_FA3_USABLE and _SOL_ATTN_USABLE),
    reason="flash_attention_3 and sol_attn are both required on this GPU",
)
def test_gpu_fa3_dense_and_sol_attn_kv_splits2_with_cuda_graph() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Pure flash_attention_3 parity vs torch SDPA on AV model (self + text cross + AV cross)
    m_ref = _build_small_av_model(num_layers=2, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    m_fa3 = _build_small_av_model(num_layers=2, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    m_fa3.load_state_dict(m_ref.state_dict())
    m_fa3.fuse_projections(fuse_self_qkv=True, fuse_cross_kv=True)
    m_fa3.enable_fused_ops()
    m_fa3.configure_attention_backend("flash_attention_3")

    inp_s1 = _make_av_input(t_v=64, t_a=32, t_txt=16, v_ctx_dim=512, a_ctx_dim=256, device=device, dtype=dtype)
    den_ref = LTXJointDenoiser(_as_av_wrapper(m_ref), compute_dtype=dtype)
    den_fa3 = LTXJointDenoiser(_as_av_wrapper(m_fa3), compute_dtype=dtype, enable_cuda_graph="whole")
    reset_attention_provider_runtime()
    with torch.inference_mode():
        y_ref = den_ref(inp_s1).samples
        y_fa3 = den_fa3(inp_s1).samples
    fa3_rep = attention_provider_runtime_report().get("flash_attention_3", {})
    assert fa3_rep.get("successes", 0) > 0
    assert fa3_rep.get("fallbacks", 0) == 0
    assert _rel_l2(y_fa3["video"], y_ref["video"]) < 2e-2
    assert _rel_l2(y_fa3["audio"], y_ref["audio"]) < 2e-2
    _release_denoiser(den_ref)
    _release_denoiser(den_fa3)

    # 2. FA3 + Stage-2 Sol-Attn (kv_splits=2, allow_multi_split=True, dense_backend=flash_attention_3) + CUDA Graph
    m_mix = _build_small_av_model(num_layers=3, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    m_mix.fuse_projections(fuse_self_qkv=True, fuse_cross_kv=True)
    m_mix.enable_fused_ops()
    m_mix.configure_attention_backend("flash_attention_3")
    wrapper_mix = _as_av_wrapper(m_mix)
    ctx_mix = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            options={
                "accelerations": {
                    "attention_policy": {
                        "self": {
                            "backend": "sol_attn",
                            "tau": 0.0,
                            "kv_splits": 2,
                            "allow_multi_split": True,
                            "tile_3d": (1, 8, 8),
                            "dense_blocks": [0, -1],
                            "min_tokens": 4096,
                            "dense_backend": "flash_attention_3",
                        }
                    }
                }
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper_mix, ctx_mix)
    blocks = [cast(BasicAVTransformerBlock, b) for b in m_mix.transformer_blocks]
    assert blocks[0].attn1.attention_backend == "flash_attention_3"
    assert blocks[1].attn1.attention_backend == "sol_attn"
    assert blocks[1].attn1._dense_attention_backend == "flash_attention_3"
    assert blocks[2].attn1.attention_backend == "flash_attention_3"

    den_mix = LTXJointDenoiser(wrapper_mix, compute_dtype=dtype, enable_cuda_graph="whole")
    # Short sequence (Stage 1: t_v=64 < min_tokens=4096) -> routes to FA3, 0 sol_attn calls
    reset_attention_provider_runtime()
    with torch.inference_mode():
        out_s1 = den_mix(inp_s1).samples
    rep_s1 = attention_provider_runtime_report()
    assert rep_s1.get("sol_attn", {}).get("successes", 0) == 0
    assert rep_s1.get("flash_attention_3", {}).get("successes", 0) > 0
    assert torch.isfinite(out_s1["video"]).all()

    # Long sequence (Stage 2: t_v=4608 = 1x48x96 -> route_groups=2 >= kv_splits=2) -> Sol-Attn kv_splits=2 + FA3 + CUDA Graph!
    g2 = torch.Generator(device="cpu").manual_seed(29)
    v_lat2 = torch.randn((1, 4608, 32), generator=g2).to(device=device, dtype=dtype)
    v_mask2 = torch.ones((1, 4608, 1), device=device, dtype=dtype)
    grid2 = torch.stack(
        torch.meshgrid(
            torch.arange(1, dtype=torch.float32),
            torch.arange(48, dtype=torch.float32),
            torch.arange(96, dtype=torch.float32),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, -1)
    v_pos2 = torch.stack((grid2, grid2 + 1.0), dim=-1).unsqueeze(0).to(device=device)
    inp_s2 = MultiModalDenoiserInput(
        modalities={
            "video": ModalityState(latent=v_lat2, clean_latent=v_lat2, denoise_mask=v_mask2, positions=v_pos2),
            "audio": inp_s1.modalities["audio"],
        },
        timestep=inp_s1.timestep,
        conditioning=inp_s1.conditioning,
        step_index=0,
        total_steps=3,
    )
    reset_attention_provider_runtime()
    with torch.inference_mode():
        for _ in range(2):
            out_s2 = den_mix(inp_s2).samples
    rep_s2 = attention_provider_runtime_report()
    assert rep_s2.get("sol_attn", {}).get("successes", 0) > 0
    assert rep_s2.get("flash_attention_3", {}).get("successes", 0) > 0
    assert torch.isfinite(out_s2["video"]).all()
    assert torch.isfinite(out_s2["audio"]).all()
    cg_rep = den_mix.graph_report()
    assert cg_rep is not None
    assert cg_rep["lifetime"]["capture"] == 2
    assert cg_rep["lifetime"]["capture_failed"] == 0
    _release_denoiser(den_mix)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_video_decoder_compile_lossless_parity() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    blocks = [
        ("res_x", {"num_layers": 1, "inject_noise": False}),
        ("compress_all", {"multiplier": 2, "residual": True}),
    ]
    dec_ref = (
        VideoDecoder(
            convolution_dimensions=3,
            in_channels=16,
            out_channels=3,
            decoder_blocks=list(reversed(blocks)),
            patch_size=2,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=True,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=PaddingModeType.REFLECT,
            base_channels=16,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    dec_comp = (
        VideoDecoder(
            convolution_dimensions=3,
            in_channels=16,
            out_channels=3,
            decoder_blocks=list(reversed(blocks)),
            patch_size=2,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=True,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=PaddingModeType.REFLECT,
            base_channels=16,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    dec_comp.load_state_dict(dec_ref.state_dict())
    toy_factors = SpatioTemporalScaleFactors(time=2, height=4, width=4)
    dec_ref.video_downscale_factors = toy_factors
    dec_comp.video_downscale_factors = toy_factors
    dec_ref.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    dec_ref.per_channel_statistics.get_buffer("mean-of-means").zero_()
    dec_comp.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    dec_comp.per_channel_statistics.get_buffer("mean-of-means").zero_()

    dec_comp.enable_channels_last_3d()
    assert dec_comp.enable_compile(mode="default", dynamic=True) is True

    lat = torch.randn((1, 16, 4, 24, 24), device=device, dtype=dtype)
    with torch.inference_mode():
        y_ref = dec_ref(lat)
        y_comp = dec_comp(lat)
        rgb_ref = list(dec_ref.decode_video(lat, None))[0]
        rgb_comp = list(dec_comp.decode_video(lat, None))[0]

        tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=64, tile_overlap_in_pixels=32),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8),
        )
        rgb_tiled_ref = torch.cat(list(dec_ref.decode_video(lat, tiling)), dim=0)
        rgb_tiled_comp = torch.cat(list(dec_comp.decode_video(lat, tiling)), dim=0)

    mse_rgb = float(torch.mean((rgb_comp.float() - rgb_ref.float()) ** 2).item())
    psnr_rgb = 999.0 if mse_rgb <= 1e-12 else 10.0 * math.log10(1.0 / mse_rgb)
    mse_tiled = float(torch.mean((rgb_tiled_comp.float() - rgb_tiled_ref.float()) ** 2).item())
    psnr_tiled = 999.0 if mse_tiled <= 1e-12 else 10.0 * math.log10(1.0 / mse_tiled)
    assert psnr_rgb > 45.0
    assert psnr_tiled > 45.0
    assert _rel_l2(y_comp, y_ref) < 1.5e-2
    assert _rel_l2(rgb_tiled_comp, rgb_tiled_ref) < 1.5e-2


# =====================================================================
# WAVE 2 REVIEW REGRESSION TESTS
# =====================================================================


@pytest.mark.unit
def test_attention_dense_backend_reconfig_and_approximate_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-configuring sol_attn preserves exact dense_backend, blocks approx leaks, and respects force_dense."""
    attn = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128)
    attn.set_attention_backend("flash_attention_3")
    attn.set_attention_backend("sol_attn", backend_options={"min_tokens": 100})
    assert attn.attention_backend == "sol_attn"
    assert attn._dense_attention_backend == "flash_attention_3"

    # Re-configuring approximate options must preserve pre-bound exact dense_backend.
    attn.set_attention_backend("sol_attn", backend_options={"min_tokens": 200})
    assert attn._dense_attention_backend == "flash_attention_3"

    # Explicitly passing dense_backend=None clears it.
    attn.set_attention_backend("sol_attn", backend_options={"dense_backend": None})
    assert attn._dense_attention_backend is None

    # Transitioning from an approximate backend (sage_attention) must never leak into _dense_attention_backend.
    attn_approx = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128)
    attn_approx.set_attention_backend("sage_attention")
    attn_approx.set_attention_backend("sol_attn")
    assert attn_approx._dense_attention_backend is None

    # _worldfoundry_force_dense=True must never downgrade an already-exact backend (e.g. flash_attention_3).
    attn_exact = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128)
    attn_exact.set_attention_backend("flash_attention_3")
    attn_exact._worldfoundry_force_dense = True
    dispatched: list[str | None] = []

    def _spy_attention_forward(q, k, v, *, backend=None, **kwargs):
        del k, v, kwargs
        dispatched.append(backend)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "worldfoundry.core.attention.backends.dispatch.attention_forward",
        _spy_attention_forward,
    )
    qkv = torch.randn((1, 4, 256))
    attn_exact._run_attention(qkv, qkv, qkv, mask=None)
    assert dispatched == ["flash_attention_3"]

    # LTXModel.configure_attention_backend dense_blocks must reset approximate backends (e.g. sage_attention) to torch.
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        cross_attention_dim=64,
    )
    model.configure_attention_backend("sage_attention", self_only=True)
    model.configure_attention_backend("sol_attn", self_only=True, dense_blocks=[0])
    assert model.transformer_blocks[0].attn1.attention_backend == "torch"
    assert model.transformer_blocks[1].attn1.attention_backend == "sol_attn"


@pytest.mark.unit
def test_inference_mode_mask_mutation_and_strided_cache_invalidation() -> None:
    """Inference-mode in-place mutations and strided views never produce stale mask cache hits."""
    from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import (
        _is_trivial_context_mask,
        _is_uniform_denoise_mask,
    )

    u_cache: dict[int, Any] = {}
    c_cache: dict[int, Any] = {}
    with torch.inference_mode():
        mask = torch.ones((1, 16, 1), dtype=torch.float32)
        assert _is_uniform_denoise_mask(u_cache, mask) is True
        mask[:, :4] = 0.0
        assert _is_uniform_denoise_mask(u_cache, mask) is False

        cmask = torch.ones((1, 8), dtype=torch.int64)
        assert _is_trivial_context_mask(c_cache, cmask) is True
        cmask[:, 4:] = 0
        assert _is_trivial_context_mask(c_cache, cmask) is False

    # LTXJointDenoiser evicts inference-mode cache entries at step_index == 0 across requests.
    av_model = _build_small_av_model(num_layers=1, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    den = LTXJointDenoiser(_as_av_wrapper(av_model), compute_dtype=torch.float32)
    base_inp = _make_av_input(t_v=16, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128, sigma=0.75)

    with torch.inference_mode():
        inf_mask = torch.ones((1, 16, 1), dtype=torch.float32)
        v_state = ModalityState(
            latent=base_inp.modalities["video"].latent.clone(),
            clean_latent=base_inp.modalities["video"].clean_latent.clone(),
            denoise_mask=inf_mask,
            positions=base_inp.modalities["video"].positions.clone(),
        )
        inp1 = MultiModalDenoiserInput(
            modalities={"video": v_state, "audio": base_inp.modalities["audio"]},
            timestep=base_inp.timestep,
            conditioning=base_inp.conditioning,
            step_index=0,
            total_steps=4,
        )
        out_uniform = den(inp1).samples["video"].clone()

        # Mutate the exact same inference-mode tensor in place for an I2V request (step_index=0).
        inf_mask[:, :4] = 0.0
        out_i2v = den(inp1).samples["video"].clone()

        # Reference computed with a fresh non-cached tensor must match out_i2v exactly.
        fresh_mask = inf_mask.clone()
        v_state_ref = ModalityState(
            latent=v_state.latent,
            clean_latent=v_state.clean_latent,
            denoise_mask=fresh_mask,
            positions=v_state.positions,
        )
        inp_ref = MultiModalDenoiserInput(
            modalities={"video": v_state_ref, "audio": base_inp.modalities["audio"]},
            timestep=base_inp.timestep,
            conditioning=base_inp.conditioning,
            step_index=0,
            total_steps=4,
        )
        out_i2v_ref = den(inp_ref).samples["video"].clone()

    assert not torch.allclose(out_uniform, out_i2v)
    assert torch.allclose(out_i2v, out_i2v_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_fused_gelu_tanh_rowwise_fp8_cpu_and_empty_and_mixed_dtype_guards() -> None:
    """CPU/empty GELU-tanh FP8, non-contiguous gated-attn, and PytorchAdaZeroFunction(backend='torch') work cleanly."""
    for dtype in (torch.bfloat16, torch.float32):
        for fp8_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            x = torch.randn((4, 64), dtype=dtype)
            codes, scales = fused_ltx_gelu_tanh_rowwise_fp8(x, fp8_dt)
            assert codes.shape == (4, 64) and codes.dtype == fp8_dt
            assert scales.shape == (4, 1) and scales.dtype == torch.float32
            ref = torch.nn.functional.gelu(x, approximate="tanh").float()
            deq = codes.float() * scales
            assert _rel_l2(deq, ref) < 0.06

    empty_codes, empty_scales = fused_ltx_gelu_tanh_rowwise_fp8(torch.empty((0, 64), dtype=torch.bfloat16))
    assert empty_codes.shape == (0, 64)
    assert empty_scales.shape == (0, 1)

    # Non-contiguous stride(-1) != 1 on CPU and mixed dtype promotion must not crash or truncate.
    x_nc = torch.randn((1, 256, 8), dtype=torch.bfloat16).transpose(1, 2)
    assert x_nc.stride(-1) != 1
    gl_fp32 = torch.randn((1, 8, 2), dtype=torch.float32)
    y = fused_ltx_gated_attention(x_nc, gl_fp32, heads=2, dim_head=128)
    ref_y = (x_nc.reshape(1, 8, 2, 128) * (2.0 * torch.sigmoid(gl_fp32)).unsqueeze(-1)).reshape(1, 8, 256)
    assert y.dtype == torch.float32
    assert torch.allclose(y, ref_y, atol=1e-6, rtol=1e-6)

    # BasicAVTransformerBlock must honor PytorchAdaZeroFunction(backend="torch").
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import TransformerConfig
    from worldfoundry.core.nn import PytorchAdaZeroFunction, TransformerOpsConfig

    block = BasicAVTransformerBlock(
        video=TransformerConfig(dim=128, heads=2, d_head=64, context_dim=128),
        audio=None,
        ops=TransformerOpsConfig(ada_zero_function=PytorchAdaZeroFunction(backend="torch")),
    )
    _init_deterministic_weights(block)
    block._use_fused_residual_gate = True

    def _forbid_fused_adaln(*args, **kwargs):
        raise AssertionError("forward_from_adaln must not bypass PytorchAdaZeroFunction(backend='torch')")

    block.ff.forward_from_adaln = _forbid_fused_adaln  # type: ignore[method-assign]
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs

    x_blk = torch.randn((1, 4, 128))
    cos_b = torch.ones((1, 2, 4, 32))
    sin_b = torch.zeros((1, 2, 4, 32))
    v_args = TransformerArgs(
        x=x_blk,
        context=torch.randn((1, 3, 128)),
        context_mask=cast(Any, None),
        timesteps=torch.zeros((1, 1, 6 * 128)),
        embedded_timestep=torch.zeros((1, 1, 128)),
        positional_embeddings=(cos_b, sin_b),
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
        prompt_timestep=None,
    )
    with torch.no_grad():
        out_blk, _ = block(v_args, None)
    assert out_blk is not None and torch.isfinite(out_blk.x).all()


@pytest.mark.unit
def test_ltx_media_decoder_video_autocast_fp16_and_mixed_dtype_safety() -> None:
    """skip_redundant_video_autocast never disables FP16 autocast (PixelNorm overflow) or mixed/timestep decoders."""
    from types import SimpleNamespace

    blocks: list[tuple[str, int | dict[str, Any]]] = [("res_x", {"num_layers": 1, "inject_noise": False})]
    dec_fp16 = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=16,
    ).to(dtype=torch.float16)
    media_fp16 = LTXMediaDecoder(
        video=cast(LTXVideoDecoderModule, SimpleNamespace(decoder=dec_fp16)),
        audio=cast(LTXAudioDecoderModule, SimpleNamespace(decoder=torch.nn.Identity())),
        vocoder=cast(LTXVocoderModule, SimpleNamespace(vocoder=torch.nn.Identity())),
        compute_dtype=torch.float16,
        tiling=None,
        precision="autocast",
        skip_redundant_video_autocast=True,
    )
    assert media_fp16._use_video_autocast("cuda") is True

    # Uniform BF16 deterministic decoder may skip redundant autocast.
    dec_bf16 = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=16,
    ).to(dtype=torch.bfloat16)
    media_bf16 = LTXMediaDecoder(
        video=cast(LTXVideoDecoderModule, SimpleNamespace(decoder=dec_bf16)),
        audio=cast(LTXAudioDecoderModule, SimpleNamespace(decoder=torch.nn.Identity())),
        vocoder=cast(LTXVocoderModule, SimpleNamespace(vocoder=torch.nn.Identity())),
        compute_dtype=torch.bfloat16,
        tiling=None,
        precision="autocast",
        skip_redundant_video_autocast=True,
    )
    assert media_bf16._use_video_autocast("cuda") is False

    # Mixed BF16 + FP32 parameter must keep autocast enabled.
    dec_bf16.conv_out.to(dtype=torch.float32)
    assert media_bf16._use_video_autocast("cuda") is True

    # VRAM-managed submodule guard rejects enable_compile cleanly.
    dec_vram = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=2,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=16,
    )
    setattr(dec_vram.conv_in, "computation_device", torch.device("cpu"))
    assert dec_vram.enable_compile(mode="default") is False


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_ltx_media_decoder_plain_vocoder_compile_and_parallel_audio_exception() -> None:
    """Plain Vocoder (LTX-2.0) compiles cleanly under autocast; parallel_audio joins stream on error."""
    from types import SimpleNamespace

    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import Vocoder

    class _TinyAudioDec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(8, 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x).to(torch.bfloat16)

    voc = (
        Vocoder(
            resblock_kernel_sizes=[3],
            upsample_rates=[2],
            upsample_kernel_sizes=[4],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_initial_channel=16,
            resblock="1",
            output_sampling_rate=24000,
        )
        .to(device="cuda", dtype=torch.float32)
        .eval()
    )
    aud = _TinyAudioDec().to(device="cuda", dtype=torch.float32).eval()
    lat = torch.randn((1, 8, 10, 64), device="cuda", dtype=torch.bfloat16)

    media_eager = LTXMediaDecoder(
        video=cast(LTXVideoDecoderModule, SimpleNamespace(decoder=torch.nn.Identity())),
        audio=cast(LTXAudioDecoderModule, SimpleNamespace(decoder=aud)),
        vocoder=cast(LTXVocoderModule, SimpleNamespace(vocoder=voc)),
        compute_dtype=torch.bfloat16,
        tiling=None,
        precision="autocast",
        compile_vocoder=False,
    )
    media_comp = LTXMediaDecoder(
        video=cast(LTXVideoDecoderModule, SimpleNamespace(decoder=torch.nn.Identity())),
        audio=cast(LTXAudioDecoderModule, SimpleNamespace(decoder=aud)),
        vocoder=cast(LTXVocoderModule, SimpleNamespace(vocoder=voc)),
        compute_dtype=torch.bfloat16,
        tiling=None,
        precision="autocast",
        compile_vocoder=True,
    )
    with torch.inference_mode():
        w_ref, sr_ref = media_eager._decode_audio_waveform(lat)
        w_comp, sr_comp = media_comp._decode_audio_waveform(lat)
    assert sr_comp == sr_ref == 24000
    assert w_comp.shape == w_ref.shape
    assert torch.isfinite(w_comp).all()
    assert _rel_l2(w_comp, w_ref) < 2e-2

    # parallel_audio=True must still synchronize audio_stream when video decode raises.
    class _FailingVideoDec(torch.nn.Module):
        def decode_video(self, *_args, **_kwargs):
            raise RuntimeError("simulated video decode failure")

    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
    from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
        AudioPatchifier,
        VideoLatentPatchifier,
    )

    class _ShapeTinyAudioDec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(8, 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Map latent mel width (16) -> vocoder 64 mel bins for shape-consistent stereo decode.
            return self.conv(x).repeat(1, 1, 1, 4).to(torch.bfloat16)

    media_par = LTXMediaDecoder(
        video=cast(LTXVideoDecoderModule, SimpleNamespace(decoder=_FailingVideoDec())),
        audio=cast(LTXAudioDecoderModule, SimpleNamespace(decoder=_ShapeTinyAudioDec().to(device="cuda").eval())),
        vocoder=cast(LTXVocoderModule, SimpleNamespace(vocoder=voc)),
        compute_dtype=torch.bfloat16,
        tiling=None,
        precision="autocast",
        parallel_audio=True,
    )
    req = DiffusionRequest(
        prompt="test",
        height=64,
        width=64,
        num_frames=9,
        sampling=SamplingConfig(seed=0, num_inference_steps=1, guidance_scale=1.0),
    )
    v_shape, a_shape = LTXMediaDecoder._shapes(req)
    v_tokens = VideoLatentPatchifier(1).get_token_count(v_shape)
    a_tokens = AudioPatchifier(1).get_token_count(a_shape)
    v_lat = torch.zeros((1, v_tokens, v_shape.channels), device="cuda", dtype=torch.bfloat16)
    a_lat = torch.zeros((1, a_tokens, a_shape.channels * a_shape.mel_bins), device="cuda", dtype=torch.bfloat16)
    v_pos = torch.zeros((1, 3, v_tokens, 2), device="cuda")
    a_pos = torch.zeros((1, 1, a_tokens, 2), device="cuda")
    states = {
        "video": ModalityState(
            latent=v_lat,
            clean_latent=v_lat,
            denoise_mask=torch.ones((1, v_tokens, 1), device="cuda"),
            positions=v_pos,
        ),
        "audio": ModalityState(
            latent=a_lat,
            clean_latent=a_lat,
            denoise_mask=torch.ones((1, a_tokens, 1), device="cuda"),
            positions=a_pos,
        ),
    }
    with torch.inference_mode(), pytest.raises(RuntimeError, match="simulated video decode failure"):
        media_par.decode_modalities(states, req)
    torch.cuda.synchronize()


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_video_decoder_reduce_overhead_consecutive_and_tensor_codec() -> None:
    """VideoDecoder enable_compile(mode='reduce-overhead') survives consecutive calls & LTXTensorVideoCodec clamp_."""
    from types import SimpleNamespace

    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import LTXTensorVideoCodec

    device = torch.device("cuda")
    dtype = torch.bfloat16
    blocks = [
        ("res_x", {"num_layers": 1, "inject_noise": False}),
        ("compress_all", {"multiplier": 2, "residual": True}),
    ]
    vdec = (
        VideoDecoder(
            convolution_dimensions=3,
            in_channels=16,
            out_channels=3,
            decoder_blocks=list(reversed(blocks)),
            patch_size=2,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=True,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=PaddingModeType.REFLECT,
            base_channels=16,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    vdec.video_downscale_factors = SpatioTemporalScaleFactors(time=2, height=4, width=4)
    vdec.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    vdec.per_channel_statistics.get_buffer("mean-of-means").zero_()
    vdec.enable_channels_last_3d()
    assert vdec.enable_compile(mode="reduce-overhead") is True

    lat1 = torch.randn((1, 16, 2, 8, 8), device=device, dtype=dtype)
    lat2 = torch.randn((1, 16, 2, 8, 8), device=device, dtype=dtype)
    with torch.inference_mode():
        r1 = list(vdec.decode_video(lat1, None))[0]
        r1_snap = r1.clone()
        r2 = list(vdec.decode_video(lat2, None))[0]
        assert torch.equal(r1, r1_snap)
        assert not torch.equal(r1, r2)

    codec = LTXTensorVideoCodec(
        encoder=cast(Any, SimpleNamespace()),
        decoder=cast(Any, SimpleNamespace(decoder=vdec, parameters=vdec.parameters)),
        tiling=None,
        compute_device=device,
        compute_dtype=dtype,
    )
    o1 = codec.decode(lat1)
    o1_snap = o1.clone()
    o2 = codec.decode(lat2)
    assert torch.equal(o1, o1_snap)
    assert not torch.equal(o1, o2)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_gpu_fused_ops_noncontiguous_and_mixed_dtype_parity() -> None:
    """Mixed-dtype gated-attn / forward_from_adaln, non-contiguous gelu FP8, and opcheck all pass on CUDA."""
    from worldfoundry.base_models.diffusion_model.models.networks.ltx import fused_ops
    from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8
    from worldfoundry.core.kernels import rms_norm_scale_shift

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Mixed dtype gated_attention promotes to float32 like PyTorch reference.
    x_bf = torch.randn((1, 8, 256), device=device, dtype=dtype)
    g_fp = torch.randn((1, 8, 2), device=device, dtype=torch.float32)
    y_mixed = fused_ltx_gated_attention(x_bf, g_fp, heads=2, dim_head=128)
    y_ref = (x_bf.view(1, 8, 2, 128) * (2.0 * torch.sigmoid(g_fp)).unsqueeze(-1)).view(1, 8, 256)
    assert y_mixed.dtype == torch.float32
    assert torch.allclose(y_mixed, y_ref, atol=1e-5, rtol=1e-5)

    # 2. Non-contiguous stride(1) != 1 GELU-tanh FP8 matches contiguous result.
    mat = torch.randn((64, 32), device=device, dtype=dtype).t()
    assert mat.stride(1) != 1
    c_nc, s_nc = fused_ltx_gelu_tanh_rowwise_fp8(mat, torch.float8_e4m3fn)
    c_c, s_c = fused_ltx_gelu_tanh_rowwise_fp8(mat.contiguous(), torch.float8_e4m3fn)
    assert torch.equal(c_nc, c_c)
    assert torch.equal(s_nc, s_c)

    # 3. FeedForward.forward_from_adaln with BF16 x + FP32 scale/shift outside autocast preserves FP32 promotion.
    ff = FeedForward(1024, 1024, mult=2).to(device=device, dtype=dtype).eval()
    replace_linear_with_float8(ff, min_features=1024, keep_dense_fallback=True, scaling="rowwise")
    ff.enable_fused_fp8(True)
    x_ff = torch.randn((1, 16, 1024), device=device, dtype=dtype)
    s_fp = torch.randn((1, 1, 1024), device=device, dtype=torch.float32) * 0.1
    sh_fp = torch.randn((1, 1, 1024), device=device, dtype=torch.float32) * 0.1
    with torch.inference_mode():
        out_ff = ff.forward_from_adaln(x_ff, 1e-6, s_fp, sh_fp)
        ref_ff = ff(rms_norm_scale_shift(x_ff, s_fp, sh_fp, eps=1e-6))
    assert out_ff.dtype == ref_ff.dtype == torch.float32
    assert torch.allclose(out_ff, ref_ff, atol=1e-5, rtol=1e-5)

    # 4. torch.library.opcheck schema + FakeTensor checks for all Wave 2 custom ops.
    x_op = torch.randn((2, 8, 256), device=device, dtype=dtype)
    s_op = torch.randn((2, 1, 256), device=device, dtype=dtype)
    w_op = torch.randn((256,), device=device, dtype=dtype)
    gl_op = torch.randn((2, 8, 2), device=device, dtype=dtype)
    h2d = torch.randn((8, 512), device=device, dtype=dtype)
    for op, args in (
        (fused_ops._ltx_post_sa_residual_rmsnorm_adaln_op, (x_op, x_op, s_op, s_op, s_op, 1e-6)),
        (fused_ops._ltx_gated_attention_op, (x_op, gl_op, 2, 128)),
        (fused_ops._ltx_gated_attention_rowwise_fp8_op, (x_op, gl_op, 2, 128, False)),
        (fused_ops._ltx_rmsnorm_scale_shift_fp8_op, (x_op, s_op, s_op, 1e-6, False)),
        (fused_ops._ltx_dual_rmsnorm_scale_shift_op, (x_op, s_op, s_op, s_op, s_op, 1e-6)),
        (fused_ops._ltx_scale_shift_op, (x_op, s_op, s_op)),
        (fused_ops._ltx_qk_rmsnorm_paired_op, (x_op, x_op, w_op, w_op, 1e-6)),
        (fused_ops._ltx_gelu_tanh_rowwise_fp8_op, (h2d, False)),
    ):
        torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"), raise_exception=True)


# =====================================================================
# WAVE 3 UNIT + GPU TESTS (PISA, Token Pruning, FirstBlockCache)
# =====================================================================


_PISA_USABLE = False
if _CUDA_AVAILABLE:
    from worldfoundry.core.attention.sparse.piecewise import piecewise_attention_available

    _PISA_USABLE = piecewise_attention_available("cuda")


@pytest.mark.unit
def test_ltx_pisa_validation_scoping_dense_edge_blocks_and_cpu_fallback() -> None:
    """PISA option validation, positional backend_options, edge-block scoping, and exact CPU fallback."""
    cross_attn = Attention(query_dim=256, context_dim=256, heads=2, dim_head=128)
    with pytest.raises(ValueError, match="self-attention"):
        cross_attn.set_attention_backend("pisa", {"density": 0.15, "block_size": 64})

    self_attn = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128).eval()
    with pytest.raises(ValueError, match="unknown pisa options"):
        self_attn.set_attention_backend("pisa", {"bogus": 1})
    with pytest.raises(ValueError, match="density"):
        self_attn.set_attention_backend("pisa", {"density": 0.0})
    with pytest.raises(ValueError, match="density"):
        self_attn.set_attention_backend("pisa", {"density": 1.5})
    with pytest.raises(ValueError, match="density"):
        self_attn.set_attention_backend("pisa", {"density": True})
    with pytest.raises(ValueError, match="block_size"):
        self_attn.set_attention_backend("pisa", {"block_size": 48})
    with pytest.raises(ValueError, match="strict must be a bool"):
        self_attn.set_attention_backend("pisa", {"strict": "yes"})
    with pytest.raises(ValueError, match="exact attention backend"):
        self_attn.set_attention_backend("pisa", {"dense_backend": "pisa"})

    # Positional backend_options dict works and CPU ineligible workload falls back to exact SDPA.
    torch.manual_seed(0)
    x = torch.randn((1, 16, 256))
    with torch.inference_mode():
        y_ref = self_attn(x)

    reset_attention_provider_runtime()
    self_attn.set_attention_backend("pisa", {"density": 0.15, "block_size": 64, "min_tokens": 8, "strict": False})
    assert self_attn.attention_backend == "pisa"
    with torch.inference_mode():
        y_pisa_cpu = self_attn(x)
    torch.testing.assert_close(y_pisa_cpu, y_ref, rtol=1e-5, atol=1e-5)
    assert attention_provider_runtime_report().get("pisa", {}).get("successes", 0) == 0

    # Non-strict masked attention also falls back to exact masked SDPA; strict fails closed.
    attn_mask = torch.zeros((1, 1, 16, 16))
    with torch.inference_mode():
        y_masked = self_attn(x, mask=attn_mask)
    torch.testing.assert_close(y_masked, y_ref, rtol=1e-5, atol=1e-5)

    self_attn.set_attention_backend("pisa", {"density": 0.15, "block_size": 64, "min_tokens": 8, "strict": True})
    with pytest.raises(RuntimeError, match="not eligible"):
        self_attn(x)
    with pytest.raises(ValueError, match="does not support attention masks"):
        self_attn(x, mask=attn_mask)

    # 5-layer AV model: configure_attention_backend("pisa", self_only=True, dense_blocks=(0, 1, -1))
    # configures only video attn1 blocks 2 & 3, leaving edge blocks (0, 1, 4) dense and audio_attn1 untouched.
    av_model = _build_small_av_model(num_layers=5, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    n_cfg = av_model.configure_attention_backend(
        "pisa",
        self_only=True,
        dense_blocks=(0, 1, -1),
        dense_backend="torch",
        backend_options={"density": 0.15, "block_size": 64, "min_tokens": 128},
    )
    assert n_cfg == 2
    blocks = [cast(BasicAVTransformerBlock, b) for b in av_model.transformer_blocks]
    assert blocks[0].attn1.attention_backend == "torch"
    assert blocks[1].attn1.attention_backend == "torch"
    assert blocks[2].attn1.attention_backend == "pisa"
    assert blocks[2].attn1._dense_attention_backend == "torch"
    assert blocks[3].attn1.attention_backend == "pisa"
    assert blocks[4].attn1.attention_backend == "torch"
    for blk in blocks:
        assert blk.audio_attn1.attention_backend is None
        assert blk.attn2.attention_backend is None

    # Accelerations plugin defaults dense_blocks to (0, 1, -1) on >=4-block models and passes validation.
    av_model2 = _build_small_av_model(num_layers=5, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    wrapper2 = _as_av_wrapper(av_model2)
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={
                "accelerations": {
                    "attention_policy": {
                        "self": {
                            "backend": "pisa",
                            "density": 0.15,
                            "block_size": 64,
                            "min_tokens": 64,
                            "tile_3d": (1, 4, 4),
                            "dense_backend": "torch",
                        }
                    }
                }
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper2, ctx)
    den = LTXJointDenoiser(wrapper2, compute_dtype=torch.float32)
    validate_acceleration_installation(den, ctx)
    blocks2 = [cast(BasicAVTransformerBlock, b) for b in av_model2.transformer_blocks]
    assert [b.attn1.attention_backend for b in blocks2] == ["torch", "torch", "pisa", "pisa", "torch"]
    report = den.runtime_optimization_report()
    scope_rep = report["accelerations"]["installed"][0]["scopes"]["self"]
    assert scope_rep["resolved"] == "pisa"
    assert scope_rep["schedule"]["dense_blocks"] == [0, 1, 4]
    assert scope_rep["schedule"]["tile_3d"] == (1, 4, 4)


@pytest.mark.unit
def test_ltx_token_pruning_validation_residual_preservation_and_i2v_conditioning_protection() -> None:
    """Interior token pruning validates configs, keeps edge blocks full, and strictly preserves I2V conditioning."""
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.acceleration import (
        LTXTokenPruningConfig,
        select_ltx_salient_tokens,
    )

    with pytest.raises(ValueError, match="keep_ratio"):
        LTXTokenPruningConfig(keep_ratio=0.0)
    with pytest.raises(ValueError, match="keep_ratio"):
        LTXTokenPruningConfig(keep_ratio=1.2)
    with pytest.raises(TypeError, match="keep_ratio"):
        LTXTokenPruningConfig(keep_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        LTXTokenPruningConfig(method="unknown")
    with pytest.raises(ValueError, match="block span"):
        LTXTokenPruningConfig(start_block=3, end_block=-2).resolve_span(4)

    # Protected I2V conditioning tokens must NEVER be dropped even when their feature norm is ~0.
    h = torch.ones((1, 64, 32)) * 10.0
    h[:, :16, :] = 1e-4
    prot = torch.zeros(64, dtype=torch.bool)
    prot[:16] = True
    for method in ("feat_norm", "delta_norm", "feat_l1", "feat_var", "uniform"):
        idx = select_ltx_salient_tokens(h, 32, method=method, entry_hidden=torch.zeros_like(h), protected_mask=prot)
        assert idx.shape == (32,)
        assert torch.equal(idx, torch.sort(idx).values)
        assert set(range(16)).issubset(set(idx.tolist()))

    # Stratified selection preserves uniform spatial/temporal coverage per group; restore modes match contract.
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.acceleration import restore_ltx_pruned_tokens

    h_strat = torch.arange(64, dtype=torch.float32).view(1, 64, 1).expand(1, 64, 8).clone()
    strat_idx = select_ltx_salient_tokens(h_strat, 48, method="feat_norm", stratify_group_size=8)
    assert strat_idx.shape == (48,)
    for g in range(8):
        in_group = [i for i in strat_idx.tolist() if g * 8 <= i < (g + 1) * 8]
        assert len(in_group) == 6
    pre = torch.zeros((1, 8, 4))
    keep_idx = torch.tensor([0, 3, 7], dtype=torch.long)
    mid_in = pre.index_select(1, keep_idx)
    mid_out = torch.tensor([[[1.0] * 4, [4.0] * 4, [10.0] * 4]])
    out_zero = restore_ltx_pruned_tokens(pre, mid_in, mid_out, keep_idx, restore_mode="zero")
    assert out_zero[0, 1, 0].item() == 0.0 and out_zero[0, 3, 0].item() == 4.0
    out_mean = restore_ltx_pruned_tokens(pre, mid_in, mid_out, keep_idx, restore_mode="mean_delta")
    assert out_mean[0, 1, 0].item() == pytest.approx(5.0) and out_mean[0, 7, 0].item() == 10.0
    out_near = restore_ltx_pruned_tokens(pre, mid_in, mid_out, keep_idx, restore_mode="nearest_delta")
    assert out_near[0, 1, 0].item() == 1.0  # nearest kept to idx 1 is 0
    assert out_near[0, 2, 0].item() == 4.0  # nearest kept to idx 2 is 3
    assert out_near[0, 6, 0].item() == 10.0  # nearest kept to idx 6 is 7

    # 5-layer AV model: verify keep_ratio=1.0 bitwise parity, distilled short-step guard, min_tokens, and I2V protection.
    av_ref = _build_small_av_model(num_layers=5, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    av_tp = _build_small_av_model(num_layers=5, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    av_tp.load_state_dict(av_ref.state_dict())

    inp_short = _make_av_input(t_v=16, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128, total_steps=8)
    inp_long_distilled = _make_av_input(t_v=64, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128, total_steps=3)
    inp_long_multi = _make_av_input(t_v=64, t_a=8, t_txt=6, v_ctx_dim=128, a_ctx_dim=128, total_steps=6)
    den_ref = LTXJointDenoiser(_as_av_wrapper(av_ref), compute_dtype=torch.float32)

    av_tp.configure_token_pruning({"keep_ratio": 1.0, "start_block": 1, "end_block": -1, "min_tokens": 0})
    wrapper_tp = _as_av_wrapper(av_tp)
    den_tp = LTXJointDenoiser(wrapper_tp, compute_dtype=torch.float32)
    with torch.inference_mode():
        ref_long = den_ref(inp_long_distilled).samples
        out_k1 = den_tp(inp_long_distilled).samples
    torch.testing.assert_close(out_k1["video"], ref_long["video"], rtol=0, atol=0)
    torch.testing.assert_close(out_k1["audio"], ref_long["audio"], rtol=0, atol=0)
    _release_denoiser(den_tp)
    del den_tp
    gc.collect()

    # Configure Stage-2 pruning with default allow_distilled_pruning=False (min_steps_to_enable=5).
    ctx_tp = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={
                "accelerations": {
                    "token_pruning": {
                        "keep_ratio": 0.75,
                        "method": "delta_norm",
                        "restore_mode": "nearest_delta",
                        "stratify_group_size": 8,
                        "start_block": 1,
                        "end_block": -1,
                        "min_tokens": 32,
                    }
                }
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper_tp, ctx_tp)
    den_tp2 = LTXJointDenoiser(wrapper_tp, compute_dtype=torch.float32)
    validate_acceleration_installation(den_tp2, ctx_tp)

    observed_seq_lens: dict[int, list[int]] = {i: [] for i in range(5)}
    handles = []
    for i, blk in enumerate(av_tp.transformer_blocks):

        def _hook(_mod: torch.nn.Module, args: tuple[Any, ...], idx: int = i) -> None:
            v_arg = args[0]
            if v_arg is not None:
                observed_seq_lens[idx].append(int(v_arg.x.shape[1]))

        handles.append(blk.register_forward_pre_hook(_hook))
    try:
        with torch.inference_mode():
            # Stage 1 (t_v=16 < min_tokens=32): runs dense on all 5 blocks without triggering guard warning
            out_s1 = den_tp2(inp_short).samples
            assert all(observed_seq_lens[i][-1] == 16 for i in range(5))

            # Short distilled Stage 2 (total_steps=3 < 5, allow_distilled_pruning=False):
            # safety guard keeps all 5 blocks dense (bitwise identical to unpruned ref) and emits warning.
            with pytest.warns(UserWarning, match="distilled short-step safety guard"):
                out_guarded = den_tp2(inp_long_distilled).samples
            assert all(observed_seq_lens[i][-1] == 64 for i in range(5))
            torch.testing.assert_close(out_guarded["video"], ref_long["video"], rtol=0, atol=0)
            torch.testing.assert_close(out_guarded["audio"], ref_long["audio"], rtol=0, atol=0)

            # Multi-step Stage 2 (total_steps=6 >= 5, t_v=64 >= 32): edge blocks 0 & 4 see 64, middle 1..3 see 48
            out_s2 = den_tp2(inp_long_multi).samples
            assert observed_seq_lens[0][-1] == 64
            assert observed_seq_lens[1][-1] == 48
            assert observed_seq_lens[2][-1] == 48
            assert observed_seq_lens[3][-1] == 48
            assert observed_seq_lens[4][-1] == 64
            assert torch.isfinite(out_s1["video"]).all()
            assert torch.isfinite(out_s2["video"]).all()

            # Non-uniform I2V denoise_mask where 56/64 tokens are conditioned (> keep_tokens=48):
            # pruning must safely skip and run dense rather than drop any conditioning token.
            v_st = inp_long_multi.modalities["video"]
            heavy_cond_mask = torch.ones_like(v_st.denoise_mask)
            heavy_cond_mask[:, :56, :] = 0.0
            inp_heavy_i2v = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(
                        latent=v_st.latent,
                        clean_latent=v_st.clean_latent,
                        denoise_mask=heavy_cond_mask,
                        positions=v_st.positions,
                    ),
                    "audio": inp_long_multi.modalities["audio"],
                },
                timestep=inp_long_multi.timestep,
                conditioning=inp_long_multi.conditioning,
                step_index=0,
                total_steps=6,
            )
            _ = den_tp2(inp_heavy_i2v)
            assert all(observed_seq_lens[i][-1] == 64 for i in range(5))
    finally:
        for h_ref in handles:
            h_ref.remove()

    rep = den_tp2.runtime_optimization_report()
    assert rep["token_pruning"]["pruned_calls"] == 1
    assert rep["token_pruning"]["dense_calls"] == 3
    assert rep["token_pruning"]["distilled_guard_skips"] == 1
    assert rep["token_pruning"]["allow_distilled_pruning"] is False
    assert rep["token_pruning"]["last_kept_tokens"] == 48
    assert rep["token_pruning"]["last_total_tokens"] == 64


@pytest.mark.unit
def test_ltx_firstblock_cache_isolation_boundaries_and_mutual_exclusion() -> None:
    """FirstBlockCache respects boundaries, max_consecutive_skips, branch/request isolation, and rejects conflicts."""
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.acceleration import LTXFirstBlockCacheConfig

    with pytest.raises(ValueError, match="residual_diff_threshold"):
        LTXFirstBlockCacheConfig(residual_diff_threshold=-0.1)
    with pytest.raises(ValueError, match="dense_first"):
        LTXFirstBlockCacheConfig(dense_first=0)

    av_model = _build_small_av_model(num_layers=4, video_heads=2, video_dim_head=64, audio_heads=2, audio_dim_head=64)
    wrapper = _as_av_wrapper(av_model)

    # Mutual exclusion with TeaCache and whole CUDA Graph.
    with pytest.raises(ValueError, match="mutually exclusive"):
        LTXJointDenoiser(wrapper, compute_dtype=torch.float32, teacache_threshold=0.1, fbcache_threshold=0.08)
    with pytest.raises(ValueError, match="piecewise"):
        LTXJointDenoiser(wrapper, compute_dtype=torch.float32, enable_cuda_graph="whole", fbcache_threshold=0.08)

    # Plugin installation + validate_acceleration_installation contract.
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=torch.device("cpu"),
            dtype=torch.float32,
            options={
                "accelerations": {
                    "fbcache": {
                        "residual_diff_threshold": 0.50,
                        "dense_first": 1,
                        "dense_last": 1,
                        "max_consecutive_skips": 2,
                        "downsample_factor": 2,
                        "min_steps_to_enable": 4,
                        "max_tokens": 32,
                    }
                }
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper, ctx)
    den = LTXJointDenoiser(wrapper, compute_dtype=torch.float32)
    validate_acceleration_installation(den, ctx)

    # Run 6 Stage-1 steps (t_v=16 <= 32) across two requests and verify hits + boundaries + max_consecutive_skips.
    with torch.inference_mode():
        for step, sigma in enumerate((0.500, 0.499, 0.498, 0.497, 0.496, 0.495)):
            inp_a = _make_av_input(
                t_v=16,
                t_a=8,
                t_txt=6,
                v_ctx_dim=128,
                a_ctx_dim=128,
                sigma=sigma,
                step_index=step,
                total_steps=6,
                request_id="fb-req-a",
                seed=11,
            )
            inp_b = _make_av_input(
                t_v=16,
                t_a=8,
                t_txt=6,
                v_ctx_dim=128,
                a_ctx_dim=128,
                sigma=sigma,
                step_index=step,
                total_steps=6,
                request_id="fb-req-b",
                seed=29,
            )
            out_a = den(inp_a).samples
            out_b = den(inp_b).samples
            assert torch.isfinite(out_a["video"]).all() and torch.isfinite(out_b["video"]).all()

        # Stage 2 (t_v=64 > max_tokens=32) bypasses FBCache cleanly without modifying Stage-1 receipts.
        for step, sigma in enumerate((0.9, 0.7, 0.4)):
            inp_s2 = _make_av_input(
                t_v=64,
                t_a=8,
                t_txt=6,
                v_ctx_dim=128,
                a_ctx_dim=128,
                sigma=sigma,
                step_index=step,
                total_steps=3,
                request_id="fb-req-a",
            )
            out_s2 = den(inp_s2).samples
            assert torch.isfinite(out_s2["video"]).all()

    rep_a = den.runtime_optimization_report("fb-req-a")["feature_cache"]
    assert rep_a["algorithm"] == "firstblock"
    assert rep_a["events"] == 6
    assert rep_a["hits"] >= 2
    assert rep_a["skipped_block_calls"] == rep_a["hits"] * 3  # 3 skipped blocks per hit on 4-layer model
    events_a = rep_a["branches"]["positive"]["event_receipts"]
    assert events_a[0]["reason"] == "seed" and events_a[0]["hit"] is False
    assert events_a[-1]["reason"] == "dense-boundary" and events_a[-1]["hit"] is False
    # Max consecutive skips = 2 means step 3 must force a dense refresh if steps 1 & 2 hit.
    if events_a[1]["hit"] and events_a[2]["hit"]:
        assert events_a[3]["hit"] is False
        assert events_a[3]["reason"] == "max-consecutive-skips"

    den.end_feature_cache_request("fb-req-a")
    snap_a = den.fbcache_report("fb-req-a")
    assert snap_a["finalized"] is True
    assert snap_a["hits"] == rep_a["hits"]
    den.end_feature_cache_request("fb-req-b")


@pytest.mark.gpu
@pytest.mark.skipif(not _PISA_USABLE, reason="In-tree PISA TMA kernel requires Hopper/DC-Blackwell CUDA + Triton TMA")
@pytest.mark.parametrize("graph_mode", ["whole", "piecewise"])
def test_gpu_pisa_tile3d_dense_blocks_and_cuda_graph(graph_mode: str) -> None:
    """In-tree PISA with dense edge blocks (0, 1, -1), tile_3d=(2,4,4), dense_steps=1, and CUDA Graph capture."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    dense_bk = "flash_attention_3" if _FA3_USABLE else "torch"

    m_av = _build_small_av_model(num_layers=5, video_heads=4, video_dim_head=128, device=device, dtype=dtype)
    m_av.fuse_projections()
    m_av.enable_fused_ops()
    wrapper = _as_av_wrapper(m_av)
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            options={
                "accelerations": {
                    "attention_policy": {
                        "self": {
                            "backend": "pisa",
                            "density": 0.25,
                            "block_size": 64,
                            "min_tokens": 128,
                            "dense_blocks": (0, 1, -1),
                            "dense_steps": 1,
                            "tile_3d": (2, 4, 4),
                            "dense_backend": dense_bk,
                            "strict": True,
                        }
                    }
                }
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper, ctx)
    den = LTXJointDenoiser(wrapper, compute_dtype=dtype, enable_cuda_graph=graph_mode)
    validate_acceleration_installation(den, ctx)

    g = torch.Generator(device="cpu").manual_seed(77)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(2, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, -1)
    v_pos = torch.stack((grid, grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
    v_lat = torch.randn((1, 128, 32), generator=g).to(device=device, dtype=dtype)
    v_mask = torch.ones((1, 128, 1), device=device, dtype=dtype)
    v_ctx = torch.randn((1, 16, 512), generator=g).to(device=device, dtype=dtype)
    a_lat = torch.randn((1, 32, 32), generator=g).to(device=device, dtype=dtype)
    a_mask = torch.ones((1, 32, 1), device=device, dtype=dtype)
    a_grid = torch.arange(32, dtype=torch.float32).reshape(1, -1)
    a_pos = torch.stack((a_grid, a_grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
    a_ctx = torch.randn((1, 16, 256), generator=g).to(device=device, dtype=dtype)

    def _inp(step: int) -> MultiModalDenoiserInput:
        return MultiModalDenoiserInput(
            modalities={
                "video": ModalityState(latent=v_lat, clean_latent=v_lat, denoise_mask=v_mask, positions=v_pos),
                "audio": ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos),
            },
            timestep=torch.tensor([0.5], device=device, dtype=dtype),
            conditioning={"video_context": v_ctx, "audio_context": a_ctx},
            step_index=step,
            total_steps=4,
        )

    reset_attention_provider_runtime()
    with torch.inference_mode():
        # Step 0: dense_steps=1 forces dense backend (0 PISA calls).
        out0 = den(_inp(0)).samples
        assert attention_provider_runtime_report().get("pisa", {}).get("successes", 0) == 0

        # Short Stage-1 input (t_v=64 < min_tokens=128) also bypasses PISA cleanly.
        short_inp = _make_av_input(
            t_v=64, t_a=32, t_txt=16, v_ctx_dim=512, a_ctx_dim=256, step_index=1, device=device, dtype=dtype
        )
        out_short = den(short_inp).samples
        assert torch.isfinite(out_short["video"]).all()
        assert attention_provider_runtime_report().get("pisa", {}).get("successes", 0) == 0

        # Steps 1..3 at t_v=128: PISA runs on blocks 2..3 with tile_3d=(2,4,4) and captures/replays in CUDA Graph!
        for step in (1, 2, 3):
            out_s = den(_inp(step)).samples
            assert torch.isfinite(out_s["video"]).all()
            assert torch.isfinite(out_s["audio"]).all()

    pisa_rep = attention_provider_runtime_report().get("pisa", {})
    assert pisa_rep.get("successes", 0) > 0
    assert pisa_rep.get("fallbacks", 0) == 0
    assert pisa_rep.get("errors", 0) == 0
    cg_rep = den.graph_report()
    assert cg_rep is not None
    assert cg_rep["capture_failed"] == 0
    assert torch.isfinite(out0["video"]).all()
    _release_denoiser(den)


@pytest.mark.gpu
@pytest.mark.skipif(not (_PISA_USABLE and _HOPPER_PLUS), reason="Requires Hopper+ CUDA with PISA TMA and FP8")
def test_gpu_full_sol_engine_stack_fbcache_pisa_fp8_token_pruning_and_i2v() -> None:
    """Full SOL-Engine stack: FBCache (Stage 1) + PISA + tile_3d + Token Pruning + FP8 + Piecewise CUDA Graph (T2V & I2V)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    dense_bk = "flash_attention_3" if _FA3_USABLE else "torch"

    m_av = _build_small_av_model(
        num_layers=6, video_heads=8, video_dim_head=128, audio_heads=8, audio_dim_head=64, device=device, dtype=dtype
    )
    m_av.fuse_projections()
    m_av.enable_fused_ops()
    enable_ltx_fp8_quantization(
        m_av,
        min_features=512,
        scaling="rowwise",
        use_fast_accum=False,
        keep_dense_fallback=False,
        fuse_fp8_ffn=True,
    )
    wrapper = _as_av_wrapper(m_av)
    ctx = ComponentBuildContext(
        model_id="ltx-2.3-t2v",
        key=ComponentKey(ComponentKind.DENOISER, "main"),
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            options={
                "cuda_graph": "piecewise",
                "accelerations": {
                    "attention_policy": {
                        "self": {
                            "backend": "pisa",
                            "density": 0.25,
                            "block_size": 64,
                            "min_tokens": 128,
                            "dense_blocks": (0, 1, -1),
                            "tile_3d": (2, 4, 4),
                            "dense_backend": dense_bk,
                            "strict": True,
                        }
                    },
                    "token_pruning": {
                        "keep_ratio": 0.75,
                        "method": "feat_norm",
                        "restore_mode": "nearest_delta",
                        "start_block": 2,
                        "end_block": -1,
                        "min_tokens": 128,
                        "allow_distilled_pruning": True,
                    },
                    "fbcache": {
                        "residual_diff_threshold": 0.50,
                        "dense_first": 1,
                        "dense_last": 1,
                        "max_consecutive_skips": 2,
                        "min_steps_to_enable": 4,
                        "max_tokens": 96,
                    },
                },
            },
        ),
        checkpoints={},
    )
    _apply_ltx_post_load_accelerations(wrapper, ctx)
    den = LTXJointDenoiser(wrapper, compute_dtype=dtype, enable_cuda_graph="piecewise")
    validate_acceleration_installation(den, ctx)

    reset_attention_provider_runtime()
    with torch.inference_mode():
        # Stage 1 (t_v=64 <= 96, 6 steps): FBCache hits, PISA & Token Pruning stay bypassed!
        for step, sigma in enumerate((0.500, 0.499, 0.498, 0.497, 0.496, 0.495)):
            inp_s1 = _make_av_input(
                t_v=64,
                t_a=32,
                t_txt=16,
                v_ctx_dim=1024,
                a_ctx_dim=512,
                sigma=sigma,
                step_index=step,
                total_steps=6,
                request_id="sol-full-req",
                device=device,
                dtype=dtype,
            )
            out_s1 = den(inp_s1).samples
            assert torch.isfinite(out_s1["video"]).all() and torch.isfinite(out_s1["audio"]).all()

        # Stage 2 T2V (t_v=256 >= 128, pruned to 192 >= 128 & multiple of block_size=64, 3 steps):
        # FBCache bypassed, PISA + tile_3d + Token Pruning + FP8 all active simultaneously!
        g = torch.Generator(device="cpu").manual_seed(91)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(4, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
                indexing="ij",
            ),
            dim=0,
        ).reshape(3, -1)
        v_pos = torch.stack((grid, grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
        v_lat = torch.randn((1, 256, 32), generator=g).to(device=device, dtype=dtype)
        v_mask_t2v = torch.ones((1, 256, 1), device=device, dtype=dtype)
        v_ctx = torch.randn((1, 16, 1024), generator=g).to(device=device, dtype=dtype)
        a_lat = torch.randn((1, 32, 32), generator=g).to(device=device, dtype=dtype)
        a_mask = torch.ones((1, 32, 1), device=device, dtype=dtype)
        a_grid = torch.arange(32, dtype=torch.float32).reshape(1, -1)
        a_pos = torch.stack((a_grid, a_grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
        a_ctx = torch.randn((1, 16, 512), generator=g).to(device=device, dtype=dtype)

        for step, sigma in enumerate((0.9, 0.7, 0.4)):
            inp_s2 = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(latent=v_lat, clean_latent=v_lat, denoise_mask=v_mask_t2v, positions=v_pos),
                    "audio": ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos),
                },
                timestep=torch.tensor([sigma], device=device, dtype=dtype),
                conditioning={"video_context": v_ctx, "audio_context": a_ctx, "request_id": "sol-full-req"},
                step_index=step,
                total_steps=3,
            )
            out_s2 = den(inp_s2).samples
            assert torch.isfinite(out_s2["video"]).all() and torch.isfinite(out_s2["audio"]).all()

        # Stage 2 I2V (non-uniform denoise_mask: frame-0 tokens 0..63 conditioned = 0.0 -> protected!)
        v_mask_i2v = torch.ones((1, 256, 1), device=device, dtype=dtype)
        v_mask_i2v[:, :64, :] = 0.0
        for step, sigma in enumerate((0.9, 0.7, 0.4)):
            inp_i2v = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(latent=v_lat, clean_latent=v_lat, denoise_mask=v_mask_i2v, positions=v_pos),
                    "audio": ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos),
                },
                timestep=torch.tensor([sigma], device=device, dtype=dtype),
                conditioning={"video_context": v_ctx, "audio_context": a_ctx, "request_id": "sol-i2v-req"},
                step_index=step,
                total_steps=3,
            )
            out_i2v = den(inp_i2v).samples
            assert torch.isfinite(out_i2v["video"]).all() and torch.isfinite(out_i2v["audio"]).all()

    rep = den.runtime_optimization_report("sol-full-req")
    assert rep["feature_cache"]["algorithm"] == "firstblock"
    assert rep["feature_cache"]["hits"] > 0
    assert rep["feature_cache"]["skipped_block_calls"] > 0
    assert rep["token_pruning"]["pruned_calls"] == 6  # 3 T2V + 3 I2V Stage-2 steps
    assert rep["token_pruning"]["last_kept_tokens"] == 192
    assert rep["cuda_graph"]["capture_failed"] == 0
    assert rep["cuda_graph"]["cache_skips"] > 0
    pisa_rep = attention_provider_runtime_report().get("pisa", {})
    assert pisa_rep.get("successes", 0) > 0
    assert pisa_rep.get("fallbacks", 0) == 0
    assert pisa_rep.get("errors", 0) == 0
    fp8_fb = sum(int(getattr(m, "dense_fallback_calls", 0)) for m in m_av.modules() if isinstance(m, Float8Linear))
    assert fp8_fb == 0
    den.end_feature_cache_request("sol-full-req")
    _release_denoiser(den)


@pytest.mark.unit
def test_ltx_wave3_edge_cases_regression_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression unit tests for NaN-safe protection, mixed-dtype restore, empty-modality FBCache, and config guards."""
    from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import _resolve_protected_tokens
    from worldfoundry.base_models.diffusion_model.models.networks.ltx import acceleration as acc_mod
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.acceleration import (
        LTXFirstBlockCache,
        LTXFirstBlockCacheConfig,
        LTXTokenPruningConfig,
        restore_ltx_pruned_tokens,
        select_ltx_salient_tokens,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality

    # 1. NaN / +inf unprotected activations must NEVER outrank protected conditioning tokens.
    h = torch.randn((1, 16, 8), dtype=torch.float32)
    h[0, 15, 0] = float("nan")
    h[0, 14, 0] = float("inf")
    prot = torch.zeros(16, dtype=torch.bool)
    prot[:4] = True
    kept = select_ltx_salient_tokens(h, 4, protected_mask=prot)
    assert kept.tolist() == [0, 1, 2, 3]

    # 2. Mixed-dtype token restoration (fp32 pre_hidden + bf16 mid_out_hidden) works in all modes.
    pre = torch.randn((1, 16, 8), dtype=torch.float32)
    mid_in = pre[:, :8].to(torch.bfloat16)
    mid_out = (mid_in + 0.5).to(torch.bfloat16)
    idx = torch.arange(8, dtype=torch.long)
    for mode in ("zero", "mean_delta", "nearest_delta"):
        restored = restore_ltx_pruned_tokens(pre, mid_in, mid_out, idx, restore_mode=mode)
        assert restored.dtype == torch.float32 and torch.isfinite(restored).all()

    # 3. FBCache treats empty/disabled modality tensors (numel() == 0) cleanly instead of producing NaN metrics.
    fb_cfg = LTXFirstBlockCacheConfig(
        residual_diff_threshold=0.5,
        audio_diff_threshold=0.5,
        dense_first=1,
        dense_last=0,
        min_steps_to_enable=4,
    )
    with torch.inference_mode():
        fb_va = LTXFirstBlockCache(fb_cfg)
        v_in = torch.ones((1, 16, 8))
        v_out = v_in + 0.1
        a_empty = torch.empty((1, 0, 8))
        hit0, r0, m0, vp0, ap0 = fb_va.evaluate_probe(
            step=0,
            total_steps=5,
            num_blocks=4,
            block0_video_in=v_in,
            block0_video_out=v_out,
            block0_audio_in=a_empty,
            block0_audio_out=a_empty,
        )
        assert not hit0
        fb_va.record_miss(
            step=0,
            num_blocks=4,
            reason=r0,
            metric=m0,
            v_probe=vp0,
            a_probe=ap0,
            video_tail_base=v_out,
            video_tail_out=v_out + 0.2,
            audio_tail_base=a_empty,
            audio_tail_out=a_empty,
        )
        hit1, r1, _, _, _ = fb_va.evaluate_probe(
            step=1,
            total_steps=5,
            num_blocks=4,
            block0_video_in=v_in,
            block0_video_out=v_out,
            block0_audio_in=a_empty,
            block0_audio_out=a_empty,
        )
        assert hit1 and r1 == "below-threshold"

        fb_ao = LTXFirstBlockCache(fb_cfg)
        v_empty = torch.empty((1, 0, 8))
        a_in = torch.ones((1, 16, 8))
        a_out = a_in + 0.1
        hit0_a, r0_a, m0_a, vp0_a, ap0_a = fb_ao.evaluate_probe(
            step=0,
            total_steps=5,
            num_blocks=4,
            block0_video_in=v_empty,
            block0_video_out=v_empty,
            block0_audio_in=a_in,
            block0_audio_out=a_out,
        )
        assert not hit0_a
        fb_ao.record_miss(
            step=0,
            num_blocks=4,
            reason=r0_a,
            metric=m0_a,
            v_probe=vp0_a,
            a_probe=ap0_a,
            video_tail_base=v_empty,
            video_tail_out=v_empty,
            audio_tail_base=a_out,
            audio_tail_out=a_out + 0.2,
        )
        hit1_a, r1_a, _, _, _ = fb_ao.evaluate_probe(
            step=1,
            total_steps=5,
            num_blocks=4,
            block0_video_in=v_empty,
            block0_video_out=v_empty,
            block0_audio_in=a_in,
            block0_audio_out=a_out,
        )
        assert hit1_a and r1_a == "below-threshold"
        hit_empty, r_empty, _, _, _ = fb_ao.evaluate_probe(
            step=1,
            total_steps=5,
            num_blocks=4,
            block0_video_in=v_empty,
            block0_video_out=v_empty,
            block0_audio_in=a_empty,
            block0_audio_out=a_empty,
        )
        assert not hit_empty and r_empty == "empty-probe"

    # 4. configure_attention_backend validates upfront with zero partial mutation on failure.
    av_model = _build_small_av_model(num_layers=6, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    blocks = [cast(BasicAVTransformerBlock, b) for b in av_model.transformer_blocks]
    before = [blk.attn1.attention_backend for blk in blocks]
    with pytest.raises(ValueError, match="exact attention backend"):
        av_model.configure_attention_backend("pisa", self_only=True, dense_blocks=(0, 1, -1), dense_backend="sol_attn")
    assert [blk.attn1.attention_backend for blk in blocks] == before

    # 5. _resolve_protected_tokens tracks keyframes_mask in-place mutations and protects uniform all-zero denoise_mask.
    pcache: dict[tuple[int, int], Any] = {}
    with torch.no_grad():
        dm = torch.ones((1, 16, 1))
        kf = torch.zeros((1, 16, 1))
        kf[:, :2, :] = 1.0
        _, c1 = _resolve_protected_tokens(pcache, dm, kf, uniform_denoise_mask=True, allow_inference_cache=True)
        kf[:, :6, :] = 1.0
        _, c2 = _resolve_protected_tokens(pcache, dm, kf, uniform_denoise_mask=True, allow_inference_cache=True)
        assert c1 == 2 and c2 == 6
        dm_zero = torch.zeros((1, 16, 1))
        m_zero, c_zero = _resolve_protected_tokens(
            pcache, dm_zero, None, uniform_denoise_mask=True, allow_inference_cache=True
        )
        assert c_zero == 16 and m_zero is not None and bool(m_zero.all().item())

    # 6. accelerations.fbcache honors fbcache_max_tokens and rejects conflicting runtime fbcache option;
    #    PISA tile_3d schedule honors min_sequence_length and defaults to 1024.
    def _ctx(opts: dict[str, Any]) -> ComponentBuildContext:
        return ComponentBuildContext(
            key=ComponentKey(kind=ComponentKind.DENOISER, name="transformer"),
            model_id="test-ltx",
            checkpoints={},
            policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.float32, options=opts),
            component_options={},
        )

    wrap_fb = _as_av_wrapper(
        _build_small_av_model(num_layers=4, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    )
    _apply_ltx_post_load_accelerations(
        wrap_fb,
        _ctx({"accelerations": {"fbcache": {"residual_diff_threshold": 0.16}}, "fbcache_max_tokens": 2048}),
    )
    den_fb = LTXJointDenoiser(wrap_fb, compute_dtype=torch.float32, enable_cuda_graph="off")
    assert den_fb._fbcache_config.max_tokens == 2048
    with pytest.raises(ValueError, match="both runtime options and accelerations"):
        _apply_ltx_post_load_accelerations(
            wrap_fb,
            _ctx({"accelerations": {"fbcache": {"residual_diff_threshold": 0.12}}, "fbcache": 0.16}),
        )

    m_pisa_ms = _build_small_av_model(num_layers=4, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    _apply_ltx_post_load_accelerations(
        _as_av_wrapper(m_pisa_ms),
        _ctx(
            {
                "accelerations": {
                    "attention_policy": {
                        "self": {"backend": "pisa", "density": 0.15, "min_sequence_length": 4096, "tile_3d": (1, 4, 4)}
                    }
                }
            }
        ),
    )
    assert m_pisa_ms.__dict__["_worldfoundry_sol_schedule"]["min_tokens"] == 4096

    m_pisa_def = _build_small_av_model(
        num_layers=4, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64
    )
    _apply_ltx_post_load_accelerations(
        _as_av_wrapper(m_pisa_def),
        _ctx(
            {
                "accelerations": {
                    "attention_policy": {"self": {"backend": "pisa", "density": 0.15, "tile_3d": (1, 4, 4)}}
                }
            }
        ),
    )
    assert m_pisa_def.__dict__["_worldfoundry_sol_schedule"]["min_tokens"] == 1024

    # 7. Invalid WORLDFOUNDRY_PISA_MIN_SEQUENCE fails closed with descriptive ValueError.
    attn = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128).eval()
    attn.set_attention_backend("pisa", {"density": 0.25, "block_size": 64})
    monkeypatch.setenv("WORLDFOUNDRY_PISA_MIN_SEQUENCE", "not-an-int")
    with pytest.raises(ValueError, match="WORLDFOUNDRY_PISA_MIN_SEQUENCE"):
        with torch.inference_mode():
            attn(torch.randn((1, 16, 256)))
    monkeypatch.delenv("WORLDFOUNDRY_PISA_MIN_SEQUENCE", raising=False)

    # 8. Direct LTXModel.forward skips derive_ltx_protected_token_mask when pruning is bypassed by distilled guard.
    m_guard = _build_small_av_model(num_layers=4, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
    m_guard.configure_token_pruning(
        LTXTokenPruningConfig(
            keep_ratio=0.75, start_block=1, end_block=-1, min_tokens=16, allow_distilled_pruning=False
        )
    )
    derive_calls = [0]
    orig_derive = acc_mod.derive_ltx_protected_token_mask

    def _spy_derive(*args: Any, **kwargs: Any) -> Any:
        derive_calls[0] += 1
        return orig_derive(*args, **kwargs)

    monkeypatch.setattr(acc_mod, "derive_ltx_protected_token_mask", _spy_derive)
    v_mod = Modality(
        enabled=True,
        latent=torch.randn((1, 32, 32)),
        sigma=torch.tensor([0.5]),
        timesteps=torch.full((1, 32, 1), 0.5),
        positions=torch.zeros((1, 3, 32, 2)),
        context=torch.randn((1, 8, 256)),
    )
    with pytest.warns(UserWarning, match="distilled short-step safety guard"):
        with torch.inference_mode():
            m_guard(video=v_mod, audio=None, feature_cache_step=0, feature_cache_total_steps=3)
    assert derive_calls[0] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not _PISA_USABLE, reason="requires Hopper/DC-Blackwell CUDA GPU with Triton TMA PISA support")
def test_gpu_wave3_pisa_fbcache_teacache_pruning_cfg_and_resolution_switches() -> None:
    """Verify strict PISA length gates, 2-stage FBCache telemetry, piecewise skip counts, CFG, and resolution switches."""
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.acceleration import (
        LTXFirstBlockCacheConfig,
        LTXTokenPruningConfig,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Strict PISA honors min_sequence_length and short K (< min_tokens) as clean dense fallbacks.
    attn = Attention(query_dim=256, context_dim=None, heads=2, dim_head=128).to(device=device, dtype=dtype).eval()
    x128 = torch.randn((1, 128, 256), device=device, dtype=dtype)
    ctx32 = torch.randn((1, 32, 256), device=device, dtype=dtype)
    with torch.inference_mode():
        y_ref = attn(x128)
    attn.set_attention_backend("pisa", {"density": 0.25, "block_size": 64, "min_sequence_length": 4096, "strict": True})
    with torch.inference_mode():
        torch.testing.assert_close(attn(x128), y_ref, rtol=1e-2, atol=1e-2)
    attn.set_attention_backend("pisa", {"density": 0.25, "block_size": 64, "min_tokens": 64, "strict": True})
    with torch.inference_mode():
        assert torch.isfinite(attn(x128, context=ctx32)).all()

    def _grid_pos(f: int, h: int, w: int) -> torch.Tensor:
        g3 = torch.stack(
            torch.meshgrid(
                torch.arange(f, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            ),
            dim=0,
        ).reshape(3, -1)
        return torch.stack((g3, g3 + 1.0), dim=-1).unsqueeze(0).to(device=device)

    def _av_inp(
        f: int,
        h: int,
        w: int,
        *,
        sigma: float,
        step: int,
        total_steps: int,
        req_id: str,
        branch: str = "positive",
    ) -> MultiModalDenoiserInput:
        t_v = f * h * w
        gen = torch.Generator(device="cpu").manual_seed(77 + step)
        v_lat = torch.randn((1, t_v, 32), generator=gen).to(device=device, dtype=dtype)
        v_mask = torch.ones((1, t_v, 1), device=device, dtype=dtype)
        v_ctx = torch.randn((1, 8, 256), generator=gen).to(device=device, dtype=dtype)
        a_lat = torch.randn((1, 16, 32), generator=gen).to(device=device, dtype=dtype)
        a_mask = torch.ones((1, 16, 1), device=device, dtype=dtype)
        a_grid = torch.arange(16, dtype=torch.float32).reshape(1, -1)
        a_pos = torch.stack((a_grid, a_grid + 1.0), dim=-1).unsqueeze(0).to(device=device)
        a_ctx = torch.randn((1, 8, 128), generator=gen).to(device=device, dtype=dtype)
        return MultiModalDenoiserInput(
            modalities={
                "video": ModalityState(
                    latent=v_lat, clean_latent=v_lat, denoise_mask=v_mask, positions=_grid_pos(f, h, w)
                ),
                "audio": ModalityState(latent=a_lat, clean_latent=a_lat, denoise_mask=a_mask, positions=a_pos),
            },
            timestep=torch.tensor([sigma], device=device, dtype=dtype),
            conditioning={"video_context": v_ctx, "audio_context": a_ctx, "request_id": req_id},
            step_index=step,
            total_steps=total_steps,
            branch=branch,
        )

    # 2. Two-stage FBCache under same request_id preserves Stage-1 telemetry + interleaved CFG dual-branch + resolution switches
    #    including non-multiple-of-64 (T=130) with PISA + tile_3d (no spurious _sink segment).
    m = (
        _build_small_av_model(num_layers=6, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
        .to(device=device, dtype=dtype)
        .eval()
    )
    m.configure_attention_backend(
        "pisa",
        self_only=True,
        dense_blocks=(0, 1, -1),
        dense_backend="torch",
        backend_options={"density": 0.25, "block_size": 64, "min_tokens": 96},
    )
    m.__dict__["_worldfoundry_sol_schedule"] = {
        "dense_steps": 0,
        "dense_blocks": (0, 1, -1),
        "tile_3d": (1, 4, 4),
        "min_tokens": 96,
    }
    fb_cfg = LTXFirstBlockCacheConfig(
        residual_diff_threshold=10.0,
        dense_first=1,
        dense_last=1,
        max_consecutive_skips=2,
        min_steps_to_enable=4,
        max_tokens=4096,
    )
    den = LTXJointDenoiser(_as_av_wrapper(m), compute_dtype=dtype, enable_cuda_graph="piecewise", fbcache_config=fb_cfg)
    reset_attention_provider_runtime()
    with torch.inference_mode():
        # Stage 1 (6 steps, T=64) -> Stage 2 (3 steps, T=130 non-mult-64 <= max_tokens) on same request_id
        for s in range(6):
            out = den(_av_inp(1, 8, 8, sigma=0.5 - 0.001 * s, step=s, total_steps=6, req_id="req-2stage"))
            assert torch.isfinite(out.samples["video"]).all()
        s1_hits = den.runtime_optimization_report("req-2stage")["feature_cache"]["hits"]
        assert s1_hits > 0
        for s in range(3):
            out = den(_av_inp(2, 5, 13, sigma=0.8 - 0.2 * s, step=s, total_steps=3, req_id="req-2stage"))
            assert torch.isfinite(out.samples["video"]).all()
        rep_2s = den.runtime_optimization_report("req-2stage")
        assert rep_2s["feature_cache"]["hits"] == s1_hits
        assert rep_2s["feature_cache"]["events"] == 9
        assert all("_sink" not in seg for seg in rep_2s["cuda_graph"]["segments"])

        # Interleaved dual-branch CFG + sequential resolution switch (256 -> 64)
        for s in range(5):
            for req in ("cfg-A", "cfg-B"):
                for br in ("positive", "negative"):
                    out = den(_av_inp(4, 8, 8, sigma=0.5 - 0.001 * s, step=s, total_steps=5, req_id=req, branch=br))
                    assert torch.isfinite(out.samples["video"]).all()
        for s in range(5):
            out = den(_av_inp(1, 8, 8, sigma=0.5 - 0.001 * s, step=s, total_steps=5, req_id="res-back"))
            assert torch.isfinite(out.samples["video"]).all()

    rep_cfg_a = den.runtime_optimization_report("cfg-A")["feature_cache"]
    assert set(rep_cfg_a["branches"]) == {"positive", "negative"} and rep_cfg_a["hits"] > 0
    assert den.runtime_optimization_report("res-back")["cuda_graph"]["capture_failed"] == 0
    pisa_rep = attention_provider_runtime_report().get("pisa", {})
    assert pisa_rep.get("successes", 0) > 0 and pisa_rep.get("fallbacks", 0) == 0 and pisa_rep.get("errors", 0) == 0
    _release_denoiser(den)

    # 3. FBCache + token_pruning + piecewise AND TeaCache + token_pruning + piecewise preserve cache_skips parity.
    for use_fb in (True, False):
        m_tp = (
            _build_small_av_model(num_layers=6, video_heads=2, video_dim_head=128, audio_heads=2, audio_dim_head=64)
            .to(device=device, dtype=dtype)
            .eval()
        )
        m_tp.configure_token_pruning(
            LTXTokenPruningConfig(
                keep_ratio=0.75, start_block=2, end_block=-2, min_tokens=64, allow_distilled_pruning=True
            )
        )
        if use_fb:
            den_tp = LTXJointDenoiser(
                _as_av_wrapper(m_tp),
                compute_dtype=dtype,
                enable_cuda_graph="piecewise",
                fbcache_config=fb_cfg,
            )
        else:
            den_tp = LTXJointDenoiser(
                _as_av_wrapper(m_tp),
                compute_dtype=dtype,
                enable_cuda_graph="piecewise",
                teacache_threshold=10.0,
                teacache_max_tokens=4096,
            )
        req_name = "fb-tp" if use_fb else "tc-tp"
        with torch.inference_mode():
            for s in range(6):
                out = den_tp(_av_inp(2, 8, 8, sigma=0.5 - 0.001 * s, step=s, total_steps=6, req_id=req_name))
                assert torch.isfinite(out.samples["video"]).all()
        rep_tp = den_tp.runtime_optimization_report(req_name)
        assert rep_tp["feature_cache"]["hits"] > 0
        assert rep_tp["cuda_graph"]["cache_skips"] == rep_tp["feature_cache"]["hits"]
        assert rep_tp["cuda_graph"]["capture_failed"] == 0
        _release_denoiser(den_tp)

    # 4. LTXVideoDenoiser (video-only) end-to-end with PISA + FBCache + token_pruning + piecewise graph.
    m_v = _build_small_video_model(num_layers=6, heads=2, dim_head=128, device=device, dtype=dtype).eval()
    m_v.configure_attention_backend(
        "pisa",
        self_only=True,
        dense_blocks=(0, 1, -1),
        dense_backend="torch",
        backend_options={"density": 0.25, "block_size": 64, "min_tokens": 128},
    )
    m_v.configure_token_pruning(
        LTXTokenPruningConfig(
            keep_ratio=0.75, start_block=2, end_block=-2, min_tokens=128, allow_distilled_pruning=True
        )
    )
    fb_v = LTXFirstBlockCacheConfig(
        residual_diff_threshold=10.0,
        dense_first=1,
        dense_last=1,
        max_consecutive_skips=2,
        min_steps_to_enable=4,
        max_tokens=96,
    )
    den_v = LTXVideoDenoiser(
        _as_video_wrapper(m_v), compute_dtype=dtype, enable_cuda_graph="piecewise", fbcache_config=fb_v
    )
    with torch.inference_mode():
        for s in range(5):
            v_lat = torch.randn((1, 64, 32), device=device, dtype=dtype)
            inp_v1 = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(
                        latent=v_lat,
                        clean_latent=v_lat,
                        denoise_mask=torch.ones((1, 64, 1), device=device, dtype=dtype),
                        positions=_grid_pos(1, 8, 8),
                    )
                },
                timestep=torch.tensor([0.5 - 0.001 * s], device=device, dtype=dtype),
                conditioning={
                    "video_context": torch.randn((1, 8, 256), device=device, dtype=dtype),
                    "request_id": "vonly",
                },
                step_index=s,
                total_steps=5,
            )
            assert torch.isfinite(den_v(inp_v1).samples["video"]).all()
        for s in range(3):
            v_lat = torch.randn((1, 256, 32), device=device, dtype=dtype)
            inp_v2 = MultiModalDenoiserInput(
                modalities={
                    "video": ModalityState(
                        latent=v_lat,
                        clean_latent=v_lat,
                        denoise_mask=torch.ones((1, 256, 1), device=device, dtype=dtype),
                        positions=_grid_pos(4, 8, 8),
                    )
                },
                timestep=torch.tensor([0.8 - 0.2 * s], device=device, dtype=dtype),
                conditioning={
                    "video_context": torch.randn((1, 8, 256), device=device, dtype=dtype),
                    "request_id": "vonly",
                },
                step_index=s,
                total_steps=3,
            )
            assert torch.isfinite(den_v(inp_v2).samples["video"]).all()
    rep_v = den_v.runtime_optimization_report("vonly")
    assert rep_v["feature_cache"]["hits"] > 0
    assert rep_v["token_pruning"]["pruned_calls"] == 3
    assert rep_v["cuda_graph"]["capture_failed"] == 0
    _release_denoiser(den_v)
