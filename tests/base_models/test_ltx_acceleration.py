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
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import (
    NormLayerType,
    PaddingModeType,
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
    fused_ltx_post_sa_residual_rms_norm,
    fused_ltx_qk_rmsnorm_rope,
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
