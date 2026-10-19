# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Real small Wan components for prepared conditions, compilation and warmup."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import install_static_cross_kv_cache
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.runners.base import NativeDiffusionRunner, RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.flow_match import FlowMatchEulerScheduler
from worldfoundry.core.model_loading.policy import RuntimePolicy

pytestmark = pytest.mark.unit

CONFIG = dict(
    dim=48,
    in_dim=4,
    ffn_dim=64,
    out_dim=4,
    text_dim=16,
    freq_dim=16,
    eps=1e-6,
    patch_size=(1, 2, 2),
    num_heads=2,
    num_layers=2,
    has_image_input=False,
    require_vae_embedding=False,
)


def model() -> WanModel:
    result = WanModel(**CONFIG).eval()
    result.set_attention_compatibility_mode(True)
    return result


def model_input(context: torch.Tensor, *, request_id: str, step: int = 0, branch: str = "positive") -> DenoiserInput:
    return DenoiserInput(
        latents=torch.arange(64, device=context.device, dtype=context.dtype).reshape(1, 4, 1, 4, 4) / 64,
        timestep=torch.tensor([10.0 - step], device=context.device),
        next_timestep=torch.tensor([9.0 - step], device=context.device),
        conditioning={"context": context},
        step_index=step,
        total_steps=2,
        branch=branch,
        request_id=request_id,
    )


@torch.no_grad()
def test_prepared_cross_kv_preserves_branches_requests_and_releases_cache() -> None:
    torch.manual_seed(7)
    reference = model()
    candidate = deepcopy(reference)
    cache = install_static_cross_kv_cache(candidate)
    denoiser = WanDenoiser(candidate, compute_dtype=torch.float32, enable_cuda_graph=True)
    a, b = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    retained = []
    for request_id, positive in (("a", a), ("b", b), ("a-again", a)):
        for step in range(2):
            for branch, context in (("positive", positive), ("negative", -positive)):
                item = model_input(context, request_id=request_id, step=step, branch=branch)
                expected = reference(x=item.latents, timestep=item.timestep, context=context)
                actual = denoiser(item).sample
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                retained.append((actual, expected.clone()))
        denoiser.end_request(request_id)
        assert cache.report()["entries"] == cache.report()["condition_entries"] == 0
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_prepared_image_cross_kv_and_per_token_timesteps() -> None:
    config = dict(CONFIG, has_image_input=True, per_token_timestep=True)
    reference = WanModel(**config).eval()
    reference.set_attention_compatibility_mode(True)
    candidate = deepcopy(reference)
    install_static_cross_kv_cache(candidate)
    denoiser = WanDenoiser(candidate, compute_dtype=torch.float32, enable_cuda_graph=True)
    x = torch.randn(2, 4, 1, 4, 4)
    context = torch.randn(2, 5, 16)
    clip = torch.randn(2, 257, 1280)
    timestep = torch.tensor([10.0, 5.0])
    item = DenoiserInput(
        latents=x,
        timestep=timestep,
        next_timestep=timestep - 1,
        conditioning={
            "context": context,
            "clip_feature": clip,
            "condition_latents": torch.empty_like(x),
            "denoise_mask": torch.ones_like(x),
        },
        step_index=0,
        total_steps=1,
        request_id="image",
    )
    expected = reference(x, timestep[:, None].expand(2, 4), context, clip_feature=clip)
    actual = denoiser(item).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    denoiser.end_request("image")


@pytest.mark.parametrize("num_layers", (2, 12))
@torch.no_grad()
def test_loader_regional_compile_preserves_model_keys_and_live_inputs(tmp_path: Path, num_layers: int) -> None:
    from safetensors.torch import save_file
    from torch._dynamo.utils import counters

    config = dict(CONFIG, num_layers=num_layers)
    original = WanModel(**config).eval()
    original.set_attention_compatibility_mode(True)
    path = tmp_path / "model.safetensors"
    save_file(original.state_dict(), str(path))
    candidate = (
        NativeModuleLoader()
        .load(
            ModuleLoadSpec(module_class=WanModel, config=config, compile_regions=("blocks",)),
            CheckpointSpec(source=str(path)),
            RuntimePolicy(
                compile=True,
                options={
                    "compile_scope": "regional",
                    "compile_backend": "eager",
                    "compile_fullgraph": True,
                    "static_cross_kv": True,
                },
            ),
        )
        .eval()
    )
    candidate.set_attention_compatibility_mode(True)
    assert type(candidate) is WanModel
    assert candidate.state_dict().keys() == original.state_dict().keys()
    assert candidate._worldfoundry_compile_config["regions"] == [f"blocks.{index}" for index in range(num_layers)]
    traces_before = counters["stats"]["unique_graphs"]
    denoiser = WanDenoiser(candidate, compute_dtype=torch.float32)
    context = torch.randn(1, 5, 16)
    for index in range(2):
        item = model_input(context + index, request_id=f"compile-{index}")
        # Constrain SDPA outside Dynamo so eager and compiled calls use the
        # same reduction implementation, including the CPU Flash backend.
        with (
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
            torch.autocast("cpu", dtype=torch.bfloat16, enabled=num_layers > 2),
        ):
            actual = denoiser(item).sample
            expected = original(item.latents, item.timestep, item.conditioning["context"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert candidate._worldfoundry_compile_runtime["request_calls"] == num_layers
        denoiser.end_request(item.request_id)
    # Mixed-precision entry/remaining blocks need two compute graphs, not one
    # object-identity specialization for every layer or request.
    assert counters["stats"]["unique_graphs"] - traces_before <= 2


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@torch.no_grad()
def test_inplace_gate_preserves_functional_dtype_promotion(dtype: torch.dtype) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import GateModule

    value = torch.randn(2, 8, dtype=dtype)
    original = value.clone()
    update = torch.randn(2, 8, dtype=torch.bfloat16)
    gate = torch.randn(1, 8, dtype=torch.float32)
    expected = value + update * gate
    actual = GateModule()(value, gate, update, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if dtype == expected.dtype:
        assert actual is value
    else:
        assert actual.dtype == torch.float32
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("graph_enabled", (False, True))
def test_component_execution_options_reach_loader(monkeypatch: pytest.MonkeyPatch, graph_enabled: bool) -> None:
    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext
    from worldfoundry.base_models.diffusion_model.models.denoisers.wan import _build_wan_denoiser

    policies = []

    def load(
        self: NativeModuleLoader, spec: ModuleLoadSpec, checkpoint: CheckpointSpec, policy: RuntimePolicy
    ) -> WanModel:
        assert spec.compile_regions == ("blocks",)
        policies.append(policy)
        return model()

    monkeypatch.setattr(NativeModuleLoader, "load", load)
    context = ComponentBuildContext(
        model_id="tiny-wan",
        key="denoiser:main",
        checkpoints={"weights": CheckpointSpec(source="unused")},
        policy=RuntimePolicy(compile=True, options={"cuda_graph": not graph_enabled, "static_cross_kv": True}),
        component_options={"cuda_graph": graph_enabled, "static_cross_kv": False, "compile_scope": "regional"},
    )
    denoiser = _build_wan_denoiser(context, config=CONFIG)
    assert policies[0].options["cuda_graph"] is graph_enabled
    assert policies[0].options["static_cross_kv"] is False
    assert policies[0].options["compile_scope"] == "regional"
    assert (denoiser._graph_runner is not None) is graph_enabled


class Conditioner:
    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        scale = 1 if request.prompts[0] == "a" else 2
        return Conditioning(
            positive={"context": torch.full((1, 5, 16), scale, device=device, dtype=dtype)},
            negative={"context": torch.zeros((1, 5, 16), device=device, dtype=dtype)},
        )


class Initializer:
    def initialize(
        self, request: DiffusionRequest, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.randn((1, 4, 1, 4, 4), device=device, dtype=dtype, generator=generator)


class Decoder:
    fail = False
    calls = 0

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        self.calls += 1
        if self.fail:
            raise RuntimeError("decoder failure")
        return latents.clone()


def pipeline() -> NativeDiffusionPipeline:
    network = model()
    install_static_cross_kv_cache(network)
    return NativeDiffusionPipeline(
        NativeDiffusionRunner(
            model_id="tiny-wan",
            components=RunnerComponents(
                denoiser=WanDenoiser(network, compute_dtype=torch.float32, enable_cuda_graph=True),
                conditioner=Conditioner(),
                latent_initializer=Initializer(),
                scheduler=FlowMatchEulerScheduler(),
                decoder=Decoder(),
            ),
        )
    )


def test_warmup_runs_full_request_and_preserves_rng_and_next_request() -> None:
    pipe = pipeline()
    request = DiffusionRequest(
        prompt="a",
        height=8,
        width=8,
        num_frames=1,
        sampling=SamplingConfig(num_inference_steps=2, guidance_scale=2, seed=11),
    )
    expected = pipe(request)
    rng = torch.get_rng_state().clone()
    reports = pipe.warmup((request, replace(request, prompt="b")))
    assert len(reports) == 2
    assert pipe.components.decoder.calls == 3
    assert torch.equal(torch.get_rng_state(), rng)
    assert pipe.components.denoiser._graph_runner.capture_on_miss is False
    actual = pipe(request)
    torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)
    assert pipe.components.denoiser._optimization_request_windows == {}
    assert pipe.components.denoiser.model._worldfoundry_static_cross_kv.report()["entries"] == 0


def test_failed_warmup_releases_request_state_and_keeps_previous_capture_policy() -> None:
    pipe = pipeline()
    request = DiffusionRequest(prompt="a", sampling=SamplingConfig(num_inference_steps=2))
    expected = pipe(request)
    pipe.components.decoder.fail = True
    with pytest.raises(RuntimeError, match="decoder failure"):
        pipe.warmup((request,))
    denoiser = pipe.components.denoiser
    assert denoiser._optimization_request_windows == {}
    assert denoiser._graph_runner.capture_on_miss is True
    assert denoiser.model._worldfoundry_static_cross_kv.report()["entries"] == 0
    pipe.components.decoder.fail = False
    torch.testing.assert_close(pipe(request).latents, expected.latents, rtol=0, atol=0)


def test_warmup_covers_both_routed_experts_and_cfg_branches() -> None:
    from worldfoundry.base_models.diffusion_model.models.denoisers.wan import Wan22DualExpertDenoiser
    from worldfoundry.base_models.diffusion_model.runners.base import Wan22DualExpertGuidanceRunner

    pipe = pipeline()
    high = pipe.components.denoiser
    low = pipeline().components.denoiser
    dual = Wan22DualExpertDenoiser(high, low, boundary_ratio=0.75, num_train_timesteps=1)
    pipe.runner = Wan22DualExpertGuidanceRunner(
        model_id="tiny-dual-wan",
        components=replace(pipe.components, denoiser=dual),
        boundary_ratio=0.75,
        num_train_timesteps=1,
        high_noise_guidance_scale=2,
        low_noise_guidance_scale=2,
    )
    calls = [0, 0]

    def count(index: int) -> Callable[..., None]:
        def hook(*args: object) -> None:
            calls[index] += 1

        return hook

    handles = [high.model.register_forward_hook(count(0)), low.model.register_forward_hook(count(1))]
    try:
        pipe.warmup((DiffusionRequest(prompt="a", sampling=SamplingConfig(num_inference_steps=2, guidance_scale=2)),))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [2, 2]
    assert dual._route_requests == {}
    for expert in (high, low):
        assert not expert._graph_runner.capture_on_miss
        assert expert.model._worldfoundry_static_cross_kv.report()["entries"] == 0


def test_warmup_preserves_masked_latent_projection_and_request_isolation() -> None:
    from worldfoundry.base_models.diffusion_model.extensions.frozen_mask import FrozenLatentMaskExtension

    class MaskedConditioner(Conditioner):
        def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
            conditions = super().encode(request, device=device, dtype=dtype)
            clean = torch.full((1, 4, 1, 4, 4), 1 if request.prompts[0] == "a" else 2, device=device, dtype=dtype)
            mask = torch.ones_like(clean)
            mask[..., :2] = 0
            return replace(conditions, shared={"clean_latents": clean, "denoise_mask": mask})

    pipe = pipeline()
    pipe.runner.extensions = (FrozenLatentMaskExtension(),)
    pipe.runner.components = replace(pipe.components, conditioner=MaskedConditioner())
    request = DiffusionRequest(prompt="a", sampling=SamplingConfig(num_inference_steps=2, guidance_scale=2, seed=11))
    expected = pipe(request)
    pipe.warmup((request, replace(request, prompt="b")))
    actual = pipe(request)
    torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)
    assert torch.equal(actual.latents[..., :2], torch.ones_like(actual.latents[..., :2]))
    assert pipe.components.denoiser._optimization_request_windows == {}


def test_warmup_rejects_unsupported_runner_before_execution() -> None:
    pipe = pipeline()
    pipe.runner.extensions = (SimpleNamespace(),)
    with pytest.raises(ValueError, match="custom extensions"):
        pipe.warmup((DiffusionRequest(prompt="a"),))
    pipe = NativeDiffusionPipeline(SimpleNamespace())
    with pytest.raises(ValueError, match="stateless runner"):
        pipe.warmup((DiffusionRequest(prompt="a"),))
    with pytest.raises(ValueError, match="one or more"):
        pipe.warmup(())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "operation",
    (
        "pointwise",
        "rmsnorm",
        "layernorm",
        "linear-rmsnorm",
        "linear-gelu",
        "adaln",
        "gate",
        "inplace-gate",
    ),
)
@torch.no_grad()
def test_cuda_regional_compile_preserves_bfloat16_pointwise_rounding(operation: str) -> None:
    from worldfoundry.base_models.diffusion_model.loaders.module import _compile_policy_from_runtime
    from worldfoundry.core.execution.compile_cache import compile_callable_cached
    from worldfoundry.core.execution.compile_numerics import preserve_normalization_reductions
    from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner
    from worldfoundry.core.kernels.diffusion import layer_norm_scale_shift, residual_gate_add, residual_gate_add_
    from worldfoundry.core.nn.transformer.dit import RMSNorm

    torch.manual_seed(42)
    if operation in {"rmsnorm", "layernorm", "linear-rmsnorm", "linear-gelu"}:
        norm_class = torch.nn.LayerNorm if operation == "layernorm" else RMSNorm
        module = norm_class(1536, device="cuda", dtype=torch.float32).eval()
        module.weight.copy_(torch.randn_like(module.weight))
        inputs = (torch.randn(1, 1280, 1536, device="cuda", dtype=torch.bfloat16),)
        if operation.startswith("linear-"):
            linear = torch.nn.Linear(1536, 1536, device="cuda", dtype=torch.float32)
            linear.weight.copy_(torch.eye(1536, device="cuda"))
            linear.bias.fill_(0.3)
            following = module if operation == "linear-rmsnorm" else torch.nn.GELU(approximate="tanh")
            module = torch.nn.Sequential(linear, following).eval()
        preserve_normalization_reductions(module)
        compute = module.forward
    elif operation in {"adaln", "gate", "inplace-gate"}:
        compute = layer_norm_scale_shift if operation == "adaln" else residual_gate_add
        if operation == "inplace-gate":

            def compute(x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                return residual_gate_add_(x.clone(), update, gate)

        inputs = (
            torch.randn(
                1, 1280, 1536, device="cuda", dtype=(torch.float32 if operation == "inplace-gate" else torch.bfloat16)
            ),
            torch.randn(1, 1, 1536, device="cuda", dtype=torch.float32),
            torch.randn(1, 1, 1536, device="cuda", dtype=torch.float32),
        )
    else:

        def compute(x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return (x + residual) * gate

        inputs = tuple(torch.randn(4096, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    policy, options = _compile_policy_from_runtime(
        RuntimePolicy(
            compile=True,
            options={"compile_scope": "regional", "cuda_graph": True, "compile_fullgraph": True},
        )
    )
    compiled = compile_callable_cached(compute, policy=policy, options=options, strict=True)
    runner = InferenceCUDAGraphRunner(compiled, warmup=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(runner(*inputs), compute(*inputs), rtol=0, atol=0)
        runner.freeze()
        updated = tuple(value + 0.125 for value in inputs)
        torch.testing.assert_close(runner(*updated), compute(*updated), rtol=0, atol=0)
    assert runner.report()["capture"] == 1 and runner.report()["replay"] == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("inplace", (False, True))
@torch.no_grad()
def test_cuda_prepared_wan_replays_new_conditions_without_retaining_prior_request(
    tmp_path: Path,
    compiled: bool,
    inplace: bool,
) -> None:
    from safetensors.torch import save_file

    reference = model().cuda()
    if compiled:
        path = tmp_path / "model.safetensors"
        save_file({key: value.cpu() for key, value in reference.state_dict().items()}, str(path))
        candidate = (
            NativeModuleLoader()
            .load(
                ModuleLoadSpec(module_class=WanModel, config=CONFIG, compile_regions=("blocks",)),
                CheckpointSpec(source=str(path)),
                RuntimePolicy(
                    device="cuda",
                    compile=True,
                    options={
                        "compile_scope": "regional",
                        "compile_fullgraph": True,
                        "cuda_graph": True,
                    },
                ),
            )
            .eval()
        )
        candidate.set_attention_compatibility_mode(True)
    else:
        candidate = deepcopy(reference)
    cache = install_static_cross_kv_cache(candidate)
    denoiser = WanDenoiser(
        candidate,
        compute_dtype=torch.float32,
        enable_cuda_graph=True,
        inplace_residual=inplace,
    )
    graph = denoiser._graph_runner
    assert graph is not None
    contexts = [torch.randn(1, 5, 16, device="cuda") for _ in range(2)]
    with denoiser.execution_warmup():
        denoiser(model_input(contexts[0], request_id="warmup"))
        denoiser.end_request("warmup")
    assert graph.capture_on_miss is False
    retained = []
    for index in (0, 1, 0):
        request_id = f"case-{len(retained)}"
        item = model_input(contexts[index], request_id=request_id)
        expected = reference(item.latents, item.timestep, contexts[index])
        actual = denoiser(item).sample
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        receipt = graph.report()
        assert receipt["replay"] == 1
        assert receipt["capture"] == receipt["eager"] == receipt["capture_failed"] == 0
        if compiled:
            assert denoiser.runtime_optimization_report()["effective"]["compile"] == "compiled-cuda-graph-replayed"
        retained.append((actual, expected.clone()))
        denoiser.end_request(request_id)
        assert cache.report()["entries"] == 0
    # Between requests, explicit cache invalidation also covers live tensor
    # updates that retain storage (new parameter storage requires rebuilding).
    contexts[0].add_(0.25)
    reference.blocks[0].cross_attn.k.weight.add_(0.01)
    candidate.blocks[0].cross_attn.k.weight.add_(0.01)
    item = model_input(contexts[0], request_id="updated")
    expected = reference(item.latents, item.timestep, contexts[0])
    torch.testing.assert_close(denoiser(item).sample, expected, rtol=1e-5, atol=1e-5)
    denoiser.end_request("updated")
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
