# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent RNG, solver, CFG and output ownership for merged native requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import Wan22DualExpertDenoiser, WanDenoiser
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import WanTextToVideoLatentInitializer
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import Head, WanModel
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import install_static_cross_kv_cache
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.runners.base import (
    NativeDiffusionRunner,
    RunnerComponents,
    Wan22DualExpertGuidanceRunner,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler, WanFlowUniPCScheduler

pytestmark = pytest.mark.unit


class Conditioner:
    calls = 0

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        self.calls += 1

        def context(prompts: tuple[str, ...]) -> torch.Tensor:
            return (
                torch.tensor([len(prompt) / 100 for prompt in prompts], device=device, dtype=dtype)[:, None, None]
                .expand(-1, 5, 16)
                .clone()
            )

        return Conditioning(
            positive={"context": context(request.prompts)},
            negative={"context": context(request.negative_prompts or ("",) * request.batch_size)}
            if request.sampling.guidance_scale != 1
            else {},
        )


class Decoder:
    def __init__(self) -> None:
        self.requests: list[DiffusionRequest] = []
        self.fail = False

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("decode failed")
        return latents


class Denoiser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.ended: list[tuple[str, BaseException | None]] = []
        self.fail = False

    def validate_request_batching(self) -> None:
        pass

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        value = model_input
        self.calls.append((value.branch, value.latents.shape[0]))
        if self.fail:
            raise RuntimeError("denoiser failed")
        context = value.conditioning["context"]
        assert isinstance(context, torch.Tensor)
        return DenoiserOutput(value.latents * 0.125 + context[:, 0, 0, None, None, None, None])

    def end_request(self, request_id: str, *, error: BaseException | None) -> None:
        self.ended.append((request_id, error))


def make_pipeline(
    *, unipc: bool = False, real: bool = False, device: str = "cpu", graph: bool = False
) -> NativeDiffusionPipeline:
    denoiser: Denoiser | WanDenoiser = Denoiser()
    if real:
        torch.manual_seed(712)
        network = (
            WanModel(
                dim=48,
                in_dim=4,
                ffn_dim=64,
                out_dim=4,
                text_dim=16,
                freq_dim=16,
                eps=1e-6,
                patch_size=(1, 2, 2),
                num_heads=4,
                num_layers=2,
                has_image_input=False,
            )
            .eval()
            .to(device)
        )
        install_static_cross_kv_cache(network)
        denoiser = WanDenoiser(network, compute_dtype=torch.float32, enable_cuda_graph=graph)
    return NativeDiffusionPipeline(
        NativeDiffusionRunner(
            model_id="tiny-wan-batching",
            device=device,
            dtype=torch.float32,
            components=RunnerComponents(
                denoiser=denoiser,
                conditioner=Conditioner(),
                latent_initializer=WanTextToVideoLatentInitializer(channels=4, spatial_compression=2),
                scheduler=WanFlowUniPCScheduler() if unipc else WanFlowMatchEulerScheduler(),
                decoder=Decoder(),
            ),
        )
    )


def requests() -> tuple[DiffusionRequest, DiffusionRequest]:
    first = DiffusionRequest(
        prompt="short",
        negative_prompt="negative",
        height=8,
        width=8,
        num_frames=5,
        sampling=SamplingConfig(num_inference_steps=4, guidance_scale=3.0, seed=11),
        metadata={"caller": "a"},
    )
    return first, replace(
        first,
        prompt=("longer prompt", "third"),
        negative_prompt=("bad", "worse"),
        sampling=replace(first.sampling, seed=29),
        metadata={"caller": "b"},
    )


@pytest.mark.parametrize("unipc", [False, True])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_batch_matches_independent_requests_and_preserves_rng(unipc: bool, guidance: float) -> None:
    pipe = make_pipeline(unipc=unipc)
    cases = tuple(replace(case, sampling=replace(case.sampling, guidance_scale=guidance)) for case in requests())
    expected = tuple(pipe(case) for case in cases)
    pipe.components.denoiser.calls.clear()
    state = torch.get_rng_state().clone()
    actual = pipe.run_batch(cases)
    assert torch.equal(state, torch.get_rng_state())
    calls = 4 if guidance == 1 else 8
    assert len(pipe.components.denoiser.calls) == calls
    assert all(size == 3 for _, size in pipe.components.denoiser.calls)
    assert pipe.components.decoder.requests[-2:] == list(cases)
    assert len(pipe.components.denoiser.ended) == 3
    assert len({item[0] for item in pipe.components.denoiser.ended}) == 3
    for index, (output, reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(output.latents, reference.latents, rtol=0, atol=0)
        torch.testing.assert_close(output.sample, reference.sample, rtol=0, atol=0)
        assert output.metadata["seed"] == cases[index].sampling.seed
        assert output.metadata["request_metadata"] == cases[index].metadata
        receipt = output.metadata["request_batching"]
        assert isinstance(receipt, dict)
        assert receipt["denoiser_calls"] == calls
        assert receipt["request_count"] == 2
        assert receipt["sample_count"] == 3
        assert receipt["sample_range"] == ([0, 1] if index == 0 else [1, 3])
        assert receipt["batch_id"] == pipe.components.denoiser.ended[-1][0]
    assert actual[0].sample.untyped_storage().data_ptr() != actual[1].sample.untyped_storage().data_ptr()
    saved = actual[1].sample.clone()
    actual[0].sample.fill_(999)
    torch.testing.assert_close(actual[1].sample, saved, rtol=0, atol=0)
    repeated = pipe.run_batch(tuple(reversed(cases)))
    torch.testing.assert_close(repeated[0].latents, expected[1].latents, rtol=0, atol=0)
    torch.testing.assert_close(repeated[1].latents, expected[0].latents, rtol=0, atol=0)


def test_scheduler_draws_continue_each_initializers_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = make_pipeline()
    scheduler = pipe.components.scheduler
    original_step = scheduler.step

    def noisy_step(
        model_output: torch.Tensor, step: SchedulerStep, latents: torch.Tensor, *, generator: torch.Generator
    ) -> torch.Tensor:
        result = original_step(model_output, step, latents, generator=generator)
        return result + torch.randn(result.shape, generator=generator) * 0.01

    monkeypatch.setattr(scheduler, "step", noisy_step)
    cases = requests()
    expected = tuple(pipe(case) for case in cases)
    actual = pipe.run_batch(cases)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.latents, reference.latents, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["shape", "steps", "guidance", "scheduler", "input", "image"])
def test_incompatible_requests_rejected_before_encoding(change: str) -> None:
    pipe = make_pipeline()
    first, second = requests()
    changes = {
        "shape": {"height": 16},
        "steps": {"sampling": replace(second.sampling, num_inference_steps=3)},
        "guidance": {"sampling": replace(second.sampling, guidance_scale=4)},
        "scheduler": {"sampling": replace(second.sampling, scheduler_options={"shift": 7})},
        "input": {"inputs": {"fps": 12}},
        "image": {"inputs": {"images": ["frame.png"]}},
    }
    with pytest.raises(ValueError, match="request batch"):
        pipe.run_batch((first, replace(second, **changes[change])))
    assert pipe.components.conditioner.calls == 0
    assert not pipe.components.denoiser.calls


@pytest.mark.parametrize("stage", ["denoiser", "decoder"])
def test_failed_batch_cleans_once_and_subsequent_requests_reproduce(stage: str) -> None:
    pipe = make_pipeline(unipc=True)
    cases = requests()
    reference = pipe.run_batch(cases)
    component = getattr(pipe.components, stage)
    component.fail = True
    with pytest.raises(RuntimeError, match="failed"):
        pipe.run_batch(tuple(reversed(cases)))
    assert len(pipe.components.denoiser.ended) == 2
    assert pipe.components.denoiser.ended[-1][1] is not None
    component.fail = False
    actual = pipe.run_batch(cases)
    for output, expected in zip(actual, reference):
        torch.testing.assert_close(output.latents, expected.latents, rtol=0, atol=0)


def test_custom_extensions_and_scheduler_rejected_before_execution() -> None:
    pipe = make_pipeline()
    assert isinstance(pipe.runner, NativeDiffusionRunner)
    pipe.runner.extensions = (DiffusionExtension(),)
    with pytest.raises(ValueError, match="without custom extensions"):
        pipe.run_batch(requests())
    pipe.runner.extensions = ()
    pipe.runner.components = replace(pipe.components, scheduler=object())
    with pytest.raises(ValueError, match="request-local"):
        pipe.run_batch(requests())
    assert pipe.components.conditioner.calls == 0


@pytest.mark.parametrize("unipc", [False, True])
def test_small_wan_batch_conditioning_and_request_cache_isolation(unipc: bool) -> None:
    pipe = make_pipeline(unipc=unipc, real=True)
    cases = requests()
    reference = tuple(pipe(case) for case in cases)
    observed: dict[str, list[int]] = {"blocks": [], "time": [], "head": []}

    def capture(name: str) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name].append(int(output.shape[0]))

        return hook

    network = pipe.components.denoiser.model
    handles = [
        module.register_forward_hook(capture(name))
        for name, module in (("blocks", network.blocks[0]), ("time", network.time_embedding), ("head", network.head))
    ]
    try:
        actual = pipe.run_batch(cases)
    finally:
        for handle in handles:
            handle.remove()
    assert observed["blocks"] == [3] * 8
    assert observed["time"] == observed["head"] == [1, 2] * 8
    for output, expected in zip(actual, reference):
        torch.testing.assert_close(output.latents, expected.latents, rtol=1e-5, atol=1e-6)
    denoiser = pipe.components.denoiser
    assert denoiser._optimization_request_windows == {}
    assert denoiser.model._worldfoundry_static_cross_kv.report()["entries"] == 0
    changed = tuple(replace(case, prompt=("changed",) * case.batch_size) for case in cases)
    pipe.run_batch(changed)
    repeated = pipe.run_batch(cases)
    for output, expected in zip(repeated, actual):
        torch.testing.assert_close(output.latents, expected.latents, rtol=0, atol=0)


def test_dual_expert_batch_keeps_both_guidance_routes() -> None:
    pipe = make_pipeline(real=True)
    high = pipe.components.denoiser
    low = make_pipeline(real=True).components.denoiser
    denoiser = Wan22DualExpertDenoiser(high, low, boundary_ratio=0.8)
    pipe.runner = Wan22DualExpertGuidanceRunner(
        model_id=pipe.model_id,
        components=replace(pipe.components, denoiser=denoiser),
        boundary_ratio=0.8,
        low_noise_guidance_scale=2,
        high_noise_guidance_scale=4,
    )
    cases = requests()
    expected = tuple(pipe(case) for case in cases)
    actual = pipe.run_batch(cases)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)
    runtime = denoiser.runtime_optimization_report()["runtime"]
    assert isinstance(runtime, dict)
    assert all(count > 0 for count in runtime["dual_expert"]["route_calls"].values())
    assert high._optimization_request_windows == low._optimization_request_windows == {}


@pytest.mark.parametrize("batch", [(), ("not a request",)])
def test_batch_requires_nonempty_typed_requests(batch: tuple[Any, ...]) -> None:
    pipe = make_pipeline()
    with pytest.raises(ValueError, match="one or more DiffusionRequest"):
        pipe.run_batch(batch)
    assert pipe.components.conditioner.calls == 0


def test_mismatched_condition_shapes_fail_before_denoising(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = make_pipeline()
    original = pipe.components.conditioner.encode

    def encode(request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        value = original(request, device=device, dtype=dtype)
        return Conditioning(
            positive={"context": value.positive["context"][:, : request.batch_size]}, negative=value.negative
        )

    monkeypatch.setattr(pipe.components.conditioner, "encode", encode)
    with pytest.raises(ValueError, match="matching shape"):
        pipe.run_batch(requests())
    assert not pipe.components.denoiser.calls


@pytest.mark.parametrize(
    "attribute",
    [
        "_worldfoundry_sequence_parallel",
        "_worldfoundry_layerwise_cpu_offload_handle",
        "_worldfoundry_approximate_attention",
    ],
)
def test_unsupported_wan_execution_is_rejected_before_encoding(attribute: str) -> None:
    pipe = make_pipeline(real=True)
    setattr(pipe.components.denoiser.model, attribute, object())
    with pytest.raises(ValueError, match="resident unsharded T2V"):
        pipe.run_batch(requests())
    assert pipe.components.conditioner.calls == 0


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_wan_head_keeps_sample_axis_separate_from_shift_scale(batch_size: int) -> None:
    torch.manual_seed(51)
    head = Head(dim=16, out_dim=4, patch_size=(1, 2, 2), eps=1e-6).eval()
    hidden = torch.randn(batch_size, 7, 16)
    timestep = torch.randn(batch_size, 16)
    with torch.no_grad():
        expected = torch.cat([head(x[None], t[None]) for x, t in zip(hidden, timestep)])
        actual = head(hidden, timestep)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_public_pipeline_forwards_batches_and_grouped_warmup() -> None:
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

    pipe = make_pipeline()
    public = Wan2p1T2VPipeline(native_pipeline=pipe, device="cpu")
    cases = requests()
    expected = public.run_batch(cases)
    reports = public.warmup((cases,))
    assert reports == public.warmup_reports
    assert reports[0]["batch_size"] == 3
    actual = public.run_batch(cases)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.latents, reference.latents, rtol=0, atol=0)


def test_product_batch_preserves_small_wan_request_arithmetic() -> None:
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

    public = Wan2p1T2VPipeline(native_pipeline=make_pipeline(real=True), device="cpu")
    calls: list[dict[str, Any]] = [
        dict(
            prompt=prompt,
            seed=seed,
            height=8,
            width=8,
            num_frames=5,
            infer_steps=4,
            negative_prompt="",
            guidance_scale=3.0,
            output_type="latent",
            return_dict=True,
        )
        for prompt, seed in (("boat", 11), (["dog", "bird"], 29))
    ]
    expected = [public(**call) for call in calls]
    actual = public.batch_call(calls)
    for index, (output, reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(output["latents"], reference["latents"], rtol=1e-5, atol=1e-6)
        assert output["metadata"]["seed"] == reference["metadata"]["seed"]
        assert output["metadata"]["request_batching"]["request_index"] == index
    repeated = public.batch_call(tuple(reversed(calls)))
    for output, reference in zip(repeated, reversed(expected)):
        torch.testing.assert_close(output["latents"], reference["latents"], rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_cuda_batch_warmup_freezes_capture_and_preserves_output_ownership() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for this contract")
    pipe = make_pipeline(real=True, device="cuda", graph=True)
    cases = requests()
    expected = pipe.run_batch(cases)
    state = torch.cuda.get_rng_state().clone()
    reports = pipe.warmup((cases, cases[0], tuple(reversed(cases))))
    assert reports[0]["request_count"] == 2
    assert reports[0]["batch_size"] == 3
    assert reports[1]["request_count"] == 1
    assert torch.equal(state, torch.cuda.get_rng_state())
    actual = pipe.run_batch(tuple(reversed(cases)))
    graph = pipe.components.denoiser.graph_report()
    assert graph["capture"] == graph["capture_failed"] == graph["eager"] == 0
    assert graph["replay"] == 8
    assert graph["capture_on_miss"] is False
    for output, reference in zip(actual, reversed(expected)):
        torch.testing.assert_close(output.latents, reference.latents, rtol=1e-5, atol=1e-6)
    held = actual[0].sample.clone()
    pipe.run_batch(tuple(replace(case, prompt=("other",) * case.batch_size) for case in cases))
    torch.testing.assert_close(actual[0].sample, held, rtol=0, atol=0)
    assert pipe.components.denoiser._optimization_request_windows == {}
