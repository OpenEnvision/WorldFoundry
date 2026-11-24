# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Opt-in multistage hooks share generators without leaking between requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    ModalityState,
    MultiModalDenoiserInput,
    MultiModalDenoiserOutput,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.runners.multistage import (
    JointMultiStageDiffusionRunner,
    MultiModalLatentDecoder,
    MultiStageComponents,
    MultiStageLatentInitializer,
)
from worldfoundry.base_models.diffusion_model.schedulers.ltx import LTXFixedEulerScheduler


def _runner(events: list, clock: list[float] | None = None) -> JointMultiStageDiffusionRunner:
    class Initializer(MultiStageLatentInitializer):
        def initialize_stage(
            self,
            request: DiffusionRequest,
            *,
            generator: torch.Generator,
            **kwargs: Any,
        ) -> Mapping[str, ModalityState]:
            noise = torch.randn(1, 2, 2, generator=generator)
            events.append(("initialize", request.num_frames, noise.clone()))
            return {
                "video": ModalityState(
                    latent=noise,
                    denoise_mask=torch.zeros(1, 2, 1),
                    clean_latent=torch.zeros_like(noise),
                    positions=torch.zeros(1, 3, 2, 2),
                )
            }

    class Scheduler(LTXFixedEulerScheduler):
        def step_state(
            self,
            prediction: torch.Tensor,
            step: SchedulerStep,
            state: ModalityState,
            *,
            generator: torch.Generator,
        ) -> torch.Tensor:
            noise = torch.randn(state.latent.shape, generator=generator)
            events.append(("step", generator.initial_seed(), noise.clone()))
            return state.latent + noise

    class Decoder(MultiModalLatentDecoder):
        uses_run_generator = True

        def decode_modalities(
            self,
            states: Mapping[str, ModalityState],
            request: DiffusionRequest,
            *,
            generator: torch.Generator | None = None,
        ) -> Mapping[str, object]:
            assert generator is not None
            noise = torch.randn(1, 2, 2, generator=generator)
            events.append(("decode", request.num_frames, noise.clone()))
            return {"video": states["video"].latent, "decode_noise": noise}

    class Conditioner:
        def encode(self, request: DiffusionRequest, **kwargs: Any) -> Conditioning:
            return Conditioning(positive={})

        def resolve_request(self, request: DiffusionRequest, conditioning: Conditioning) -> DiffusionRequest:
            return replace(request, num_frames=17)

    class Processor:
        def process(
            self,
            states: Mapping[str, ModalityState],
            request: DiffusionRequest,
        ) -> Mapping[str, torch.Tensor]:
            return {"video": states["video"].latent}

    def denoise(inputs: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        if clock is not None:
            clock[0] += 20
        return MultiModalDenoiserOutput(
            samples={name: torch.zeros_like(state.latent) for name, state in inputs.modalities.items()},
        )

    return JointMultiStageDiffusionRunner(
        model_id="hook-fixture",
        stage_steps=(2, 1),
        device="cpu",
        dtype=torch.float32,
        step_noise_seed_offset=10000,
        components=MultiStageComponents(
            denoiser=denoise,
            conditioner=Conditioner(),
            latent_initializer=Initializer(),
            schedulers=(Scheduler((1.0, 0.5, 0.0)), Scheduler((0.5, 0.0))),
            processor=Processor(),
            decoder=Decoder(),
        ),
    )


def _request(seed: int, deadline: float | None = None) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="A scene.",
        sampling=SamplingConfig(num_inference_steps=3, seed=seed),
        inputs={"deadline_monotonic": deadline},
    )


def test_multistage_hooks_preserve_run_generator_and_reset_step_noise_each_stage() -> None:
    events = []
    runner = _runner(events)
    output = runner.run(_request(42))
    initial = [event for event in events if event[0] == "initialize"]
    steps = [event for event in events if event[0] == "step"]
    decoded = [event for event in events if event[0] == "decode"]
    assert [event[1] for event in initial] == [17, 17]
    assert [event[1] for event in steps] == [10042, 10042, 10042]
    torch.testing.assert_close(steps[0][2], steps[2][2], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(42)
    torch.testing.assert_close(initial[0][2], torch.randn(1, 2, 2, generator=generator), rtol=0, atol=0)
    torch.testing.assert_close(initial[1][2], torch.randn(1, 2, 2, generator=generator), rtol=0, atol=0)
    torch.testing.assert_close(decoded[0][2], torch.randn(1, 2, 2, generator=generator), rtol=0, atol=0)
    torch.testing.assert_close(output.sample, initial[1][2] + steps[-1][2], rtol=0, atol=0)


def test_multistage_hooks_are_repeatable_across_changed_requests() -> None:
    runner = _runner([])
    first = runner.run(_request(42))
    changed = runner.run(_request(43))
    repeated = runner.run(_request(42))
    assert not torch.equal(first.sample, changed.sample)
    torch.testing.assert_close(first.sample, repeated.sample, rtol=0, atol=0)
    torch.testing.assert_close(first.artifacts["decode_noise"], repeated.artifacts["decode_noise"], rtol=0, atol=0)


def test_multistage_deadline_stops_at_the_next_step_and_allows_subsequent_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.core.execution import deadline

    events = []
    clock = [0.0]
    monkeypatch.setattr(deadline.time, "monotonic", lambda: clock[0])
    runner = _runner(events, clock)
    with pytest.raises(TimeoutError, match="stage 1 step 2"):
        runner.run(_request(42, deadline=10))
    assert len([event for event in events if event[0] == "step"]) == 1
    assert not [event for event in events if event[0] == "decode"]
    recovered = runner.run(_request(42))
    expected = _runner([]).run(_request(42))
    torch.testing.assert_close(recovered.sample, expected.sample, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("stage_specific", [False, True], ids=["single-denoiser", "per-stage-denoisers"])
def test_multistage_selects_weights_by_stage_and_keeps_default_denoiser(stage_specific: bool) -> None:
    calls = []
    runner = _runner([])

    def denoiser(name: str) -> Callable[[MultiModalDenoiserInput], MultiModalDenoiserOutput]:
        def predict(inputs: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
            calls.append((name, inputs.step_index, inputs.total_steps))
            return MultiModalDenoiserOutput(samples={key: value.latent for key, value in inputs.modalities.items()})

        return predict

    default = denoiser("default")
    runner.components = replace(
        runner.components,
        denoiser=default,
        stage_denoisers=(denoiser("adapted"), denoiser("clean")) if stage_specific else (),
    )
    runner.run(_request(42))
    expected_names = ["adapted", "adapted", "clean"] if stage_specific else ["default"] * 3
    assert calls == list(zip(expected_names, [0, 1, 0], [2, 2, 1], strict=True))
    assert runner.components.denoiser is default


@pytest.mark.unit
def test_multistage_rejects_incomplete_stage_denoiser_bundle() -> None:
    runner = _runner([])
    with pytest.raises(ValueError, match="one denoiser per stage"):
        JointMultiStageDiffusionRunner(
            model_id=runner.model_id,
            components=replace(runner.components, stage_denoisers=(runner.components.denoiser,)),
            stage_steps=runner.stage_steps,
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.unit
def test_multistage_injects_request_id_and_finalizes_feature_caches_on_success_and_error() -> None:
    runner = _runner([])
    seen_ids: list[str] = []
    ended: list[tuple[str, type[BaseException] | None]] = []

    class TrackingDenoiser:
        def __init__(self, *, fail_on_step: int | None = None) -> None:
            self.fail_on_step = fail_on_step

        def __call__(self, inputs: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
            req_id = inputs.conditioning.get("request_id")
            assert isinstance(req_id, str) and req_id.strip()
            seen_ids.append(req_id)
            if self.fail_on_step is not None and inputs.step_index == self.fail_on_step:
                raise RuntimeError("injected stage failure")
            return MultiModalDenoiserOutput(
                samples={k: torch.zeros_like(v.latent) for k, v in inputs.modalities.items()}
            )

        def end_feature_cache_request(self, request_id: str, *, error: BaseException | None = None) -> None:
            ended.append((request_id, None if error is None else type(error)))

    runner.components = replace(runner.components, denoiser=TrackingDenoiser())
    runner.run(_request(7))
    assert len(seen_ids) == 3
    assert len(set(seen_ids)) == 1
    assert ended == [(seen_ids[0], None)]

    # Explicit request_id from request.inputs is honored across stages and cleanup.
    seen_ids.clear()
    ended.clear()
    explicit_req = DiffusionRequest(
        prompt="A scene.",
        sampling=SamplingConfig(num_inference_steps=3, seed=7),
        inputs={"request_id": "req-explicit-123"},
    )
    runner.run(explicit_req)
    assert seen_ids == ["req-explicit-123"] * 3
    assert ended == [("req-explicit-123", None)]

    # Errors still invoke end_feature_cache_request with the original exception type.
    seen_ids.clear()
    ended.clear()
    runner.components = replace(runner.components, denoiser=TrackingDenoiser(fail_on_step=1))
    with pytest.raises(RuntimeError, match="injected stage failure"):
        runner.run(_request(7))
    assert len(ended) == 1
    assert ended[0] == (seen_ids[0], RuntimeError)
