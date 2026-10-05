from __future__ import annotations

from types import SimpleNamespace

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
from worldfoundry.base_models.diffusion_model.runners.base import (
    NativeDiffusionRunner,
    RunnerComponents,
)
from worldfoundry.base_models.diffusion_model.runners.strategies import _cfg_gate_step

_BRANCH_VALUES = {
    "default": {
        "positive": (10.0, 20.0, 30.0, 40.0),
        "negative": (2.0, 5.0, 7.0, 11.0),
    },
    "first": {
        "positive": (10.0, 20.0, 30.0),
        "negative": (2.0, 5.0, 7.0),
    },
    "second": {
        "positive": (100.0,),
        "negative": (99.0,),
    },
}


class _Conditioner:
    def encode(self, request, *, device, dtype) -> Conditioning:
        del device, dtype
        request_name = request.prompts[0]
        values = _BRANCH_VALUES[request_name]
        return Conditioning(
            positive={"request_name": request_name, "values": values["positive"]},
            negative={"request_name": request_name, "values": values["negative"]},
        )


class _Denoiser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []
        self.request_ids: list[str | None] = []
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        request_name = str(model_input.conditioning["request_name"])
        values = model_input.conditioning["values"]
        assert isinstance(values, tuple)
        value = float(values[model_input.step_index])
        self.calls.append((model_input.branch, model_input.step_index, request_name))
        self.request_ids.append(model_input.request_id)
        return DenoiserOutput(sample=torch.full_like(model_input.latents, value))

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.end_calls.append((request_id, error))


class _Initializer:
    def initialize(self, request, *, generator, device, dtype) -> torch.Tensor:
        del request, generator
        return torch.zeros((1,), device=device, dtype=dtype)


class _Scheduler:
    def __init__(self) -> None:
        self.predictions: list[float] = []

    def schedule(self, sampling, *, device, dtype) -> tuple[SchedulerStep, ...]:
        return tuple(
            SchedulerStep(
                index=index,
                timestep=torch.tensor(float(index), device=device, dtype=dtype),
                next_timestep=torch.tensor(float(index + 1), device=device, dtype=dtype),
            )
            for index in range(sampling.num_inference_steps)
        )

    def scale_model_input(self, latents, step) -> torch.Tensor:
        del step
        return latents

    def step(self, model_output, step, latents, *, generator) -> torch.Tensor:
        del step, generator
        self.predictions.append(float(model_output.item()))
        return latents


class _Decoder:
    def decode(self, latents, request) -> torch.Tensor:
        del request
        return latents.clone()


def _runner(
    *,
    cfg_gate_step: float,
    guidance_mode: str = "standard",
) -> tuple[NativeDiffusionRunner, _Denoiser, _Scheduler]:
    denoiser = _Denoiser()
    scheduler = _Scheduler()
    runner = NativeDiffusionRunner(
        model_id="cfg-gate-test",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_Initializer(),
            scheduler=scheduler,
            decoder=_Decoder(),
        ),
        guidance_mode=guidance_mode,
        cfg_gate_step=cfg_gate_step,
    )
    return runner, denoiser, scheduler


def _request(
    request_name: str = "default",
    *,
    steps: int = 4,
    guidance_scale: float = 3.0,
) -> DiffusionRequest:
    return DiffusionRequest(
        prompt=request_name,
        negative_prompt="negative",
        sampling=SamplingConfig(
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ),
    )


def _gate_runtime(output) -> dict[str, object]:
    report = output.metadata["cfg_gate_optimization_report"]
    assert isinstance(report, dict)
    runtime = report["runtime"]
    assert isinstance(runtime, dict)
    gate_runtime = runtime["cfg_delta_cache"]
    assert isinstance(gate_runtime, dict)
    return gate_runtime


def test_cfg_gate_default_runs_both_branches_on_every_guided_step() -> None:
    runner, denoiser, scheduler = _runner(cfg_gate_step=1.0)

    output = runner.run(_request())

    assert denoiser.calls == [
        ("positive", 0, "default"),
        ("negative", 0, "default"),
        ("positive", 1, "default"),
        ("negative", 1, "default"),
        ("positive", 2, "default"),
        ("negative", 2, "default"),
        ("positive", 3, "default"),
        ("negative", 3, "default"),
    ]
    assert len(set(denoiser.request_ids)) == 1
    assert denoiser.request_ids[0] == denoiser.end_calls[0][0]
    assert denoiser.end_calls[0][1] is None
    assert scheduler.predictions == pytest.approx([26.0, 50.0, 76.0, 98.0])
    report = output.metadata["cfg_gate_optimization_report"]
    assert isinstance(report, dict)
    assert report["requested"] == {"cfg_delta_cache": False}
    assert report["effective"] == {"cfg_delta_cache": "not-requested"}
    assert report["quality_tier"] == "exact"
    assert _gate_runtime(output) == {
        "gate_fraction": 1.0,
        "total_steps": 4,
        "gate_step_index": 4,
        "delta_refreshes": 0,
        "cached_delta_steps": 0,
        "positive_branch_calls": 4,
        "negative_branch_calls": 4,
    }


def test_cfg_gate_half_reuses_latest_dense_delta_for_later_steps() -> None:
    runner, denoiser, scheduler = _runner(cfg_gate_step=0.5)

    output = runner.run(_request())

    assert denoiser.calls == [
        ("positive", 0, "default"),
        ("negative", 0, "default"),
        ("positive", 1, "default"),
        ("negative", 1, "default"),
        ("positive", 2, "default"),
        ("positive", 3, "default"),
    ]
    # The latest dense delta is step 1's 20 - 5 = 15. Standard CFG therefore
    # becomes cond + (scale - 1) * stale_delta on the gated steps.
    assert scheduler.predictions == pytest.approx([26.0, 50.0, 60.0, 70.0])
    report = output.metadata["cfg_gate_optimization_report"]
    assert isinstance(report, dict)
    assert report["requested"] == {"cfg_delta_cache": 0.5}
    assert report["effective"] == {"cfg_delta_cache": "stale-uncond-delta-reuse"}
    assert report["fallbacks"] == []
    assert report["quality_tier"] == "approximate"
    assert _gate_runtime(output) == {
        "gate_fraction": 0.5,
        "total_steps": 4,
        "gate_step_index": 2,
        "delta_refreshes": 2,
        "cached_delta_steps": 2,
        "positive_branch_calls": 4,
        "negative_branch_calls": 2,
    }


def test_cfg_gate_zero_still_seeds_delta_with_dense_step_zero() -> None:
    runner, denoiser, _ = _runner(cfg_gate_step=0.0)

    output = runner.run(_request("first", steps=3))

    assert denoiser.calls == [
        ("positive", 0, "first"),
        ("negative", 0, "first"),
        ("positive", 1, "first"),
        ("positive", 2, "first"),
    ]
    runtime = _gate_runtime(output)
    assert runtime["gate_step_index"] == 0
    assert runtime["delta_refreshes"] == 1
    assert runtime["cached_delta_steps"] == 2
    assert runtime["negative_branch_calls"] == 1


def test_cfg_gate_positive_guidance_uses_positive_plus_scaled_delta() -> None:
    runner, _, scheduler = _runner(cfg_gate_step=0.5, guidance_mode="positive")

    runner.run(_request())

    # Positive guidance is cond + scale * (cond - uncond), including when the
    # latest dense delta is reused after the gate.
    assert scheduler.predictions == pytest.approx([34.0, 65.0, 75.0, 85.0])


def test_cfg_gate_clears_stale_delta_and_receipt_between_requests() -> None:
    runner, denoiser, scheduler = _runner(cfg_gate_step=0.0)

    first_output = runner.run(_request("first", steps=3))
    first_runtime = _gate_runtime(first_output)
    assert first_runtime["cached_delta_steps"] == 2

    second_output = runner.run(_request("second", steps=1))

    assert denoiser.calls[-2:] == [
        ("positive", 0, "second"),
        ("negative", 0, "second"),
    ]
    assert scheduler.predictions[-1] == pytest.approx(102.0)
    assert _gate_runtime(second_output) == {
        "gate_fraction": 0.0,
        "total_steps": 1,
        "gate_step_index": 0,
        "delta_refreshes": 1,
        "cached_delta_steps": 0,
        "positive_branch_calls": 1,
        "negative_branch_calls": 1,
    }
    report = second_output.metadata["cfg_gate_optimization_report"]
    assert isinstance(report, dict)
    assert report["effective"] == {"cfg_delta_cache": "requested-not-exercised"}
    assert report["fallbacks"]


def test_runner_finally_ends_denoiser_request_on_error() -> None:
    runner, denoiser, scheduler = _runner(cfg_gate_step=1.0)

    def fail_step(*_args, **_kwargs):
        raise RuntimeError("scheduler failed")

    scheduler.step = fail_step  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="scheduler failed"):
        runner.run(_request())

    assert len(denoiser.end_calls) == 1
    request_id, error = denoiser.end_calls[0]
    assert isinstance(request_id, str) and request_id
    assert isinstance(error, RuntimeError)


def test_cfg_gate_and_cfg_parallel_conflict_before_distributed_setup() -> None:
    components = RunnerComponents(
        denoiser=object(),
        conditioner=object(),
        latent_initializer=object(),
        scheduler=object(),
        decoder=object(),
    )

    with pytest.raises(ValueError, match="not composable with cfg_parallel"):
        NativeDiffusionRunner(
            model_id="cfg-gate-conflict",
            components=components,
            cfg_parallel_degree=2,
            cfg_gate_step=0.5,
        )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 1.0),
        ({"cfg_gate_step": 0.25}, 0.25),
        ({"cfg_gate_fraction": 0.75}, 0.75),
        ({"cfg_gate_step": 0.5, "cfg_gate_fraction": 0.5}, 0.5),
    ],
)
def test_cfg_gate_runtime_aliases(options: dict[str, object], expected: float) -> None:
    context = SimpleNamespace(policy=SimpleNamespace(options=options))
    assert _cfg_gate_step(context) == expected


def test_cfg_gate_runtime_aliases_must_match() -> None:
    context = SimpleNamespace(
        policy=SimpleNamespace(
            options={"cfg_gate_step": 0.25, "cfg_gate_fraction": 0.5},
        )
    )
    with pytest.raises(ValueError, match="must match"):
        _cfg_gate_step(context)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_cfg_gate_runtime_fraction_must_be_in_range(value: float) -> None:
    context = SimpleNamespace(policy=SimpleNamespace(options={"cfg_gate_step": value}))
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        _cfg_gate_step(context)
