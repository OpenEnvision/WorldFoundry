from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput, DenoiserOutput
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    Wan22DualExpertDenoiser,
)
from worldfoundry.base_models.diffusion_model.recipes import (
    default_native_diffusion_registry,
    wan22_i2v_a14b_recipe,
    wan22_t2v_a14b_recipe,
)
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


class _FakeDenoiser:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self.calls += 1
        return DenoiserOutput(sample=torch.full_like(model_input.latents, self.value))

    def runtime_optimization_report(self) -> dict[str, object]:
        return {
            "requested": {"attention": "flash_attention_3"},
            "effective": {"expert_marker": self.value},
            "fallbacks": [] if self.value == 2.0 else ["low expert dense fallback"],
            "quality_tier": "exact",
            "runtime": {"calls": self.calls},
        }

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.end_calls.append((request_id, error))


def _input(
    timestep: torch.Tensor | float,
    *,
    step_index: int = 0,
) -> DenoiserInput:
    return DenoiserInput(
        latents=torch.zeros(1, 16, 2, 2, 2),
        timestep=torch.as_tensor(timestep),
        next_timestep=torch.tensor(0.0),
        conditioning={},
        step_index=step_index,
        total_steps=2,
    )


def test_wan22_a14b_dual_expert_routes_at_official_boundary() -> None:
    high = _FakeDenoiser(2.0)
    low = _FakeDenoiser(1.0)
    denoiser = Wan22DualExpertDenoiser(high, low, boundary_ratio=0.875)

    assert denoiser(_input(900.0)).extras["expert"] == "high-noise"
    assert denoiser(_input(874.0, step_index=1)).extras["expert"] == "low-noise"
    assert (high.calls, low.calls) == (1, 1)

    report = denoiser.runtime_optimization_report()
    assert report["requested"]["attention"] == "flash_attention_3"
    assert report["effective"]["expert_marker"] == {
        "high-noise": 2.0,
        "low-noise": 1.0,
    }
    assert report["effective"]["dual_expert_route_calls"] == {
        "high-noise": 1,
        "low-noise": 1,
    }
    assert report["fallbacks"] == ["low-noise: low expert dense fallback"]

    with pytest.raises(ValueError, match="cannot mix"):
        denoiser(_input(torch.tensor([900.0, 800.0])))


def test_wan22_a14b_route_receipts_reset_between_requests() -> None:
    denoiser = Wan22DualExpertDenoiser(
        _FakeDenoiser(2.0),
        _FakeDenoiser(1.0),
        boundary_ratio=0.875,
    )
    denoiser(_input(900.0))
    denoiser(_input(100.0, step_index=1))
    assert denoiser.runtime_optimization_report()["runtime"]["dual_expert"][
        "route_calls"
    ] == {"high-noise": 1, "low-noise": 1}

    denoiser(_input(900.0))

    assert denoiser.runtime_optimization_report()["runtime"]["dual_expert"][
        "route_calls"
    ] == {"high-noise": 1, "low-noise": 0}


def test_wan22_a14b_route_receipts_are_request_local_when_interleaved() -> None:
    high = _FakeDenoiser(2.0)
    low = _FakeDenoiser(1.0)
    denoiser = Wan22DualExpertDenoiser(high, low, boundary_ratio=0.875)

    denoiser(
        _input(900.0).with_updates(
            request_id="request-a",
            branch="positive",
        )
    )
    denoiser(
        _input(900.0).with_updates(
            request_id="request-b",
            branch="negative",
        )
    )
    denoiser(
        _input(100.0, step_index=1).with_updates(
            request_id="request-a",
            branch="negative",
        )
    )

    request_a = denoiser.route_receipt("request-a")
    request_b = denoiser.route_receipt("request-b")
    assert request_a["route_calls"] == {"high-noise": 1, "low-noise": 1}
    assert request_a["branch_calls"] == {"positive": 1, "negative": 1}
    assert request_b["route_calls"] == {"high-noise": 1, "low-noise": 0}
    assert request_b["branch_calls"] == {"negative": 1}
    assert request_a["request_epoch"] != request_b["request_epoch"]

    denoiser.end_request("request-a")
    finalized = denoiser.route_receipt("request-a")
    assert finalized["finalized"] is True
    assert finalized["release_reason"] == "completed"
    assert finalized["request_local"] is True
    assert denoiser.route_lifecycle_report() == {
        "live_requests": 1,
        "receipt_snapshots": 1,
        "max_receipt_snapshots": 32,
    }
    assert high.end_calls[-1] == ("request-a", None)
    assert low.end_calls[-1] == ("request-a", None)

    # Snapshots are JSON copies; callers cannot mutate the stored receipt.
    finalized["route_calls"]["high-noise"] = 99
    assert denoiser.route_receipt("request-a")["route_calls"]["high-noise"] == 1


def test_wan22_a14b_cfg_parallel_step_zero_does_not_reset_same_request() -> None:
    denoiser = Wan22DualExpertDenoiser(
        _FakeDenoiser(2.0),
        _FakeDenoiser(1.0),
        boundary_ratio=0.875,
    )
    parallel = {"_worldfoundry_cfg_parallel_request": True}
    for branch in ("positive", "negative"):
        denoiser(
            _input(900.0).with_updates(
                request_id="parallel-request",
                branch=branch,
                conditioning=parallel,
            )
        )

    receipt = denoiser.route_receipt("parallel-request")
    assert receipt["route_calls"] == {"high-noise": 2, "low-noise": 0}
    assert receipt["branch_calls"] == {"positive": 1, "negative": 1}


def test_wan22_a14b_recipes_bind_both_experts_and_official_cfg() -> None:
    t2v = wan22_t2v_a14b_recipe()
    i2v = wan22_i2v_a14b_recipe()

    assert t2v.execution.strategy == i2v.execution.strategy == "wan22-dual-expert-guidance"
    assert t2v.execution.options["boundary_ratio"] == 0.875
    assert t2v.execution.options["low_noise_guidance_scale"] == 3.0
    assert t2v.execution.options["high_noise_guidance_scale"] == 4.0
    assert i2v.execution.options["boundary_ratio"] == 0.9
    assert i2v.execution.options["low_noise_guidance_scale"] == 3.5
    assert i2v.execution.options["high_noise_guidance_scale"] == 3.5
    for recipe in (t2v, i2v):
        assert recipe.checkpoints["high-dit"].files[0].startswith("high_noise_model/")
        assert recipe.checkpoints["low-dit"].files[0].startswith("low_noise_model/")

    registry = default_native_diffusion_registry()
    assert registry.resolve("Wan-AI/Wan2.2-T2V-A14B").model_id == t2v.model_id
    assert registry.resolve("Wan-AI/Wan2.2-I2V-A14B").model_id == i2v.model_id


@pytest.mark.parametrize("model_id", ["wan2.2-t2v-a14b", "wan2.2-i2v-a14b"])
def test_wan22_a14b_runtime_profiles_are_resolvable(model_id: str) -> None:
    profile = load_runtime_profile(model_id, check_conda_env_exists=False)

    assert profile.model_id == model_id
    assert profile.execution["pipeline_binding"] == model_id
