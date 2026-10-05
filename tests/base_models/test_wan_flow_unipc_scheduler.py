from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import (
    FlowUniPCMultistepScheduler,
    _solve_small_system_on_cpu,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowUniPCScheduler


def test_karras_unipc_one_step_matches_native_scheduler_contract() -> None:
    scheduler = WanFlowUniPCScheduler(shift=5.0, use_karras_sigma=True)

    schedule = scheduler.schedule(
        SamplingConfig(num_inference_steps=1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(schedule) == 1
    assert schedule[0].index == 0
    assert float(schedule[0].next_timestep) == 0.0


def test_karras_unipc_multi_step_still_returns_requested_evaluations() -> None:
    scheduler = WanFlowUniPCScheduler(shift=5.0, use_karras_sigma=True)

    schedule = scheduler.schedule(
        SamplingConfig(num_inference_steps=3),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(schedule) == 3
    assert [step.index for step in schedule] == [0, 1, 2]


def test_unipc_small_coefficient_solve_stays_on_cpu(monkeypatch) -> None:
    observed_devices = []
    original_solve = torch.linalg.solve

    def recording_solve(matrix, rhs):
        observed_devices.append((matrix.device.type, rhs.device.type))
        return original_solve(matrix, rhs)

    monkeypatch.setattr(torch.linalg, "solve", recording_solve)
    matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0])

    solution = _solve_small_system_on_cpu(
        matrix,
        rhs,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert observed_devices == [("cpu", "cpu")]
    torch.testing.assert_close(matrix @ solution, rhs)


def test_flow_unipc_multistep_updates_remain_finite_after_cpu_solve() -> None:
    scheduler = FlowUniPCMultistepScheduler(shift=5.0, solver_order=2)
    scheduler.set_timesteps(4, device=torch.device("cpu"))
    sample = torch.randn(1, 2, 2, 2, generator=torch.Generator().manual_seed(0))

    for timestep in scheduler.timesteps:
        model_output = torch.full_like(sample, 0.125)
        sample = scheduler.step(
            model_output,
            timestep,
            sample,
            return_dict=False,
        )[0]

    assert scheduler.step_index == 4
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("corrector", [False, True])
def test_unipc_explicit_sigmas_match_adjusted_schedule(order, corrector) -> None:
    scheduler = FlowUniPCMultistepScheduler(solver_order=2)
    scheduler.set_timesteps(6, device=torch.device("cpu"))
    sample = torch.randn(1, 2, 2, 2, generator=torch.Generator().manual_seed(7))
    for timestep in scheduler.timesteps[:2]:
        sample = scheduler.step(torch.full_like(sample, 0.125), timestep, sample, return_dict=False)[0]

    reference = deepcopy(scheduler)
    original_sigmas = scheduler.sigmas.clone()
    model_output = torch.full_like(sample, 0.25)
    index = scheduler.step_index
    if corrector:
        reference.sigmas[index - 1] = 0.72
        reference.sigmas[index] = 0.48
        kwargs = dict(last_sample=scheduler.last_sample, this_sample=sample, order=order)
        expected = reference.multistep_uni_c_bh_update(model_output, **kwargs)
        actual = scheduler.multistep_uni_c_bh_update(
            model_output, sigma_before=reference.sigmas[index - 1],
            sigma=reference.sigmas[index], **kwargs,
        )
    else:
        reference.sigmas[index] = 0.48
        reference.sigmas[index + 1] = 0.24
        kwargs = dict(sample=sample, order=order)
        expected = reference.multistep_uni_p_bh_update(model_output, **kwargs)
        actual = scheduler.multistep_uni_p_bh_update(
            model_output, sigma=reference.sigmas[index],
            sigma_next=reference.sigmas[index + 1], **kwargs,
        )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(scheduler.sigmas, original_sigmas)
