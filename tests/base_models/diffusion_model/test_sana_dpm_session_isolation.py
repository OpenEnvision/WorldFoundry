from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig, SchedulerStep
from worldfoundry.base_models.diffusion_model.schedulers.flow_dpm import FlowDPMSolverMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.sana import SanaFlowDPMScheduler


def _schedule(scheduler, *, steps, shift, dtype):
    return scheduler.schedule(
        SamplingConfig(num_inference_steps=steps, scheduler_options={"shift": shift}),
        device=torch.device("cpu"), dtype=dtype,
    )


def _flow(latents, step):
    return 0.125 + 0.2 * latents + step.timestep / 10000


def _advance(scheduler, latents, step, generator):
    return scheduler.step(_flow(latents, step), step, latents, generator=generator)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("other_steps,other_shift", [(4, 7.0), (4, 2.0), (7, 2.0)])
@pytest.mark.parametrize("interrupt_after", [0, 1, 2])
def test_interleaved_dpm_requests_keep_their_own_history(dtype, other_steps, other_shift, interrupt_after):
    shared, isolated_a, isolated_b = (SanaFlowDPMScheduler(shift=7.0) for _ in range(3))
    steps_a = _schedule(shared, steps=4, shift=7.0, dtype=dtype)
    expected_a = _schedule(isolated_a, steps=4, shift=7.0, dtype=dtype)
    noise = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(dtype)
    a, reference_a = noise.clone(), noise.clone()
    rng_a, rng_reference_a, rng_b, rng_reference_b = (
        torch.Generator().manual_seed(43) for _ in range(4)
    )
    for index in range(interrupt_after):
        a = _advance(shared, a, steps_a[index], rng_a)
        reference_a = _advance(isolated_a, reference_a, expected_a[index], rng_reference_a)
    steps_b = _schedule(shared, steps=other_steps, shift=other_shift, dtype=dtype)
    expected_b = _schedule(isolated_b, steps=other_steps, shift=other_shift, dtype=dtype)
    b, reference_b = noise.neg(), noise.neg()
    for index in range(max(len(steps_a) - interrupt_after, len(steps_b))):
        if index < len(steps_b):
            b = _advance(shared, b, steps_b[index], rng_b)
            reference_b = _advance(isolated_b, reference_b, expected_b[index], rng_reference_b)
            torch.testing.assert_close(b, reference_b, rtol=0, atol=0)
        a_index = interrupt_after + index
        if a_index < len(steps_a):
            a = _advance(shared, a, steps_a[a_index], rng_a)
            reference_a = _advance(isolated_a, reference_a, expected_a[a_index], rng_reference_a)
            torch.testing.assert_close(a, reference_a, rtol=0, atol=0)
    assert torch.equal(rng_a.get_state(), rng_reference_a.get_state())
    assert torch.equal(rng_b.get_state(), rng_reference_b.get_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("count", [1, 2, 4, 20])
def test_serial_dpm_matches_native_solver_at_every_step(dtype, count):
    scheduler = SanaFlowDPMScheduler(shift=7.0)
    native = FlowDPMSolverMultistepScheduler(**dict(scheduler.solver.config))
    native.set_timesteps(count, device=torch.device("cpu"), shift=7.0)
    steps = _schedule(scheduler, steps=count, shift=7.0, dtype=dtype)
    actual = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(dtype)
    expected = actual.clone()
    actual_rng, expected_rng = (torch.Generator().manual_seed(91) for _ in range(2))
    global_rng = torch.get_rng_state().clone()
    for step, timestep in zip(steps, native.timesteps):
        assert torch.equal(step.timestep, timestep)
        actual = _advance(scheduler, actual, step, actual_rng)
        expected = native.step(_flow(expected, step), timestep, expected,
                               generator=expected_rng, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


def test_dpm_rejects_steps_without_request_owned_history():
    scheduler = SanaFlowDPMScheduler()
    steps = _schedule(scheduler, steps=4, shift=7.0, dtype=torch.float32)
    invalid = SchedulerStep(index=0, timestep=steps[0].timestep, next_timestep=steps[0].next_timestep)
    with pytest.raises(TypeError, match="schedule"):
        scheduler.step(torch.zeros(1), invalid, torch.zeros(1), generator=torch.Generator())
