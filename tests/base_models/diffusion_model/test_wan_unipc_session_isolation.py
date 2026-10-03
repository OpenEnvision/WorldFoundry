from __future__ import annotations

import gc
import weakref

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig, SchedulerStep
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowUniPCScheduler


def _schedule(scheduler, *, count=6, shift=5.0, karras=False, dtype=torch.float32):
    return scheduler.schedule(
        SamplingConfig(num_inference_steps=count, scheduler_options={"shift": shift, "use_karras_sigma": karras}),
        device=torch.device("cpu"), dtype=dtype,
    )


def _prediction(latents, step):
    return 0.125 + 0.2 * latents + step.timestep / 10000


def _advance(scheduler, latents, step, generator):
    return scheduler.step(_prediction(latents, step), step, latents, generator=generator)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("karras,intervals", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("other_count,other_shift", [(6, 5.0), (6, 2.0), (4, 2.0)])
@pytest.mark.parametrize("interrupt_after", [0, 1, 2])
def test_interleaved_unipc_requests_match_isolated_sampling(dtype, karras, intervals, other_count,
                                                          other_shift, interrupt_after):
    shared, isolated_a, isolated_b = (WanFlowUniPCScheduler(karras_steps_are_intervals=intervals) for _ in range(3))
    steps_a = _schedule(shared, karras=karras, dtype=dtype)
    expected_a = _schedule(isolated_a, karras=karras, dtype=dtype)
    initial = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(dtype)
    a, reference_a = initial.clone(), initial.clone()
    rng_a, reference_rng_a, rng_b, reference_rng_b = (torch.Generator().manual_seed(43) for _ in range(4))
    for index in range(interrupt_after):
        a = _advance(shared, a, steps_a[index], rng_a)
        reference_a = _advance(isolated_a, reference_a, expected_a[index], reference_rng_a)
    steps_b = _schedule(shared, count=other_count, shift=other_shift, karras=karras, dtype=dtype)
    expected_b = _schedule(isolated_b, count=other_count, shift=other_shift, karras=karras, dtype=dtype)
    b, reference_b = initial.neg(), initial.neg()
    for index in range(max(len(steps_a) - interrupt_after, len(steps_b))):
        if index < len(steps_b):
            b = _advance(shared, b, steps_b[index], rng_b)
            reference_b = _advance(isolated_b, reference_b, expected_b[index], reference_rng_b)
            torch.testing.assert_close(b, reference_b, rtol=0, atol=0)
        a_index = interrupt_after + index
        if a_index < len(steps_a):
            a = _advance(shared, a, steps_a[a_index], rng_a)
            reference_a = _advance(isolated_a, reference_a, expected_a[a_index], reference_rng_a)
            torch.testing.assert_close(a, reference_a, rtol=0, atol=0)
    assert torch.equal(rng_a.get_state(), reference_rng_a.get_state())
    assert torch.equal(rng_b.get_state(), reference_rng_b.get_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("karras,intervals", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("count", [1, 2, 6, 12])
def test_serial_unipc_preserves_native_solver_math_and_evaluation_count(dtype, karras, intervals, count):
    scheduler = WanFlowUniPCScheduler(karras_steps_are_intervals=intervals)
    native = FlowUniPCMultistepScheduler(**dict(scheduler.solver.config))
    effective_karras = karras and (intervals or count > 1)
    native_count = count - 1 if effective_karras and not intervals else count
    native.set_timesteps(native_count, device=torch.device("cpu"), shift=5.0, use_kerras_sigma=effective_karras)
    steps = _schedule(scheduler, count=count, karras=karras, dtype=dtype)
    assert len(steps) == count + int(intervals and karras)
    assert len(steps) == scheduler.expected_step_count(
        SamplingConfig(num_inference_steps=count, scheduler_options={"use_karras_sigma": karras})
    )
    actual = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(dtype)
    expected = actual.clone()
    actual_rng, expected_rng = (torch.Generator().manual_seed(91) for _ in range(2))
    global_rng = torch.get_rng_state().clone()
    for step, timestep in zip(steps, native.timesteps):
        assert torch.equal(step.timestep, timestep)
        actual = _advance(scheduler, actual, step, actual_rng)
        expected = native.step(_prediction(expected, step), timestep, expected,
                               generator=expected_rng, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


def test_unipc_preserves_nondefault_template_configuration():
    scheduler = WanFlowUniPCScheduler()
    scheduler.solver = FlowUniPCMultistepScheduler(num_train_timesteps=700, solver_order=1,
                                                  disable_corrector=[1])
    native = FlowUniPCMultistepScheduler(**dict(scheduler.solver.config))
    native.set_timesteps(6, device=torch.device("cpu"), shift=5.0)
    actual = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2)
    expected = actual.clone()
    for step, timestep in zip(_schedule(scheduler), native.timesteps):
        actual = _advance(scheduler, actual, step, torch.Generator())
        expected = native.step(_prediction(expected, step), timestep, expected, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unipc_rejects_steps_without_request_history():
    scheduler = WanFlowUniPCScheduler()
    steps = _schedule(scheduler)
    invalid = SchedulerStep(index=0, timestep=steps[0].timestep, next_timestep=steps[0].next_timestep)
    with pytest.raises(TypeError, match="schedule"):
        scheduler.step(torch.zeros(1), invalid, torch.zeros(1), generator=torch.Generator())


def test_finished_unipc_request_does_not_retain_history_in_shared_adapter():
    scheduler = WanFlowUniPCScheduler()
    steps = _schedule(scheduler)
    assert steps[0].solver is not scheduler.solver
    owner = weakref.ref(steps[0].solver)
    sample = torch.ones(1, 2, 2, 2)
    for step in steps:
        sample = _advance(scheduler, sample, step, torch.Generator())
    assert scheduler.solver.step_index is None
    assert all(value is None for value in scheduler.solver.model_outputs)
    del steps, step
    gc.collect()
    assert owner() is None
