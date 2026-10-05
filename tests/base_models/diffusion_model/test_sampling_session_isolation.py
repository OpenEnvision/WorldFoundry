from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.schedulers.flow_match import FlowMatchEulerScheduler
from worldfoundry.base_models.diffusion_model.schedulers.sana import SanaLongLiveScheduler
from worldfoundry.base_models.diffusion_model.schedulers.wan import FastVideoCausalWanSelfForcingScheduler


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_longlive_schedule_keeps_request_shift_when_another_request_is_scheduled(dtype):
    shared = SanaLongLiveScheduler()
    isolated = SanaLongLiveScheduler()
    sampling = SamplingConfig(num_inference_steps=4, guidance_scale=1.0, scheduler_options={"shift": 7.0})
    steps = shared.schedule(sampling, device=torch.device("cpu"), dtype=dtype)
    expected_steps = isolated.schedule(sampling, device=torch.device("cpu"), dtype=dtype)
    shared.schedule(SamplingConfig(num_inference_steps=4, guidance_scale=1.0,
                                   scheduler_options={"shift": 2.0}), device=torch.device("cpu"), dtype=dtype)
    noise = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(dtype)
    flow = torch.full_like(noise, 0.125)
    actual, expected = noise.clone(), noise.clone()
    actual_rng, expected_rng = (torch.Generator().manual_seed(123) for _ in range(2))
    for step, expected_step in zip(steps, expected_steps):
        actual = shared.step(flow, step, actual, generator=actual_rng)
        expected = isolated.step(flow, expected_step, expected, generator=expected_rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


@pytest.mark.parametrize("kind", ["euler", "longlive", "self_forcing"])
def test_sampling_uses_request_rng_and_preserves_global_rng(kind):
    scheduler = {"euler": FlowMatchEulerScheduler, "longlive": SanaLongLiveScheduler,
                 "self_forcing": FastVideoCausalWanSelfForcingScheduler}[kind]()
    config = SamplingConfig(num_inference_steps=8 if kind == "self_forcing" else 4, guidance_scale=1.0)
    initial = torch.zeros((1, 2, 3, 2, 2))
    global_state = torch.get_rng_state().clone()
    outputs = []
    for seed in (43, 91, 43):
        rng = torch.Generator().manual_seed(seed)
        before = rng.get_state().clone()
        sample = initial.clone()
        for step in scheduler.schedule(config, device=torch.device("cpu"), dtype=torch.float32):
            sample = scheduler.step(torch.zeros_like(sample), step, sample, generator=rng)
        outputs.append(sample)
        assert torch.equal(rng.get_state(), before) == (kind == "euler")
        assert torch.equal(torch.get_rng_state(), global_state)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert torch.equal(outputs[0], outputs[1]) == (kind == "euler")
