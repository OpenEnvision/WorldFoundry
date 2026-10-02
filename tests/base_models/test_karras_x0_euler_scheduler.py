from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.schedulers.karras_x0 import KarrasX0EulerScheduler


def test_karras_x0_euler_matches_cosmos_predict1_sigma_contract() -> None:
    scheduler = KarrasX0EulerScheduler(sigma_min=0.0002, sigma_max=80.0, rho=7.0)
    steps = scheduler.schedule(
        SamplingConfig(num_inference_steps=35, seed=1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(steps) == 35
    assert steps[0].timestep.item() == 80.0
    assert torch.isclose(steps[-1].timestep, torch.tensor(0.0002), rtol=1e-5)
    assert steps[-1].next_timestep.item() == 0.0


def test_karras_x0_euler_step_uses_clean_prediction() -> None:
    scheduler = KarrasX0EulerScheduler()
    step = scheduler.schedule(
        SamplingConfig(num_inference_steps=2, seed=1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0]
    latents = torch.tensor([[[[[4.0]]]]])
    clean = torch.tensor([[[[[1.5]]]]])

    actual = scheduler.step(clean, step, latents, generator=torch.Generator())
    sigma = step.timestep
    next_sigma = step.next_timestep
    expected = latents + (next_sigma - sigma) * (latents - clean) / sigma

    torch.testing.assert_close(actual, expected)
