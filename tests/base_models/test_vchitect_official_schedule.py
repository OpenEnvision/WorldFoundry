"""Keep Vchitect's sampling timetable aligned with the released pipeline."""

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.runners.vchitect import vchitect_guidance_scale
from worldfoundry.base_models.diffusion_model.schedulers.vchitect import VchitectFlowMatchEulerScheduler


@pytest.mark.parametrize("step_count", [1, 2, 8, 20, 40, 50, 100])
def test_vchitect_schedule_matches_release_diffusers_scheduler(step_count: int) -> None:
    released = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    released.set_timesteps(step_count, device="cpu")
    native = VchitectFlowMatchEulerScheduler(shift=3.0)
    steps = native.schedule(SamplingConfig(num_inference_steps=step_count), device=torch.device("cpu"), dtype=torch.bfloat16)

    assert len(steps) == step_count
    actual = torch.stack([step.timestep for step in steps])
    assert torch.equal(actual, released.timesteps)
    assert steps[-1].next_timestep.item() == released.sigmas[-1].item() == 0.0


def test_vchitect_guidance_varies_with_released_timestep_curve() -> None:
    released = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    released.set_timesteps(100, device="cpu")
    observed = [vchitect_guidance_scale(7.5, float(released.timesteps[index]), 100)
                for index in (0, 25, 50, 75, 99)]
    assert observed == pytest.approx([
        8.5,
        1.7680820513279873,
        8.109923599892976,
        1.0220466111707944,
        6.201211201397824,
    ], abs=1e-6)
