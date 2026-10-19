# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""CUDA scalar promotion and full SD2/Marigold scheduler trajectories."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.schedulers.stable_diffusion import SDDiscreteScheduler

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("algorithm", ["ddim", "euler-unscaled"])
@pytest.mark.parametrize("prediction", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("steps", [1, 4, 30])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_trajectory_matches_released_scalar_precision(
    algorithm: str, prediction: str, steps: int, dtype: torch.dtype
) -> None:
    config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": prediction,
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
        "timestep_spacing": "leading" if algorithm == "ddim" else "linspace",
    }
    native = SDDiscreteScheduler(config, algorithm=algorithm)
    reference = cast(
        DDIMScheduler | EulerDiscreteScheduler,
        (DDIMScheduler if algorithm == "ddim" else EulerDiscreteScheduler).from_config(config),
    )
    reference.set_timesteps(steps, device="cuda")
    if isinstance(reference, EulerDiscreteScheduler):
        reference.is_scale_input_called = True
    generator = torch.Generator(device="cuda").manual_seed(31)
    actual = torch.randn((1, 4, 16, 16), device="cuda", dtype=dtype, generator=generator)
    expected = actual.clone()
    schedule = native.schedule(SamplingConfig(num_inference_steps=steps), device=torch.device("cuda"), dtype=dtype)
    for step in schedule:
        prediction_tensor = torch.randn(actual.shape, device="cuda", dtype=dtype, generator=generator)
        probe = native.step(prediction_tensor, step, actual, generator=generator)
        actual = native.step(prediction_tensor, step, actual, generator=generator)
        torch.testing.assert_close(actual, probe, rtol=0, atol=0)
        if isinstance(reference, DDIMScheduler):
            expected = reference.step(prediction_tensor, int(step.timestep), expected, return_dict=False)[0]
        else:
            expected = reference.step(prediction_tensor, step.timestep, expected, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
