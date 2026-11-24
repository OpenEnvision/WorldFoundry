# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.core.nn.diffusion.schedulers import SchedulerInterface
from worldfoundry.synthesis.visual_generation.echo_infinity.echo_infinity_runtime.utils import scheduler

pytestmark = pytest.mark.unit


class _Scheduler(SchedulerInterface):
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return clean_latent + noise


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_scheduler_interface_keeps_double_precision_prediction_conversion(dtype: torch.dtype) -> None:
    assert scheduler.SchedulerInterface is SchedulerInterface
    model = _Scheduler()
    model.alphas_cumprod = torch.linspace(0.1, 0.9, 7)
    x, xt = torch.randn(2, 3, 4, 5, dtype=dtype), torch.randn(2, 3, 4, 5, dtype=dtype)
    t = torch.tensor([1, 5])
    alpha = model.alphas_cumprod.double()[t].reshape(-1, 1, 1, 1)
    expected = ((xt.double() - alpha**0.5 * x.double()) / (1 - alpha) ** 0.5).to(dtype)
    torch.testing.assert_close(model.convert_x0_to_noise(x, xt, t), expected, rtol=0, atol=0)
