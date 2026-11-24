# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""LTX sigma and res_2s coefficients retain the source schedules."""

import pytest
import torch

from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import LinearQuadraticScheduler, LTX2Scheduler
from worldfoundry.base_models.diffusion_model.schedulers.res2s import get_res2s_coefficients, phi

pytestmark = pytest.mark.unit


def test_ltx_sigma_terminal_stretch_and_linear_quadratic_single_step() -> None:
    schedule = LTX2Scheduler().execute(4, max_shift=0, base_shift=0, stretch=False)
    torch.testing.assert_close(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), rtol=0, atol=0)
    schedule = LTX2Scheduler().execute(4, max_shift=0, base_shift=0, terminal=0.25)
    torch.testing.assert_close(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), rtol=0, atol=0)
    torch.testing.assert_close(LinearQuadraticScheduler().execute(1), torch.tensor([1.0, 0.0]), rtol=0, atol=0)


def test_res2s_zero_step_limit_and_cache_reuse() -> None:
    cache: dict[tuple[int, float], float] = {}
    assert phi(1, 0) == 1.0
    assert phi(2, 0) == 0.5
    assert get_res2s_coefficients(0, cache) == (0.5, 0.0, 1.0)
    coefficients = get_res2s_coefficients(0.25, cache)
    cached = dict(cache)
    assert get_res2s_coefficients(0.25, cache) == coefficients
    assert cache == cached
