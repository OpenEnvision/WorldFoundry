# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from worldfoundry.core.attention.rotary.rope_nd import (
    get_1d_rotary_pos_embed,
    get_1d_rotary_pos_embed_linear,
    get_nd_rotary_pos_embed,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["count", "numpy", "tensor"])
@pytest.mark.parametrize("real,repeated", [(False, True), (True, True), (True, False)])
def test_get_1d_rotary_pos_embed_linear_keeps_native_scaling_and_real_layout(
    dtype: torch.dtype, kind: str, real: bool, repeated: bool
) -> None:
    positions = {"count": 7, "numpy": np.arange(7), "tensor": torch.arange(7)}[kind]
    theta, ntk, linear = 100.0, 2.5, 1.7
    angles = torch.outer(torch.arange(7), 1.0 / ((theta * ntk) ** (torch.arange(0, 16, 2, dtype=dtype) / 16)) / linear)
    actual = get_1d_rotary_pos_embed_linear(16, positions, theta, real, linear, ntk, repeated, freqs_dtype=dtype)
    if real:
        assert isinstance(actual, tuple)
        expected = (
            (angles.cos().repeat_interleave(2, dim=-1), angles.sin().repeat_interleave(2, dim=-1))
            if repeated
            else (torch.cat([angles.cos()] * 2, dim=-1), torch.cat([angles.sin()] * 2, dim=-1))
        )
        for value, reference in zip(actual, expected, strict=True):
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, reference.float(), rtol=0, atol=0)
    else:
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, torch.polar(torch.ones_like(angles), angles), rtol=0, atol=0)


def test_existing_nd_builder_retains_power_rescaling_and_interpolation() -> None:
    positions = torch.arange(7)
    actual = get_1d_rotary_pos_embed(16, positions, theta_rescale_factor=2.0, interpolation_factor=0.5)
    angles = torch.outer(positions * 0.5, 1.0 / ((10000 * 2 ** (16 / 14)) ** (torch.arange(0, 16, 2).float() / 16)))
    torch.testing.assert_close(actual, torch.polar(torch.ones_like(angles), angles), rtol=0, atol=0)
    factors = get_nd_rotary_pos_embed([8, 8], (2, 3), use_real=True, device="cpu")
    assert isinstance(factors, tuple)
    assert factors[0].shape == factors[1].shape == (6, 16)
