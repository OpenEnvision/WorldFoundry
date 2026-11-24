# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only RMS bindings preserve positional epsilon and existing array behavior."""

import importlib

import numpy as np
import pytest
import torch

from worldfoundry.core.nn.blocks.normalization import rms_norm, rms_norm_tensor
from worldfoundry.core.nn.transformer.ops import PytorchPostSAFunction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rms_norm_tensor_preserves_positional_epsilon(dtype: torch.dtype) -> None:
    value = torch.randn(2, 3, 8, dtype=dtype)
    weight = torch.randn(8, dtype=dtype)
    expected = torch.nn.functional.rms_norm(value, (8,), weight, 1e-4)
    torch.testing.assert_close(rms_norm_tensor(value, weight, 1e-4), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rms_norm_retains_numpy_arithmetic(dtype: type[np.floating]) -> None:
    value = np.arange(24, dtype=dtype).reshape(3, 8) / 7
    weight = np.linspace(0.4, 1.4, 8, dtype=dtype)
    expected = value / ((value * value).mean(axis=-1, keepdims=True) + 1e-4) ** 0.5
    np.testing.assert_array_equal(rms_norm(value, weight, eps=1e-4), expected * weight)


@pytest.mark.parametrize(
    "module",
    [
        "worldfoundry.synthesis.visual_generation.joyai_echo.echo_wm_runtime.ltx_core.utils",
        "worldfoundry.synthesis.visual_generation.joyai_echo.echo_longvideo_runtime.ltx_core.utils",
        "worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.utils",
    ],
)
def test_ltx_public_rms_name_resolves_the_tensor_entry_point(module: str) -> None:
    assert importlib.import_module(module).rms_norm is rms_norm_tensor


def test_existing_post_attention_rms_caller_keeps_its_contract() -> None:
    residual = torch.randn(2, 3, 8)
    value = torch.randn_like(residual)
    gate = torch.randn_like(residual)
    weight = torch.randn(8)
    output, normalized = PytorchPostSAFunction()(residual, value, weight, 1e-6, gate)
    expected = residual + value * gate
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(normalized, torch.nn.functional.rms_norm(expected, (8,), weight, 1e-6), rtol=0, atol=0)
