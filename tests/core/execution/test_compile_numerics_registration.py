# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Registered normalization schemas execute with the unified Torch environment."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from worldfoundry.core.execution.compile_numerics import _layer_norm, _rms_norm
from worldfoundry.core.nn.transformer.dit import _rms_norm_reference

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalized_shape", [[4], [3, 4]])
@pytest.mark.parametrize("affine", [False, True])
@torch.no_grad()
def test_registered_layer_norm_preserves_shape_optional_affine_and_values(
    dtype: torch.dtype, normalized_shape: list[int], affine: bool
) -> None:
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(2, 3, 4, generator=generator).to(dtype)
    original = value.clone()
    weight = torch.randn(normalized_shape, generator=generator).to(dtype) if affine else None
    bias = torch.randn(normalized_shape, generator=generator).to(dtype) if affine else None
    expected = F.layer_norm(value, normalized_shape, weight, bias, 1e-5)
    actual = _layer_norm(value, normalized_shape, weight, bias, 1e-5, False, torch.bfloat16)
    assert actual.shape == value.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compute_mode", ["fp32", "input"])
@pytest.mark.parametrize("affine", [False, True])
@torch.no_grad()
def test_registered_rms_norm_preserves_optional_affine_and_native_reduction(
    dtype: torch.dtype, compute_mode: str, affine: bool
) -> None:
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(2, 3, 4, generator=generator).to(dtype)
    original = value.clone()
    weight = torch.randn(4, generator=generator).to(dtype) if affine else None
    expected = _rms_norm_reference(value, weight, 1e-6, compute_mode)
    actual = _rms_norm(value, weight, 1e-6, compute_mode)
    assert actual.shape == value.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("operation", ["layernorm", "rmsnorm"])
def test_registered_normalization_fake_dispatch_retains_shape_and_dtype(operation: str) -> None:
    with FakeTensorMode():
        value = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        if operation == "layernorm":
            result = _layer_norm(value, [3, 4], None, None, 1e-5, False, torch.bfloat16)
        else:
            result = _rms_norm(value, None, 1e-6, "fp32")
        assert result.shape == value.shape and result.dtype == value.dtype
        assert result.device == value.device
