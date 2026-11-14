# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for sequence-parallel padding and depadding."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.comm.padding import (
    depad_by_length,
    minimal_pad_to_divisible,
    pad_by_length,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("shape", [(2, 5, 3), (1, 7), (4,), (2, 3, 4, 5)])
@pytest.mark.parametrize("value", [0.0, -1.5])
def test_padding_appends_values_on_each_dimension(dtype: torch.dtype, shape: tuple[int, ...], value: float) -> None:
    tensor = torch.arange(torch.Size(shape).numel()).reshape(shape).to(dtype)
    for dim in range(tensor.ndim):
        padded, length = minimal_pad_to_divisible(tensor, 4, dim=dim, pad_value=value)
        assert length == (-shape[dim]) % 4
        assert padded.dtype == tensor.dtype
        assert padded.device == tensor.device
        torch.testing.assert_close(padded.narrow(dim, 0, shape[dim]), tensor, rtol=0, atol=0)
        assert padded.shape[dim] == shape[dim] + length
        if length == 0:
            assert padded is tensor
        else:
            assert torch.all(padded.narrow(dim, shape[dim], length) == torch.tensor(value, dtype=dtype))
        torch.testing.assert_close(depad_by_length(padded, length, dim), tensor, rtol=0, atol=0)
        explicit = pad_by_length(tensor, 2, dim, value)
        assert explicit.shape[dim] == shape[dim] + 2
        torch.testing.assert_close(explicit.narrow(dim, 0, shape[dim]), tensor, rtol=0, atol=0)
        assert torch.all(explicit.narrow(dim, shape[dim], 2) == torch.tensor(value, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_empty_dimension_matches_native_padding(dtype: torch.dtype) -> None:
    tensor = torch.empty(2, 0, 3, dtype=dtype)
    assert minimal_pad_to_divisible(tensor, 4)[0] is tensor
    assert depad_by_length(tensor, 0).shape == tensor.shape
    padded = pad_by_length(tensor, 2, pad_value=-1.5)
    assert padded.shape == (2, 2, 3)
    assert padded.dtype == dtype
    assert torch.all(padded == torch.tensor(-1.5, dtype=dtype))
    for dim in (0, 2):
        for minimal in (True, False):
            length = (-tensor.shape[dim]) % 4 if minimal else 2
            padding = (0, 0, 0, 0, 0, length) if dim == 0 else (0, length, 0, 0, 0, 0)

            def apply_padding() -> torch.Tensor:
                if minimal:
                    result, actual_length = minimal_pad_to_divisible(tensor, 4, dim=dim)
                    assert actual_length == length
                    return result
                return pad_by_length(tensor, length, dim=dim)

            try:
                expected = torch.nn.functional.pad(tensor, padding, mode="constant", value=0.0)
            except RuntimeError as native_error:
                with pytest.raises(type(native_error)) as actual_error:
                    apply_padding()
                assert str(actual_error.value) == str(native_error)
            else:
                torch.testing.assert_close(apply_padding(), expected, rtol=0, atol=0)


def test_no_padding_identity_and_depadding_view() -> None:
    tensor = torch.arange(24.0).reshape(2, 4, 3).transpose(0, 2)
    padded, length = minimal_pad_to_divisible(tensor, 4)
    assert padded is tensor
    assert length == 0
    explicit = pad_by_length(tensor, 0)
    assert explicit is not tensor
    assert explicit.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(explicit, tensor, rtol=0, atol=0)
    view = depad_by_length(tensor, 0)
    assert view is not tensor
    assert view.data_ptr() == tensor.data_ptr()
    assert view.stride() == tensor.stride()


def test_padding_and_depadding_preserve_gradients() -> None:
    tensor = torch.arange(30.0).reshape(2, 5, 3).requires_grad_()
    padded, length = minimal_pad_to_divisible(tensor, 4)
    restored = depad_by_length(padded, length)
    extended = pad_by_length(restored, 2, dim=2, pad_value=-1.5)
    weights = torch.arange(extended.numel(), dtype=tensor.dtype).reshape_as(extended)
    (extended * weights).sum().backward()
    torch.testing.assert_close(tensor.grad, weights[:, :, :3], rtol=0, atol=0)


def test_padding_retains_dimension_and_length_failures() -> None:
    tensor = torch.ones(2, 5, 3)
    with pytest.raises(ValueError, match="Invalid dimension index"):
        minimal_pad_to_divisible(tensor, 4, dim=-1)
    assert minimal_pad_to_divisible(tensor, 3, dim=-1)[0] is tensor
    with pytest.raises(IndexError):
        minimal_pad_to_divisible(tensor, 4, dim=3)
    with pytest.raises(ZeroDivisionError):
        minimal_pad_to_divisible(tensor, 0)
    for dim in (-1, 3):
        with pytest.raises(ValueError, match="Dimension index"):
            pad_by_length(tensor, 0, dim=dim)
    with pytest.raises(ValueError, match="padding_len must be non-negative"):
        pad_by_length(tensor, -1)
    with pytest.raises(ValueError, match="depadding_len must be non-negative"):
        depad_by_length(tensor, -1)
    with pytest.raises(ValueError, match="Cannot remove 6 elements"):
        depad_by_length(tensor, 6)
    assert depad_by_length(tensor, 5).shape == (2, 0, 3)
    assert depad_by_length(tensor, 3, dim=-1).shape == (2, 5, 0)
