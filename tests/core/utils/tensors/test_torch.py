# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Episode concatenation and leading-axis multiplication contracts."""

import pytest
import torch

from worldfoundry.core.utils.tensors import torch as torch_utils

pytestmark = pytest.mark.unit


def test_merge_info_concatenates_shared_keys_and_retains_unique_tensors() -> None:
    first = {"shared": torch.ones(2, 3), "first": torch.ones(1)}
    second = {"shared": torch.zeros(1, 3), "second": torch.zeros(4)}
    merged = torch_utils.merge_info(first, second)
    assert merged["first"] is first["first"]
    assert merged["second"] is second["second"]
    torch.testing.assert_close(merged["shared"], torch.cat([first["shared"], second["shared"]]), rtol=0, atol=0)


def test_mult_along_first_dims_only_expands_trailing_axes() -> None:
    value = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    factors = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        torch_utils.mult_along_first_dims(value, factors), value * factors[..., None], rtol=0, atol=0
    )
