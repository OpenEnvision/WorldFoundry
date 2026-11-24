# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Adjacent-pair rotation preserves feature slices, scale placement and dtype."""

import pytest
import torch

from worldfoundry.core.attention.rotary.broadcast import broadcat
from worldfoundry.core.attention.rotary.interleaved import apply_rotary_emb, rotate_pairs

pytestmark = pytest.mark.unit


def test_rotate_pairs_uses_adjacent_features() -> None:
    torch.testing.assert_close(
        rotate_pairs(torch.tensor([[1, 2, 3, 4]])), torch.tensor([[-2, 1, -4, 3]]), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_emb_keeps_partial_slice_and_scale_order(dtype: torch.dtype, ndim: int) -> None:
    torch.manual_seed(11)
    value = torch.randn((2, 5, 12) if ndim == 3 else (2, 3, 5, 12), dtype=dtype)
    frequencies = torch.randn(9 if ndim == 3 else 5, 4).repeat_interleave(2, dim=-1)
    applied = frequencies[-5:]
    middle = value[..., 2:10]
    rotated = middle * applied.cos() * 0.75 + rotate_pairs(middle) * applied.sin() * 0.75
    expected = torch.cat([value[..., :2], rotated, value[..., 10:]], dim=-1).to(dtype)
    actual = apply_rotary_emb(frequencies, value, start_index=2, scale=0.75)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., :2], value[..., :2], atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 10:], value[..., 10:], atol=0, rtol=0)


def test_broadcat_broadcasts_the_concatenation_axis_too() -> None:
    first, second = torch.randn(2, 1, 4), torch.randn(1, 3, 4)
    expected = torch.cat(torch.broadcast_tensors(first, second), dim=0)
    torch.testing.assert_close(broadcat([first, second], dim=0), expected, atol=0, rtol=0)
