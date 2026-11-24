# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Positive-index RoPE keeps unscaled frequencies and clone-on-read grid caching."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.attention.rotary.indexed_2d import IndexedRotaryPositionEmbedding2D, ShapeCachedPositionGetter
from worldfoundry.core.attention.rotary.rope_2d import RotaryPositionEmbedding2D

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexed_rotary_preserves_unscaled_frequencies(dtype: torch.dtype) -> None:
    positions = ShapeCachedPositionGetter()(2, 3, 5, torch.device("cpu"))
    tokens = torch.randn(2, 4, 15, 8, dtype=dtype)
    indexed = IndexedRotaryPositionEmbedding2D(frequency=100, scaling_factor=2)
    expected = RotaryPositionEmbedding2D(frequency=100, scaling_factor=1)(tokens, positions)
    torch.testing.assert_close(indexed(tokens, positions), expected, rtol=0, atol=0)
    torch.testing.assert_close(indexed(tokens, positions), expected, rtol=0, atol=0)
    assert len(indexed.frequency_cache) == 1
    assert indexed.scaling_factor == 2
    assert not indexed.state_dict()


def test_shape_cached_position_getter_returns_independent_caller_storage() -> None:
    getter = ShapeCachedPositionGetter()
    positions = getter(2, 3, 5, torch.device("cpu"))
    expected = positions.clone()
    positions.zero_()
    restored = getter(2, 3, 5, torch.device("cpu"))
    assert torch.equal(restored, expected)
    assert len(getter.position_cache) == 1


def test_indexed_rotary_rejects_negative_indices_instead_of_reflecting_them() -> None:
    rotary = IndexedRotaryPositionEmbedding2D()
    positions = torch.tensor([[[0, 1], [-1, 2]]])
    with pytest.raises(IndexError):
        rotary(torch.randn(1, 2, 2, 8), positions)
