# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact range-union attention contracts without model weights."""

import pytest
import torch

from worldfoundry.core.attention.sparse.ranges import attention_range_plan, range_attention

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("heads", [2, 4])
def test_overlapping_ranges_match_independent_dense_union(heads: int) -> None:
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(9, heads, 8, generator=generator)
    k = torch.randn(9, 2, 8, generator=generator)
    v = torch.randn(9, 2, 8, generator=generator)
    queries = torch.tensor([[0, 4], [2, 8], [0, 8], [8, 9], [2, 8]])
    keys = torch.tensor([[0, 3], [3, 7], [7, 9], [0, 9], [4, 6]])
    mask = torch.zeros(9, 9, dtype=torch.bool)
    for (start, stop), (left, right) in zip(queries, keys):
        mask[start:stop, left:right] = True
    expanded_k = k.repeat_interleave(heads // 2, dim=1)
    expanded_v = v.repeat_interleave(heads // 2, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, expanded_k) / 8**0.5
    expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~mask, -torch.inf).softmax(-1), expanded_v)
    actual = range_attention(q, k, v, attention_range_plan(queries, keys), query_chunk_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
