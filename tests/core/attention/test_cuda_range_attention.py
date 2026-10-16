# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-backend CUDA precision checks for exact range-union attention."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from worldfoundry.core.attention.sparse.ranges import attention_range_plan, range_attention

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [2, 4])
def test_cuda_ranges_match_dense_union(dtype: torch.dtype, heads: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(9, heads, 8, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(9, 2, 8, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(9, 2, 8, device="cuda", dtype=dtype, generator=generator)
    queries = torch.tensor([[0, 4], [2, 8], [0, 8], [8, 9]], device="cuda")
    keys = torch.tensor([[0, 3], [3, 7], [7, 9], [0, 9]], device="cuda")
    mask = torch.zeros(9, 9, device="cuda", dtype=torch.bool)
    for (start, stop), (left, right) in zip(queries.tolist(), keys.tolist()):
        mask[start:stop, left:right] = True
    expanded_k = k.repeat_interleave(heads // 2, dim=1)
    expanded_v = v.repeat_interleave(heads // 2, dim=1)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            expanded_k.transpose(0, 1)[None],
            expanded_v.transpose(0, 1)[None],
            attn_mask=mask,
        )[0].transpose(0, 1)
        actual = range_attention(q, k, v, attention_range_plan(queries, keys), query_chunk_size=2)
    torch.testing.assert_close(actual, expected)
