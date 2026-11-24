# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Half-split Q/K rotation shares the canonical Torch/NumPy rotation helper."""

import pytest
import torch

from worldfoundry.core.attention.rotary.embedding import apply_rotary_pos_emb
from worldfoundry.core.attention.rotary.rope import rotate_half

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_axis", [1, 2])
def test_apply_rotary_pos_emb_preserves_layout_and_table_precision(dtype: torch.dtype, head_axis: int) -> None:
    q = torch.randn(2, 4, 3, 8).to(dtype)
    k = torch.randn(2, 2, 3, 8).to(dtype)
    if head_axis == 2:
        q, k = q.transpose(1, 2), k.transpose(1, 2)
    positions = torch.tensor([[0, 2, 4], [5, 1, 3]])
    cos = torch.randn(6, 8)[positions]
    sin = torch.randn(6, 8)[positions]
    actual = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=head_axis)
    for x, result in zip((q, k), actual, strict=True):
        rotated = torch.cat((-x[..., 4:], x[..., :4]), dim=-1)
        expected = x * cos.unsqueeze(head_axis) + rotated * sin.unsqueeze(head_axis)
        assert result.dtype == expected.dtype
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(rotate_half(x), rotated, atol=0, rtol=0)
