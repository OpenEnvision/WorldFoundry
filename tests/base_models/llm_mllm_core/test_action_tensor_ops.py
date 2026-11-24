# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Action conditioning preserves mask semantics and each family's period precision."""

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.tensor_ops import (
    apply_rope,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
)

pytestmark = pytest.mark.unit


def test_block_causal_mask_excludes_padded_queries_and_keys() -> None:
    padding = torch.tensor([[True, True, True, False]])
    blocks = torch.tensor([[0, 0, 1, 1]])
    assert make_att_2d_masks(padding, blocks).tolist() == [
        [[True, True, False, False], [True, True, False, False], [True, True, True, False], [False] * 4]
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sinusoidal_conditioning_keeps_requested_period_precision(dtype: torch.dtype) -> None:
    t = torch.tensor([0.13, 0.75])
    actual = create_sinusoidal_pos_embedding(t, 8, 0.004, 4.0, dtype=dtype)
    fractions = torch.linspace(0.0, 1.0, 4, dtype=dtype)
    periods = 0.004 * (4.0 / 0.004) ** fractions
    scaling = 1.0 / periods * 2 * torch.pi
    angles = scaling[None, :] * t[:, None]
    expected = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_action_rope_preserves_input_dtype_and_half_layout() -> None:
    x = torch.randn(2, 5, 3, 8).bfloat16()
    positions = torch.arange(5).expand(2, -1)
    powers = (2.0 / 8) * torch.arange(4, dtype=torch.float32)
    angles = torch.einsum("bl,h->blh", positions.float(), 1.0 / (10000.0**powers))[..., None, :]
    first, second = x.float().split(4, dim=-1)
    expected = torch.cat(
        [first * angles.cos() - second * angles.sin(), second * angles.cos() + first * angles.sin()], dim=-1
    ).bfloat16()
    torch.testing.assert_close(apply_rope(x, positions), expected, atol=0, rtol=0)


def test_pad_vector_keeps_existing_width_and_zero_fills_suffix() -> None:
    x = torch.randn(2, 3, 5)
    assert pad_vector(x, 5) is x
    actual = pad_vector(x, 8)
    torch.testing.assert_close(actual[..., :5], x, atol=0, rtol=0)
    assert torch.count_nonzero(actual[..., 5:]) == 0
