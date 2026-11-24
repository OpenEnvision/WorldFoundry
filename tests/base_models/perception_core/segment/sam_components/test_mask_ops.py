# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Point sampling retains each SAM generation's empty-box and noise policies."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.perception_core.segment.sam_components.mask_ops import (
    batched_mask_to_box,
    mask_to_box,
    mask_to_box_zero_empty,
    mask_to_rle_pytorch,
    rle_to_mask,
    sample_box_points,
    sample_box_points_zero_empty,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("zero_empty", [False, True])
@pytest.mark.parametrize("noise", [0.0, 0.1, 2.0])
def test_box_sampler_preserves_seed_empty_policy_and_clamp(zero_empty: bool, noise: float) -> None:
    masks = torch.zeros(3, 1, 7, 9, dtype=torch.bool)
    masks[0, 0, 1:5, 2:7] = True
    masks[2] = True
    expected = torch.tensor([[[2, 1, 6, 4]], [[9, 7, -1, -1]], [[0, 0, 8, 6]]], dtype=torch.int32)
    if zero_empty:
        expected[1] = 0
    torch.manual_seed(7)
    if noise > 0:
        widths, heights = expected[..., 2] - expected[..., 0], expected[..., 3] - expected[..., 1]
        dx = torch.min(widths * noise, torch.tensor(20))
        dy = torch.min(heights * noise, torch.tensor(20))
        perturbation = (2 * torch.rand(3, 1, 4) - 1) * torch.stack((dx, dy, dx, dy), dim=-1)
        expected = expected + perturbation
        bounds = torch.tensor([8, 6, 8, 6])
        expected.clamp_(torch.zeros_like(bounds), bounds)
    sampler = sample_box_points_zero_empty if zero_empty else sample_box_points
    torch.manual_seed(7)
    actual, labels = sampler(masks, noise=noise)
    torch.testing.assert_close(actual, expected.reshape(3, 2, 2), rtol=0, atol=0)
    assert torch.equal(labels, torch.tensor([[2, 3]] * 3, dtype=torch.int32))


def test_mask_bounds_keep_sentinel_and_zero_empty_contracts_distinct() -> None:
    masks = torch.zeros(1, 1, 7, 9, dtype=torch.bool)
    assert mask_to_box(masks).tolist() == [[[9, 7, -1, -1]]]
    assert mask_to_box_zero_empty(masks).tolist() == [[[0, 0, 0, 0]]]
    assert batched_mask_to_box(masks[:, 0]).tolist() == [[0, 0, 0, 0]]


def test_mask_rle_round_trip_keeps_column_major_order() -> None:
    masks = torch.zeros(2, 5, 7, dtype=torch.bool)
    masks[0, 1:4, 2:6] = True
    encoded = mask_to_rle_pytorch(masks)
    restored = torch.stack([torch.from_numpy(rle_to_mask(item)) for item in encoded])
    assert torch.equal(restored, masks)
