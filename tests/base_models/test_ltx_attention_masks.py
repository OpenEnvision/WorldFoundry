# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""LTX appended-reference masks preserve noisy and prior-reference attention."""

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import ModalityState
from worldfoundry.base_models.diffusion_model.models.representations.ltx.attention_masks import (
    extend_keyframes_mask,
    resolve_cross_mask,
    update_attention_mask,
)

pytestmark = pytest.mark.unit


def test_ltx_attention_mask_isolates_reference_groups() -> None:
    state = ModalityState(
        latent=torch.zeros(1, 3, 4),
        clean_latent=torch.zeros(1, 3, 4),
        denoise_mask=torch.ones(1, 3, 1),
        positions=torch.zeros(1, 3, 3, 2),
    )
    mask = update_attention_mask(state, torch.tensor([0.25, 0.5]), 2, 2, 1, torch.device("cpu"), torch.float32)
    expected = torch.tensor(
        [[[1, 1, 1, 0.25, 0.5], [1, 1, 1, 0.25, 0.5], [1, 1, 1, 0, 0], [0.25, 0.25, 0, 1, 1], [0.5, 0.5, 0, 1, 1]]]
    )
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)
    expanded_state = state.with_updates(latent=torch.zeros(1, 5, 4), attention_mask=mask)
    expanded = update_attention_mask(expanded_state, None, 2, 1, 1, torch.device("cpu"), torch.float32)
    assert expanded is not None
    torch.testing.assert_close(expanded[:, :5, :5], expected, rtol=0, atol=0)
    torch.testing.assert_close(expanded[:, :2, 5], torch.ones(1, 2), rtol=0, atol=0)
    assert torch.count_nonzero(expanded[:, 2:5, 5]) == 0


@pytest.mark.parametrize("mask", [0.5, torch.tensor(0.5), torch.tensor([0.5, 0.5]), torch.tensor([[0.5, 0.5]])])
def test_ltx_cross_mask_broadcasts_scalar_and_single_batch(mask: float | torch.Tensor) -> None:
    actual = resolve_cross_mask(mask, 2, 3, torch.device("cpu"), torch.bfloat16)
    torch.testing.assert_close(actual, torch.full((3, 2), 0.5, dtype=torch.bfloat16), rtol=0, atol=0)


def test_ltx_keyframe_mask_marks_only_appended_generated_slots() -> None:
    state = ModalityState(
        latent=torch.zeros(1, 3, 4),
        clean_latent=torch.zeros(1, 3, 4),
        denoise_mask=torch.zeros(1, 3, 1),
        positions=torch.zeros(1, 3, 3, 2),
    )
    assert extend_keyframes_mask(state, 2, marked=False) is None
    mask = extend_keyframes_mask(state, 2, marked=True)
    torch.testing.assert_close(mask, torch.tensor([[[0], [0], [0], [1], [1]]], dtype=torch.float32), rtol=0, atol=0)
    expanded = extend_keyframes_mask(state.with_updates(keyframes_mask=mask), 1, marked=False)
    torch.testing.assert_close(
        expanded, torch.tensor([[[0], [0], [0], [1], [1], [0]]], dtype=torch.float32), rtol=0, atol=0
    )
