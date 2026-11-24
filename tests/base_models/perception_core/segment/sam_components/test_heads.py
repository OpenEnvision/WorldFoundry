# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared SAM prompt and mask decoders preserve their output and stability contracts."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.perception_core.segment.sam_components.mask_decoder import MaskDecoder
from worldfoundry.base_models.perception_core.segment.sam_components.prompt_encoder import PromptEncoder
from worldfoundry.base_models.perception_core.segment.sam_components.transformer import TwoWayTransformer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("high_resolution", [False, True])
def test_mask_decoder_keeps_mask_tokens_scores_and_high_resolution_features(high_resolution: bool) -> None:
    decoder = MaskDecoder(
        transformer_dim=32,
        transformer=TwoWayTransformer(1, 32, 4, 48),
        iou_head_hidden_dim=24,
        use_high_res_features=high_resolution,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
    ).eval()
    features = [torch.randn(2, 4, 8, 12), torch.randn(2, 8, 4, 6)] if high_resolution else None
    masks, iou, tokens, scores = decoder(
        torch.randn(1, 32, 2, 3),
        torch.randn(1, 32, 2, 3),
        torch.randn(2, 3, 32),
        torch.randn(2, 32, 2, 3),
        multimask_output=True,
        repeat_image=True,
        high_res_features=features,
    )
    assert [value.shape for value in (masks, iou, tokens, scores)] == [(2, 3, 8, 12), (2, 3), (2, 3, 32), (2, 1)]
    assert "pred_obj_score_head.layers.2.weight" in decoder.state_dict()
    assert ("conv_s0.weight" in decoder.state_dict()) is high_resolution


def test_dynamic_multimask_keeps_stable_single_masks_and_selects_best_alternative() -> None:
    decoder = MaskDecoder(transformer_dim=32, transformer=TwoWayTransformer(1, 32, 4, 48)).eval()
    logits = torch.zeros(2, 4, 2, 3)
    logits[0, 3] = 2
    logits[1] = 3
    iou = torch.tensor([[0.1, 0.2, 0.3, 0.9], [0.2, 0.3, 0.8, 0.1]])
    masks, scores = decoder._dynamic_multimask_via_stability(logits, iou)
    assert torch.equal(masks, torch.stack((logits[0, 3:4], logits[1, :1])))
    assert torch.equal(scores, torch.tensor([[0.9], [0.2]]))


def test_prompt_encoder_preserves_points_boxes_masks_and_gaussian_buffer() -> None:
    encoder = PromptEncoder(32, (2, 3), (8, 12), 8).eval()
    restored = PromptEncoder(32, (2, 3), (8, 12), 8).eval()
    restored.load_state_dict(encoder.state_dict(), strict=True)
    points = (torch.rand(2, 3, 2) * 7, torch.tensor([[-1, 0, 1], [-1, 0, 1]]))
    boxes, masks = torch.rand(2, 4) * 7, torch.randn(2, 1, 8, 12)
    actual = encoder(points, boxes, masks)
    assert [value.shape for value in actual] == [(2, 5, 32), (2, 32, 2, 3)]
    torch.testing.assert_close(restored(points, boxes, masks), actual, rtol=0, atol=0)
    assert "pe_layer.positional_encoding_gaussian_matrix" in encoder.state_dict()
    assert encoder.get_dense_pe().shape == (1, 32, 2, 3)
