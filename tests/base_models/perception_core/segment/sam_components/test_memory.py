# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SAM memory fusion clones layers and retains channel-first residual arithmetic."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.perception_core.segment.sam_components.memory import CXBlock, Fuser, MemoryEncoder

pytestmark = pytest.mark.unit


def test_fuser_clones_independent_layers_and_optional_input_projection() -> None:
    layer = CXBlock(8, kernel_size=3, padding=1, layer_scale_init_value=0.1)
    fuser = Fuser(layer, 2, dim=8, input_projection=True).eval()
    assert fuser.layers[0].pwconv1.weight.data_ptr() != fuser.layers[1].pwconv1.weight.data_ptr()
    x = torch.randn(2, 8, 3, 5)
    expected = fuser.layers[1](fuser.layers[0](fuser.proj(x)))
    torch.testing.assert_close(fuser(x), expected, rtol=0, atol=0)
    assert "layers.0.gamma" in fuser.state_dict()
    assert "proj.weight" in fuser.state_dict()


@pytest.mark.parametrize("skip_sigmoid", [False, True])
def test_memory_encoder_preserves_feature_projection_and_position_output(skip_sigmoid: bool) -> None:
    encoder = MemoryEncoder(4, nn.Conv2d(1, 8, 1), Fuser(CXBlock(8), 2), nn.Identity(), in_dim=8).eval()
    pix, masks = torch.randn(2, 8, 3, 5), torch.randn(2, 1, 3, 5)
    expected_masks = masks if skip_sigmoid else masks.sigmoid()
    expected = encoder.out_proj(encoder.fuser(encoder.pix_feat_proj(pix) + encoder.mask_downsampler(expected_masks)))
    result = encoder(pix, masks, skip_mask_sigmoid=skip_sigmoid)
    torch.testing.assert_close(result["vision_features"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["vision_pos_enc"], [expected], rtol=0, atol=0)
    assert "pix_feat_proj.weight" in encoder.state_dict()
    assert "fuser.layers.1.dwconv.weight" in encoder.state_dict()
