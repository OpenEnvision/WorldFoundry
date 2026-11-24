# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.resampler import FeedForward
from worldfoundry.base_models.diffusion_model.models.networks.lvdm import attention, attention_vid2world
from worldfoundry.base_models.three_dimensions.general_3d.mvdiffusion.mvdiffusion_runtime.src.models.modules import (
    transformer,
)

pytestmark = pytest.mark.unit


def test_lvdm_and_multiview_transformer_reuse_native_geglu() -> None:
    assert attention_vid2world.GEGLU is transformer.GEGLU is attention.GEGLU
    layer = transformer.GEGLU(8, 12)
    assert list(layer.state_dict()) == ["proj.weight", "proj.bias"]
    x = torch.randn(2, 3, 8)
    value, gate = layer.proj(x).chunk(2, dim=-1)
    torch.testing.assert_close(layer(x), value * torch.nn.functional.gelu(gate), rtol=0, atol=0)


def test_resampler_feedforward_keeps_bias_free_sequential_layout() -> None:
    layer = FeedForward(8, mult=2.5)
    assert list(layer.state_dict()) == ["0.weight", "0.bias", "1.weight", "3.weight"]
    assert layer[1].weight.shape == (20, 8)
    assert layer(torch.randn(2, 3, 8)).shape == (2, 3, 8)
