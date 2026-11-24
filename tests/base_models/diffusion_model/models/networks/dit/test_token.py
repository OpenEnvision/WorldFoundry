# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""The token DiT final layer preserves 2-D AdaLN and nonzero Linear initialization."""

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.dit.token import (
    TokenFinalLayer,
)

pytestmark = pytest.mark.unit


def test_token_final_layer_preserves_two_dimensional_modulation_and_state() -> None:
    torch.manual_seed(42)
    model = TokenFinalLayer(8, 4)
    assert set(model.state_dict()) == {
        "linear.weight",
        "linear.bias",
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
    }
    assert torch.count_nonzero(model.linear.weight) == model.linear.weight.numel()
    x, conditioning = torch.randn(5, 8), torch.randn(5, 8)
    shift, scale = model.adaLN_modulation(conditioning).chunk(2, dim=-1)
    expected = model.linear(model.norm_final(x) * (1 + scale) + shift)
    assert model(x, conditioning).shape == (5, 4)
    torch.testing.assert_close(model(x, conditioning), expected, atol=0, rtol=0)


def test_hma_complete_denoising_head_uses_token_final_layer() -> None:
    from worldfoundry.synthesis.visual_generation.hma.hma_runtime.hma.model.diffloss import SimpleMLPAdaLN

    model = SimpleMLPAdaLN(4, 16, 4, 8, 2).eval()
    assert isinstance(model.final_layer, TokenFinalLayer)
    expected_keys = {
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias",
    }
    assert {key for key in model.state_dict() if key.startswith("final_layer.")} == expected_keys
    result = model(torch.randn(3, 4), torch.tensor([0.0, 1.0, 3.25]), torch.randn(3, 8))
    assert result.shape == (3, 4)
    assert torch.isfinite(result).all()
