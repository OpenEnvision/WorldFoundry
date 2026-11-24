# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Small, weight-free contracts for UniWorld's canonical MoGe-2 variant."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones
from worldfoundry.base_models.perception_core.general_perception.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
from worldfoundry.base_models.three_dimensions.depth.moge.model.modules import MLP
from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel_v2
from worldfoundry.base_models.three_dimensions.depth.moge.variants.uniworld_view import UniWorldMoGeModel
from worldfoundry.core.nn.transformer.vit_qkv import QKVSelfAttention

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tokens", [23, torch.tensor(23)])
def test_uniworld_grid_truncates_each_dimension(tokens: int | torch.Tensor) -> None:
    model = UniWorldMoGeModel.__new__(UniWorldMoGeModel)
    base = MoGeModel_v2.__new__(MoGeModel_v2)

    assert model._token_grid_shape(tokens, 1.5) == (3, 5)
    assert base._token_grid_shape(tokens, 1.5) == (4, 6)


def test_scale_mlp_has_one_linear_per_adjacent_dimension() -> None:
    head = MLP([32, 16, 16, 1])

    assert [(layer.in_features, layer.out_features) for layer in head if isinstance(layer, torch.nn.Linear)] == [
        (32, 16),
        (16, 16),
        (16, 1),
    ]
    assert list(head.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"]
    assert head(torch.ones(2, 32)).shape == (2, 1)


def test_variant_reuses_weights_and_forward_on_an_integral_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    def small_backbone(*, pretrained: bool) -> DinoVisionTransformer:
        assert not pretrained
        return DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=2, num_heads=4, block_chunks=0)

    monkeypatch.setattr(backbones, "dinov2_vits14", small_backbone)
    config: dict[str, Any] = {
        "encoder": {"backbone": "dinov2_vits14", "intermediate_layers": [0, 1], "dim_out": 32},
        "neck": {
            "dim_in": [34, 2, 2, 2, 2],
            "dim_res_blocks": [32] * 5,
            "dim_out": None,
            "num_res_blocks": 0,
            "resamplers": ["bilinear"] * 4,
        },
        "points_head": {
            "dim_in": [32] * 5,
            "dim_res_blocks": [32] * 5,
            "dim_out": [None, None, None, None, 3],
            "num_res_blocks": 0,
            "resamplers": ["bilinear"] * 4,
        },
        "scale_head": {"dims": [32, 16, 1]},
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        base = MoGeModel_v2(**config).eval()
        variant = UniWorldMoGeModel(**config).eval()
        for module in variant.encoder.backbone.modules():
            if isinstance(module, QKVSelfAttention):
                assert not module.fused_attn
        assert base.state_dict().keys() == variant.state_dict().keys()
        variant.load_state_dict(base.state_dict(), strict=True)
        base.enable_pytorch_native_sdpa()
        variant.enable_pytorch_native_sdpa()
        image = torch.rand(1, 3, 28, 42)
        with torch.inference_mode():
            reference = base(image, num_tokens=6)
            actual = variant(image, num_tokens=6)
        assert reference.keys() == actual.keys()
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
