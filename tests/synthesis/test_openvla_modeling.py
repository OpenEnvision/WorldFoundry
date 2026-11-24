# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""OpenVLA vision backbone handles timm>=1.0 list outputs."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from worldfoundry.synthesis.action_generation.openvla.modeling import (
    PrismaticVisionBackbone,
    unpack_tuple,
)


class _DummyBackbone(nn.Module):
    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

    def get_intermediate_layers(self, x: torch.Tensor, n: set[int]) -> list[torch.Tensor]:
        assert n == {1}
        return [torch.ones((x.shape[0], 5, self.embed_dim), dtype=x.dtype)]


def test_unpack_tuple_handles_tuple_and_list() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert unpack_tuple(lambda _x: (tensor,))(None) is tensor
    assert unpack_tuple(lambda _x: [tensor])(None) is tensor
    assert unpack_tuple(lambda _x: tensor)(None) is tensor


def test_fused_prismatic_vision_backbone_accepts_list_intermediate_layers(monkeypatch) -> None:
    monkeypatch.setattr(timm, "create_model", lambda *_args, **_kwargs: _DummyBackbone(embed_dim=8))
    backbone = PrismaticVisionBackbone(
        use_fused_vision_backbone=True,
        image_sizes=[4, 4],
        timm_model_ids=["dummy-a", "dummy-b"],
        timm_override_act_layers=[None, None],
    )
    out = backbone(torch.randn(2, 6, 4, 4))
    assert out.shape == (2, 5, 16)
