# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""MIRA resolves frozen DINO backbones through the canonical in-tree owner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from worldfoundry.base_models.perception_core.general_perception.dinov3.hub import backbones
from worldfoundry.synthesis.visual_generation.mira_wm.src.mira.codec.dino import DINO_WEIGHT_FILENAMES, DinoModel


@pytest.mark.parametrize("model_name", ["dinov3_vitb16", "dinov3_vitl16"])
@pytest.mark.parametrize("local_weights", [False, True])
def test_mira_dino_uses_canonical_builder(model_name: str, local_weights: bool, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RS_DINO_WEIGHTS_DIR", str(tmp_path))
    weights = tmp_path / DINO_WEIGHT_FILENAMES[model_name]
    if local_weights:
        weights.touch()
    backbone = torch.nn.Linear(2, 2)
    builder = Mock(return_value=backbone)
    monkeypatch.setattr(backbones, model_name, builder)
    monkeypatch.setattr(torch.hub, "load", Mock(side_effect=AssertionError("remote code must not execute")))

    model = DinoModel(dino_model=model_name, compile=False, require_pretrained=local_weights)

    builder.assert_called_once_with(**({"weights": str(weights)} if local_weights else {"pretrained": False}))
    assert model.dino_model is backbone
    assert not backbone.training
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert set(model.state_dict()) == {"dino_model.weight", "dino_model.bias"}


def test_mira_dino_requires_local_weights_when_not_restoring_a_codec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RS_DINO_WEIGHTS_DIR", str(tmp_path))
    builder = Mock()
    monkeypatch.setattr(backbones, "dinov3_vitb16", builder)
    with pytest.raises(FileNotFoundError, match="RS_DINO_WEIGHTS_DIR"):
        DinoModel(compile=False)
    builder.assert_not_called()


def test_mira_dino_preserves_preloaded_backbone_and_layer_selection() -> None:
    backbone = torch.nn.Linear(2, 2)
    model = DinoModel(preloaded_dino_module=backbone, last_layer_only=False, layer_indices=(1, 3), compile=False)
    assert model.dino_model is backbone
    assert model.layers == (1, 3)
