# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""ArtScore checkpoint loading uses local weights and explicit backbone defaults."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {"backbone_config": "resnet50", "no_dense_layer": False, "dropout": 0.0}),
        (
            {"backbone_config": "resnet101", "no_dense_layer": True, "dropout": 0.2},
            {"backbone_config": "resnet101", "no_dense_layer": True, "dropout": 0.2},
        ),
    ],
)
def test_loader_defaults_and_overrides_use_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    from worldfoundry.evaluation.tasks.metrics.artscore import load_artscore_model, models

    checkpoint = tmp_path / "local.pt"
    weights = {"weight": torch.tensor([[2.0, 3.0]]), "bias": torch.tensor([4.0])}
    torch.save(weights, checkpoint)
    arguments = []

    def build_model(args: SimpleNamespace) -> torch.nn.Module:
        arguments.append(vars(args))
        return torch.nn.Linear(2, 1)

    monkeypatch.setattr(models, "get_resnet", build_model)
    model = load_artscore_model(checkpoint, device="cpu", **options)
    assert arguments == [expected]
    assert model.training is False
    torch.testing.assert_close(model(torch.tensor([[5.0, 6.0]])), torch.tensor([[32.0]]))
    torch.testing.assert_close(model.state_dict()["weight"], weights["weight"])


@pytest.mark.parametrize("backbone,no_dense", [("resnet50", False), ("resnet101", True)])
def test_backbone_factory_does_not_request_external_pretrained_weights(
    monkeypatch: pytest.MonkeyPatch,
    backbone: str,
    no_dense: bool,
) -> None:
    from worldfoundry.evaluation.tasks.metrics.artscore import models

    calls = []

    def construct(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(fc=None)

    monkeypatch.setattr(models.models, backbone, construct)
    model = models.get_resnet(SimpleNamespace(backbone_config=backbone, no_dense_layer=no_dense, dropout=0.2))
    assert calls == [{"weights": None}]
    if no_dense:
        assert isinstance(model.fc, torch.nn.Linear)
        assert model.fc.in_features == 2048 and model.fc.out_features == 1
    else:
        assert isinstance(model.fc, torch.nn.Sequential)
        assert model.fc[0].in_features == 2048 and model.fc[-1].out_features == 1
