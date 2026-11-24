# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""TransNetV2 reports options its PyTorch port does not implement."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from torch import nn

from worldfoundry.base_models.perception_core.video_quality import transnetv2

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: transnetv2.StackedDDCNNV2(3, 2, 4, use_octave_conv=True), "Octave convolution"),
        (lambda: transnetv2.DilatedDCNNV2(3, 4, octave_conv=True), "Octave convolution"),
        (lambda: transnetv2.Conv3DConfigurable(3, 4, 1, octave=True), "Octave convolution"),
        (lambda: transnetv2.Conv3DConfigurable(3, 4, 1, kernel_initializer="he_normal"), "Kernel initializers"),
        (lambda: transnetv2.FrameSimilarity(4, stop_gradient=True), "Stop gradient"),
    ],
)
def test_unsupported_layer_options_raise_not_implemented(build: Callable[[], nn.Module], message: str) -> None:
    with pytest.raises(NotImplementedError, match=message):
        build()


def test_unsupported_model_options_raise_before_building_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    # The constructor seeds the global RNGs and enables deterministic algorithms first; keep this process unchanged.
    monkeypatch.setattr(transnetv2.TransNetV2, "_setup_deterministic_behavior", lambda self: None)

    with pytest.raises(NotImplementedError, match="Some options not implemented"):
        transnetv2.TransNetV2(use_resnet_features=True, device="cpu")
