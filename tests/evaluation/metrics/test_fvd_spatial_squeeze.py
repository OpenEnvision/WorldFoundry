# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""I3D keeps or removes singleton spatial axes as requested."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.evals.tasks.metrics.distribution.fvd.pytorch_i3d import InceptionI3d

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("spatial_squeeze", [True, False])
def test_forward_preserves_requested_spatial_layout(spatial_squeeze: bool) -> None:
    model = InceptionI3d.__new__(InceptionI3d)
    nn.Module.__init__(model)
    model.end_points = {}
    model.add_module("avg_pool", nn.Identity())
    model.add_module("dropout", nn.Identity())
    model.add_module("logits", nn.Identity())
    model._spatial_squeeze = spatial_squeeze
    model.eval()
    inputs = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4, 1, 1)
    expected = torch.tensor([[1.5, 5.5, 9.5], [13.5, 17.5, 21.5]], dtype=torch.float32, device="cpu")
    if not spatial_squeeze:
        expected = expected.reshape(2, 3, 1, 1)

    result = model(inputs)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == inputs.dtype
    assert result.device == inputs.device
