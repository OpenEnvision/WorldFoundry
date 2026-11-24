# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared detector box conversions preserve coordinate conventions."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.geometry.boxes import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

pytestmark = pytest.mark.unit


def test_box_center_and_extent_round_trip() -> None:
    centers = torch.tensor([[1.0, 2.0, 4.0, 6.0], [-2.0, -1.0, 2.0, 4.0]])
    corners = torch.tensor([[-1.0, -1.0, 3.0, 5.0], [-3.0, -3.0, -1.0, 1.0]])
    assert torch.equal(box_cxcywh_to_xyxy(centers), corners)
    assert torch.equal(box_xyxy_to_cxcywh(corners), centers)
