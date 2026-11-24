# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Padded SAM and unpadded Swin windows preserve pixel order on reconstruction."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.attention.window import (
    window_partition,
    window_partition_padded,
    window_reverse,
    window_unpartition,
)

pytestmark = pytest.mark.unit


def test_padded_window_round_trip_preserves_crop_and_stride() -> None:
    x = torch.arange(2 * 5 * 7 * 3).reshape(2, 5, 7, 3)
    windows, padding = window_partition_padded(x, 2)
    assert windows.shape == (24, 2, 2, 3)
    assert padding == (6, 8)
    restored = window_unpartition(windows, 2, padding, (5, 7))
    assert torch.equal(restored, x)
    assert not restored.is_contiguous()


def test_unpadded_window_round_trip_preserves_contiguous_layout() -> None:
    x = torch.arange(2 * 6 * 8 * 3).reshape(2, 6, 8, 3)
    windows = window_partition(x, 2)
    restored = window_reverse(windows, 2, 6, 8)
    assert torch.equal(restored, x)
    assert windows.is_contiguous() and restored.is_contiguous()
