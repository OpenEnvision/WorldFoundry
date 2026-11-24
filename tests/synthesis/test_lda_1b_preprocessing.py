# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for LDA-1B per-part sin/cos state feature encoding."""

from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.synthesis.action_generation.lda_1b.preprocessing import pack_parts, state_sincos

pytestmark = pytest.mark.unit


def test_lda_state_sincos_encodes_per_part_before_concatenating() -> None:
    parts: list[dict[str, object]] = [
        {"key": "left_arm", "dim": 7},
        {"key": "right_arm", "dim": 7},
        {"key": "left_hand", "dim": 6},
        {"key": "right_hand", "dim": 6},
        {"key": "waist", "dim": 3},
    ]
    observation = {
        str(part["key"]): np.arange(1, 1 + 2 * int(str(part["dim"])), dtype=np.float32).reshape(
            2, int(str(part["dim"]))
        )
        * 0.05
        for part in parts
    }
    raw = pack_parts(observation, parts, label="state")
    actual = state_sincos(
        raw,
        steps=2,
        expected_dim=58,
        part_widths=[int(str(part["dim"])) for part in parts],
    )
    expected = np.concatenate(
        [
            np.concatenate((np.sin(observation[str(p["key"])]), np.cos(observation[str(p["key"])])), axis=-1)
            for p in parts
        ],
        axis=-1,
    ).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
