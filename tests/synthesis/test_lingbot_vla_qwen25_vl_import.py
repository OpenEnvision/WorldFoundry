# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Regression test ensuring LingBot-VLA v1 qwen25_vl imports cleanly without flash_attn."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_lingbot_v1_qwen25_vl_imports_when_flash_attn_is_absent() -> None:
    qwen25_vl = importlib.import_module("worldfoundry.synthesis.action_generation.lingbot_vla.modeling.qwen25_vl")

    assert hasattr(qwen25_vl, "Qwen2_5_VLVisionFlashAttention2")
