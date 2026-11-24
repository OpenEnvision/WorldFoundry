# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU rotary helpers do not import the optional fused GPU kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_cpu_rotary_helpers_leave_the_fused_kernel_unloaded() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
from worldfoundry.core.attention.rotary.rope import (
    RotaryPositionEmbedding3D, apply_rotary_embedding, rotary_frequencies,
)

rope = RotaryPositionEmbedding3D(head_dim=48, len_t=1, len_h=2, len_w=3, device="cpu")
assert rope.shift_t(1).shape == (6, 1, 1, 48)
cos, sin = rotary_frequencies(2, 4)
assert cos.shape == sin.shape == (2, 4)
x = torch.ones(1, 4)
assert torch.equal(apply_rotary_embedding(x, torch.ones_like(x), torch.zeros_like(x)), x)
assert "worldfoundry.core.attention.rotary.rope_kernel" not in sys.modules
assert not torch.cuda.is_initialized()
""",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
