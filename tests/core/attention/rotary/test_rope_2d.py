# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.core.attention.rotary.rope_2d import (
    CrocoRotaryPositionEmbedding2D,
    RotaryPositionEmbedding2D,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_croco_rotary_position_embedding_retains_keyword_binding_and_signed_positions(dtype: torch.dtype) -> None:
    native = CrocoRotaryPositionEmbedding2D(freq=37, F0=1.5)
    canonical = RotaryPositionEmbedding2D(frequency=37, scaling_factor=1.5)
    tokens = torch.randn(2, 4, 3, 16, dtype=dtype)
    positions = torch.tensor([[[-1, -1], [0, 1], [2, 3]]] * 2)
    torch.testing.assert_close(native(tokens, positions), canonical(tokens, positions), rtol=0, atol=0)
    assert native.state_dict() == {}
    assert native.base_frequency == 37
    assert native.scaling_factor == 1.5
    tables = next(iter(native.frequency_cache.values()))
    native(tokens, positions)
    assert next(iter(native.frequency_cache.values())) is tables
