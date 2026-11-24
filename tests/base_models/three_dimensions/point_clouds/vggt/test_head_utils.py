# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared UV and head-position helpers retain generation-specific frequency precision."""

from __future__ import annotations

import ast
from functools import partial
from pathlib import Path

import pytest
import torch

import worldfoundry
from worldfoundry.base_models.three_dimensions.depth.vigeo.conv_head import (
    normalized_view_plane_uv as vigeo_uv_grid,
)
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.conv_head import (
    normalized_view_plane_uv as pi3_uv_grid,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.heads.utils import (
    create_uv_grid,
    position_grid_to_embed,
    position_grid_to_embed_fp32,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position_dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("frequency_dtype", [torch.float32, torch.float64])
def test_position_grid_to_embed_frequency_precision(position_dtype: torch.dtype, frequency_dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    grid = torch.randn(3, 5, 2, dtype=position_dtype)
    exponent = torch.arange(5, dtype=frequency_dtype) / 5.0
    frequencies = 1.0 / 100.0**exponent
    phase = grid[..., None] * frequencies
    expected = torch.cat(
        [phase[..., 0, :].sin(), phase[..., 0, :].cos(), phase[..., 1, :].sin(), phase[..., 1, :].cos()], dim=-1
    ).float()
    before = torch.get_rng_state()
    result = position_grid_to_embed(grid, 20, frequency_dtype=frequency_dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


def test_depth_anything_position_binding_preserves_fp32_instead_of_vggt_default() -> None:
    binding = position_grid_to_embed_fp32
    assert isinstance(binding, partial)
    assert binding.func is position_grid_to_embed
    assert binding.keywords == {"frequency_dtype": torch.float32}
    grid = torch.linspace(-3, 4, 30).reshape(3, 5, 2)
    expected = position_grid_to_embed(grid, 20, frequency_dtype=torch.float32)
    torch.testing.assert_close(binding(grid, 20), expected, rtol=0, atol=0)
    default = position_grid_to_embed(grid, 20)
    torch.testing.assert_close(default, position_grid_to_embed(grid, 20, frequency_dtype=torch.float64), rtol=0, atol=0)
    assert not torch.equal(default, expected)
    for filename in ("dpt.py", "dualdpt.py"):
        path = (
            Path(worldfoundry.__file__).parent
            / "base_models/three_dimensions/depth/depth_anything/depth_anything_v3/model"
            / filename
        )
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == position_grid_to_embed.__module__
            and any(
                alias.name == "position_grid_to_embed_fp32" and alias.asname == "position_grid_to_embed"
                for alias in node.names
            )
            for node in ast.walk(ast.parse(path.read_text()))
        )


def test_normalized_view_plane_consumers_share_create_uv_grid() -> None:
    assert vigeo_uv_grid is pi3_uv_grid is create_uv_grid
    grid = create_uv_grid(3, 2, aspect_ratio=1.0, dtype=torch.float64)
    assert grid.shape == (2, 3, 2)
    assert grid.dtype == torch.float64
    span = 1.0 / 2.0**0.5
    torch.testing.assert_close(
        grid[0, 0], torch.tensor([-span * 2 / 3, -span / 2], dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(grid[-1, -1], -grid[0, 0], rtol=0, atol=0)
