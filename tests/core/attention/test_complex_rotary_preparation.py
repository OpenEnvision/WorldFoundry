# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared video RoPE tables preserve absolute positions and complex precision."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.attention.rotary.complex_rope import (
    assemble_complex_rotary_frequencies_3d,
    complex_rotary_frequencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("grid,start", [((1, 2, 3), 0), ((4, 3, 5), 12), ((4, 2, 2), 80)])
def test_prepared_tables_match_each_absolute_axis(dtype: torch.dtype, grid: tuple[int, int, int], start: int) -> None:
    tables = tuple(complex_rotary_frequencies(width, 128).to(dtype) for width in (8, 4, 4))
    actual = assemble_complex_rotary_frequencies_3d(tables, grid, start_frame=start)
    expected = torch.stack(
        [
            torch.cat((tables[0][frame + start], tables[1][row], tables[2][column]))
            for frame in range(grid[0])
            for row in range(grid[1])
            for column in range(grid[2])
        ]
    ).unsqueeze(1)
    assert torch.equal(actual, expected) and actual.dtype == dtype
    assert actual.is_contiguous() and torch.isfinite(actual).all()
    changed = assemble_complex_rotary_frequencies_3d(tables, grid, start_frame=start + 1)
    assert not torch.equal(actual[..., :4], changed[..., :4])
    assert torch.equal(actual[..., 4:], changed[..., 4:])
    independent = assemble_complex_rotary_frequencies_3d(tables, grid, start_frame=start)
    assert independent.data_ptr() != actual.data_ptr()
