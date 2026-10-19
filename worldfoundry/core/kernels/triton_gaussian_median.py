# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Front-to-back median depth from gsplat's projected Gaussian tile lists."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit
def _median_depth_kernel(
    means,
    conics,
    opacities,
    depths,
    offsets,
    indices,
    output,
    WIDTH: tl.constexpr,
    HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
    TILE_COUNT: tl.constexpr,
    INTERSECTIONS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    x = (tile % TILE_WIDTH) * TILE_SIZE + lane % TILE_SIZE
    y = (tile // TILE_WIDTH) * TILE_SIZE + lane // TILE_SIZE
    inside = (lane < TILE_SIZE * TILE_SIZE) & (x < WIDTH) & (y < HEIGHT)
    start = tl.load(offsets + tile)
    end = tl.load(offsets + tile + 1, tile + 1 < TILE_COUNT, other=INTERSECTIONS)
    transmission = tl.full((BLOCK,), 1.0, tl.float32)
    median = tl.full((BLOCK,), -1.0, tl.float32)
    for position in range(start, end):
        index = tl.load(indices + position)
        dx = tl.load(means + index * 2) - (x + 0.5)
        dy = tl.load(means + index * 2 + 1) - (y + 0.5)
        a = tl.load(conics + index * 3)
        b = tl.load(conics + index * 3 + 1)
        c = tl.load(conics + index * 3 + 2)
        exponent = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        alpha = tl.minimum(0.99, tl.load(opacities + index) * tl.exp(-exponent))
        # Equality leaves later surfaces eligible for the first strict crossing.
        contributes = inside & (exponent >= 0) & (alpha >= 1.0 / 255) & (transmission >= 0.5)
        next_transmission = transmission * (1 - alpha)
        crosses = contributes & (next_transmission < 0.5)
        median = tl.where(crosses, tl.load(depths + index), median)
        transmission = tl.where(contributes, next_transmission, transmission)
    tl.store(output + y * WIDTH + x, median, inside)


def gaussian_median_depth(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    height: int,
    width: int,
    tile_size: int,
) -> torch.Tensor:
    """Return [1,1,H,W] camera depth at the first 50% opacity crossing.

    Inputs are one camera's dense gsplat projection and sorted tile lists.
    Pixels whose accumulated opacity never exceeds 0.5 have depth -1.
    Depth is a geometry receipt, not a differentiable fitting objective.
    """
    output = depths.new_empty((1, 1, height, width))
    tile_width = triton.cdiv(width, tile_size)
    tile_count = tile_width * triton.cdiv(height, tile_size)
    cast(Any, _median_depth_kernel)[(tile_count,)](
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        depths.contiguous(),
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        output,
        WIDTH=width,
        HEIGHT=height,
        TILE_WIDTH=tile_width,
        TILE_COUNT=tile_count,
        INTERSECTIONS=flatten_ids.numel(),
        TILE_SIZE=tile_size,
        BLOCK=triton.next_power_of_2(tile_size * tile_size),
    )
    return output
