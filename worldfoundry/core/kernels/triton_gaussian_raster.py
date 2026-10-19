# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Differentiable RGB/alpha compositing of one camera's Gaussian tile lists."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["INTERSECTIONS"])
def _raster_forward(
    means,
    conics,
    opacities,
    colors,
    offsets,
    indices,
    rgb,
    alpha_out,
    final_t,
    last,
    WIDTH: tl.constexpr,
    HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
    TILE_COUNT: tl.constexpr,
    INTERSECTIONS,
    BACKGROUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    x = tile % TILE_WIDTH * 16 + lane % 16
    y = tile // TILE_WIDTH * 16 + lane // 16
    inside = (x < WIDTH) & (y < HEIGHT)
    start = tl.load(offsets + tile)
    end = tl.load(offsets + tile + 1, tile + 1 < TILE_COUNT, other=INTERSECTIONS)
    transmission = tl.full((BLOCK,), 1.0, tl.float32)
    red = tl.full((BLOCK,), 0.0, tl.float32)
    green = tl.full((BLOCK,), 0.0, tl.float32)
    blue = tl.full((BLOCK,), 0.0, tl.float32)
    latest = tl.full((BLOCK,), -1, tl.int32)
    for position in range(start, end):
        index = tl.load(indices + position)
        dx = tl.load(means + index * 2) - (x + 0.5)
        dy = tl.load(means + index * 2 + 1) - (y + 0.5)
        a = tl.load(conics + index * 3)
        b = tl.load(conics + index * 3 + 1)
        c = tl.load(conics + index * 3 + 2)
        exponent = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        opacity = tl.minimum(0.99, tl.load(opacities + index) * tl.exp(-exponent))
        contributes = inside & (exponent >= 0) & (opacity >= 1.0 / 255) & (transmission >= 1e-4)
        weight = tl.where(contributes, transmission * opacity, 0.0)
        red += weight * tl.load(colors + index * 3)
        green += weight * tl.load(colors + index * 3 + 1)
        blue += weight * tl.load(colors + index * 3 + 2)
        latest = tl.where(contributes, position, latest)
        # The surface that crosses the termination threshold still contributes.
        transmission = tl.where(contributes, transmission * (1 - opacity), transmission)
    pixel = y * WIDTH + x
    size = HEIGHT * WIDTH
    tl.store(rgb + pixel, red + transmission * BACKGROUND, inside)
    tl.store(rgb + size + pixel, green + transmission * BACKGROUND, inside)
    tl.store(rgb + size * 2 + pixel, blue + transmission * BACKGROUND, inside)
    tl.store(alpha_out + pixel, 1 - transmission, inside)
    tl.store(final_t + pixel, transmission, inside)
    tl.store(last + pixel, latest, inside)


@triton.jit(do_not_specialize=["INTERSECTIONS"])
def _raster_backward(
    means,
    conics,
    opacities,
    colors,
    offsets,
    indices,
    final_t,
    last,
    grad_rgb,
    grad_alpha,
    grad_means,
    grad_conics,
    grad_opacities,
    grad_colors,
    WIDTH: tl.constexpr,
    HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
    TILE_COUNT: tl.constexpr,
    INTERSECTIONS,
    BACKGROUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    x = tile % TILE_WIDTH * 16 + lane % 16
    y = tile // TILE_WIDTH * 16 + lane // 16
    inside = (x < WIDTH) & (y < HEIGHT)
    pixel = y * WIDTH + x
    size = WIDTH * HEIGHT
    gr = tl.load(grad_rgb + pixel, inside, other=0)
    gg = tl.load(grad_rgb + size + pixel, inside, other=0)
    gb = tl.load(grad_rgb + size * 2 + pixel, inside, other=0)
    ga = tl.load(grad_alpha + pixel, inside, other=0)
    transmission = tl.load(final_t + pixel, inside, other=1)
    latest = tl.load(last + pixel, inside, other=-1)
    start = tl.load(offsets + tile)
    end = tl.load(offsets + tile + 1, tile + 1 < TILE_COUNT, other=INTERSECTIONS)
    future_t = tl.full((BLOCK,), 1.0, tl.float32)
    red = tl.full((BLOCK,), BACKGROUND, tl.float32)
    green = tl.full((BLOCK,), BACKGROUND, tl.float32)
    blue = tl.full((BLOCK,), BACKGROUND, tl.float32)
    for reverse in range(0, end - start):
        position = end - reverse - 1
        index = tl.load(indices + position)
        dx = tl.load(means + index * 2) - (x + 0.5)
        dy = tl.load(means + index * 2 + 1) - (y + 0.5)
        a = tl.load(conics + index * 3)
        b = tl.load(conics + index * 3 + 1)
        c = tl.load(conics + index * 3 + 2)
        exponent = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        visibility = tl.exp(-exponent)
        raw = tl.load(opacities + index) * visibility
        opacity = tl.minimum(0.99, raw)
        contributes = inside & (position <= latest) & (exponent >= 0) & (opacity >= 1.0 / 255)
        before = transmission / (1 - opacity)
        cr = tl.load(colors + index * 3)
        cg = tl.load(colors + index * 3 + 1)
        cb = tl.load(colors + index * 3 + 2)
        derivative = before * ((cr - red) * gr + (cg - green) * gg + (cb - blue) * gb + future_t * ga)
        derivative = tl.where(contributes & (raw < 0.99), derivative, 0.0)
        d_exponent = -raw * derivative
        tl.atomic_add(grad_opacities + index, tl.sum(derivative * visibility, 0))
        tl.atomic_add(grad_means + index * 2, tl.sum(d_exponent * (a * dx + b * dy), 0))
        tl.atomic_add(grad_means + index * 2 + 1, tl.sum(d_exponent * (c * dy + b * dx), 0))
        tl.atomic_add(grad_conics + index * 3, tl.sum(d_exponent * 0.5 * dx * dx, 0))
        tl.atomic_add(grad_conics + index * 3 + 1, tl.sum(d_exponent * dx * dy, 0))
        tl.atomic_add(grad_conics + index * 3 + 2, tl.sum(d_exponent * 0.5 * dy * dy, 0))
        weight = tl.where(contributes, before * opacity, 0.0)
        tl.atomic_add(grad_colors + index * 3, tl.sum(weight * gr, 0))
        tl.atomic_add(grad_colors + index * 3 + 1, tl.sum(weight * gg, 0))
        tl.atomic_add(grad_colors + index * 3 + 2, tl.sum(weight * gb, 0))
        red = tl.where(contributes, opacity * cr + (1 - opacity) * red, red)
        green = tl.where(contributes, opacity * cg + (1 - opacity) * green, green)
        blue = tl.where(contributes, opacity * cb + (1 - opacity) * blue, blue)
        transmission = tl.where(contributes, before, transmission)
        future_t = tl.where(contributes, future_t * (1 - opacity), future_t)


class _GaussianRaster(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        offsets: torch.Tensor,
        indices: torch.Tensor,
        height: int,
        width: int,
        background: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = colors.new_empty((1, 3, height, width))
        alpha = colors.new_empty((1, 1, height, width))
        transmission = torch.empty_like(alpha)
        last = torch.empty_like(alpha, dtype=torch.int32)
        tile_width = triton.cdiv(width, 16)
        tile_count = tile_width * triton.cdiv(height, 16)
        configuration = dict(
            WIDTH=width,
            HEIGHT=height,
            TILE_WIDTH=tile_width,
            TILE_COUNT=tile_count,
            INTERSECTIONS=indices.numel(),
            BACKGROUND=background,
            BLOCK=256,
        )
        cast(Any, _raster_forward)[(tile_count,)](
            means,
            conics,
            opacities,
            colors,
            offsets,
            indices,
            rgb,
            alpha,
            transmission,
            last,
            **configuration,
        )
        ctx.save_for_backward(means, conics, opacities, colors, offsets, indices, transmission, last)
        ctx.configuration = configuration
        return rgb, alpha

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        grad_rgb, grad_alpha = cast(tuple[torch.Tensor, torch.Tensor], grad_outputs)
        means, conics, opacities, colors, offsets, indices, transmission, last = ctx.saved_tensors
        grad_means, grad_conics, grad_opacities, grad_colors = (
            torch.zeros_like(value) for value in (means, conics, opacities, colors)
        )
        cast(Any, _raster_backward)[(ctx.configuration["TILE_COUNT"],)](
            means,
            conics,
            opacities,
            colors,
            offsets,
            indices,
            transmission,
            last,
            grad_rgb.contiguous(),
            grad_alpha.contiguous(),
            grad_means,
            grad_conics,
            grad_opacities,
            grad_colors,
            **ctx.configuration,
        )
        return grad_means, grad_conics, grad_opacities, grad_colors, None, None, None, None, None


def gaussian_raster_rgb(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    height: int,
    width: int,
    background: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite dense one-camera FP32 projections into BCHW RGB and alpha.

    Uses pixel centers, alpha cap 0.99 and minimum alpha 1/255. Transmission
    below 1e-4 terminates only after including the crossing surface. Gradients
    propagate through means, conics, opacity and color; tile selection is fixed.
    Tile-list length is runtime data so changing intersections reuse the kernels.
    """
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        _GaussianRaster.apply(
            means2d.contiguous(),
            conics.contiguous(),
            opacities.contiguous(),
            colors.contiguous(),
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            height,
            width,
            background,
        ),
    )
