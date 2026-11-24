# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""LTX coordinate and latent blends retain axis order and staged rounding."""

from typing import NamedTuple

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.representations.ltx.conditioning import post_process_latent
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
    get_pixel_coords,
    get_pixel_coords_tuple_order,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import SpatioTemporalScaleFactors

pytestmark = pytest.mark.unit


class _TupleScaleFactors(NamedTuple):
    time: int
    width: int
    height: int


@pytest.mark.parametrize("causal", [False, True])
def test_coordinate_binding_preserves_tuple_axis_order(causal: bool) -> None:
    coordinates = torch.arange(24).reshape(1, 3, 4, 2)
    scales = _TupleScaleFactors(time=4, width=3, height=7)
    actual = get_pixel_coords_tuple_order(coordinates, scales, causal)
    expected = get_pixel_coords(coordinates, SpatioTemporalScaleFactors(time=4, height=3, width=7), causal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    named_axes = get_pixel_coords(coordinates, SpatioTemporalScaleFactors(time=4, height=7, width=3), causal)
    assert not torch.equal(actual, named_axes)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mask_dtype", [torch.bfloat16, torch.float32])
def test_latent_blend_keeps_first_product_rounding_and_output_dtype(
    dtype: torch.dtype, mask_dtype: torch.dtype
) -> None:
    denoised = torch.randn(2, 3, 8, dtype=dtype)
    clean = torch.randn(2, 3, 8, dtype=torch.float64)
    mask = torch.rand(2, 3, 1, dtype=mask_dtype)
    original = denoised.clone()
    first_product = denoised * mask
    expected = (first_product + clean.float() * (1 - mask)).to(dtype)
    actual = post_process_latent(denoised, mask, clean)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(denoised, original, rtol=0, atol=0)
