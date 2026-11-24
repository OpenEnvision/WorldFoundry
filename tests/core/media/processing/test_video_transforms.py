# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared video transforms retain layouts, interpolation and Torch RNG consumption."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.media.processing.image_utils import numpy_to_pil
from worldfoundry.core.media.processing.video_transforms import (
    normalize,
    random_shift_crop,
    resize,
    resize_scale,
    to_tensor,
)

pytestmark = pytest.mark.unit


def test_to_tensor_preserves_uint8_axis_order_and_requires_four_dimensions() -> None:
    clip = torch.arange(120, dtype=torch.uint8).reshape(2, 3, 4, 5)
    result = to_tensor(clip)
    assert result.shape == clip.shape
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, clip.float() / 255.0, rtol=0, atol=0)
    with pytest.raises(TypeError, match="uint8"):
        to_tensor(clip.float())
    with pytest.raises(ValueError, match="4D"):
        to_tensor(clip[0])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_normalize_uses_first_axis_channels(inplace: bool, dtype: torch.dtype) -> None:
    clip = torch.arange(120, dtype=dtype).reshape(3, 2, 4, 5)
    mean, std = [0.25, 0.5, 1.0], [0.2, 0.5, 2.0]
    expected = clip.clone()
    expected.sub_(torch.tensor(mean, dtype=dtype)[:, None, None, None])
    expected.div_(torch.tensor(std, dtype=dtype)[:, None, None, None])
    actual = normalize(clip, mean, std, inplace=inplace)
    assert (actual is clip) is inplace
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("height,width", [(7, 11), (11, 7), (7, 7)])
def test_random_shift_crop_consumes_two_draws_even_for_square_input(height: int, width: int) -> None:
    clip = torch.arange(2 * 3 * height * width).reshape(2, 3, height, width)
    short = min(height, width)
    torch.manual_seed(83)
    i = int(torch.randint(0, height - short + 1, size=(1,)).item())
    j = int(torch.randint(0, width - short + 1, size=(1,)).item())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(83)
    torch.testing.assert_close(random_shift_crop(clip), clip[..., i : i + short, j : j + short], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_resize_preserves_antialias_and_short_edge_scale(mode: str) -> None:
    clip = torch.arange(2 * 3 * 7 * 11, dtype=torch.float32).reshape(2, 3, 7, 11)
    expected = F.interpolate(clip, size=(5, 9), mode=mode, align_corners=True, antialias=True)
    torch.testing.assert_close(resize(clip, (5, 9), mode), expected, rtol=0, atol=0)
    scaled = F.interpolate(clip, scale_factor=5 / 7, mode=mode, align_corners=True, antialias=True)
    torch.testing.assert_close(resize_scale(clip, (5, 9), mode), scaled, rtol=0, atol=0)
    with pytest.raises(ValueError, match="align_corners"):
        resize(clip, (5, 9), "nearest")


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_numpy_to_pil_rounds_before_uint8_conversion(channels: int) -> None:
    images = np.linspace(0, 1, 2 * 3 * 5 * channels).reshape(2, 3, 5, channels)
    result = numpy_to_pil(images)
    expected = (images * 255).round().astype("uint8")
    assert len(result) == 2
    for index, image in enumerate(result):
        assert image.mode == {1: "L", 3: "RGB", 4: "RGBA"}[channels]
        np.testing.assert_array_equal(np.asarray(image), expected[index, ..., 0] if channels == 1 else expected[index])
