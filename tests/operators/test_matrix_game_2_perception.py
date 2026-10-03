"""MG2 perception applies each request's canvas before pixel normalization."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from worldfoundry.operators.matrix_game_2_operator import MatrixGame2Operator


@pytest.fixture
def perception_transforms(monkeypatch):
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    class ToTensor:
        def __call__(self, image):
            return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div(255)

    class Normalize:
        def __init__(self, *, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def __call__(self, value):
            return (value - self.mean) / self.std

    class Resize:
        def __init__(self, *, size, antialias=True):
            self.size = tuple(size)
            self.antialias = antialias

        def __call__(self, value):
            if isinstance(value, Image.Image):
                return value.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            return F.interpolate(
                value.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False, antialias=self.antialias
            ).squeeze(0)

    torchvision = ModuleType("torchvision")
    transforms = ModuleType("torchvision.transforms")
    v2 = ModuleType("torchvision.transforms.v2")
    for transform in (Compose, ToTensor, Normalize, Resize):
        setattr(v2, transform.__name__, transform)
    torchvision.transforms = transforms
    transforms.v2 = v2
    for module in (torchvision, transforms, v2):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def _reference_pixels(image, height, width, dtype):
    source_width, source_height = image.size
    if source_height * width > source_width * height:
        crop_width = source_width
        crop_height = int(source_width * height / width)
    else:
        crop_width = int(source_height * width / height)
        crop_height = source_height
    left = (source_width - crop_width) / 2
    top = (source_height - crop_height) / 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    resized = cropped.resize((width, height), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float()
    return pixels.div(255).sub(0.5).div(0.5).to(dtype)[None, :, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reused_perception_matches_fresh_requests_for_each_canvas(perception_transforms, dtype):
    row, column = np.indices((71, 113))
    pixels = np.stack(((row * 7 + column * 3) % 256, (row * 11) % 256, (column * 13) % 256), axis=-1)
    source = Image.fromarray(pixels.astype(np.uint8))
    reused = MatrixGame2Operator()
    results = []
    for height, width in ((32, 64), (64, 128), (32, 64)):
        options = {"resize_H": height, "resize_W": width, "device": "cpu", "weight_dtype": dtype}
        actual = reused.process_perception(source, num_output_frames=3, **options)
        fresh = MatrixGame2Operator().process_perception(source, num_output_frames=3, **options)
        expected = _reference_pixels(source, height, width, dtype)

        assert actual["image"].shape == (1, 3, 1, height, width)
        assert actual["img_cond"].shape == (1, 3, 9, height, width)
        assert actual["image"].dtype is dtype
        assert actual["image"].device.type == "cpu"
        torch.testing.assert_close(actual["image"], expected, atol=0, rtol=0)
        torch.testing.assert_close(actual["image"], fresh["image"], atol=0, rtol=0)
        torch.testing.assert_close(actual["img_cond"], fresh["img_cond"], atol=0, rtol=0)
        torch.testing.assert_close(actual["img_cond"][:, :, :1], expected, atol=0, rtol=0)
        assert torch.count_nonzero(actual["img_cond"][:, :, 1:]) == 0
        assert (
            actual["tiler_kwargs"]
            == fresh["tiler_kwargs"]
            == {
                "tiled": True,
                "tile_size": [height // 8, width // 8],
                "tile_stride": [height // 16 + 1, width // 16 - 2],
            }
        )
        results.append(actual)
    torch.testing.assert_close(results[0]["image"], results[2]["image"], atol=0, rtol=0)
