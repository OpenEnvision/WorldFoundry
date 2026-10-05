from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat.worldfoundry_runtime import (
    PixelSplatRuntime,
)


def test_stage_demo_re10k_packages_sequential_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (96, 64), color).save(image_dir / f"{index:02d}.png")

    image_paths = PixelSplatRuntime._input_image_paths(image_dir, {})
    dataset_root, evaluation_index, selected = PixelSplatRuntime._stage_demo_re10k(
        image_paths
    )
    sample = torch.load(dataset_root / "test" / "000000.torch", map_location="cpu")[
        0
    ]

    assert len(selected) == 3
    assert evaluation_index.is_file()
    assert sample["key"] == "worldfoundry-demo"
    assert tuple(sample["cameras"].shape) == (3, 18)
    torch.testing.assert_close(
        sample["cameras"][:, 9], torch.tensor([0.0, -0.05, -0.1])
    )
    assert len(sample["images"]) == 3
    with Image.open(BytesIO(sample["images"][0].numpy().tobytes())) as image:
        assert image.size == (640, 360)


def test_stage_demo_re10k_repeats_a_single_image(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), (16, 32, 64)).save(image_path)

    dataset_root, _, selected = PixelSplatRuntime._stage_demo_re10k([image_path])
    sample = torch.load(dataset_root / "test" / "000000.torch", map_location="cpu")[
        0
    ]

    assert selected == [str(image_path)] * 3
    assert len(sample["images"]) == 3
