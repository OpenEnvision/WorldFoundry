# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free WonderWorld preprocessing and layered reconstruction contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.wonderworld import runtime

pytestmark = pytest.mark.unit


def test_image_normalization_is_contiguous_and_computed_before_transfer() -> None:
    array = np.arange(3 * 17 * 19, dtype=np.uint8).reshape(17, 19, 3)
    image = Image.fromarray(array)
    actual = runtime.image_tensor(image, "cpu")
    expected = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()[None] / 255
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.image_pil(actual).tobytes() == image.tobytes()


def test_observation_resize_uses_the_released_bicubic_filter() -> None:
    from worldfoundry.operators.wonderworld_operator import WonderWorldOperator

    image = Image.fromarray(np.arange(3 * 17 * 19, dtype=np.uint8).reshape(17, 19, 3))
    actual, cameras = WonderWorldOperator().prepare(images=image)
    assert actual.tobytes() == image.resize((512, 512), Image.Resampling.BICUBIC).tobytes()
    assert cameras.shape == (0, 4, 4)


def test_larger_sam_regions_override_smaller_regions_and_ignore_tiny_masks() -> None:
    disparity = torch.full((1, 1, 20, 20), 500.0)
    disparity[..., :10, :] = 507
    depth = disparity.reciprocal()
    large = np.ones((20, 20), dtype=bool)
    small = large.copy()
    small[10:] = False
    tiny = large.copy()
    tiny[4:] = False
    regions = [{"segmentation": mask} for mask in (large, tiny, small)]
    actual = runtime.refine_disparity(depth, regions, torch.zeros(20, 20).bool())
    torch.testing.assert_close(actual, torch.full_like(depth, 1 / 503.5))


def test_background_disparity_uses_known_anchors_and_preserves_unmodified_regions() -> None:
    depth = torch.full((1, 1, 20, 20), 0.002)
    original = torch.full_like(depth, 0.0025)
    known = torch.zeros_like(depth).bool()
    known[..., :10] = True
    region = {"segmentation": np.ones((20, 20), dtype=bool)}
    actual = runtime.refine_disparity(
        depth, [region], torch.zeros(20, 20).bool(), existing_mask=known, existing_depth=original
    )
    torch.testing.assert_close(actual, original)
    unchanged = runtime.refine_disparity(
        depth, [region], torch.zeros(20, 20).bool(), existing_mask=torch.ones_like(known), existing_depth=original
    )
    torch.testing.assert_close(unchanged, depth)


def test_occluder_support_keeps_forced_objects_and_excludes_ground() -> None:
    import cv2

    semantic = torch.full((32, 32), 3)
    semantic[12:20, 12:20] = 4
    depth = torch.full((1, 1, 32, 32), 0.003)
    actual = runtime.occluder_mask(semantic, depth)
    expected = cv2.dilate((semantic.numpy() == 4).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    np.testing.assert_array_equal(actual[0, 0].numpy(), expected)


@pytest.mark.parametrize("category", [3, 6, 9, 11, 13, 26, 29, 46, 128])
def test_ground_mask_includes_the_released_surface_categories(category: int) -> None:
    semantic = torch.full((7, 9), category)
    assert runtime.ground_mask(semantic).all()
    semantic[:2, :2] = 0
    actual = runtime.ground_mask(semantic)
    assert not actual[:3, :3].any()
    assert actual[3:, 3:].all()


def test_ground_mask_does_not_treat_path_as_a_ground_plane() -> None:
    assert not runtime.ground_mask(torch.full((7, 9), 52)).any()


@pytest.mark.parametrize("with_sky", [False, True])
def test_covered_views_skip_generation_and_next_view_uses_current_render_and_depth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_sky: bool
) -> None:
    depth_calls: list[dict[str, Any]] = []
    inpaint_calls: list[dict[str, Any]] = []
    initialization: list[dict[str, Any]] = []
    fits: list[int] = []
    alpha = torch.ones(1, 1, 512, 512)
    alpha[..., 300:312, 200:212] = 0
    alpha[..., 318:322, 200:212] = 0.3
    median = torch.where(alpha > 0.5, 2.5, -1)
    expected_new = torch.nn.functional.max_pool2d((alpha < 0.01).float(), 7, 1, 3) > 0

    class Scene:
        def __init__(self, layers: torch.Tensor) -> None:
            self.layers = layers
            self.means = torch.zeros(len(layers), 3)

        @classmethod
        def from_rgbd(cls, image: torch.Tensor, depth: torch.Tensor, normals: torch.Tensor, **kwargs: Any) -> Scene:
            initialization.append(kwargs)
            count = 3 if kwargs["mask"].any() else 0
            return cls(torch.full((count,), kwargs["layer"], dtype=torch.long))

        def append(self, other: Scene) -> Scene:
            return Scene(torch.cat((self.layers, other.layers)))

        def remove_floaters(self, **kwargs: Any) -> Scene:
            return self

        def refine(self, *args: Any, stop: int, **kwargs: Any) -> Scene:
            fits.append(stop)
            return self

        def render(
            self, camera: torch.Tensor, intrinsic: torch.Tensor, *, height: int, width: int, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            novel = camera[0, 3] < -0.15
            return (
                torch.full((1, 3, height, width), 0.42 if novel else 0.31),
                alpha.clone() if novel else torch.ones_like(alpha),
                median.clone() if novel else torch.full_like(median, 2.5),
            )

        def save_splat(self, destination: Path) -> None:
            destination.write_bytes(bytes(32 * len(self.means)))

    semantic = torch.zeros(512, 512, dtype=torch.long)
    if with_sky:
        semantic[:128] = 2
    synthesis = runtime.WonderWorldSynthesis({}, device="cpu")
    monkeypatch.setattr(synthesis, "_ensure_models", lambda: None)
    monkeypatch.setattr(synthesis, "segmenter", lambda image: (semantic, []))

    def depth(pixels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        depth_calls.append({"pixels": pixels.clone(), **kwargs})
        return torch.full((1, 1, 512, 512), 0.003)

    def inpaint(pixels: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        inpaint_calls.append({"pixels": pixels.clone(), "mask": mask.clone(), **kwargs})
        return torch.full_like(pixels, 0.4)

    monkeypatch.setattr(synthesis, "_depth", depth)
    monkeypatch.setattr(synthesis, "_inpaint", inpaint)
    monkeypatch.setattr(
        synthesis,
        "_normals",
        lambda pixels, **kwargs: torch.tensor([0.0, 0, -1])[None, :, None, None].expand_as(pixels),
    )
    monkeypatch.setattr(runtime, "GaussianSurfels", Scene)
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    cameras[:, 0, 3] = [-0.1, -0.2]
    result = synthesis.predict(
        image=Image.new("RGB", (512, 512), (51, 51, 51)),
        camera_path=cameras,
        run_dir=tmp_path,
        gen_layer=False,
        refinement_steps=0,
    )
    assert len(depth_calls) == 2 and len(inpaint_calls) == 1 and fits == [3, 3]
    torch.testing.assert_close(inpaint_calls[0]["pixels"], torch.full((1, 3, 512, 512), 0.42), rtol=0, atol=0)
    assert torch.equal(inpaint_calls[0]["mask"], expected_new)
    assert torch.equal(inpaint_calls[0]["fill_mask"], alpha < 0.6)
    align = (median > 1) & (median < 4.8)
    assert torch.equal(depth_calls[1]["align"], align)
    torch.testing.assert_close(depth_calls[1]["target"][align], (median / 1000)[align], rtol=0, atol=0)
    torch.testing.assert_close(depth_calls[1]["pixels"], torch.full((1, 3, 512, 512), 0.4), rtol=0, atol=0)
    bases = [item for item in initialization if item["layer"] == 1]
    assert len(bases) == 2 and torch.equal(bases[1]["mask"], expected_new)
    torch.testing.assert_close(bases[1]["world_to_camera"], torch.from_numpy(cameras[1]), rtol=0, atol=0)
    views = result["metadata"]["views"]
    assert len(views) == 3 and views[1]["new_pixels"] == views[1]["new_surfels"] == 0
    assert views[2]["new_pixels"] == int(expected_new.sum()) and views[2]["new_surfels"] == 3
    for index in range(3):
        assert (tmp_path / f"view_{index:03d}.png").is_file()


def test_background_is_fitted_before_frozen_base_and_foreground_composition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitting: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    initialization: list[dict[str, Any]] = []

    class Scene:
        def __init__(self, layers: torch.Tensor) -> None:
            self.layers = layers
            self.means = torch.zeros(len(layers), 3)

        @classmethod
        def from_rgbd(cls, image: torch.Tensor, depth: torch.Tensor, normals: torch.Tensor, **kwargs: Any) -> Scene:
            initialization.append(kwargs)
            return cls(torch.full((3,), kwargs["layer"]))

        def append(self, other: Scene) -> Scene:
            return Scene(torch.cat((self.layers, other.layers)))

        def remove_floaters(self, *, mean_squared_distance: float) -> Scene:
            assert mean_squared_distance == 0.0002
            return self

        def refine(
            self, image: torch.Tensor, camera: torch.Tensor, intrinsic: torch.Tensor, *, steps: int, stop: int
        ) -> Scene:
            fitting.append((image.clone(), stop, self.layers.clone()))
            return self

        def render(self, *args: Any, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.full((1, 3, height, width), 0.2),
                torch.ones(1, 1, height, width),
                torch.ones(1, 1, height, width),
            )

        def save_splat(self, destination: Path) -> None:
            destination.write_bytes(bytes(32 * len(self.means)))

    synthesis = runtime.WonderWorldSynthesis({}, device="cpu")
    monkeypatch.setattr(synthesis, "_ensure_models", lambda: None)
    monkeypatch.setattr(synthesis, "segmenter", lambda image: (torch.zeros(512, 512).long(), []))
    monkeypatch.setattr(synthesis, "_depth", lambda pixels, **kwargs: torch.full((1, 1, 512, 512), 0.003))
    negatives = []

    def inpaint(pixels: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        negatives.append(kwargs["negative_prompt"])
        return torch.full_like(pixels, 0.803125)

    monkeypatch.setattr(synthesis, "_inpaint", inpaint)
    monkeypatch.setattr(
        synthesis,
        "_normals",
        lambda pixels, **kwargs: torch.tensor([0.0, 0, -1])[None, :, None, None].expand_as(pixels),
    )
    support = torch.zeros(1, 1, 512, 512).bool()
    support[..., 128:256, 128:256] = True
    monkeypatch.setattr(runtime, "occluder_mask", lambda semantic, depth: support)
    monkeypatch.setattr(runtime, "GaussianSurfels", Scene)
    synthesis.predict(
        image=Image.new("RGB", (512, 512), (51, 51, 51)),
        camera_path=np.empty((0, 4, 4), dtype=np.float32),
        run_dir=tmp_path,
        refinement_steps=100,
    )
    assert len(fitting) == 2 and [item[1] for item in fitting] == [3, 3]
    assert fitting[0][2].tolist() == [1, 1, 1]
    assert fitting[1][2].tolist() == [2, 2, 2, 1, 1, 1]
    assert negatives == ["tree, plant"]
    assert not torch.equal(fitting[0][0], fitting[1][0])
    torch.testing.assert_close(fitting[0][0][0, :, 192, 192], torch.full((3,), 0.8), rtol=0, atol=0)
    torch.testing.assert_close(fitting[1][0], torch.full_like(fitting[1][0], 0.2), rtol=0, atol=0)
    assert [item["layer"] for item in initialization] == [1, 2]
    assert all(
        item["opacity"] == 0.15 and item["footprint"] == 0.7 and item["thickness"] == 0.7 for item in initialization
    )
