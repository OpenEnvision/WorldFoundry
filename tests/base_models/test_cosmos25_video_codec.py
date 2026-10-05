from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos2p5.component import (
    Cosmos25VideoCodec,
    _extract_canny_edges,
)


def _rgb_step_frame() -> np.ndarray:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, 8:] = 255
    return frame


def test_cosmos25_edge_control_extracts_canny_from_raw_rgb() -> None:
    frames = np.stack([_rgb_step_frame()], axis=0)
    codec = Cosmos25VideoCodec(None, device=torch.device("cpu"))  # type: ignore[arg-type]
    request = DiffusionRequest(
        prompt="test",
        height=16,
        width=16,
        num_frames=1,
        inputs={"control_video": frames, "controlnet_variant": "edge"},
    )

    pixels = codec._control_pixels(request, device=torch.device("cpu"), dtype=torch.float32)

    assert pixels is not None
    assert pixels.shape == (1, 3, 1, 16, 16)
    assert torch.equal(pixels[:, 0], pixels[:, 1])
    assert torch.equal(pixels[:, 1], pixels[:, 2])
    assert set(torch.unique(pixels).tolist()) == {-1.0, 1.0}


def test_cosmos25_preprocessed_edge_control_is_not_extracted_twice() -> None:
    edges = _extract_canny_edges(np.stack([_rgb_step_frame()], axis=0))
    codec = Cosmos25VideoCodec(None, device=torch.device("cpu"))  # type: ignore[arg-type]
    request = DiffusionRequest(
        prompt="test",
        height=16,
        width=16,
        num_frames=1,
        inputs={
            "control_video": edges,
            "controlnet_variant": "edge",
            "control_is_preprocessed": True,
        },
    )

    pixels = codec._control_pixels(request, device=torch.device("cpu"), dtype=torch.float32)

    assert pixels is not None
    expected = torch.from_numpy(edges).permute(0, 3, 1, 2).permute(1, 0, 2, 3).unsqueeze(0)
    expected = expected.float().div(127.5).sub(1)
    assert torch.equal(pixels, expected)


def test_cosmos25_image_conditioning_zero_pads_future_frames() -> None:
    captured = []

    class RecordingVAE:
        def encode(self, images, device, **kwargs):
            captured.extend(images)
            return torch.zeros(1, 16, 2, 2, 2)

    codec = Cosmos25VideoCodec(RecordingVAE(), device=torch.device("cpu"))  # type: ignore[arg-type]
    request = DiffusionRequest(
        prompt="test",
        height=16,
        width=16,
        num_frames=5,
        inputs={"image": Image.new("RGB", (16, 16), (128, 128, 128))},
    )

    initialized = codec.initialize(
        request,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(captured) == 1
    assert captured[0].shape == (3, 5, 16, 16)
    assert torch.all(captured[0][:, 0] > -1)
    assert torch.all(captured[0][:, 1:] == -1)
    assert torch.all(initialized.conditioning["condition_indicator"][:, :, 0] == 1)
    assert torch.all(initialized.conditioning["condition_indicator"][:, :, 1] == 0)
    assert initialized.conditioning["conditional_frame_timestep"] == -1.0
    expected_noise = np.random.RandomState(0).standard_normal((1, 16, 2, 2, 2)).astype(np.float32)
    torch.testing.assert_close(initialized.latents, torch.from_numpy(expected_noise), rtol=0, atol=0)


def test_cosmos25_direct_initializer_uses_generator_initial_seed() -> None:
    codec = Cosmos25VideoCodec(None, device=torch.device("cpu"))  # type: ignore[arg-type]
    request = DiffusionRequest(prompt="test", height=16, width=16, num_frames=1)

    initialized = codec.initialize(
        request,
        generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_noise = np.random.RandomState(17).standard_normal((1, 16, 1, 2, 2)).astype(np.float32)
    torch.testing.assert_close(initialized.latents, torch.from_numpy(expected_noise), rtol=0, atol=0)


def test_cosmos25_variant_conditional_timestep_default_can_be_overridden() -> None:
    codec = Cosmos25VideoCodec(
        None, device=torch.device("cpu"), conditional_frame_timestep=0.1
    )  # type: ignore[arg-type]
    request = DiffusionRequest(prompt="test", height=16, width=16, num_frames=1)
    initialized = codec.initialize(
        request,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert initialized.conditioning["conditional_frame_timestep"] == 0.1

    explicit = DiffusionRequest(
        prompt="test", height=16, width=16, num_frames=1,
        inputs={"conditional_frame_timestep": 0.4},
    )
    initialized = codec.initialize(
        explicit,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert initialized.conditioning["conditional_frame_timestep"] == 0.4


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1, 100), (100, 100), (200, 100), (256, 256)],
)
def test_cosmos25_canny_thresholds_are_validated(low: int, high: int) -> None:
    with pytest.raises(ValueError, match="Canny thresholds"):
        _extract_canny_edges(
            np.stack([_rgb_step_frame()], axis=0),
            low_threshold=low,
            high_threshold=high,
        )
