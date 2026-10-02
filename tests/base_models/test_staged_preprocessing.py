from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.runners.staged import StagedDiffusionPipeline


def _preprocessor() -> StagedDiffusionPipeline:
    pipe = object.__new__(StagedDiffusionPipeline)
    pipe.torch_dtype = torch.float32
    pipe.device = torch.device("cpu")
    return pipe


def test_preprocess_image_accepts_bchw_tensor_without_numpy_conversion() -> None:
    image = torch.tensor([[[[0.0, 0.5], [1.0, 0.25]]]])

    result = _preprocessor().preprocess_image(image)

    assert result.shape == (1, 1, 2, 2)
    torch.testing.assert_close(result, image * 2 - 1)


def test_preprocess_video_accepts_single_frame_tchw_tensor() -> None:
    video = torch.full((1, 3, 2, 4), 0.25)

    result = _preprocessor().preprocess_video(video)

    assert result.shape == (1, 3, 1, 2, 4)
    torch.testing.assert_close(result, torch.full_like(result, -0.5))
