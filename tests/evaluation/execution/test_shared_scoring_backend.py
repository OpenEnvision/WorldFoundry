# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the shared visual/text scoring interface."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from worldfoundry.evaluation.tasks.execution.runners._scorers import score as scoring

pytestmark = pytest.mark.unit


class _Backend:
    def __init__(self, *, video_mode: str = "direct", pixels: bool = False, fail: bool = False) -> None:
        self.video_mode = video_mode
        self.pixels = pixels
        self.fail = fail
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []
        self.images: list[np.ndarray] = []

    def forward(self, images: list[str], texts: list[str], **kwargs: Any) -> torch.Tensor:
        self.calls.append((list(images), list(texts), kwargs))
        values = []
        for image, text in zip(images, texts):
            if self.pixels:
                pixels = cv2.imread(image)
                assert pixels is not None
                self.images.append(pixels)
                value = float(pixels.mean())
            else:
                value = float(Path(image).stem.rsplit("-", 1)[-1])
            values.append(value + float(text))
        if self.fail:
            raise RuntimeError("backend failure")
        return torch.tensor(values)


class _Score(scoring.Score):
    def prepare_scoremodel(self, model: str, device: str, cache_dir: str, **kwargs: Any) -> _Backend:
        return _Backend(**kwargs)

    def list_all_models(self) -> list[str]:
        return ["fixture"]


def _write_video(path: Path, frames: int, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (16, 16))
    assert writer.isOpened(), "MJPG encoder required for the local video contract"
    try:
        for index in range(frames):
            writer.write(np.full((16, 16, 3), offset + 20 * index, dtype=np.uint8))
    finally:
        writer.release()


def _reference_mosaic(path: Path, count: int) -> np.ndarray:
    """Independently reproduce floor-step sampling, JPEG frames, and ten-pixel gaps."""
    video = cv2.VideoCapture(str(path))
    assert video.isOpened()
    step = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // count
    frames = []
    try:
        for index in range(count):
            video.set(cv2.CAP_PROP_POS_FRAMES, index * step)
            success, frame = video.read()
            assert success
            success, jpeg = cv2.imencode(".jpg", frame)
            assert success
            frames.append(cv2.imdecode(jpeg, cv2.IMREAD_COLOR))
    finally:
        video.release()
    mosaic = np.zeros((16, 16 * count + 10 * (count - 1), 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        mosaic[:, index * 26 : index * 26 + 16] = frame
    success, jpeg = cv2.imencode(".jpg", mosaic)
    assert success
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)


def test_forward_matrix_and_scalar_pairing() -> None:
    score = _Score("fixture", device="cpu")
    torch.testing.assert_close(
        score(["image-2.png", "image-7.png"], ["3", "5"], temperature=0.2), torch.tensor([[5.0, 7.0], [10.0, 12.0]])
    )
    torch.testing.assert_close(score("image-2.png", "3"), torch.tensor([[5.0]]))
    assert score.model.calls[0] == (["image-2.png"] * 2, ["3", "5"], {"temperature": 0.2})


@pytest.mark.parametrize("media_type", ["images", "videos"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_pairing_and_partial_final_batch(media_type: str, batch_size: int) -> None:
    score = _Score("fixture", device="cpu")
    suffix = "avi" if media_type == "videos" else "png"
    dataset = [
        {media_type: [f"image-{i}.{suffix}", f"image-{i + 10}.{suffix}"], "texts": [str(i + 20), str(i + 30)]}
        for i in range(5)
    ]
    expected = torch.tensor(
        [[[2 * i + 20, 2 * i + 30], [2 * i + 30, 2 * i + 40]] for i in range(5)], dtype=torch.float32
    )
    torch.testing.assert_close(score.batch_forward(dataset, batch_size=batch_size, question="q"), expected)
    assert len(score.model.calls) == 20
    assert all(
        len(images) == len(texts) == 1 and kwargs == {"question": "q"} for images, texts, kwargs in score.model.calls
    )


@pytest.mark.parametrize("frames,count", [(8, 3), (2, 4)])
def test_concat_matches_reference_pixels_and_cleans_files(tmp_path: Path, frames: int, count: int) -> None:
    video = tmp_path / "source.avi"
    _write_video(video, frames)
    score = _Score("fixture", device="cpu", video_mode="concat", pixels=True)
    score(str(video), "0", num_frames=count)
    np.testing.assert_array_equal(score.model.images[0], _reference_mosaic(video, count))
    assert not Path(score.model.calls[0][0][0]).parent.parent.exists()
    assert video.is_file()


@pytest.mark.parametrize("fail", [False, True])
def test_same_basename_videos_and_mixed_images_are_isolated(tmp_path: Path, fail: bool) -> None:
    videos = [tmp_path / name / "same.avi" for name in ("left", "right")]
    for index, path in enumerate(videos):
        _write_video(path, 4, offset=index * 100)
    image = tmp_path / "image.png"
    assert cv2.imwrite(str(image), np.full((16, 16, 3), 200, dtype=np.uint8))
    score = _Score("fixture", device="cpu", video_mode="concat", pixels=True, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="backend failure"):
            score([str(path) for path in videos] + [str(image)], "0")
    else:
        score([str(path) for path in videos] + [str(image)], "0")
        assert score.model.calls[0][0] != score.model.calls[1][0]
        assert score.model.images[0].mean() < score.model.images[1].mean()
        assert score.model.calls[2][0] == [str(image)]
    for images, _, _ in score.model.calls:
        if images[0] != str(image):
            assert not Path(images[0]).parent.parent.exists()
    assert image.is_file()


def test_concat_batch_prepares_each_video_once_for_all_text_options(tmp_path: Path) -> None:
    video = tmp_path / "source.avi"
    _write_video(video, 8)
    score = _Score("fixture", device="cpu", video_mode="concat", pixels=True)
    scores = score.batch_forward([{"videos": [str(video)], "texts": ["0", "1", "2"]}])
    images = [call[0][0] for call in score.model.calls]
    assert len(set(images)) == 1
    assert scores.shape == (1, 1, 3)
    np.testing.assert_array_equal(score.model.images[0], _reference_mosaic(video, 4))
    assert not Path(images[0]).parent.parent.exists()


def test_images_and_direct_video_do_not_allocate_temporary_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_temporary(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("image/direct-video paths must not allocate a mosaic")

    monkeypatch.setattr(scoring, "TemporaryDirectory", unexpected_temporary)
    score = _Score("fixture", device="cpu")
    torch.testing.assert_close(score(["image-2.png", "video-4.avi"], "3"), torch.tensor([[5.0], [7.0]]))


@pytest.mark.parametrize(
    "family,model,registry,getter,lister",
    [
        (
            "clip_score.clipscore",
            "blip2-itc",
            "clip_score.models.clipscore_models",
            "get_clipscore_model",
            "list_all_clipscore_models",
        ),
        (
            "itm_score.itmscore",
            "blip2-itm",
            "itm_score.models.itmscore_models",
            "get_itmscore_model",
            "list_all_itmscore_models",
        ),
        (
            "vqa_score.vqascore",
            "clip-flant5-xxl",
            "worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models",
            "get_vqascore_model",
            "list_all_vqascore_models",
        ),
    ],
)
def test_families_use_one_score_base(
    monkeypatch: pytest.MonkeyPatch, family: str, model: str, registry: str, getter: str, lister: str
) -> None:
    prefix = "worldfoundry.evaluation.tasks.execution.runners._scorers."
    registry_name = registry if registry.startswith("worldfoundry.") else prefix + registry
    backend = _Backend()
    module = ModuleType(registry_name)
    setattr(module, getter, lambda *args, **kwargs: backend)
    setattr(module, lister, lambda: [model])
    monkeypatch.setitem(sys.modules, registry_name, module)
    family_name = prefix + family
    monkeypatch.delitem(sys.modules, family_name, raising=False)
    try:
        family_module = importlib.import_module(family_name)
        cls = getattr(
            family_module,
            {"clip_score.clipscore": "CLIPScore", "itm_score.itmscore": "ITMScore", "vqa_score.vqascore": "VQAScore"}[
                family
            ],
        )
        score = cls(model, device="cpu", cache_dir="unused")
        assert cls.__bases__ == (scoring.Score,)
        torch.testing.assert_close(score("image-2.png", "3"), torch.tensor([[5.0]]))
    finally:
        sys.modules.pop(family_name, None)


def test_invalid_model_and_video_mode_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="unsupported score model"):
        _Score("unknown", device="cpu")
    score = _Score("fixture", device="cpu", video_mode="unknown")
    with pytest.raises(ValueError, match="unsupported video_mode"):
        score("video-2.avi", "3")
