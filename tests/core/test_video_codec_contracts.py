# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""CPU codec contracts for reader ownership and supported media backend APIs."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional

from worldfoundry.core.media.codecs import video

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure", [None, "decode", "metadata"])
def test_read_video_closes_reader_before_releasing_local_input(
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    events: list[str] = []
    frames = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)

    class Reader:
        def __iter__(self) -> Iterator[np.ndarray]:
            yield frames[0]
            if failure == "decode":
                raise RuntimeError("decode failed")
            yield frames[1]

        def get_meta_data(self) -> dict[str, Any]:
            if failure == "metadata":
                raise RuntimeError("metadata failed")
            return {"fps": 12}

        def close(self) -> None:
            events.append("reader closed")

    @contextmanager
    def local_input(path: str) -> Generator[Path, None, None]:
        assert path == "https://example.test/input.mp4"
        try:
            yield Path("local.mp4")
        finally:
            events.append("local input released")

    monkeypatch.setattr(imageio, "get_reader", lambda path: Reader())
    monkeypatch.setattr(video, "local_path_for_uri", local_input)
    if failure is not None:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            video.read_video("https://example.test/input.mp4")
    else:
        decoded, metadata = video.read_video("https://example.test/input.mp4")
        np.testing.assert_array_equal(decoded, frames)
        assert metadata == {"fps": 12}
    assert events == ["reader closed", "local input released"]


@pytest.mark.parametrize("remote", [False, True])
def test_imageio_write_preserves_array_format_and_encoder_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote: bool,
) -> None:
    frames = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    target = "memory://video.mp4" if remote else str(tmp_path / "video.mp4")
    written: list[Path] = []
    uploaded: list[tuple[str, bytes]] = []

    def save(path: str, pixels: np.ndarray, **kwargs: Any) -> None:
        assert pixels is frames
        assert kwargs == {"fps": 12, "macro_block_size": 1, "quality": 7, "format": "FFMPEG", "codec": "libx264"}
        written.append(Path(path))
        Path(path).write_bytes(b"encoded")

    monkeypatch.setattr(imageio, "mimsave", save)
    monkeypatch.setattr(video, "write_binary_uri", lambda path, data: uploaded.append((path, data)))
    video.write_video(frames, target, fps=12, quality=7, format="FFMPEG", codec="libx264")
    assert len(written) == 1
    if remote:
        assert uploaded == [(target, b"encoded")]
        assert not written[0].exists()
    else:
        assert written == [Path(target)]
        assert Path(target).read_bytes() == b"encoded"
        assert not uploaded


def test_imageio_array_and_format_contract_matches_direct_backend(tmp_path: Path) -> None:
    frames = np.random.default_rng(7).integers(0, 256, (3, 16, 24, 3), dtype=np.uint8)
    original = frames.copy()
    reference, candidate = tmp_path / "reference.mp4", tmp_path / "candidate.mp4"
    imageio.mimsave(str(reference), frames, fps=12, macro_block_size=1, quality=7, format="FFMPEG")  # ty: ignore[no-matching-overload]
    video.write_video(frames, candidate, fps=12, quality=7, format="FFMPEG")
    expected, expected_metadata = video.read_video(reference)
    actual, metadata = video.read_video(candidate)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frames, original)
    assert actual.shape == frames.shape
    assert metadata["fps"] == expected_metadata["fps"] == 12


@pytest.mark.parametrize("directory_kind", ["scratch", "str", "path"])
def test_materialize_video_input_accepts_default_and_explicit_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_kind: str,
) -> None:
    directory = tmp_path / "materialized"
    directory.mkdir()
    monkeypatch.setattr(video, "scratch_directory", lambda prefix: directory)
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    calls: list[tuple[str, int]] = []

    def save(pixels: np.ndarray, path: str, *, fps: int) -> None:
        assert pixels is frames
        calls.append((path, fps))

    monkeypatch.setattr(video, "save_video_frames", save)
    output_dir = None if directory_kind == "scratch" else str(directory) if directory_kind == "str" else directory
    result = video.materialize_video_input(frames, output_dir=output_dir, filename="input.mp4", fps=12)
    assert result == str(directory / "input.mp4")
    assert calls == [(result, 12)]


def test_image_conditioning_preserves_resize_crop_and_zero_tail(tmp_path: Path) -> None:
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    path = tmp_path / "input.png"
    image = Image.fromarray(pixels)
    image.save(path)
    tensor = (functional.to_tensor(image) * 255).to(torch.uint8).unsqueeze(0)
    expected = functional.center_crop(functional.resize(tensor, [6, 12]), [6, 6])
    result = video.read_image_as_video_tensor(path, (6, 6), 3)
    assert result.shape == (1, 3, 3, 6, 6)
    assert result.dtype == torch.uint8
    torch.testing.assert_close(result[:, :, 0], expected, rtol=0, atol=0)
    assert torch.count_nonzero(result[:, :, 1:]) == 0


@pytest.mark.parametrize("fail_write", [False, True])
def test_opencv_fallback_uses_fourcc_and_releases_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_write: bool,
) -> None:
    import cv2
    import torchvision.io

    frames = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    path = tmp_path / "output.mp4"
    original_fourcc = cv2.VideoWriter.fourcc
    written: list[np.ndarray] = []
    closed: list[bool] = []

    class Writer:
        fourcc = staticmethod(original_fourcc)

        def __init__(self, filename: str, codec: int, fps: float, size: tuple[int, int]) -> None:
            assert filename == str(path)
            assert codec == original_fourcc(*"mp4v")
            assert fps == 12
            assert size == (6, 4)

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            if fail_write:
                raise RuntimeError("write failed")
            written.append(frame)

        def release(self) -> None:
            closed.append(True)

    monkeypatch.delattr(torchvision.io, "write_video")
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    if fail_write:
        with pytest.raises(RuntimeError, match="write failed"):
            video.write_video_torchvision(path, frames, 12)
    else:
        video.write_video_torchvision(path, frames, 12)
        np.testing.assert_array_equal(np.stack(written), frames[..., ::-1])
    assert closed == [True]
