from __future__ import annotations

import av
import numpy as np
import pytest
import torch
from torchvision.io import read_video

from worldfoundry.core.media.codecs.video import read_video_window_rgb, save_video_h264


class _Container:
    def __init__(self, *, fail=False):
        self.decoded = []
        self.converted = []
        self.closed = False
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def decode(self, *, video):
        assert video == 0
        for index in range(100):
            self.decoded.append(index)
            container = self

            class Frame:
                def to_rgb(self):
                    return self

                def to_ndarray(self):
                    if container.fail:
                        raise RuntimeError("decode failed")
                    container.converted.append(index)
                    return np.full((4, 6, 3), index, dtype=np.uint8)

            yield Frame()


def test_window_stops_before_decoding_tail_and_closes_decoder(monkeypatch):
    container = _Container()
    monkeypatch.setattr(av, "open", lambda path: container)
    frames = read_video_window_rgb("unused.mp4", start_frame=2, frame_count=3)
    assert container.decoded == [0, 1, 2, 3, 4]
    assert container.converted == [2, 3, 4]
    assert container.closed
    assert frames.shape == (3, 4, 6, 3)
    np.testing.assert_array_equal(frames[:, 0, 0, 0], [2, 3, 4])


def test_decode_failure_still_closes_decoder(monkeypatch):
    container = _Container(fail=True)
    monkeypatch.setattr(av, "open", lambda path: container)
    with pytest.raises(RuntimeError, match="decode failed"):
        read_video_window_rgb("unused.mp4", frame_count=1)
    assert container.closed


@pytest.mark.parametrize(
    "kwargs", [{"start_frame": -1, "frame_count": 1}, {"frame_count": 0}, {"frame_count": 1.5}, {"frame_count": True}]
)
def test_invalid_windows_fail_before_opening_decoder(monkeypatch, kwargs):
    monkeypatch.setattr(av, "open", lambda _: pytest.fail("decoder must not be opened"))
    with pytest.raises((TypeError, ValueError)):
        read_video_window_rgb("unused.mp4", **kwargs)


def test_real_window_matches_existing_torchvision_pixels_and_frame_order(tmp_path):
    frames = np.random.default_rng(7).integers(0, 256, size=(12, 32, 48, 3), dtype=np.uint8)
    path = tmp_path / "input.mp4"
    save_video_h264(frames, path, fps=12)
    full, _, _ = read_video(str(path), pts_unit="sec")
    np.testing.assert_array_equal(read_video_window_rgb(path, start_frame=2, frame_count=3), full[2:5].numpy())
    np.testing.assert_array_equal(read_video_window_rgb(path, start_frame=10, frame_count=4), full[10:].numpy())
    with pytest.raises(ValueError, match="No frames found"):
        read_video_window_rgb(path, start_frame=12, frame_count=1)


def test_oasis_window_retains_original_normalized_conditioning(tmp_path):
    from torchvision.transforms.functional import resize

    from worldfoundry.synthesis.visual_generation.open_oasis.utils import load_prompt

    frames = np.random.default_rng(9).integers(0, 256, size=(8, 32, 48, 3), dtype=np.uint8)
    path = tmp_path / "input.mp4"
    save_video_h264(frames, path, fps=12)
    full, _, _ = read_video(str(path), pts_unit="sec")
    expected = resize(full[2:5].permute(0, 3, 1, 2), (360, 640)).unsqueeze(0).float() / 255
    actual = load_prompt(str(path), video_offset=2, n_prompt_frames=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_tail = resize(full[-2:].permute(0, 3, 1, 2), (360, 640)).unsqueeze(0).float() / 255
    torch.testing.assert_close(
        load_prompt(str(path), video_offset=-2, n_prompt_frames=2), expected_tail, rtol=0, atol=0
    )


class _ImageIOReader:
    def __init__(self, *, fail=False):
        self.decoded = []
        self.closed = False
        self.fail = fail

    def __iter__(self):
        buffer = np.zeros((4, 6, 3), dtype=np.uint8)
        for index in range(10):
            if self.fail and index == 2:
                raise RuntimeError("reader failed")
            self.decoded.append(index)
            buffer.fill(index)
            yield buffer

    def close(self):
        self.closed = True


@pytest.mark.parametrize("from_end,expected,decoded", [(False, [0, 1, 2], 3), (True, [7, 8, 9], 10)])
def test_imageio_windows_close_and_preserve_reused_reader_buffers(monkeypatch, from_end, expected, decoded):
    import imageio

    from worldfoundry.core.media.codecs.video import load_video_frames

    reader = _ImageIOReader()
    monkeypatch.setattr(imageio, "get_reader", lambda _, **kw: reader)
    actual = load_video_frames("unused.mp4", max_frames=3, from_end=from_end)
    np.testing.assert_array_equal(actual[:, 0, 0, 0], expected)
    assert len(reader.decoded) == decoded
    assert reader.closed


def test_imageio_window_failure_closes_reader_and_temporary_uri(monkeypatch):
    from contextlib import contextmanager

    import imageio

    from worldfoundry.core.media.codecs import video

    reader = _ImageIOReader(fail=True)
    cleaned = []

    @contextmanager
    def local_copy(path):
        try:
            yield "local.mp4"
        finally:
            cleaned.append(path)

    monkeypatch.setattr(imageio, "get_reader", lambda _, **kw: reader)
    monkeypatch.setattr(video, "local_path_for_uri", local_copy)
    with pytest.raises(RuntimeError, match="reader failed"):
        video.load_video_frames("https://example.test/video.mp4", max_frames=5)
    assert reader.closed
    assert cleaned == ["https://example.test/video.mp4"]


@pytest.mark.parametrize("from_end", [False, True])
def test_bounded_imageio_input_matches_original_full_reader_and_short_eof(tmp_path, from_end):
    from worldfoundry.core.media.codecs.video import coerce_video_frames, load_video_frames

    path = tmp_path / "control.mp4"
    save_video_h264(np.random.default_rng(13).integers(0, 256, (12, 32, 48, 3), dtype=np.uint8), path)
    full = load_video_frames(path)
    expected = full[-3:] if from_end else full[:3]
    np.testing.assert_array_equal(coerce_video_frames(path, max_frames=3, from_end=from_end), expected)
    np.testing.assert_array_equal(coerce_video_frames(path, max_frames=20, from_end=from_end), full)


@pytest.mark.parametrize("kind", ["numpy", "list", "torch"])
def test_truncation_preserves_whole_input_value_range_inference(kind):
    from worldfoundry.core.media.codecs.video import coerce_video_frames

    array = np.ones((5, 4, 6, 3), dtype=np.float32) * 0.75
    # The excluded end frame changes auto normalization. Normalize before slicing.
    array[-1] = 100 if kind != "torch" else -1
    value = torch.from_numpy(array) if kind == "torch" else list(array) if kind == "list" else array
    full = coerce_video_frames(value)
    np.testing.assert_array_equal(coerce_video_frames(value, max_frames=2), full[:2])
    np.testing.assert_array_equal(coerce_video_frames(value, max_frames=2, from_end=True), full[-2:])


@pytest.mark.parametrize(
    "options,error",
    [({"max_frames": 0}, ValueError), ({"max_frames": -1}, ValueError),
     ({"max_frames": True}, TypeError), ({"max_frames": 1.5}, TypeError),
     ({"from_end": True}, ValueError), ({"max_frames": 1, "from_end": "false"}, TypeError)],
)
def test_invalid_limits_fail_before_opening_reader(monkeypatch, options, error):
    import imageio

    from worldfoundry.core.media.codecs.video import coerce_video_frames

    monkeypatch.setattr(imageio, "get_reader", lambda _: pytest.fail("must not open reader"))
    with pytest.raises(error):
        coerce_video_frames("unused.mp4", **options)
