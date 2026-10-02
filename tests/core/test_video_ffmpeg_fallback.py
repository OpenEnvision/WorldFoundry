import io
import threading
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.io

from worldfoundry.core.media.codecs import video


def test_ffmpeg_resolver_uses_imageio_bundle_when_system_binary_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_FFMPEG_PREFERENCE", raising=False)
    monkeypatch.setattr(video.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "/opt/imageio/ffmpeg"),
    )

    assert video._resolve_ffmpeg_executable() == "/opt/imageio/ffmpeg"


def test_ffmpeg_resolver_prefers_pinned_imageio_bundle_by_default(monkeypatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_FFMPEG_PREFERENCE", raising=False)
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        video,
        "_imageio_ffmpeg_executable",
        lambda: "/opt/imageio/ffmpeg",
    )

    assert video._resolve_ffmpeg_executable() == "/opt/imageio/ffmpeg"


def test_ffmpeg_resolver_can_prefer_system_binary(monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_FFMPEG_PREFERENCE", "system")
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        video,
        "_imageio_ffmpeg_executable",
        lambda: "/opt/imageio/ffmpeg",
    )

    assert video._resolve_ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_ffmpeg_resolver_explicit_path_wins_over_preference(monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_FFMPEG_PREFERENCE", "system")

    assert video._resolve_ffmpeg_executable("/custom/ffmpeg") == "/custom/ffmpeg"


def test_save_video_h264_drains_stderr_before_streaming_frames(monkeypatch, tmp_path) -> None:
    drain_started = threading.Event()

    class RecordingStderr:
        def __init__(self) -> None:
            self._chunks = [b"ffmpeg diagnostic", b""]

        def read(self, _size: int = -1) -> bytes:
            drain_started.set()
            return self._chunks.pop(0)

    class GuardedStdin(io.BytesIO):
        def write(self, data: bytes) -> int:
            assert drain_started.wait(timeout=1.0)
            return super().write(data)

    process = SimpleNamespace(
        stdin=GuardedStdin(),
        stderr=RecordingStderr(),
        wait=lambda: 0,
        kill=lambda: None,
    )
    monkeypatch.setattr(video, "_resolve_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", lambda *_args, **_kwargs: process)

    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    video.save_video_h264(frames, tmp_path / "video.mp4")


def test_save_video_h264_only_relocates_metadata_when_requested(
    monkeypatch,
    tmp_path,
) -> None:
    commands: list[list[str]] = []

    class _Stdin(io.BytesIO):
        pass

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdin=_Stdin(),
            stderr=io.BytesIO(),
            wait=lambda: 0,
            kill=lambda: None,
        )

    monkeypatch.setattr(video, "_resolve_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", fake_popen)
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)

    video.save_video_h264(frames, tmp_path / "latency.mp4")
    video.save_video_h264(frames, tmp_path / "streamable.mp4", faststart=True)

    assert "-movflags" not in commands[0]
    assert commands[1][-3:-1] == ["-movflags", "+faststart"]


def test_ffmpeg_stderr_drain_retains_a_bounded_tail() -> None:
    result: list[bytes] = []

    video._drain_ffmpeg_stderr(io.BytesIO(b"0123456789"), result, max_bytes=4)

    assert result == [video._FFMPEG_STDERR_TRUNCATED_PREFIX + b"6789"]


def test_default_local_mp4_streams_directly_to_ffmpeg(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_h264_writer(frames, output_path, *, fps, crf) -> None:
        calls.update(frames=frames, output_path=output_path, fps=fps, crf=crf)

    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    output_path = tmp_path / "video.mp4"
    monkeypatch.setattr(video, "save_video_h264", fake_h264_writer)

    video.write_video(frames, output_path, fps=30, format="mp4")

    np.testing.assert_array_equal(calls.pop("frames"), frames)
    assert calls == {"output_path": output_path, "fps": 30, "crf": 23}


def test_torchvision_pyav_failure_falls_back_to_h264(monkeypatch, tmp_path) -> None:
    calls = {}

    def incompatible_torchvision_writer(*_args, **_kwargs) -> None:
        raise TypeError("an integer is required")

    def fake_h264_writer(frames, output_path, *, fps, crf) -> None:
        calls.update(frames=frames, output_path=output_path, fps=fps, crf=crf)

    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    output_path = tmp_path / "video.mp4"
    monkeypatch.setattr(torchvision.io, "write_video", incompatible_torchvision_writer)
    monkeypatch.setattr(video, "save_video_h264", fake_h264_writer)

    video.write_video_torchvision(
        output_path,
        frames,
        30,
        video_codec="libx264",
        options={"crf": "16"},
    )

    np.testing.assert_array_equal(calls.pop("frames"), frames)
    assert calls == {"output_path": output_path, "fps": 30, "crf": 16}


def test_torchvision_pyav_fallback_preserves_float_0_255_tensor(monkeypatch, tmp_path) -> None:
    captured = {}

    def incompatible_torchvision_writer(*_args, **_kwargs) -> None:
        raise TypeError("an integer is required")

    def fake_h264_writer(frames, _output_path, *, fps, crf) -> None:
        captured.update(frames=frames, fps=fps, crf=crf)

    frames = torch.tensor(
        [[[[0.0, 64.0, 255.0], [12.0, 128.0, 240.0]]]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(torchvision.io, "write_video", incompatible_torchvision_writer)
    monkeypatch.setattr(video, "save_video_h264", fake_h264_writer)

    video.write_video_torchvision(tmp_path / "video.mp4", frames, 16)

    assert captured["frames"].dtype == np.uint8
    np.testing.assert_array_equal(captured["frames"], frames.to(torch.uint8).numpy())
    assert captured["fps"] == 16
    assert captured["crf"] == 18
