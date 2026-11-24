# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SDR tensor conversion, encoder settings, complete tracks and deadline cleanup."""

from __future__ import annotations

import time
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from worldfoundry.core.media.codecs.pyav import _write_stereo_aac, save_audio_video_h264
from worldfoundry.core.media.processing.color import rgb_to_i420_bt709


def test_rgb_to_i420_bt709_uses_limited_range_and_packed_chroma() -> None:
    colors = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    frames = colors[:, :, None, None].expand(-1, -1, 2, 2)
    actual = rgb_to_i420_bt709(frames)
    expected = torch.tensor(
        [
            [[16, 16], [16, 16], [128, 128]],
            [[235, 235], [235, 235], [128, 128]],
            [[62, 62], [62, 62], [102, 240]],
            [[172, 172], [172, 172], [41, 26]],
            [[31, 31], [31, 31], [240, 117]],
        ],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="even height and width"):
        rgb_to_i420_bt709(torch.zeros(3, 3, 4))


def test_pcm16_conversion_truncates_without_changing_channel_order(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    audio = torch.tensor([[-1.1, 0.5, 0.00006], [1.1, -0.5, -0.00006]])
    path = tmp_path / "pcm.wav"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=48000)
        stream.codec_context.layout = "stereo"
        stream.codec_context.time_base = Fraction(1, 48000)
        _write_stereo_aac(container, stream, audio, 48000)
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 48000
        assert handle.getnchannels() == 2
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert pcm.tolist() == [-32767, 32767, 16383, -16383, 1, -1]


def test_h264_aac_writer_uses_explicit_defaults_and_keeps_full_audio(tmp_path: Path, monkeypatch) -> None:
    av = pytest.importorskip("av")
    original_open = av.open
    settings = []

    class RecordingContainer:
        def __init__(self, *args, **kwargs) -> None:
            self.container = original_open(*args, **kwargs)
            self.streams = []

        def add_stream(self, codec, **kwargs):
            settings.append((codec, kwargs))
            stream = self.container.add_stream(codec, **kwargs)
            self.streams.append(stream)
            return stream

        def mux(self, packet) -> None:
            self.container.mux(packet)

        def close(self) -> None:
            video, audio = self.streams
            settings.append(
                (
                    "video_context",
                    {
                        "pix_fmt": video.pix_fmt,
                        "colorspace": video.codec_context.colorspace,
                        "range": video.codec_context.color_range,
                        "threads": video.codec_context.thread_count,
                    },
                )
            )
            settings.append(("audio_context", {"layout": audio.codec_context.layout.name, "rate": audio.rate}))
            self.container.close()

    path = tmp_path / "result.mp4"
    with monkeypatch.context() as patch:
        patch.setattr(av, "open", RecordingContainer)
        save_audio_video_h264(
            [torch.zeros(2, 16, 16, 3), torch.ones(2, 16, 16, 3)],
            path,
            fps=24,
            waveform=torch.zeros(2, 48000),
            sample_rate=48000,
        )
    assert settings[:2] == [
        ("libx264", {"rate": 24, "options": {"crf": "19", "preset": "veryfast"}}),
        ("aac", {"rate": 48000}),
    ]
    assert settings[2][1]["pix_fmt"] == "yuv420p"
    assert settings[2][1]["colorspace"] == 1
    assert settings[2][1]["range"] == 1
    assert settings[3] == ("audio_context", {"layout": "stereo", "rate": 48000})
    with av.open(str(path)) as container:
        assert container.streams.video[0].frames == 4
        assert float(container.streams.audio[0].duration * container.streams.audio[0].time_base) >= 1
        assert len(list(container.decode(video=0))) == 4


def test_expired_writer_does_not_create_an_artifact(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "result.mp4"
    with pytest.raises(TimeoutError, match="artifact creation"):
        save_audio_video_h264(torch.zeros(1, 16, 16, 3), path, fps=24, deadline_monotonic=time.monotonic() - 1)
    assert not path.exists()


def test_codec_failure_removes_partial_artifact(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "result.mp4"

    def chunks():
        yield torch.zeros(1, 16, 16, 3)
        raise RuntimeError("frame source failed")

    with pytest.raises(RuntimeError, match="frame source failed"):
        save_audio_video_h264(chunks(), path, fps=24)
    assert not path.exists()
