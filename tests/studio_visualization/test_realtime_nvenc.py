from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.studio.serving.realtime import nvenc


def test_supported_probe_reports_missing_dependency_without_import() -> None:
    ok, reason = nvenc.nvenc_h264_supported()
    # PyNvVideoCodec is not installed in CI; the probe must say so cleanly
    # rather than raising, and must not import the (side-effectful) library.
    if not ok:
        assert reason
    assert isinstance(ok, bool)


def test_bitrate_for_resolution_is_clamped_and_scales() -> None:
    tiny = nvenc.bitrate_for_resolution(64, 64, 16)
    huge = nvenc.bitrate_for_resolution(3840, 2160, 60)
    mid = nvenc.bitrate_for_resolution(1280, 720, 30)
    assert tiny == 1_000_000  # clamped to floor
    assert huge == 20_000_000  # clamped to ceiling
    assert 1_000_000 < mid < 20_000_000
    # Monotonic in pixel throughput below the ceiling.
    assert nvenc.bitrate_for_resolution(640, 360, 30) < mid


def test_abgr_conversion_rejects_non_rgb_shapes() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="HWC RGB frame"):
        nvenc._rgb_frame_to_abgr_cuda(np.zeros((8, 8), dtype=np.uint8), torch)
    with pytest.raises(ValueError, match="HWC RGB frame"):
        nvenc._rgb_frame_to_abgr_cuda(np.zeros((8, 8, 4), dtype=np.uint8), torch)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="ABGR upload requires a CUDA device",
)
def test_abgr_conversion_channel_order_and_alpha() -> None:
    import torch

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[..., 0] = 10  # R
    frame[..., 1] = 20  # G
    frame[..., 2] = 30  # B
    rgba = nvenc._rgb_frame_to_abgr_cuda(frame, torch).cpu().numpy()
    assert rgba.shape == (2, 3, 4)
    # Channel-last [R, G, B, A] is what NVENC's ABGR word token expects.
    assert (rgba[..., 0] == 10).all()
    assert (rgba[..., 1] == 20).all()
    assert (rgba[..., 2] == 30).all()
    assert (rgba[..., 3] == 255).all()


def test_payload_nal_scan_finds_idr() -> None:
    idr = b"\x00\x00\x01" + bytes([nvenc._H264_NAL_TYPE_IDR])
    non_idr = b"\x00\x00\x01" + bytes([1])
    assert nvenc._payload_contains_nal_type(idr, nvenc._H264_NAL_TYPE_IDR) is True
    assert nvenc._payload_contains_nal_type(non_idr, nvenc._H264_NAL_TYPE_IDR) is False
    assert nvenc._payload_contains_nal_type(b"", nvenc._H264_NAL_TYPE_IDR) is False


def test_build_video_track_falls_back_when_nvenc_unsupported(monkeypatch) -> None:
    pytest.importorskip("aiortc")  # software fallback path constructs an aiortc track
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NVENC", "1")
    monkeypatch.setattr(
        nvenc, "nvenc_h264_supported", lambda: (False, "forced-unavailable")
    )

    frames = LatestFrameBuffer(maxsize=2)
    track = backend._build_video_track(frames=frames, fps=16)
    # Fell back to the software RealtimeVideoTrack, not the NVENC track.
    assert type(track).__name__ == "RealtimeVideoTrack"


def test_build_video_track_default_is_software(monkeypatch) -> None:
    pytest.importorskip("aiortc")  # software path constructs an aiortc track
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_NVENC", raising=False)
    frames = LatestFrameBuffer(maxsize=2)
    track = backend._build_video_track(frames=frames, fps=16)
    assert type(track).__name__ == "RealtimeVideoTrack"
