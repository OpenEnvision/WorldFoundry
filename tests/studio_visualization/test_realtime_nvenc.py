from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import weakref
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from worldfoundry.studio.serving.realtime import nvenc


def _install_media_stubs(monkeypatch):
    class MediaStreamError(Exception):
        pass

    class Packet:
        def __init__(self, payload):
            self.payload = payload

    aiortc = ModuleType("aiortc")
    aiortc.MediaStreamTrack = type("MediaStreamTrack", (), {})
    streams = ModuleType("aiortc.mediastreams")
    streams.MediaStreamError = MediaStreamError
    av = ModuleType("av")
    av.VideoFrame = SimpleNamespace(from_ndarray=lambda *args, **kwargs: None)
    packet = ModuleType("av.packet")
    packet.Packet = Packet
    for name, module in (("aiortc", aiortc), ("aiortc.mediastreams", streams), ("av", av), ("av.packet", packet)):
        monkeypatch.setitem(sys.modules, name, module)
    return Packet, MediaStreamError


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


@pytest.mark.gpu
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
    _install_media_stubs(monkeypatch)
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NVENC", "1")
    monkeypatch.setattr(nvenc, "nvenc_h264_supported", lambda **kwargs: (False, "forced-unavailable"))

    frames = LatestFrameBuffer(maxsize=2)
    track = backend._build_video_track(frames=frames, fps=16)
    # Fell back to the software RealtimeVideoTrack, not the NVENC track.
    assert type(track).__name__ == "RealtimeVideoTrack"


def test_build_video_track_default_is_software(monkeypatch) -> None:
    _install_media_stubs(monkeypatch)
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_NVENC", raising=False)
    frames = LatestFrameBuffer(maxsize=2)
    track = backend._build_video_track(frames=frames, fps=16)
    assert type(track).__name__ == "RealtimeVideoTrack"


def test_build_video_track_selects_requested_gpu_and_presentation_contract(monkeypatch):
    from worldfoundry.studio.serving.realtime.media import ChunkPresentationBuffer
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    calls = []
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NVENC", "1")
    monkeypatch.setattr(nvenc, "nvenc_h264_supported", lambda **kwargs: (calls.append(kwargs) or True, ""))
    track = object()
    monkeypatch.setattr(nvenc, "build_nvenc_track", lambda **kwargs: (calls.append(kwargs) or track))
    frames = ChunkPresentationBuffer(fps=20)
    assert backend._build_video_track(frames=frames, fps=20, gpu_id=2, presentation_mode="real-frames") is track
    assert calls == [{"gpu_id": 2}, {"frames": frames, "fps": 20, "gpu_id": 2, "presentation_mode": "real-frames"}]
    assert backend._realtime_nvenc_device("cuda:3") == 3
    assert backend._realtime_nvenc_device("cuda") == 0


def test_nvenc_negotiates_h264_rtp_instead_of_default_video_codec(monkeypatch):
    from worldfoundry.studio.visualization.backends import world_realtime as backend

    _install_media_stubs(monkeypatch)
    codecs = [SimpleNamespace(mimeType=mime) for mime in ("video/VP8", "video/H264", "video/rtx")]
    sys.modules["aiortc"].RTCRtpSender = SimpleNamespace(getCapabilities=lambda kind: SimpleNamespace(codecs=codecs))
    track = SimpleNamespace(preserves_cuda_frames=True)
    selected = []
    transceiver = SimpleNamespace(sender=SimpleNamespace(track=track), setCodecPreferences=selected.append)
    peer = SimpleNamespace(getTransceivers=lambda: [transceiver])
    backend._configure_nvenc_h264(peer, track)
    assert selected == [[codecs[1]]]
    codecs.clear()
    with pytest.raises(RuntimeError, match="H.264 RTP support"):
        backend._configure_nvenc_h264(peer, track)


@pytest.mark.parametrize("caps,match", [({}, "no NVENC"), ({"width_max": 64}, "outside"), (None, "query")])
def test_driver_capabilities_reject_missing_engine_and_unsupported_geometry(caps, match):
    def query(**kwargs):
        assert kwargs == {"gpuid": 2, "codec": "h264"}
        if caps is None:
            raise RuntimeError("engine unavailable")
        return caps

    ok, reason = nvenc._encoder_supported(SimpleNamespace(GetEncoderCaps=query), 2, width=128, height=64)
    assert not ok
    assert match in reason


def test_encoder_sdk_contract_waits_for_pixels_routes_gpu_and_preserves_input_clock(monkeypatch):
    Packet, _ = _install_media_stubs(monkeypatch)
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("ready", stream))

        def synchronize(self):
            calls.append("synchronize")

    sdk = ModuleType("PyNvVideoCodec")
    sdk.FORCEIDR = 7
    sdk.GetEncoderCaps = lambda **kwargs: {"width_min": 1, "height_min": 1}
    bitstreams = iter([b"", b"packet"])

    def encode(frame, *flags):
        assert calls[-1] == "synchronize"
        calls.append(("encode", flags))
        return next(bitstreams)

    native = SimpleNamespace(Encode=encode, EndEncode=lambda: calls.append("close"))
    sdk.CreateEncoder = lambda **kwargs: (calls.append(kwargs) or native)
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(Event=Event, current_stream=lambda device: f"stream:{device}")
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", sdk)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    packed = SimpleNamespace(shape=(4, 8, 4), device="cuda:2")
    monkeypatch.setattr(nvenc, "_rgb_frame_to_abgr_cuda", lambda frame, torch_module, **kwargs: packed)
    encoder = nvenc.NVENCFrameEncoder(width=8, height=4, fps=16, bitrate=1_000_000, gpu_id=2)
    config = calls[0]
    assert config["gpu_id"] == 2
    assert config["fmt"] == "ABGR"
    assert (config["bf"], config["lookahead"], config["repeatspspps"]) == (0, 0, 1)
    packets = []
    assert encoder.encode_frame(object(), force_keyframe=True, on_packet=packets.append) == 0
    assert encoder.encode_frame(object(), on_packet=packets.append) == 1
    assert isinstance(packets[0], Packet)
    assert packets[0].pts == packets[0].dts == 5625
    assert packets[0].time_base == Fraction(1, 90_000)
    assert ("encode", (7,)) in calls
    encoder.close()
    encoder.close()
    assert calls.count("close") == 1
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode_frame(object())


def test_native_track_rebuilds_geometry_and_bounds_packet_backpressure(monkeypatch):
    from worldfoundry.studio.serving.realtime.media import FrameQueuePolicy, LatestFrameBuffer

    Packet, _ = _install_media_stubs(monkeypatch)
    calls = []

    class Encoder:
        def __init__(self, **kwargs):
            self.width = kwargs["width"]
            calls.append(("create", self.width, threading.get_ident()))

        def encode_frame(self, frame, *, force_keyframe, on_packet, pts):
            calls.append(("encode", self.width, force_keyframe, pts, threading.get_ident()))
            packet = Packet(b"packet")
            packet.pts = pts
            on_packet(packet)

        def close(self):
            calls.append(("close", self.width, threading.get_ident()))

    monkeypatch.setattr(nvenc, "NVENCFrameEncoder", Encoder)

    async def exercise():
        frames = LatestFrameBuffer(maxsize=5, policy=FrameQueuePolicy.ORDERED_QUALITY)
        track = nvenc.build_nvenc_track(frames=frames, fps=16, maxsize=1, presentation_mode="real-frames")
        assert frames.preserve_cuda_frames
        await frames.put_chunk([np.zeros((2, width, 3), np.uint8) for width in (2, 3, 3, 3)])
        while len([call for call in calls if call[0] == "encode"]) < 2:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        # One emitted packet and one pending put bound native work: frames 3/4
        # cannot be encoded while the consumer is blocked.
        assert len([call for call in calls if call[0] == "encode"]) == 2
        packets = [await asyncio.wait_for(track.recv(), 1) for _ in range(4)]
        assert [packet.pts for packet in packets] == sorted({packet.pts for packet in packets})
        await track.close()
        frames.close()
        assert track._last_frame is None
        assert track._encoder is None

    asyncio.run(exercise())
    assert [call[:2] for call in calls if call[0] in ("create", "close")] == [
        ("create", 2),
        ("close", 2),
        ("create", 3),
        ("close", 3),
    ]
    encoded = [call for call in calls if call[0] == "encode"]
    assert [call[2] for call in encoded] == [True, True, False, False]
    assert len({call[-1] for call in calls}) == 1


def test_native_track_cancel_drains_encode_before_closing_encoder(monkeypatch):
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer

    _install_media_stubs(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Encoder:
        def __init__(self, **kwargs):
            pass

        def encode_frame(self, frame, **kwargs):
            entered.set()
            assert release.wait(2)
            calls.append("encode-complete")

        def close(self):
            assert release.is_set()
            calls.append("close")

    monkeypatch.setattr(nvenc, "NVENCFrameEncoder", Encoder)

    async def exercise():
        frames = LatestFrameBuffer(maxsize=1)
        track = nvenc.build_nvenc_track(frames=frames, fps=16)
        await frames.put_chunk([np.zeros((2, 2, 3), np.uint8)])
        while not entered.is_set():
            await asyncio.sleep(0)
        closing = asyncio.create_task(track.close())
        await asyncio.sleep(0.01)
        assert not closing.done()
        assert not calls
        release.set()
        await asyncio.wait_for(closing, 1)
        frames.close()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert calls == ["encode-complete", "close"]


def test_native_track_close_survives_repeated_caller_cancellation_and_releases_resources_once(monkeypatch):
    _, MediaStreamError = _install_media_stubs(monkeypatch)
    encode_entered, encode_release = threading.Event(), threading.Event()
    end_entered, end_release = threading.Event(), threading.Event()
    calls, storage_refs, session_refs, frame_refs, workers = [], [], [], [], []

    class OwnedStorage:
        shape = (2, 2, 4)
        device = "cuda:0"

        def __del__(self):
            calls.append(("storage-release", threading.get_ident()))

    class SourceFrame:
        shape = (2, 2, 3)

        def __del__(self):
            calls.append(("frame-release", threading.get_ident()))

    def pack(frame, torch_module, **kwargs):
        storage = OwnedStorage()
        storage_refs.append(weakref.ref(storage))
        return storage

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    class NativeSession:
        def Encode(self, storage, *flags):
            calls.append(("encode-start", threading.get_ident()))
            encode_entered.set()
            assert encode_release.wait(3), "test did not release native Encode"
            assert storage_refs[0]() is storage
            calls.append(("encode-complete", threading.get_ident()))
            return b""

        def EndEncode(self):
            calls.append(("end-start", threading.get_ident()))
            end_entered.set()
            assert end_release.wait(3), "test did not release native EndEncode"
            calls.append(("end-complete", threading.get_ident()))

        def __del__(self):
            calls.append(("session-release", threading.get_ident()))

    def create_encoder(**kwargs):
        calls.append(("create", threading.get_ident()))
        session = NativeSession()
        session_refs.append(weakref.ref(session))
        return session

    sdk = ModuleType("PyNvVideoCodec")
    sdk.FORCEIDR = 7
    sdk.GetEncoderCaps = lambda **kwargs: {"width_min": 1, "height_min": 1}
    sdk.CreateEncoder = create_encoder
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(Event=Event, current_stream=lambda device: device)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", sdk)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(nvenc, "_rgb_frame_to_abgr_cuda", pack)
    executor_type = nvenc.ThreadPoolExecutor

    class RecordingExecutor(executor_type):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.shutdown_calls = 0
            workers.append(self)

        def shutdown(self, **kwargs):
            self.shutdown_calls += 1
            return super().shutdown(**kwargs)

    monkeypatch.setattr(nvenc, "ThreadPoolExecutor", RecordingExecutor)

    class Frames:
        def __init__(self):
            self.queue = asyncio.Queue()

        async def get(self):
            return await self.queue.get()

        def get_nowait(self):
            return self.queue.get_nowait()

    async def wait_until(predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(poll(), 1)

    async def exercise():
        frames = Frames()
        frame = SourceFrame()
        frame_refs.append(weakref.ref(frame))
        frames.queue.put_nowait(frame)
        del frame
        track = nvenc.build_nvenc_track(frames=frames, fps=16)
        try:
            await wait_until(encode_entered.is_set)
            for _ in range(3):
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(track.close(), 0.01)
                assert track._close_task is not None
                assert not track._close_task.done()
                assert storage_refs[0]() is not None
                assert frame_refs[0]() is not None
                assert not end_entered.is_set()
                assert workers[0].shutdown_calls == 0
            cleanup = track._close_task
            with pytest.raises(MediaStreamError):
                await track.recv()
            encode_release.set()
            await wait_until(end_entered.is_set)
            assert storage_refs[0]() is None
            assert frame_refs[0]() is None
            assert session_refs[0]() is not None
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(track.close(), 0.01)
            assert track._close_task is cleanup
            assert not cleanup.done()
            assert workers[0].shutdown_calls == 0
            end_release.set()
            await asyncio.wait_for(track.close(), 1)
            await track.close()
            await track.close()
            assert track._close_task is cleanup
            assert cleanup.done() and not cleanup.cancelled()
            assert track._encoder is track._last_frame is track._encode_future is None
            assert session_refs[0]() is None
            assert workers[0].shutdown_calls == 1
            assert track._packets.qsize() == 1
            assert track._packets.get_nowait() is None
            await wait_until(lambda: all(not thread.is_alive() for thread in workers[0]._threads))
        finally:
            encode_release.set()
            end_release.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(track.close(), 1)

    try:
        asyncio.run(exercise())
    finally:
        encode_release.set()
        end_release.set()
    names = [name for name, _ in calls]
    for name in (
        "create",
        "encode-start",
        "encode-complete",
        "end-start",
        "end-complete",
        "storage-release",
        "frame-release",
        "session-release",
    ):
        assert names.count(name) == 1
    assert names.index("encode-complete") < names.index("storage-release") < names.index("end-start")
    assert names.index("end-complete") < names.index("session-release")
    assert len({thread for name, thread in calls if name in {"create", "encode-start", "end-start"}}) == 1


def test_native_track_surfaces_encoder_failure_to_consumer(monkeypatch):
    from worldfoundry.studio.serving.realtime.media import LatestFrameBuffer

    _, MediaStreamError = _install_media_stubs(monkeypatch)

    class Encoder:
        def __init__(self, **kwargs):
            raise RuntimeError("driver encoder failure")

    monkeypatch.setattr(nvenc, "NVENCFrameEncoder", Encoder)

    async def exercise():
        frames = LatestFrameBuffer(maxsize=1)
        track = nvenc.build_nvenc_track(frames=frames, fps=16)
        await frames.put_chunk([np.zeros((2, 2, 3), np.uint8)])
        with pytest.raises(MediaStreamError) as caught:
            await asyncio.wait_for(track.recv(), 1)
        assert str(caught.value.__cause__) == "driver encoder failure"
        await track.close()
        frames.close()

    asyncio.run(exercise())
