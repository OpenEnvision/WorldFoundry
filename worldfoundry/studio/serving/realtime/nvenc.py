# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NVENC hardware H.264 encoding for the Studio realtime WebRTC track.

Isolated from the default software path so importing the realtime serving
surface never drags ``PyNvVideoCodec`` in with it: that library's import has
global side effects (CUDA driver init, shared-library loading) that the
software path should not pay. :func:`nvenc_h264_supported` probes availability
with :func:`importlib.util.find_spec` and this module is only imported when the
hardware track is actually requested.

Device byte frames retain their producer event and owned CUDA storage through
the presentation buffer. CPU frames are uploaded when necessary. Encoding
waits for packed RGBA pixels before handing them to the native codec, whose
internal stream is independent of PyTorch's producer stream.
"""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from av.packet import Packet

# H.264 Annex-B NAL type identifier for an IDR (keyframe) slice.
_H264_NAL_TYPE_IDR = 5

# RTP video clock, per RFC 6184. aiortc's H264Encoder.pack() rescales from the
# packet's declared time_base into this base, so stamping packets on 1/90000 is
# the lowest-conversion choice.
_RTP_VIDEO_CLOCK = 90_000


def nvenc_h264_supported(*, gpu_id: int = 0) -> tuple[bool, str]:
    """Return whether the NVENC hardware path can be imported and used.

    Importing this module remains cheap. The opt-in probe queries the native
    driver without allocating an encoder session: CUDA availability alone is
    insufficient because many compute GPUs have no NVENC engine.
    """
    if find_spec("PyNvVideoCodec") is None:
        return False, "PyNvVideoCodec is not installed"
    try:
        import torch
    except ImportError:
        return False, "torch is not available"
    if not torch.cuda.is_available():
        return False, "no CUDA device is available"
    if type(gpu_id) is not int or not 0 <= gpu_id < torch.cuda.device_count():
        return False, f"NVENC GPU {gpu_id} is not an available CUDA device"
    try:
        nvc = import_module("PyNvVideoCodec")
    except (ImportError, OSError) as exc:
        return False, f"PyNvVideoCodec could not load: {exc}"
    return _encoder_supported(nvc, gpu_id)


def _encoder_supported(nvc: Any, gpu_id: int, *, width: int = 0, height: int = 0) -> tuple[bool, str]:
    try:
        caps = nvc.GetEncoderCaps(gpuid=gpu_id, codec="h264")
    except Exception as exc:
        return False, f"NVENC H.264 capability query for GPU {gpu_id} failed: {type(exc).__name__}: {exc}"
    if not caps:
        return False, f"GPU {gpu_id} reports no NVENC H.264 capabilities"
    for dimension, value in (("width", width), ("height", height)):
        minimum = int(caps.get(f"{dimension}_min", 0) or 0)
        maximum = int(caps.get(f"{dimension}_max", 0) or 0)
        if value and ((minimum and value < minimum) or (maximum and value > maximum)):
            return False, f"NVENC {dimension} {value} is outside GPU {gpu_id} limits {minimum}..{maximum}"
    return True, ""


def _payload_contains_nal_type(payload: bytes, nal_type: int) -> bool:
    """Scan an Annex-B H.264 payload for the presence of a specific NAL type."""
    i = 0
    while True:
        idx = payload.find(b"\x00\x00\x01", i)
        if idx < 0:
            return False
        nal_start = idx + 3
        if nal_start >= len(payload):
            return False
        if (payload[nal_start] & 0x1F) == nal_type:
            return True
        i = nal_start + 1


def _rgb_frame_to_abgr_cuda(frame_rgb_uint8: Any, torch_module: Any, *, gpu_id: int = 0) -> Any:
    """Pack one CPU or resident CUDA RGB frame into owned NVENC ``ABGR`` storage.

    NVENC ``NV_ENC_BUFFER_FORMAT_ABGR`` is a *word-ordered* token, not
    memory-ordered: a pixel is the 32-bit word ``0xAABBGGRR``, which in
    little-endian memory is the byte sequence ``[R, G, B, A]``. So the
    channel-last tensor handed to the encoder must have channel 0=R, 1=G, 2=B,
    3=A. ABGR (rather than NV12) lets NVENC's driver-side RGB->YUV conversion
    handle the colour transform instead of a bespoke NV12 kernel.
    """
    materialize = getattr(frame_rgb_uint8, "to_cuda_tensor", None)
    source_event = None
    if callable(materialize):
        event = getattr(frame_rgb_uint8, "to_cuda_event", None)
        source_event = event() if callable(event) else None
        rgb = materialize()
    elif torch_module.is_tensor(frame_rgb_uint8):
        rgb = frame_rgb_uint8
    else:
        rgb = torch_module.from_numpy(np.ascontiguousarray(frame_rgb_uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB frame with 3 channels; got shape {tuple(rgb.shape)}")
    device = torch_module.device("cuda", gpu_id)
    if rgb.is_cuda and rgb.device != device:
        raise ValueError(f"CUDA frame device {rgb.device} does not match NVENC device {device}")
    with torch_module.cuda.device(device):
        if source_event is not None:
            torch_module.cuda.current_stream(device).wait_event(source_event)
        rgb = rgb.detach().to(device=device)
        if rgb.dtype != torch_module.uint8:
            rgb = rgb.clamp(0, 255).to(torch_module.uint8)
        h, w, _ = rgb.shape
        rgba = torch_module.empty((h, w, 4), dtype=torch_module.uint8, device=device)
        rgba[..., :3].copy_(rgb)
        rgba[..., 3].fill_(255)
    return rgba


class NVENCFrameEncoder:
    """NVENC H.264 encoder backed by ``PyNvVideoCodec``.

    Accepts CPU RGB arrays or owned lazy CUDA frames, and emits Annex-B H.264 packets stamped on the RTP
    90 kHz clock so aiortc's ``H264Encoder.pack()`` can rescale losslessly.
    """

    backend = "pynvvideocodec"

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        gpu_id: int = 0,
        gop: int = 30,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate}")
        if gop <= 0:
            raise ValueError(f"gop must be > 0, got {gop}")
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer")

        import PyNvVideoCodec as nvc
        import torch

        supported, reason = _encoder_supported(nvc, gpu_id, width=width, height=height)
        if not supported:
            raise RuntimeError(reason)
        self._torch = torch
        self._gpu_id = gpu_id
        self._shape = (height, width, 3)
        self.fps = fps
        self._time_base = Fraction(1, _RTP_VIDEO_CLOCK)
        self._pts_counter = 0
        self._closed = False
        self._force_idr_flag = int(nvc.FORCEIDR)

        # repeatspspps=1 prepends SPS+PPS to every IDR: aiortc's
        # H264Encoder.pack() does not synthesize parameter sets, so the RTP
        # stream must carry them in-band or the receiver cannot lock on. bf=0
        # and lookahead=0 keep output strictly 1:1 with input frames, which is
        # what interactive streaming needs.
        self._encoder = nvc.CreateEncoder(
            width=width,
            height=height,
            fmt="ABGR",
            usecpuinputbuffer=False,
            gpu_id=gpu_id,
            codec="h264",
            preset="P4",
            tuning_info="ultra_low_latency",
            rc="cbr",
            fps=fps,
            bitrate=bitrate,
            bf=0,
            lookahead=0,
            repeatspspps=1,
            idrperiod=gop,
        )

    def encode_frame(
        self,
        frame_rgb_uint8: Any,
        *,
        force_keyframe: bool = False,
        on_packet: Callable[[Packet], None] | None = None,
        pts: int | None = None,
    ) -> int:
        """Encode one RGB frame; return the number of packets emitted."""
        from av.packet import Packet

        if self._closed:
            raise RuntimeError("NVENC encoder is closed")
        cuda_frame = _rgb_frame_to_abgr_cuda(frame_rgb_uint8, self._torch, gpu_id=self._gpu_id)
        if tuple(cuda_frame.shape[:2]) != self._shape[:2]:
            raise ValueError("NVENC frame resolution changed after encoder construction")
        # The SDK owns its stream. Complete only this frame's packing before
        # native consumption, retaining device storage until Encode returns.
        ready = self._torch.cuda.Event()
        ready.record(self._torch.cuda.current_stream(cuda_frame.device))
        ready.synchronize()
        if force_keyframe:
            bitstream = self._encoder.Encode(cuda_frame, self._force_idr_flag)
        else:
            bitstream = self._encoder.Encode(cuda_frame)
        frame_pts = pts if pts is not None else (self._pts_counter * _RTP_VIDEO_CLOCK) // self.fps
        self._pts_counter += 1
        if not bitstream:
            return 0
        payload = bytes(bitstream)
        packet = Packet(payload)
        packet.pts = frame_pts
        packet.dts = frame_pts
        packet.time_base = self._time_base
        if on_packet is not None:
            on_packet(packet)
        return 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._encoder.EndEncode()


def bitrate_for_resolution(width: int, height: int, fps: int) -> int:
    """Return a CBR H.264 target bitrate (bps) from resolution and fps.

    Uses ~0.10 bits per pixel per frame, clamped to a sane interactive range.
    This is a heuristic default; override via the caller when a tuned value is
    known for the target stream.
    """
    bits_per_pixel_per_frame = 0.10
    estimated = int(width * height * fps * bits_per_pixel_per_frame)
    return max(1_000_000, min(estimated, 20_000_000))


def build_nvenc_track(
    *,
    frames: Any,
    fps: int,
    maxsize: int = 8,
    gpu_id: int = 0,
    gop: int = 30,
    bitrate: int | None = None,
    presentation_mode: str = "hold-last",
) -> Any:
    """Build an aiortc track that streams NVENC-encoded packets.

    ``frames`` is a realtime presentation buffer. A background task pulls
    CPU or resident CUDA RGB frames, encodes them, and feeds packets to
    the track. ``recv()`` returns ``av.Packet`` (not ``VideoFrame``), which
    aiortc routes through ``H264Encoder.pack()`` for RTP fragmentation only —
    the software encoder is bypassed.

    One worker owns encoder construction, encoding and shutdown. Resolution
    changes start a new session with an IDR while preserving the RTP clock.
    ``bitrate`` defaults to :func:`bitrate_for_resolution`.
    """
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError

    if fps <= 0 or maxsize < 1:
        raise ValueError("fps and maxsize must be positive")
    if presentation_mode not in ("hold-last", "real-frames"):
        raise ValueError(f"unsupported NVENC presentation mode: {presentation_mode}")

    class NVENCVideoTrack(MediaStreamTrack):
        kind = "video"
        preserves_cuda_frames = True

        def __init__(self) -> None:
            super().__init__()
            self._packets: asyncio.Queue[Packet | None] = asyncio.Queue(maxsize=maxsize)
            self._interval = 1.0 / fps
            self._closed = False
            self._error: Exception | None = None
            self._pump: asyncio.Task[None] | None = None
            self._close_task: asyncio.Task[None] | None = None
            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world-realtime-nvenc")
            self._encode_future: asyncio.Future[Any] | None = None
            self._last_frame: Any | None = None
            self._first_frame = True
            self._encoder: NVENCFrameEncoder | None = None
            self._encoder_shape: tuple[int, int] | None = None
            self._pts_counter = 0
            self._real_frames = presentation_mode == "real-frames"

        def start(self) -> None:
            if self._pump is None:
                self._pump = asyncio.ensure_future(self._pump_frames())

        def _encode(self, frame: Any, pts: int) -> list[Packet]:
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if self._encoder is None or self._encoder_shape != (height, width):
                if self._encoder is not None:
                    self._encoder.close()
                self._encoder = NVENCFrameEncoder(
                    width=width,
                    height=height,
                    fps=fps,
                    bitrate=bitrate or bitrate_for_resolution(width, height, fps),
                    gpu_id=gpu_id,
                    gop=gop,
                )
                self._encoder_shape = (height, width)
                self._first_frame = True
            packets: list[Packet] = []
            self._encoder.encode_frame(frame, force_keyframe=self._first_frame, on_packet=packets.append, pts=pts)
            self._first_frame = False
            return packets

        async def _pump_frames(self) -> None:
            loop = asyncio.get_running_loop()
            next_pull = loop.time()
            first_real_frame_at: float | None = None
            try:
                while not self._closed:
                    if self._last_frame is None or self._real_frames:
                        try:
                            self._last_frame = await frames.get()
                        except EOFError:
                            break
                    else:
                        next_pull += self._interval
                        now = loop.time()
                        if next_pull > now:
                            await asyncio.sleep(next_pull - now)
                        elif now - next_pull > self._interval:
                            next_pull = now
                        try:
                            self._last_frame = frames.get_nowait()
                        except asyncio.QueueEmpty:
                            # Repeat the held frame to keep a steady RTP clock
                            # and flush the encoder's final source frame.
                            pass
                        except EOFError:
                            break
                    if self._real_frames:
                        now = loop.time()
                        if first_real_frame_at is None:
                            first_real_frame_at = now
                        pts = max(self._pts_counter, round((now - first_real_frame_at) * _RTP_VIDEO_CLOCK))
                    else:
                        pts = self._pts_counter
                    self._pts_counter = pts + max(_RTP_VIDEO_CLOCK // fps, 1)
                    self._encode_future = loop.run_in_executor(self._worker, self._encode, self._last_frame, pts)
                    packets = await asyncio.shield(self._encode_future)
                    self._encode_future = None
                    for packet in packets:
                        await self._packets.put(packet)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error = exc
            finally:
                # A canceled asyncio wait cannot stop native Encode. Drain it
                # before releasing frame storage or closing the encoder.
                if self._encode_future is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self._encode_future)
                    self._encode_future = None
                self._last_frame = None
                if not self._closed:
                    await self._packets.put(None)

        async def recv(self) -> Packet:
            if self._closed:
                raise MediaStreamError
            self.start()
            packet = await self._packets.get()
            if packet is None:
                raise MediaStreamError from self._error
            return packet

        async def close(self) -> None:
            # A caller's deadline may cancel its wait while native Encode or
            # EndEncode is still running. One retained task owns cleanup, and
            # every close call joins it without forwarding cancellation.
            if self._close_task is None:
                self._closed = True
                self._close_task = asyncio.create_task(self._close_owned(), name="world-realtime-nvenc-cleanup")
            await asyncio.shield(self._close_task)

        async def _close_owned(self) -> None:
            try:
                if self._pump is not None:
                    self._pump.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await self._pump
                if self._encode_future is not None:
                    # The pump itself may have been canceled again during its
                    # drain. Retain the native operation until it completes.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self._encode_future)
                    self._encode_future = None
                if self._encoder is not None:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self._worker, self._encoder.close)
            finally:
                self._encoder = None
                self._last_frame = None
                self._worker.shutdown(wait=False)
                while not self._packets.empty():
                    self._packets.get_nowait()
                self._packets.put_nowait(None)

    track = NVENCVideoTrack()
    frames.preserve_cuda_frames = True
    track.start()
    return track


__all__ = [
    "NVENCFrameEncoder",
    "bitrate_for_resolution",
    "build_nvenc_track",
    "nvenc_h264_supported",
]
