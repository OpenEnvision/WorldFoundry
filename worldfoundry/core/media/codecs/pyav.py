# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""SDR H.264/AAC artifacts with explicit Torch BT.709 color conversion.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). PyAV stays optional until encoding is requested.
The existing FFmpeg RGB and WAV writers retain their own defaults.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _write_stereo_aac(container: Any, stream: Any, waveform: torch.Tensor, sample_rate: int) -> None:
    """Truncate clipped PCM16, then resample to the AAC encoder's sample format."""
    import av
    import torch
    from av.audio.resampler import AudioResampler

    samples = waveform.T if waveform.shape[1] != 2 else waveform
    if samples.dtype != torch.int16:
        samples = (torch.clip(samples, -1.0, 1.0) * 32767.0).to(torch.int16)
    frame = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame.sample_rate = sample_rate
    codec = stream.codec_context
    resampler = AudioResampler(
        format=codec.format or "fltp",
        layout=codec.layout or "stereo",
        rate=codec.sample_rate or sample_rate,
    )
    next_pts = 0
    for resampled in resampler.resample(frame):
        if resampled.pts is None:
            resampled.pts = next_pts
        next_pts += resampled.samples
        resampled.sample_rate = sample_rate
        container.mux(stream.encode(resampled))
    for packet in stream.encode():
        container.mux(packet)


def save_audio_video_h264(
    video: torch.Tensor | Iterable[torch.Tensor],
    output_path: str | Path,
    *,
    fps: float,
    waveform: torch.Tensor | None = None,
    sample_rate: int | None = None,
    crf: int = 19,
    preset: str = "veryfast",
    thread_count: int = 0,
    deadline_monotonic: float | None = None,
) -> None:
    """Write float FHWC RGB chunks and optional stereo audio to an H.264/AAC MP4.

    Frames use BT.709 limited-range I420, not FFmpeg's implicit RGB conversion.
    Audio uses truncated PCM16 and PyAV's default AAC bitrate. Both full tracks
    are retained. Deadlines are checked before encoding chunks and audio, not
    inside an active codec call or CUDA kernel.
    """
    import av
    import torch

    from worldfoundry.core.execution.deadline import check_deadline
    from worldfoundry.core.media.processing.color import rgb_to_i420_bt709

    if waveform is not None:
        if waveform.numel() == 0 or waveform.ndim != 2 or 2 not in waveform.shape:
            raise ValueError("AAC waveform must be non-empty stereo shaped (2,N) or (N,2)")
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("AAC output requires a positive sample_rate")
    chunks = iter([video]) if isinstance(video, torch.Tensor) else iter(video)
    first = next(chunks, None)
    if first is None or first.shape[0] == 0:
        raise ValueError("video must contain at least one frame")
    height, width = first.shape[-3:-1]
    check_deadline(deadline_monotonic, stage="H.264 artifact creation")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(target), mode="w")
    success = False
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(crf), "preset": preset})
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = thread_count
        stream.codec_context.thread_type = "FRAME"
        stream.codec_context.colorspace = 1  # AVCOL_SPC_BT709
        stream.codec_context.color_range = 1  # AVCOL_RANGE_MPEG
        audio_stream = None
        if waveform is not None and sample_rate is not None:
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.codec_context.sample_rate = sample_rate
            audio_stream.codec_context.layout = "stereo"
            audio_stream.codec_context.time_base = Fraction(1, sample_rate)

        def encode_chunk(chunk: torch.Tensor) -> None:
            check_deadline(deadline_monotonic, stage="H.264 frame encoding")
            packed = rgb_to_i420_bt709(chunk.movedim(-1, -3)).cpu().numpy()
            for frame in packed:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="yuv420p")):
                    container.mux(packet)

        encode_chunk(first)
        for chunk in chunks:
            encode_chunk(chunk)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None and waveform is not None and sample_rate is not None:
            check_deadline(deadline_monotonic, stage="AAC audio encoding")
            _write_stereo_aac(container, audio_stream, waveform, sample_rate)
        success = True
    finally:
        container.close()
        if not success:
            target.unlink(missing_ok=True)
