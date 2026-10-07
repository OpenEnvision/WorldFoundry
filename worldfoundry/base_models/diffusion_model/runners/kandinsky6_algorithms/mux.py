from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from torch import Tensor


def _require_av():
    """Load PyAV only when video or audio output is requested."""
    try:
        import av
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SR video/audio output requires PyAV and FFmpeg. Install with `pip install av`."
        ) from exc
    return av


def _video_np(frames: Tensor) -> np.ndarray:
    """(3, T, H, W) uint8 → (T, H, W, 3) uint8."""
    return frames.detach().permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8, copy=False)


def _audio_np(audio: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """int16 / float waveform → float32 mono in [-1, 1]."""
    audio_np = np.asarray(audio[0] if isinstance(audio, list) else audio)
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = audio_np.astype(np.float32) / np.iinfo(audio_np.dtype).max
    return np.clip(audio_np, -1.0, 1.0).astype(np.float32, copy=False)


def extract_audio_from_video(
    source_video: str | Path,
    audio_sample_rate: int = 44100,
) -> np.ndarray | None:
    """Decode a source video's audio as mono float32 samples.

    SR changes only the video stream.  Decoding the source audio here lets the
    output mux keep that audio without requiring the SR caller to load the
    entire source container itself.  Resampling also gives the AAC writer a
    stable sample rate for videos whose source audio uses another rate.
    """
    if audio_sample_rate <= 0:
        raise ValueError("audio_sample_rate must be positive")

    av = _require_av()
    chunks: list[np.ndarray] = []
    with av.open(str(source_video), mode="r") as container:
        audio_streams = list(container.streams.audio)
        if not audio_streams:
            return None

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=audio_sample_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32, copy=False)


def mux_video_audio(  # noqa: PLR0913
    frames: Tensor,
    audio: list[np.ndarray] | np.ndarray | None,
    output_path: str | Path,
    fps: int = 24,
    audio_sample_rate: int = 44100,
    video_crf: int = 18,
    source_video: str | Path | None = None,
    lossless: bool = False,
) -> Path:
    """Mux video (+ optional audio) into ``output_path`` via PyAV — no temp files.

    frames: (3, T, H, W) uint8 — single video, channels-first.
    audio:  list of (samples,) int16, a single ndarray, or None for video-only.
    source_video: optional source container from which audio is copied when
        ``audio`` is not supplied.  This is useful when SR receives frames
        extracted from a generated T2VA video.
    lossless: encode video as FFV1 instead of libx264. Audio remains AAC.
    """
    av = _require_av()
    if audio is not None and source_video is not None:
        raise ValueError("pass either audio or source_video, not both")
    if source_video is not None:
        audio = extract_audio_from_video(source_video, audio_sample_rate)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_np = _video_np(frames)

    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("ffv1" if lossless else "libx264", rate=fps)
        video_stream.width = video_np.shape[2]
        video_stream.height = video_np.shape[1]
        if lossless:
            video_stream.pix_fmt = "bgra"
            video_stream.options = {"level": "3"}
        else:
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": str(video_crf)}

        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "mono"
            audio_stream.bit_rate = 192_000

        for i, frame_np in enumerate(video_np):
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in video_stream.encode(frame):
                container.mux(packet)

        if audio_stream is not None:
            audio_np = _audio_np(audio)
            audio_frame = av.AudioFrame.from_ndarray(audio_np[np.newaxis, :], format="fltp", layout="mono")
            audio_frame.sample_rate = audio_sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio_sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)

    return output_path
