"""Dependency-light audio artifact helpers shared by model pipelines.

Float32 channel convert, write, and mux onto video. Avoids pulling
torchaudio into every inference job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Waveform normalize / WAV write / ffmpeg mux — no torchaudio import
# ──────────────────────────────────────────────────────────────────────────


def _ffmpeg_executables() -> tuple[str, ...]:
    """Return system and bundled ffmpeg candidates without requiring imageio-ffmpeg."""

    candidates: list[str] = []
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if configured:
        candidates.append(configured)
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError):
        pass
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def audio_to_float32_channels(waveform: Any, *, channel_first: bool = True) -> np.ndarray:
    """Normalize a waveform to finite float32 ``[channels, samples]``."""

    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "float"):
        waveform = waveform.float()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()

    audio = np.asarray(waveform)
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim != 2:
        raise ValueError(f"Expected audio shaped [N], [C, N], or [N, C], got {audio.shape}.")
    elif not channel_first:
        audio = audio.T

    if audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError(f"Audio waveform cannot be empty, got {audio.shape}.")
    if audio.shape[0] > 64:
        raise ValueError(
            f"Audio appears to be sample-first ({audio.shape}); pass channel_first=False when using [N, C]."
        )

    audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("Audio waveform contains NaN or infinite values.")
    return np.clip(audio, -1.0, 1.0)


def write_audio(
    waveform: Any,
    output_path: str | Path,
    *,
    sample_rate: int,
    channel_first: bool = True,
) -> str:
    """Write a generated waveform as signed 16-bit PCM WAV and return its path."""

    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}.")
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".wav":
        raise ValueError(f"write_audio currently emits WAV artifacts; expected a .wav path, got {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = audio_to_float32_channels(waveform, channel_first=channel_first)
    pcm = np.rint(audio.T * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(audio.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes(order="C"))
    return str(path)


def mux_audio_video(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    output_path: str | Path | None = None,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    shortest: bool = True,
) -> str:
    """Mux with checked, atomic output; ``shortest=False`` retains both full tracks."""

    video = Path(video_path).expanduser()
    audio = Path(audio_path).expanduser()
    target = Path(output_path).expanduser() if output_path is not None else video
    if not video.is_file():
        raise FileNotFoundError(f"Video artifact does not exist: {video}")
    if not audio.is_file():
        raise FileNotFoundError(f"Audio artifact does not exist: {audio}")
    target.parent.mkdir(parents=True, exist_ok=True)

    suffix = target.suffix or video.suffix or ".mp4"
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.mux-",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        failures: list[str] = []
        for executable in _ffmpeg_executables():
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                audio_codec,
                "-b:a",
                audio_bitrate,
                *(["-shortest"] if shortest else []),
                str(temporary),
            ]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
            except OSError as exc:
                failures.append(f"{executable}: {exc}")
                continue
            if completed.returncode == 0:
                break
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
            failures.append(f"{executable}: {detail}")
        else:
            detail = "; ".join(failures) if failures else "no ffmpeg executable is available"
            raise RuntimeError(f"ffmpeg audio/video mux failed: {detail}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


__all__ = ["audio_to_float32_channels", "mux_audio_video", "write_audio"]
