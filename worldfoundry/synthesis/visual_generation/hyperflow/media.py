# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Adapted from Diffusers v0.40.0 MiniMax-H3 reference preprocessing and HyperFlow 1dd2f342.
"""Local HyperFlow references, released canvas rules, and H.264/AAC output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import minimax_h3_align_frame_count
from worldfoundry.core.media.codecs.pyav import _write_stereo_aac


@dataclass
class ReferenceMedia:
    """A normalized local reference: RGB pixels at 24 fps and stereo audio at 32 kHz."""

    kind: str
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    audio: torch.Tensor | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


@dataclass
class PreparedMedia:
    workflow: str
    height: int
    width: int
    num_frames: int
    references: list[ReferenceMedia]
    keyframe_indices: tuple[int, ...] = ()


def resolve_canvas(width: int, height: int) -> tuple[int, int]:
    """Resolve H3's 768-short-edge, capped-area canvas as (height, width)."""
    ratio = width / height
    if not 0.25 <= ratio <= 4:
        raise ValueError("HyperFlow requires an aspect ratio between 1:4 and 4:1.")
    out_width, out_height = (768 * ratio, 768.0) if ratio >= 1 else (768.0, 768 / ratio)
    if out_width * out_height > 768 * 1344:
        scale = (768 * 1344 / (out_width * out_height)) ** 0.5
        out_width, out_height = out_width * scale, out_height * scale
    return max(32, round(out_height / 32) * 32), max(32, round(out_width / 32) * 32)


def _read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _read_soundtrack(container: Any, stream: Any) -> tuple[torch.Tensor, int]:
    from av.audio.resampler import AudioResampler

    rate = int(stream.codec_context.sample_rate)
    resampler = AudioResampler(format="fltp", layout=stream.layout, rate=rate)
    chunks = [
        torch.from_numpy(part.to_ndarray()) for frame in container.decode(stream) for part in resampler.resample(frame)
    ]
    chunks.extend(torch.from_numpy(part.to_ndarray()) for part in resampler.resample(None))
    return torch.cat(chunks, dim=-1).float(), rate


def normalize_audio(waveform: torch.Tensor, sample_rate: int, num_frames: int) -> torch.Tensor:
    """Truncate at the source rate, upmix mono, then resample exactly once."""
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError("HyperFlow reference audio must be mono or stereo.")
    waveform = waveform.float()[:, : int(num_frames / 24 * sample_rate)]
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate != 32000:
        from torchaudio.transforms import Resample

        waveform = Resample(sample_rate, 32000)(waveform)
    return waveform


def normalize_video(frames: np.ndarray, fps: float, num_frames: int) -> np.ndarray:
    """Place decoded RGB frames on the release's 24 fps grid and reference canvas."""
    if fps <= 0:
        raise ValueError("HyperFlow reference video must have a positive frame rate.")
    if fps != 24:
        scale = 24 / fps
        slots = np.floor(np.arange(len(frames)) * scale + 0.5).astype(np.int64)
        repeats = np.diff(slots, append=math.floor(len(frames) * scale + 0.5))
        frames = np.repeat(frames, repeats, axis=0)
    frames = frames[:num_frames]
    height, width = resolve_canvas(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def _read_reference(kind: str, path: Path, num_frames: int) -> ReferenceMedia:
    if kind == "image":
        image = _read_image(path)
        width, height = image.size
        if not 0.25 <= width / height <= 4:
            raise ValueError("HyperFlow reference images must have an aspect ratio between 1:4 and 4:1.")
        scale = 2048 / min(width, height)
        size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
        return ReferenceMedia(kind, image=image.resize(size, Image.Resampling.LANCZOS) if image.size != size else image)

    import av

    waveform = None
    with av.open(str(path)) as container:
        if kind == "audio":
            if not container.streams.audio:
                raise ValueError(f"No audio stream in HyperFlow reference: {path}")
            audio, rate = _read_soundtrack(container, container.streams.audio[0])
            return ReferenceMedia(kind, audio=normalize_audio(audio, rate, num_frames))
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            rotation = frame.rotation
            frames.append(frame.to_ndarray(format="rgb24"))
        fps = float(stream.average_rate or stream.guessed_rate)
        if container.streams.audio:
            container.seek(0)
            audio, rate = _read_soundtrack(container, container.streams.audio[0])
            waveform = normalize_audio(audio, rate, num_frames)
    if not frames:
        raise ValueError(f"No video frames in HyperFlow reference: {path}")
    pixels = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        pixels = np.ascontiguousarray(np.rot90(pixels, k=-turns, axes=(1, 2)))
    return ReferenceMedia(kind, frames=normalize_video(pixels, fps, num_frames), audio=waveform)


def prepare_media(
    *,
    image: Path | None = None,
    last_image: Path | None = None,
    references: list[tuple[str, Path]] | None = None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int = 124,
) -> PreparedMedia:
    """Read local inputs before loading models; preserve reference order and keyframe anchors."""
    if (height is None) != (width is None):
        raise ValueError("HyperFlow height and width must be passed together.")
    if height is not None and (height <= 0 or width <= 0 or height % 32 or width % 32):
        raise ValueError("HyperFlow height and width must be positive multiples of 32.")
    frames = minimax_h3_align_frame_count(num_frames)
    if not 120 <= frames <= 360:
        raise ValueError("HyperFlow generates 5–15 seconds at 24 fps after frame alignment.")
    if references:
        if image is not None or last_image is not None:
            raise ValueError("HyperFlow references cannot be combined with keyframes.")
        height, width = (height, width) if height is not None else resolve_canvas(16, 9)
        return PreparedMedia("ref2va", height, width, frames, [_read_reference(k, p, frames) for k, p in references])

    keyframes = [_read_image(path) for path in (image, last_image) if path is not None]
    anchors = tuple(index for index, path in ((0, image), (-1, last_image)) if path is not None)
    height, width = (
        (height, width) if height is not None else resolve_canvas(*(keyframes[0].size if keyframes else (16, 9)))
    )
    prepared = []
    for index, keyframe in enumerate(keyframes):
        if keyframe.size != (width, height):
            if index == 0:
                keyframe = keyframe.resize((width, height), Image.Resampling.LANCZOS)
            else:
                scale = max(width / keyframe.width, height / keyframe.height)
                size = (max(width, round(keyframe.width * scale)), max(height, round(keyframe.height * scale)))
                left, top = max(0, (size[0] - width) // 2), max(0, (size[1] - height) // 2)
                keyframe = keyframe.resize(size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))
        prepared.append(ReferenceMedia("image", image=keyframe))
    return PreparedMedia("fl2va" if prepared else "t2va", height, width, frames, prepared, anchors)


def write_video_audio(video: torch.Tensor, audio: torch.Tensor, output: Path) -> None:
    """Mux uint8 FHWC RGB and stereo audio using the released PyAV encoder defaults."""
    import av

    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.height, stream.width = video.shape[1:3]
        stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=32000)
        audio_stream.codec_context.sample_rate = 32000
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, 32000)
        for frame in video.cpu().numpy():
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        _write_stereo_aac(container, audio_stream, audio, 32000)
