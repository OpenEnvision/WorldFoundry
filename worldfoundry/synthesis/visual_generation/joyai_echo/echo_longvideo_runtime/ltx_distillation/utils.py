"""Shared utilities for inference: latent computation, noise, media I/O, video concat."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .inference.memory_multishot import audio_waveform_stats, normalize_audio_waveform_for_media


def _write_video(*args, **kwargs) -> None:
    """Import the pinned torchvision video backend only when media is written."""

    from torchvision.io import write_video

    write_video(*args, **kwargs)


def compute_latent_shapes(
    *,
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")

    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=original.device, dtype=original.dtype)
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    return (1 - sigma) * original + sigma * noise


@torch.no_grad()
def decode_generated_sample(
    video_vae,
    audio_vae,
    video_latent,
    audio_latent,
    *,
    video_tiling_config=None,
):
    if video_tiling_config is None:
        video_pixel = video_vae.decode_to_pixel(video_latent)
        video_uint8 = video_pixel[0]
        if video_uint8.shape[0] == 3:
            video_uint8 = video_uint8.permute(1, 0, 2, 3)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)
        video_uint8 = (video_uint8.clamp(0, 1) * 255).cpu().to(torch.uint8).contiguous()
    else:
        video_chunks = list(video_vae.decode_to_uint8_chunks(video_latent, video_tiling_config))
        if not video_chunks:
            raise RuntimeError("tiled video VAE decode produced no frames")
        video_uint8 = torch.cat(video_chunks, dim=0)

    audio_waveform = audio_vae.decode_to_waveform(audio_latent) if audio_latent is not None else None

    audio_float = normalize_audio_waveform_for_media(audio_waveform)
    return video_uint8, audio_float


def fit_audio_to_video_frames(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    video_frames: int,
    video_fps: float,
) -> torch.Tensor:
    """Trim or zero-pad audio to the exact encoded video-frame duration."""

    if sample_rate <= 0 or video_frames <= 0 or video_fps <= 0:
        return audio
    target_samples = max(1, round(video_frames * sample_rate / video_fps))
    current_samples = int(audio.shape[-1])
    if current_samples > target_samples:
        return audio[..., :target_samples].contiguous()
    if current_samples < target_samples:
        return torch.nn.functional.pad(audio, (0, target_samples - current_samples))
    return audio


def _write_video_with_aligned_audio(
    *,
    video_uint8: torch.Tensor,
    output_path: Path,
    audio_path: Path,
    fps: float,
) -> None:
    """Mux aligned audio without ``-shortest``, matching production."""

    frame_count = int(video_uint8.shape[0])
    if frame_count <= 0:
        raise ValueError("cannot write an empty video")
    if fps <= 0:
        raise ValueError(f"invalid frame rate: {fps}")

    duration_arg = f"{frame_count / float(fps):.9f}"
    silent_path = output_path.with_name(f"{output_path.stem}_silent{output_path.suffix}")
    _write_video(str(silent_path), video_uint8, fps=int(fps))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg not found")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-af",
        f"apad,atrim=duration={duration_arg},asetpts=N/SR/TB",
        "-t",
        duration_arg,
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        silent_path.unlink(missing_ok=True)


def write_generated_media(
    *,
    output_path: Path,
    video_uint8: torch.Tensor,
    audio_waveform: torch.Tensor | None,
    fps: int,
    audio_sr: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_waveform = normalize_audio_waveform_for_media(audio_waveform)
    if audio_waveform is not None:
        audio_waveform = fit_audio_to_video_frames(
            audio_waveform,
            sample_rate=audio_sr,
            video_frames=int(video_uint8.shape[0]),
            video_fps=float(fps),
        )
    stats = audio_waveform_stats(audio_waveform)

    wrote_with_audio = False
    wrote_sidecar_wav = False
    if audio_waveform is not None:
        audio_path = output_path.with_suffix(".wav")
        torchaudio.save(str(audio_path), audio_waveform, audio_sr)
        wrote_sidecar_wav = True
        try:
            _write_video_with_aligned_audio(
                video_uint8=video_uint8,
                output_path=output_path,
                audio_path=audio_path,
                fps=float(fps),
            )
            wrote_with_audio = True
        except Exception as exc:
            print(
                f"[warn] aligned audio mux failed for {output_path}: {exc}; retrying direct write_video",
                flush=True,
            )
            try:
                _write_video(
                    str(output_path),
                    video_uint8,
                    fps=fps,
                    audio_array=audio_waveform,
                    audio_fps=audio_sr,
                    audio_codec="aac",
                )
                wrote_with_audio = True
            except Exception as fallback_exc:
                print(
                    f"[warn] direct audio mux failed for {output_path}: "
                    f"{fallback_exc}; writing video only; audio_stats={stats}",
                    flush=True,
                )

    if not wrote_with_audio:
        _write_video(str(output_path), video_uint8, fps=fps)

    return {
        "wrote_audio_in_mp4": wrote_with_audio,
        "wrote_sidecar_wav": wrote_sidecar_wav,
        "audio_stats": stats,
    }


def concat_shot_videos(shot_paths: list[Path], output_path: Path) -> None:
    if not shot_paths:
        raise ValueError("No shot videos provided for concatenation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fp:
        concat_file = Path(fp.name)
        for shot_path in shot_paths:
            fp.write(f"file '{shot_path.resolve().as_posix()}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ]
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
            if fallback_result.returncode != 0:
                raise RuntimeError(
                    "Failed to concatenate shot videos with ffmpeg.\n"
                    f"copy stderr:\n{result.stderr}\n"
                    f"reencode stderr:\n{fallback_result.stderr}"
                )
    finally:
        concat_file.unlink(missing_ok=True)


def concat_shot_audios(audios: list[torch.Tensor]) -> torch.Tensor | None:
    if not audios:
        return None
    audio = audios[0]
    if audio.ndim == 1:
        sample_dim = 0
    elif audio.ndim == 2:
        sample_dim = 1 if audio.shape[0] <= audio.shape[1] else 0
    else:
        raise ValueError(f"Expected audio tensor with 1 or 2 dims, got shape={tuple(audio.shape)}")
    return torch.cat([a.contiguous() for a in audios], dim=sample_dim).contiguous()
