"""Audio-video memory helpers aligned with the Echo 1.5 online runtime."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..audio_voice_filter import VoiceFilterConfig, filter_voice_only


def prompt_payload_to_text(payload: Any, prompt_max_chars: int | None = None) -> str:
    if not isinstance(payload, str):
        raise TypeError(
            f"unsupported prompt payload type: {type(payload).__name__}; each shot must be one prompt string"
        )
    text = payload.strip()
    return text[:prompt_max_chars] if prompt_max_chars else text


def json_to_prompts(data: dict[str, Any], prompt_max_chars: int | None = None) -> list[str]:
    values = data.get("prompts", data.get("shots", []))
    if not isinstance(values, list):
        return []
    prompts = [prompt_payload_to_text(item, prompt_max_chars) for item in values]
    return [prompt for prompt in prompts if prompt]


def load_multishot_prompts(
    prompts_file: str | Path,
    prompt_max_chars: int | None = None,
) -> list[str]:
    path = Path(prompts_file)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = json_to_prompts(payload, prompt_max_chars=prompt_max_chars)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def normalize_audio_waveform_for_media(
    audio_waveform: torch.Tensor | None,
) -> torch.Tensor | None:
    if audio_waveform is None:
        return None
    waveform = getattr(audio_waveform, "waveform", audio_waveform)
    waveform = torch.as_tensor(waveform).detach().cpu().float()
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got shape={tuple(waveform.shape)}")
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] not in {1, 2} and waveform.shape[1] in {1, 2}:
        waveform = waveform.transpose(0, 1)
    elif waveform.ndim != 2:
        raise ValueError(f"expected decoded audio with 1-3 dims, got shape={tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.contiguous()


def audio_waveform_stats(audio_waveform: torch.Tensor | None) -> dict[str, Any]:
    waveform = normalize_audio_waveform_for_media(audio_waveform)
    if waveform is None:
        return {
            "present": False,
            "shape": None,
            "num_samples": 0,
            "rms": 0.0,
            "peak": 0.0,
        }
    waveform_f = waveform.float()
    return {
        "present": True,
        "shape": list(waveform.shape),
        "num_samples": int(waveform.shape[-1]),
        "rms": float(waveform_f.square().mean().sqrt().item()),
        "peak": float(waveform_f.abs().max().item()),
    }


@dataclass
class MemoryEntry:
    video_latent: torch.Tensor
    audio_waveform: torch.Tensor | None
    audio_sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AudioVideoMemoryBank:
    """Stores one generated video-latent frame and full voice waveform per shot."""

    def __init__(self, max_size: int, num_fix_frames: int = 0) -> None:
        self.max_size = max(0, int(max_size))
        self.num_fix_frames = max(0, int(num_fix_frames))
        self.memory: list[MemoryEntry] = []

    def _trim(self) -> None:
        if self.max_size <= 0:
            self.memory = []
            return
        if len(self.memory) <= self.max_size:
            return
        fixed_count = min(self.num_fix_frames, self.max_size)
        fixed = self.memory[:fixed_count]
        keep_tail = self.max_size - fixed_count
        tail = self.memory[-keep_tail:] if keep_tail else []
        self.memory = fixed + tail

    def save_generated_shot(
        self,
        video_latent: torch.Tensor,
        audio_waveform: torch.Tensor | None,
        audio_sample_rate: int,
        *,
        enable_audio_memory: bool,
        voice_filter_config: VoiceFilterConfig,
    ) -> dict[str, Any]:
        if video_latent.ndim != 5 or video_latent.shape[0] != 1:
            raise ValueError(f"expected video latent [1, F, C, H, W], got {tuple(video_latent.shape)}")
        num_frames = int(video_latent.shape[1])
        if num_frames <= 0:
            raise ValueError("cannot save memory from an empty video latent")
        frame_index = random.randrange(num_frames)
        selected_video = video_latent[:, frame_index : frame_index + 1].detach().cpu().contiguous()

        filtered_audio = None
        if enable_audio_memory and audio_waveform is not None:
            normalized = normalize_audio_waveform_for_media(audio_waveform)
            filtered_audio = filter_voice_only(
                normalized,
                int(audio_sample_rate),
                voice_filter_config,
            )
            if filtered_audio is not None:
                filtered_audio = filtered_audio.detach().cpu().contiguous()

        metadata = {
            "selection": "random_latent_frame_full_audio",
            "video_frame_index": frame_index,
            "video_total_latent_frames": num_frames,
            "audio_present": filtered_audio is not None,
            "audio_samples": int(filtered_audio.shape[-1]) if filtered_audio is not None else 0,
            "audio_sample_rate": int(audio_sample_rate),
            "voice_filter_backend": voice_filter_config.backend if enable_audio_memory else "disabled",
        }
        self.memory.append(
            MemoryEntry(
                video_latent=selected_video,
                audio_waveform=filtered_audio,
                audio_sample_rate=int(audio_sample_rate),
                metadata=metadata,
            )
        )
        self._trim()
        return metadata

    def get_memory_video(self) -> torch.Tensor:
        if not self.memory:
            raise RuntimeError("memory bank is empty")
        return torch.cat([entry.video_latent for entry in self.memory], dim=1).contiguous()

    @torch.no_grad()
    def encode_memory_audio(self, audio_vae) -> list[torch.Tensor | None]:
        encoded: list[torch.Tensor | None] = []
        for entry in self.memory:
            waveform = entry.audio_waveform
            if waveform is None or waveform.numel() <= 1 or waveform.shape[-1] <= 1:
                encoded.append(None)
            else:
                encoded.append(audio_vae.encode(waveform, entry.audio_sample_rate).detach().cpu().contiguous())
        return encoded

    def get_memory_metadata(self) -> list[dict[str, Any]]:
        return [dict(entry.metadata) for entry in self.memory]

    def __len__(self) -> int:
        return len(self.memory)


@torch.no_grad()
def build_memory_audio_pipeline_kwargs(
    memory_bank: AudioVideoMemoryBank,
    audio_vae,
    *,
    enable_audio_memory: bool,
    memory_position_mode: str,
    memory_position_offset: float,
    memory_position_slot_stride: float,
) -> dict[str, Any]:
    """Encode full per-slot waveforms and assemble position-aligned audio memory."""

    if not enable_audio_memory:
        return {}
    audio_slices = memory_bank.encode_memory_audio(audio_vae)
    template = next((item for item in audio_slices if item is not None), None)
    if template is None:
        return {}
    aligned = [item if item is not None else torch.zeros_like(template) for item in audio_slices]
    memory_audio = torch.cat(aligned, dim=1).contiguous()
    segment_lengths = tuple(int(item.shape[1]) for item in aligned)
    return {
        "memory_audio": memory_audio,
        "memory_audio_timestep": torch.zeros(memory_audio.shape[:2], dtype=torch.float32),
        "memory_audio_segment_lengths": (segment_lengths,),
        "memory_position_mode": str(memory_position_mode),
        "memory_position_offset": float(memory_position_offset),
        "memory_position_slot_stride": float(memory_position_slot_stride),
    }
