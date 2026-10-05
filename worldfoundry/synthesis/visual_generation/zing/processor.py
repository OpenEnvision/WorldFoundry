# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.zing.config import ZingConfig


@dataclass
class InferenceRequest:
    clean_latents: torch.Tensor
    label_mask: torch.Tensor
    prompts: list[str]
    prompt_lengths: torch.Tensor
    chunk_spans: list[tuple[int, int]]
    action: torch.Tensor | None


class MessageProcessor:
    def __init__(self, config: ZingConfig, encode_reference: Callable[[torch.Tensor], torch.Tensor]):
        self.config = config
        self.encode_reference = encode_reference

    def process(self, sample: dict) -> InferenceRequest:
        messages = sample.get("messages")
        if not isinstance(messages, list):
            raise ValueError("sample.messages must be a list")
        user_messages = [item for item in messages if item.get("role") == "user" and item.get("type") == "text"]
        target_messages = [item for item in messages if item.get("role") == "target" and item.get("type") == "video"]
        if len(user_messages) != 1 or len(target_messages) != 1:
            raise ValueError("sample must contain one user text and one target video")
        initial_prompt = str(user_messages[0].get("content") or "").strip()
        if not initial_prompt:
            raise ValueError("user text content must not be empty")
        target = target_messages[0]
        if "frames" in target or "latent" in target:
            raise ValueError("target video must not contain frames or latent")
        output = target.get("output")
        if not isinstance(output, dict):
            raise ValueError("target video output is required")
        missing = [key for key in ("frames", "height", "width") if output.get(key) is None]
        if missing:
            raise ValueError(f"target video output is missing {', '.join(missing)}")
        output_frames, height, width = (int(output[key]) for key in ("frames", "height", "width"))
        if min(output_frames, height, width) < 1:
            raise ValueError("target video output values must be positive")
        patch = self.config.generator.patch_size
        height_factor = self.config.vae.spatial_scale * patch[1]
        width_factor = self.config.vae.spatial_scale * patch[2]
        if height % height_factor or width % width_factor:
            raise ValueError("output height and width must align with the VAE and generator patch size")

        reference_count = int(target.get("reference_frame_count", 0) or 0)
        if reference_count not in (0, 1):
            raise ValueError("reference_frame_count must be 0 or 1")
        uri = target.get("uri")
        if reference_count and uri is None:
            raise ValueError("reference_frame_count=1 requires a local image uri")
        if not reference_count and uri is not None:
            raise ValueError("uri requires reference_frame_count=1")

        temporal_scale = self.config.vae.temporal_scale
        total_pixel_frames = reference_count + output_frames
        if (total_pixel_frames - 1) % temporal_scale:
            raise ValueError("the total frame count must map exactly to VAE latent frames")
        total_latent_frames = 1 + (total_pixel_frames - 1) // temporal_scale
        if total_latent_frames % patch[0]:
            raise ValueError("latent frames must align with the temporal patch size")
        latent_h = height // self.config.vae.spatial_scale
        latent_w = width // self.config.vae.spatial_scale
        clean_latents = torch.zeros(
            (1, total_latent_frames, self.config.vae.z_dim, latent_h, latent_w), dtype=torch.float16
        )
        if reference_count:
            path = Path(str(uri)).expanduser()
            if not path.is_file():
                raise ValueError(f"reference image does not exist: {path}")
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
                pixels = np.asarray(rgb, dtype=np.uint8).copy()
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            reference = self.encode_reference(tensor)
            if tuple(reference.shape) != (1, 1, self.config.vae.z_dim, latent_h, latent_w):
                raise ValueError(f"reference latent shape mismatch: {tuple(reference.shape)}")
            clean_latents[:, :1] = reference.to(dtype=torch.float16, device="cpu")

        label_mask = (torch.arange(total_latent_frames) >= reference_count).unsqueeze(0)
        controls = target.get("controls", []) or []
        if not isinstance(controls, list):
            raise ValueError("target controls must be a list")
        if any(
            not isinstance(item, dict)
            or item.get("type") not in {"text_prompt_interval", "keyboard_direction_frame_interval"}
            for item in controls
        ):
            raise ValueError("Unsupported Zing control type")
        prompts, prompt_lengths = self._prompt_spans(initial_prompt, controls, total_pixel_frames, total_latent_frames)
        hard_boundaries = set(torch.cumsum(prompt_lengths, dim=0).tolist())
        hard_boundaries.add(reference_count)
        chunk_spans = self._chunk_spans(total_latent_frames, hard_boundaries)
        action = self._action(controls, reference_count, output_frames, total_pixel_frames, total_latent_frames)
        return InferenceRequest(clean_latents, label_mask, prompts, prompt_lengths, chunk_spans, action)

    def _prompt_spans(
        self, initial_prompt: str, controls: list[dict], total_pixel_frames: int, total_latent_frames: int
    ) -> tuple[list[str], torch.Tensor]:
        matches = [item for item in controls if item.get("type") == "text_prompt_interval"]
        if len(matches) > 1:
            raise ValueError("only one text_prompt_interval control is allowed")
        if not matches:
            return [initial_prompt], torch.tensor([total_latent_frames], dtype=torch.long)
        segments = []
        for item in sorted(matches[0].get("segments", []) or [], key=lambda value: int(value.get("start", 0))):
            text = str(item.get("text") or "").strip()
            if not text:
                raise ValueError("prompt segment text must not be empty")
            raw_start = int(item.get("start", 0))
            raw_end = int(item.get("end", total_pixel_frames))
            if raw_start < 0 or raw_end > total_pixel_frames or raw_end <= raw_start:
                raise ValueError("prompt segment boundaries are invalid")
            start = self._frame_boundary_to_latent(raw_start)
            end = self._frame_boundary_to_latent(raw_end)
            if end <= start:
                raise ValueError("prompt segment maps to an empty latent span")
            segments.append((start, end, text))
        if not segments:
            return [initial_prompt], torch.tensor([total_latent_frames], dtype=torch.long)
        previous_end = segments[0][1]
        for start, end, _ in segments[1:]:
            if start != previous_end:
                raise ValueError("prompt segments overlap or contain a gap")
            previous_end = end
        if segments[-1][1] != total_latent_frames:
            raise ValueError("prompt segments leave a tail gap")
        prompts, lengths = [], []
        if segments[0][0] > 0:
            prompts.append(initial_prompt)
            lengths.append(segments[0][0])
        for start, end, text in segments:
            prompts.append(text)
            lengths.append(end - start)
        return prompts, torch.tensor(lengths, dtype=torch.long)

    def _frame_boundary_to_latent(self, boundary: int) -> int:
        if boundary <= 0:
            return 0
        factor = self.config.vae.temporal_scale
        lower_n = max((boundary - 1) // factor, 0)
        lower = 1 + lower_n * factor
        upper = 1 + (lower_n + 1) * factor
        snapped = lower if abs(boundary - lower) <= abs(upper - boundary) else upper
        return 1 + (snapped - 1) // factor

    def _chunk_spans(self, total_frames: int, hard_boundaries: set[int]) -> list[tuple[int, int]]:
        boundaries = {0, total_frames}
        if total_frames > 1:
            boundaries.add(1)
        boundaries.update(value for value in hard_boundaries if 0 < value < total_frames)
        spans = []
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            frame = start
            while frame < end:
                block_end = min(frame + self.config.inference.frames_per_block, end)
                spans.append((frame, block_end))
                frame = block_end
        return spans

    def _action(
        self,
        controls: list[dict],
        reference_count: int,
        output_frames: int,
        total_pixel_frames: int,
        total_latent_frames: int,
    ) -> torch.Tensor | None:
        matches = [
            item
            for item in controls
            if item.get("type") == "keyboard_direction_frame_interval" and item.get("actions") is not None
        ]
        if len(matches) > 1:
            raise ValueError("only one keyboard_direction_frame_interval control is allowed")
        if not matches:
            return None
        array = np.asarray(matches[0]["actions"])
        if array.ndim != 2 or array.shape[1] != 8 or array.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("actions must have shape [N, 8]")
        values = array.astype(np.float32, copy=False)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("action values must be finite and in [0, 1]")
        expected = output_frames if reference_count else output_frames - 1
        if values.shape[0] != expected:
            raise ValueError(f"action row count must be {expected}")
        full = torch.zeros((total_pixel_frames - 1, 8), dtype=torch.float32)
        start = max(reference_count - 1, 0)
        full[start : start + values.shape[0]] = torch.from_numpy(np.ascontiguousarray(values))
        factor = self.config.vae.temporal_scale
        windows = torch.zeros((total_latent_frames, factor, 8), dtype=torch.float32)
        if total_latent_frames > 1:
            windows[1:] = full.reshape(total_latent_frames - 1, factor, 8)
        return windows.unsqueeze(0)
