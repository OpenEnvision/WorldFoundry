# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 OpenEnvision
"""Input validation and request normalization operator for FastFrames."""

from __future__ import annotations

from typing import Any

from worldfoundry.base_models.diffusion_model.schedulers.fastframes import (
    FASTFRAMES_CONTEXT_MODES,
    FASTFRAMES_NOISE_MODES,
    FastFramesVariantConfig,
    resolve_fastframes_variant,
)

from .base_operator import BaseOperator

__all__ = ["FastFramesOperator"]


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"FastFrames {name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"FastFrames {name} must be positive, got {value}")
    return value


def _validate_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"FastFrames seed must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"FastFrames seed must be non-negative, got {value}")
    return value


class FastFramesOperator(BaseOperator):
    """Validate and normalize text-to-video generation requests for FastFrames."""

    def __init__(
        self,
        operation_types: list[str] | None = None,
        *,
        default_variant: str | FastFramesVariantConfig = "fastframes-s",
    ) -> None:
        super().__init__(operation_types=operation_types or ["textual_instruction"])
        self.default_variant = resolve_fastframes_variant(default_variant)
        self.interaction_template = ["prompt"]
        self.interaction_template_init()

    def check_interaction(self, interaction: Any) -> bool:
        """Validate a text prompt interaction."""
        if not isinstance(interaction, str) or not interaction.strip():
            raise ValueError("FastFrames requires a non-empty text prompt string.")
        return True

    def get_interaction(self, interaction: Any) -> None:
        """Validate and record a prompt interaction."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction.strip())

    def process_interaction(self) -> dict[str, Any]:
        """Consume the latest recorded prompt interaction."""
        if not self.current_interaction:
            raise ValueError("No FastFrames prompt interaction registered. Call get_interaction() first.")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {"prompt": prompt}

    def process_perception(self, *, images: Any = None, video: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Reject unsupported image/video conditioning for the T2V FastFrames route."""
        del kwargs
        if images is not None:
            raise ValueError("FastFrames T2V does not accept image inputs.")
        if video is not None:
            raise ValueError("FastFrames T2V does not accept video inputs.")
        return {"images": None, "video": None}

    def prepare_request(
        self,
        *,
        prompt: str | None,
        variant: str | FastFramesVariantConfig | None = None,
        context: str = "refiner",
        seed: int = 0,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        noise: str | None = None,
        images: Any = None,
        video: Any = None,
        allow_empty_prompt: bool = False,
    ) -> dict[str, Any]:
        """Validate all FastFrames generation arguments and derive latent geometry."""
        self.process_perception(images=images, video=video)

        if prompt is None:
            if not allow_empty_prompt:
                raise ValueError("FastFrames requires a non-empty text prompt.")
            cleaned_prompt = ""
        else:
            if not isinstance(prompt, str):
                raise TypeError(f"FastFrames prompt must be a string, got {type(prompt).__name__}")
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt and not allow_empty_prompt:
                raise ValueError("FastFrames requires a non-empty text prompt.")

        variant_cfg = resolve_fastframes_variant(variant if variant is not None else self.default_variant)

        if not isinstance(context, str) or context not in FASTFRAMES_CONTEXT_MODES:
            raise ValueError(f"FastFrames context must be one of {FASTFRAMES_CONTEXT_MODES}, got {context!r}")

        validated_seed = _validate_seed(seed)

        resolved_num_frames = (
            variant_cfg.num_pixel_frames if num_frames is None else _validate_positive_int(num_frames, "num_frames")
        )
        if (resolved_num_frames - 1) % 4 != 0:
            raise ValueError(
                f"FastFrames num_frames must satisfy 4k+1 (e.g. 1, 5, 9, ..., 81), got {resolved_num_frames}"
            )
        num_latent_frames = (resolved_num_frames + 3) // 4

        default_h = variant_cfg.latent_size[0] * 8
        default_w = variant_cfg.latent_size[1] * 8
        resolved_height = default_h if height is None else _validate_positive_int(height, "height")
        resolved_width = default_w if width is None else _validate_positive_int(width, "width")
        if resolved_height % 16 != 0 or resolved_width % 16 != 0:
            raise ValueError(
                f"FastFrames height and width must be positive multiples of 16, got {resolved_width}x{resolved_height}"
            )
        latent_size = (resolved_height // 8, resolved_width // 8)

        resolved_fps = variant_cfg.fps if fps is None else _validate_positive_int(fps, "fps")

        resolved_noise = variant_cfg.noise if noise is None else noise
        if not isinstance(resolved_noise, str) or resolved_noise not in FASTFRAMES_NOISE_MODES:
            raise ValueError(f"FastFrames noise mode must be one of {FASTFRAMES_NOISE_MODES}, got {resolved_noise!r}")

        return {
            "prompt": cleaned_prompt,
            "variant": variant_cfg,
            "context": context,
            "seed": validated_seed,
            "num_frames": resolved_num_frames,
            "num_latent_frames": num_latent_frames,
            "height": resolved_height,
            "width": resolved_width,
            "latent_size": latent_size,
            "fps": resolved_fps,
            "noise": resolved_noise,
        }
