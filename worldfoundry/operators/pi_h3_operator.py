# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3.
"""Input validation and request normalization operator for pi-H3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base_operator import BaseOperator

PI_H3_VARIANTS: tuple[str, ...] = ("2step-dwd", "4step-dwd", "2step", "4step")
PI_H3_TASKS: tuple[str, ...] = ("t2va", "i2va", "l2va", "fl2va")
PI_H3_SUPPORTED_STEPS: tuple[int, ...] = (2, 4)
PI_H3_SUPPORTED_FPS: int = 24
PI_H3_SPATIAL_MULTIPLE: int = 32
PI_H3_MIN_DURATION_SECONDS: float = 4.0
PI_H3_MAX_DURATION_SECONDS: float = 15.0

PI_H3_ASPECT_RATIOS: frozenset[str] = frozenset(
    {
        "21:9",
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16",
        "3:2",
        "5:4",
        "4:5",
        "2:3",
        "9:21",
    }
)

PI_H3_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "2step-dwd": {"subfolder": "pi_h3_2step_dwd_adapter", "steps": 2, "dwd": True},
    "4step-dwd": {"subfolder": "pi_h3_4step_dwd_adapter", "steps": 4, "dwd": True},
    "2step": {"subfolder": "pi_h3_2step_adapter", "steps": 2, "dwd": False},
    "4step": {"subfolder": "pi_h3_4step_adapter", "steps": 4, "dwd": False},
}

_VARIANT_ALIASES: dict[str, str] = {
    "2step-dwd": "2step-dwd",
    "2step_dwd": "2step-dwd",
    "pi-h3-2step-dwd": "2step-dwd",
    "pi_h3_2step_dwd_adapter": "2step-dwd",
    "4step-dwd": "4step-dwd",
    "4step_dwd": "4step-dwd",
    "pi-h3-4step-dwd": "4step-dwd",
    "pi_h3_4step_dwd_adapter": "4step-dwd",
    "2step": "2step",
    "2step-vae": "2step",
    "pi-h3-2step": "2step",
    "pi_h3_2step_adapter": "2step",
    "4step": "4step",
    "4step-vae": "4step",
    "pi-h3-4step": "4step",
    "pi_h3_4step_adapter": "4step",
}


def normalize_pi_h3_variant(variant: str) -> str:
    """Normalize a pi-H3 variant name or adapter subfolder to one of ``PI_H3_VARIANTS``."""

    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(f"pi-H3 variant must be one of {PI_H3_VARIANTS}, got {variant!r}.")
    key = variant.strip().lower()
    if key not in _VARIANT_ALIASES:
        raise ValueError(f"Unsupported pi-H3 variant {variant!r}; supported: {PI_H3_VARIANTS}.")
    return _VARIANT_ALIASES[key]


def _load_rgb_image(value: Any) -> Any:
    from PIL import Image, ImageOps

    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"pi-H3 input image not found: {path}")
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _normalize_image_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


class PiH3Operator(BaseOperator):
    """Validate pi-H3 prompts, variants, tasks, steps, keyframes, and video geometry."""

    def __init__(self, *, variant: str = "4step-dwd") -> None:
        super().__init__(operation_types=["textual_instruction", "visual_instruction"])
        self.variant = normalize_pi_h3_variant(variant)
        self.interaction_template = ["text_prompt", "image_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction: Any) -> None:
        """Validate and append a text prompt interaction."""

        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: Any) -> bool:
        """Verify that a prompt is a non-empty string within the H3 character limit."""

        if not isinstance(interaction, str) or not interaction.strip() or len(interaction) > 8000:
            raise ValueError("pi-H3 prompt must be a non-empty string with 1-8000 characters.")
        return True

    def process_interaction(self, **kwargs: Any) -> dict[str, Any]:
        """Process the most recent recorded prompt together with optional request parameters."""

        if not self.current_interaction:
            raise ValueError("No pi-H3 prompt to process.")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return self.validate_request(prompt=prompt, **kwargs)

    def process_perception(
        self,
        *,
        images: Any = None,
        image: Any = None,
        last_image: Any = None,
    ) -> dict[str, Any]:
        """Load optional first/last keyframe images into RGB PIL instances."""

        raw_images = self._collect_keyframes(images=images, image=image, last_image=last_image)
        return {"images": [_load_rgb_image(item) for item in raw_images]}

    @staticmethod
    def _collect_keyframes(*, images: Any, image: Any, last_image: Any) -> list[Any]:
        if images is not None and (image is not None or last_image is not None):
            raise ValueError("Specify either images or image/last_image, not both.")
        if images is not None:
            return _normalize_image_list(images)
        collected: list[Any] = []
        if image is not None:
            collected.append(image)
        if last_image is not None:
            collected.append(last_image)
        return collected

    def validate_request(
        self,
        *,
        prompt: Any,
        images: Any = None,
        image: Any = None,
        last_image: Any = None,
        task: str = "auto",
        variant: str | None = None,
        num_inference_steps: int | None = None,
        num_frames: int | None = None,
        duration_seconds: float | None = None,
        fps: int = PI_H3_SUPPORTED_FPS,
        height: int | None = None,
        width: int | None = None,
        short_edge: int = 768,
        aspect_ratio: str = "16:9",
        seed: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a complete pi-H3 inference request before running any model component."""

        if extra:
            raise TypeError(f"Unsupported pi-H3 call options: {sorted(extra)}")

        self.check_interaction(prompt)
        resolved_variant = self.variant if variant is None else normalize_pi_h3_variant(variant)
        variant_spec = PI_H3_VARIANT_SPECS[resolved_variant]

        if num_inference_steps is None:
            resolved_steps = int(variant_spec["steps"])
        else:
            if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int):
                raise ValueError(f"pi-H3 num_inference_steps must be 2 or 4, got {num_inference_steps!r}.")
            resolved_steps = int(num_inference_steps)
            if resolved_steps not in PI_H3_SUPPORTED_STEPS:
                raise ValueError(f"pi-H3 supports {PI_H3_SUPPORTED_STEPS} inference steps, got {resolved_steps}.")
            if resolved_steps != int(variant_spec["steps"]):
                raise ValueError(
                    f"pi-H3 variant {resolved_variant!r} requires {variant_spec['steps']} steps, got {resolved_steps}."
                )

        keyframes = self._collect_keyframes(images=images, image=image, last_image=last_image)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Unsupported pi-H3 task: {task!r}.")
        resolved_task = task.strip().lower()
        if resolved_task == "auto":
            if image is None and last_image is not None:
                resolved_task = "l2va"
            else:
                resolved_task = {0: "t2va", 1: "i2va", 2: "fl2va"}.get(len(keyframes), "")
        if resolved_task not in PI_H3_TASKS:
            raise ValueError(f"Unsupported pi-H3 task {task!r}; supported: {PI_H3_TASKS}.")
        expected_keyframes = {"t2va": 0, "i2va": 1, "l2va": 1, "fl2va": 2}[resolved_task]
        if len(keyframes) != expected_keyframes:
            raise ValueError(
                f"pi-H3 task {resolved_task!r} requires {expected_keyframes} keyframe image(s), got {len(keyframes)}."
            )

        if isinstance(fps, bool) or int(fps) != PI_H3_SUPPORTED_FPS:
            raise ValueError(f"pi-H3 uses a fixed frame rate of {PI_H3_SUPPORTED_FPS} fps, got {fps!r}.")

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"pi-H3 seed must be a non-negative integer, got {seed!r}.")

        if height is not None or width is not None:
            if height is None or width is None:
                raise ValueError("pi-H3 height and width must be supplied together.")
            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or not isinstance(height, int)
                or not isinstance(width, int)
                or height <= 0
                or width <= 0
                or height % PI_H3_SPATIAL_MULTIPLE != 0
                or width % PI_H3_SPATIAL_MULTIPLE != 0
            ):
                raise ValueError(
                    f"pi-H3 height and width must be positive multiples of {PI_H3_SPATIAL_MULTIPLE}, "
                    f"got {width}x{height}."
                )
        else:
            if isinstance(short_edge, bool) or not isinstance(short_edge, int) or short_edge <= 0:
                raise ValueError(f"pi-H3 short_edge must be a positive integer, got {short_edge!r}.")
            if aspect_ratio not in PI_H3_ASPECT_RATIOS:
                raise ValueError(
                    f"Unsupported pi-H3 aspect_ratio {aspect_ratio!r}; supported: {sorted(PI_H3_ASPECT_RATIOS)}."
                )

        if num_frames is not None:
            if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
                raise ValueError(f"pi-H3 num_frames must be a positive integer, got {num_frames!r}.")
            resolved_duration = float(num_frames) / float(PI_H3_SUPPORTED_FPS)
        else:
            resolved_duration = 5.0 if duration_seconds is None else float(duration_seconds)
            if not (PI_H3_MIN_DURATION_SECONDS <= resolved_duration <= PI_H3_MAX_DURATION_SECONDS):
                raise ValueError(
                    f"pi-H3 duration_seconds must be in [{PI_H3_MIN_DURATION_SECONDS}, "
                    f"{PI_H3_MAX_DURATION_SECONDS}], got {resolved_duration}."
                )

        return {
            "prompt": prompt,
            "keyframes": keyframes,
            "task": resolved_task,
            "variant": resolved_variant,
            "dwd": bool(variant_spec["dwd"]),
            "num_inference_steps": resolved_steps,
            "num_frames": num_frames,
            "duration_seconds": resolved_duration,
            "fps": PI_H3_SUPPORTED_FPS,
            "height": height,
            "width": width,
            "short_edge": int(short_edge),
            "aspect_ratio": aspect_ratio,
            "seed": int(seed),
        }


__all__ = [
    "PI_H3_ASPECT_RATIOS",
    "PI_H3_MAX_DURATION_SECONDS",
    "PI_H3_MIN_DURATION_SECONDS",
    "PI_H3_SPATIAL_MULTIPLE",
    "PI_H3_SUPPORTED_FPS",
    "PI_H3_SUPPORTED_STEPS",
    "PI_H3_TASKS",
    "PI_H3_VARIANTS",
    "PI_H3_VARIANT_SPECS",
    "PiH3Operator",
    "normalize_pi_h3_variant",
]
