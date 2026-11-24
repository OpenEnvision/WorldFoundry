# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from fuyi25/H3-Tiny at a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c (Apache-2.0).
"""Input validation operator for H3-Tiny joint video and audio generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.base_models.diffusion_model.schedulers.h3_tiny import (
    H3_TINY_FEWSTEP_DEFAULT_STEPS,
    H3_TINY_FEWSTEP_MIN_STEPS,
    H3_TINY_FULLSTEP_DEFAULT_STEPS,
)

from .base_operator import BaseOperator

H3_TINY_TASKS: tuple[str, ...] = ("t2va", "i2va", "l2va", "fl2va", "ref2va")
H3_TINY_MAX_PROMPT_CHARS = 8000
H3_TINY_FPS = 24
H3_TINY_DEFAULT_NUM_FRAMES = 124
H3_TINY_DEFAULT_SEED = 1234


def _is_valid_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_endpoint_images(
    *,
    images: Any = None,
    image: Any = None,
    last_image: Any = None,
) -> tuple[list[Any], tuple[int, ...] | None]:
    if images is not None and (image is not None or last_image is not None):
        raise ValueError("Pass images or image/last_image, not both.")
    if image is not None or last_image is not None:
        frames: list[Any] = []
        indices: list[int] = []
        if image is not None:
            frames.append(image)
            indices.append(0)
        if last_image is not None:
            frames.append(last_image)
            indices.append(-1)
        return frames, tuple(indices)
    if images is None:
        return [], None
    if isinstance(images, (list, tuple)):
        if len(images) > 2:
            raise ValueError("Endpoint conditioning accepts at most two images (first and last frame).")
        return list(images), None
    return [images], None


def _normalize_references(references: Any) -> list[Any]:
    if references is None:
        return []
    entries = list(references) if isinstance(references, (list, tuple)) else [references]
    normalized: list[Any] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            kind = str(entry.get("kind", "image")).lower()
            if kind != "image":
                raise ValueError("H3-Tiny in-tree reference conditioning currently accepts image references.")
            if "path" not in entry or entry["path"] is None:
                raise ValueError("Image reference mapping must include a non-empty 'path'.")
            normalized.append(entry["path"])
        elif isinstance(entry, str) and ":" in entry:
            kind, location = entry.split(":", 1)
            if kind.lower() in ("image", "video", "audio"):
                if kind.lower() != "image":
                    raise ValueError("H3-Tiny in-tree reference conditioning currently accepts image references.")
                normalized.append(location)
            else:
                normalized.append(entry)
        else:
            normalized.append(entry)
    return normalized


class H3TinyOperator(BaseOperator):
    """Validate H3-Tiny prompts, tasks/workflows, frames, steps, references, and seeds."""

    def __init__(self) -> None:
        super().__init__(operation_types=["textual_instruction", "visual_instruction"])
        self.interaction_template = ["text_prompt"]
        self.interaction_template_init()

    def check_interaction(self, interaction: Any) -> bool:
        if not isinstance(interaction, str) or not interaction.strip():
            raise ValueError("H3-Tiny prompt must be a non-empty string.")
        if len(interaction) > H3_TINY_MAX_PROMPT_CHARS:
            raise ValueError(f"H3-Tiny prompt must contain at most {H3_TINY_MAX_PROMPT_CHARS} characters.")
        return True

    def get_interaction(self, interaction: str) -> None:
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        if not self.current_interaction:
            raise ValueError("No prompt to process.")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {"prompt": prompt}

    def process_perception(
        self,
        *,
        images: Any = None,
        image: Any = None,
        last_image: Any = None,
        references: Any = None,
    ) -> dict[str, Any]:
        keyframes, keyframe_indices = _normalize_endpoint_images(images=images, image=image, last_image=last_image)
        reference_items = _normalize_references(references)
        if keyframes and reference_items:
            raise ValueError("Use references or endpoint images, not both.")
        return {
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "references": reference_items,
        }

    def validate_request(
        self,
        *,
        prompt: Any,
        task: str | None = None,
        workflow: str | None = None,
        images: Any = None,
        image: Any = None,
        last_image: Any = None,
        references: Any = None,
        num_frames: int = H3_TINY_DEFAULT_NUM_FRAMES,
        num_inference_steps: int | None = None,
        endpoint_mode: str = "meanflow",
        seed: int = H3_TINY_DEFAULT_SEED,
        fps: int = H3_TINY_FPS,
        height: int | None = None,
        width: int | None = None,
        short_edge: int = 768,
        aspect_ratio: str = "16:9",
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize one H3-Tiny inference request."""
        self.check_interaction(prompt)
        perception = self.process_perception(images=images, image=image, last_image=last_image, references=references)
        keyframes: list[Any] = perception["keyframes"]
        named_indices: tuple[int, ...] | None = perception["keyframe_indices"]
        reference_items: list[Any] = perception["references"]

        norm_task = None if task is None else str(task).strip().lower()
        norm_workflow = None if workflow is None else str(workflow).strip().lower()
        if norm_task not in (None, "auto") and norm_workflow not in (None, "auto"):
            compatible_fl2va = norm_workflow == "fl2va" and norm_task in ("t2va", "i2va", "l2va", "fl2va")
            if norm_task != norm_workflow and not compatible_fl2va:
                raise ValueError(f"Conflicting task={norm_task!r} and workflow={norm_workflow!r}.")
        requested = norm_task if norm_task not in (None, "auto") else norm_workflow

        if requested in (None, "auto"):
            if reference_items:
                resolved_task = "ref2va"
            elif named_indices == (0,):
                resolved_task = "i2va"
            elif named_indices == (-1,):
                resolved_task = "l2va"
            elif named_indices == (0, -1):
                resolved_task = "fl2va"
            elif len(keyframes) == 0:
                resolved_task = "t2va"
            elif len(keyframes) == 1:
                resolved_task = "i2va"
            else:
                resolved_task = "fl2va"
        else:
            if requested not in H3_TINY_TASKS:
                raise ValueError(f"Unsupported H3-Tiny task/workflow {requested!r}; expected one of {H3_TINY_TASKS}.")
            if requested == "fl2va" and norm_task in (None, "auto"):
                # Upstream `workflow="fl2va"` dispatches 0/1/2 keyframes to t2va/i2va/l2va/fl2va.
                if named_indices == (0,):
                    resolved_task = "i2va"
                elif named_indices == (-1,):
                    resolved_task = "l2va"
                elif len(keyframes) == 0:
                    resolved_task = "t2va"
                elif len(keyframes) == 1:
                    resolved_task = "i2va"
                else:
                    resolved_task = "fl2va"
            else:
                resolved_task = requested

        if resolved_task == "ref2va":
            if not reference_items:
                raise ValueError("ref2va requires at least one image reference.")
            if keyframes:
                raise ValueError("ref2va does not accept endpoint images.")
            keyframe_indices = None
        else:
            if reference_items:
                raise ValueError(f"task {resolved_task!r} does not accept references; use task='ref2va'.")
            if resolved_task == "t2va":
                if keyframes:
                    raise ValueError("t2va does not accept endpoint images.")
                keyframe_indices = None
            elif resolved_task == "i2va":
                if len(keyframes) != 1 or named_indices == (-1,):
                    raise ValueError("i2va requires exactly one first-frame image.")
                keyframe_indices = (0,)
            elif resolved_task == "l2va":
                if len(keyframes) != 1 or named_indices == (0,):
                    raise ValueError("l2va requires exactly one last-frame image.")
                keyframe_indices = (-1,)
            else:  # fl2va
                if len(keyframes) != 2:
                    raise ValueError("fl2va requires two endpoint images (first and last frame).")
                keyframe_indices = (0, -1)

        if not _is_valid_int(num_frames) or num_frames <= 0:
            raise ValueError(f"num_frames must be a positive integer, got {num_frames!r}")
        if not _is_valid_int(fps) or fps != H3_TINY_FPS:
            raise ValueError(f"H3-Tiny uses {H3_TINY_FPS} fps, got {fps!r}")

        default_steps = H3_TINY_FEWSTEP_DEFAULT_STEPS if endpoint_mode == "meanflow" else H3_TINY_FULLSTEP_DEFAULT_STEPS
        steps = default_steps if num_inference_steps is None else num_inference_steps
        if not _is_valid_int(steps) or steps <= 0:
            raise ValueError(f"num_inference_steps must be a positive integer, got {steps!r}")
        if endpoint_mode == "meanflow" and steps < H3_TINY_FEWSTEP_MIN_STEPS:
            raise ValueError(
                f"H3-Tiny MeanFlow checkpoints require at least {H3_TINY_FEWSTEP_MIN_STEPS} inference steps, got {steps}"
            )

        if not _is_valid_int(seed) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        if (height is None) != (width is None):
            raise ValueError("height and width must be provided together.")
        for label, value in (("height", height), ("width", width)):
            if value is not None and (not _is_valid_int(value) or value <= 0 or value % 32 != 0):
                raise ValueError(f"{label} must be a positive multiple of 32, got {value!r}")
        if not _is_valid_int(short_edge) or short_edge <= 0:
            raise ValueError(f"short_edge must be a positive integer, got {short_edge!r}")

        resolved_duration = float(num_frames / H3_TINY_FPS) if duration_seconds is None else float(duration_seconds)
        if resolved_duration <= 0.0:
            raise ValueError(f"duration_seconds must be positive, got {resolved_duration!r}")

        self.current_interaction.append(prompt)
        self.interaction_history.append(prompt)
        return {
            "prompt": prompt,
            "task": resolved_task,
            "model_task": "ref2va" if resolved_task == "ref2va" else ("fl2va" if keyframes else "t2va"),
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "references": reference_items,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "seed": seed,
            "fps": fps,
            "height": height,
            "width": width,
            "short_edge": short_edge,
            "aspect_ratio": str(aspect_ratio),
            "duration_seconds": resolved_duration,
        }


__all__ = [
    "H3_TINY_DEFAULT_NUM_FRAMES",
    "H3_TINY_DEFAULT_SEED",
    "H3_TINY_FPS",
    "H3_TINY_MAX_PROMPT_CHARS",
    "H3_TINY_TASKS",
    "H3TinyOperator",
]
