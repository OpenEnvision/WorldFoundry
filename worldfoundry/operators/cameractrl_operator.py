"""Module for the CameraCtrl operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class CameraCtrlOperator(BaseOperator):
    """Operator for CameraCtrl camera-trajectory conditioned video generation."""

    MODEL_ID = "cameractrl"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["camera_path"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["camera_path", "trajectory_file"]
        self.interaction_template_init()

    @staticmethod
    def _trajectories(interaction) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, (str, Path)):
            text = str(interaction).strip()
            return [text] if text else []
        if isinstance(interaction, Iterable):
            return [str(item).strip() for item in interaction if str(item).strip()]
        raise TypeError(
            f"CameraCtrl interaction must be a trajectory path or sequence, got {type(interaction).__name__}."
        )

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._trajectories(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        trajectories = self._trajectories(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(trajectories)
        result: dict[str, Any] = {"actions": trajectories}
        if trajectories:
            result["trajectory_file"] = trajectories[0]
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        text = prompt if prompt is not None else kwargs.get("caption")
        return {"prompt": "" if text is None else str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if video is not None:
            raise ValueError(
                "CameraCtrl inference consumes prompts, optional image LoRA/reference inputs, and camera trajectories, not input video."
            )
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
