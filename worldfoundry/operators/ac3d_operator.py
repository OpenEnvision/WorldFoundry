"""Module for the AC3D operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class AC3DOperator(BaseOperator):
    """Operator for AC3D camera-controlled text-to-video generation."""

    MODEL_ID = "ac3d"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": False,
        "video": True,
        "actions": ["camera_index", "camera_index_range", "camera_pose_rows"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["camera_index", "camera_index_range"]
        self.interaction_template_init()

    @staticmethod
    def _camera_indices(interaction: Any) -> list[int]:
        if interaction is None:
            return []
        if isinstance(interaction, int):
            return [interaction]
        if isinstance(interaction, str):
            text = interaction.strip()
            if not text:
                return []
            if ":" in text:
                start, end = text.split(":", 1)
                return list(range(int(start), int(end)))
            return [int(text)]
        if isinstance(interaction, Iterable) and not isinstance(interaction, (bytes, bytearray, Path)):
            return [int(item) for item in interaction]
        raise TypeError(f"AC3D interaction must be a camera index or range, got {type(interaction).__name__}.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._camera_indices(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        indices = self._camera_indices(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(indices)
        result: dict[str, Any] = {"actions": indices}
        if indices:
            result["start_camera_idx"] = min(indices)
            result["end_camera_idx"] = max(indices) + 1
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
        del ref_image_path
        if images is not None:
            raise ValueError("AC3D accepts a source video, not an image input.")
        return {
            "images": None,
            "video": video,
            "extra_inputs": dict(kwargs),
        }


__all__ = ["AC3DOperator"]
