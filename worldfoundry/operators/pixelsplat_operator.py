"""Module for the PixelSplat operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class PixelSplatOperator(BaseOperator):
    """Operator for pixelSplat dataset/index conditioned 3D reconstruction."""

    MODEL_ID = "pixelsplat"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": False,
        "image": True,
        "video": True,
        "actions": ["view_sequence"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["view_sequence", "evaluation_index"]
        self.interaction_template_init()

    @staticmethod
    def _views(interaction) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, (str, Path)):
            text = str(interaction).strip()
            return [text] if text else []
        if isinstance(interaction, Iterable):
            return [str(item).strip() for item in interaction if str(item).strip()]
        raise TypeError(f"pixelSplat interaction must be view/index paths, got {type(interaction).__name__}.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._views(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        views = self._views(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(views)
        result: dict[str, Any] = {"actions": views}
        if views:
            result["view_sequence"] = views
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        del kwargs
        return {"prompt": "" if prompt is None else str(prompt)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
