"""Module for the Kairos operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class KairosOperator(BaseOperator):
    """Operator for Kairos Sensenova text/image/video generation modes."""

    MODEL_ID = "kairos-sensenova"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["seed"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["seed"]
        self.interaction_template_init()

    @staticmethod
    def _actions(interaction: Any) -> list[Any]:
        if interaction is None:
            return []
        if isinstance(interaction, (str, int)):
            return [interaction]
        if isinstance(interaction, Iterable) and not isinstance(interaction, (bytes, bytearray, Path)):
            return list(interaction)
        return [interaction]

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._actions(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        actions = self._actions(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(actions)
        result: dict[str, Any] = {"actions": actions}
        if actions and isinstance(actions[0], int):
            result["seed"] = int(actions[0])
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
            raise ValueError("Kairos Sensenova accepts prompt and optional image input, not direct video input.")
        image_input = images if images is not None else ref_image_path
        return {
            "images": image_input,
            "video": None,
            "extra_inputs": dict(kwargs),
        }


__all__ = ["KairosOperator"]
