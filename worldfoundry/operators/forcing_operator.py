"""Module for the _BaseForcing operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class _BaseForcingOperator(BaseOperator):
    """Shared prompt/image normalization for forcing-family video methods."""

    MODEL_ID = ""
    DISPLAY_NAME = ""
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
        if text is None or not str(text).strip():
            raise ValueError(f"{self.DISPLAY_NAME} requires a non-empty text prompt.")
        return {"prompt": str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if video is not None:
            raise ValueError(f"{self.DISPLAY_NAME} accepts prompt and optional image input, not direct video input.")
        image_input = images if images is not None else ref_image_path
        return {
            "images": image_input,
            "video": None,
            "extra_inputs": dict(kwargs),
        }


class RollingForcingOperator(_BaseForcingOperator):
    """Operator contract for RollingForcing text-to-video inference."""

    MODEL_ID = "rolling-forcing"
    DISPLAY_NAME = "RollingForcing"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": False,
        "video": False,
        "actions": ["seed"],
    }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if images is not None or ref_image_path is not None or video is not None:
            raise ValueError("RollingForcing's released inference route accepts text only.")
        return {"images": None, "video": None, "extra_inputs": dict(kwargs)}


__all__ = ["RollingForcingOperator"]
