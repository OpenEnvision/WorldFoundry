"""Module for the WorldModelRuntime operator implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class WorldModelRuntimeOperator(BaseOperator):
    """Normalize generic world-model runtime-manifest inputs."""

    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": True,
        "actions": ["world_action", "robot_action", "interaction"],
    }

    def __init__(self, input_schema: Mapping[str, Any] | None = None, model_id: str | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.model_id = model_id or "world-model-runtime"
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = list(self.input_schema.get("actions") or ())
        self.interaction_template_init()

    @staticmethod
    def _actions(interaction: Any) -> list[Any]:
        if interaction is None:
            return []
        if isinstance(interaction, str):
            text = interaction.strip()
            return [text] if text else []
        if isinstance(interaction, Mapping):
            return [dict(interaction)]
        if isinstance(interaction, Iterable) and not isinstance(interaction, (bytes, bytearray)):
            return [item for item in interaction if item not in (None, "")]
        return [interaction]

    def check_interaction(self, interaction: Any) -> bool:
        """Validate the given interaction sequence or parameters."""
        self._actions(interaction)
        return True

    def get_interaction(self, interaction: Any) -> None:
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        actions = self._actions(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(actions)
        return {"actions": actions}

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        text = prompt if prompt is not None else kwargs.get("instruction")
        return {"prompt": "" if text is None else str(text)}

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


__all__ = ["WorldModelRuntimeOperator"]
