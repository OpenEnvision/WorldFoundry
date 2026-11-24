"""Module for the OpenMAGVIT2 operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class OpenMAGVIT2Operator(BaseOperator):
    """Operator for Open-MAGVIT2 class-conditional sampling.

    The official `generate.py` entry point samples ImageNet classes via
    `--classes`; prompt text is only kept as WorldFoundry metadata.
    """

    MODEL_ID = "open-magvit2"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": False,
        "video": False,
        "actions": ["class_id"],
    }

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["class_id"]
        self.interaction_template_init()

    @staticmethod
    def _class_ids(interaction) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, (int, float)):
            return [str(int(interaction))]
        if isinstance(interaction, str):
            text = interaction.strip()
            if not text:
                return []
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
            return [text]
        if isinstance(interaction, Iterable):
            return [
                str(int(item)) if isinstance(item, (int, float)) else str(item).strip()
                for item in interaction
                if str(item).strip()
            ]
        raise TypeError(f"Open-MAGVIT2 class_id must be int, string, or sequence, got {type(interaction).__name__}.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        for class_id in self._class_ids(interaction):
            if not class_id.isdigit():
                raise ValueError(f"Open-MAGVIT2 class_id values must be integer labels, got {class_id!r}.")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        classes = self._class_ids(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(classes)
        result: dict[str, Any] = {"actions": classes}
        if classes:
            result["class_id"] = classes[0] if len(classes) == 1 else ",".join(classes)
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        class_id = kwargs.get("class_id")
        if class_id is not None:
            self.check_interaction(class_id)
        return {"prompt": "" if prompt is None else str(prompt)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None or video is not None or ref_image_path is not None:
            raise ValueError("Open-MAGVIT2 generation is class-conditional and does not accept visual inputs.")
        return {"images": None, "video": None, "ref_image_path": None, "extra_inputs": dict(kwargs)}
