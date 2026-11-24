"""Module for the AnimateDiff operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class AnimateDiffOperator(BaseOperator):
    """Operator for AnimateDiff prompt/config driven animation."""

    MODEL_ID = "animatediff"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {"prompt": True, "image": False, "video": False, "actions": []}

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = []
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction in (None, [], ()):
            return True
        raise ValueError("AnimateDiff integration is driven by prompt/config and does not accept action interactions.")

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(None)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        self.interaction_history.append([])
        return {"actions": []}

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        text = prompt if prompt is not None else kwargs.get("config_prompt")
        return {"prompt": "" if text is None else str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None or video is not None or ref_image_path is not None:
            raise ValueError(
                "This AnimateDiff entry is text/config-to-video; visual conditioning belongs to SparseCtrl variants."
            )
        return {"images": None, "video": None, "ref_image_path": None, "extra_inputs": dict(kwargs)}
