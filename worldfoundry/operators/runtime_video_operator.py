"""Module for the RuntimeVideo operator implementation."""

from __future__ import annotations

from typing import Any

from .base_operator import BaseOperator


class RuntimeVideoOperator(BaseOperator):
    """Shared prompt/image operator for individually wrapped local video models."""

    def __init__(self, generation_type: str = "i2v"):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction"])
        self.generation_type = generation_type
        self.interaction_template = ["text_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, str):
            raise TypeError(f"Prompt must be a string, got {type(interaction)}")
        return True

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No prompt to process")
        prompt = self.current_interaction[-1]
        self.interaction_history.append(prompt)
        return {"processed_prompt": prompt}

    def process_perception(self, images=None, **kwargs) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        del kwargs
        if self.generation_type == "t2v":
            return {"images": None}
        return {"images": images}
