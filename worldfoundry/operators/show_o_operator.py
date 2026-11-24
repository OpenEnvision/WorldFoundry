"""Module for the ShowO operator implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class ShowOOperator(BaseOperator):
    """Operator for Show-o multimodal generation and understanding.

    Official entry points cover text-to-image, multimodal understanding, and
    inpainting/extrapolation modes. WorldFoundry keeps the mode as interaction.
    """

    MODEL_ID = "show-o"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {"prompt": True, "image": True, "video": False, "actions": []}

    def __init__(self, input_schema: dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["t2i", "mmu", "inpainting", "extrapolation"]
        self.interaction_template_init()

    @staticmethod
    def _modes(interaction) -> list[str]:
        if interaction is None:
            return []
        if isinstance(interaction, str):
            return [interaction.strip()] if interaction.strip() else []
        if isinstance(interaction, Iterable):
            return [str(item).strip() for item in interaction if str(item).strip()]
        raise TypeError(f"Show-o mode interaction must be a string or sequence, got {type(interaction).__name__}.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        allowed = {"t2i", "mmu", "inpainting", "extrapolation", "mixed_modality"}
        for mode in self._modes(interaction):
            if mode not in allowed:
                raise ValueError(f"Unsupported Show-o mode {mode!r}; expected one of {sorted(allowed)}.")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        modes = self._modes(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(modes)
        result: dict[str, Any] = {"actions": modes}
        if modes:
            result["mode"] = modes[0]
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        question = kwargs.get("question")
        text = prompt if prompt is not None else question
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
            raise ValueError("Show-o official inference does not expose video conditioning in this integration.")
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
