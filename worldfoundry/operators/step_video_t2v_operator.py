"""Module for the StepVideoT2V operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base_operator import BaseOperator


class StepVideoT2VOperator(BaseOperator):
    """Operator for Step-Video-T2V text-to-video inference.

    The official runtime takes a text prompt and generation knobs such as
    parallelism, CFG, VAE URL, and caption URL. It does not consume image,
    video, or action controls.
    """

    MODEL_ID = "step-video-t2v"
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
        raise ValueError("Step-Video-T2V is prompt-only; interactions/actions are not accepted.")

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
        del kwargs
        text = "" if prompt is None else str(prompt).strip()
        if not text:
            raise ValueError("Step-Video-T2V official inference requires a non-empty prompt.")
        return {"prompt": text}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None or video is not None or ref_image_path is not None:
            raise ValueError("Step-Video-T2V does not accept image or video conditioning inputs.")
        return {"images": None, "video": None, "ref_image_path": None, "extra_inputs": dict(kwargs)}
