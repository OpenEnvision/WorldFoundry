"""Module for the LumaRay2 operator implementation."""

from typing import Any

from ._media import require_public_url
from .base_operator import BaseOperator


def _validate_public_url(url: str, field_name: str) -> str:
    return require_public_url(url, field_name, "Luma")


def _keyframe(
    url: str | None, url_field: str, generation_id: str | None, generation_field: str
) -> dict[str, Any] | None:
    """Build a Luma keyframe from a public image URL, else from a generation ID, else None."""
    if url is not None:
        return {
            "type": "image",
            "url": _validate_public_url(url, url_field),
        }
    if generation_id is None:
        return None
    if not isinstance(generation_id, str):
        raise TypeError(f"{generation_field} must be a string, got {type(generation_id)}")
    return {
        "type": "generation",
        "id": generation_id,
    }


class LumaRay2Operator(BaseOperator):
    """Prepare prompts and keyframes for Luma Ray2 requests.

    Each Luma keyframe is either a public image URL or a generation ID.
    """

    def __init__(
        self,
        operation_types: list | None = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["prompt_processing", "keyframe_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "generation_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {"processed_prompt": now_interaction}

    def process_perception(
        self,
        images: str | None = None,
        last_frame: str | None = None,
        start_generation_id: str | None = None,
        end_generation_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None and start_generation_id is not None:
            raise ValueError("Luma frame0 accepts either images or start_generation_id, not both.")
        if last_frame is not None and end_generation_id is not None:
            raise ValueError("Luma frame1 accepts either last_frame or end_generation_id, not both.")

        keyframes: dict[str, Any] = {}
        frame0 = _keyframe(images, "images", start_generation_id, "start_generation_id")
        if frame0 is not None:
            keyframes["frame0"] = frame0
        frame1 = _keyframe(last_frame, "last_frame", end_generation_id, "end_generation_id")
        if frame1 is not None:
            keyframes["frame1"] = frame1

        return {
            "keyframes": keyframes or None,
            "images": images,
            "last_frame": last_frame,
            "start_generation_id": start_generation_id,
            "end_generation_id": end_generation_id,
        }
