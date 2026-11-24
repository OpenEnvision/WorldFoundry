"""Module for the Wan2p7 operator implementation."""

from typing import Any

from ._media import require_public_url
from .base_operator import BaseOperator


def _validate_public_url(url: str, field_name: str) -> str:
    return require_public_url(url, field_name, "Wan2.7")


class Wan2p7Operator(BaseOperator):
    """Prepare prompts and media inputs for Wan2.7 requests.

    Wan2.7 accepts only publicly reachable media URLs.
    """

    def __init__(
        self,
        operation_types: list | None = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["prompt_processing", "media_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["image_prompt", "video_prompt", "multimodal_prompt"]
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
        audio_url: str | None = None,
        first_clip: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None and first_clip is not None:
            raise ValueError("Wan2.7 expects either images or first_clip, not both.")

        media = []

        if images is not None:
            media.append(
                {
                    "type": "first_frame",
                    "url": _validate_public_url(images, "images"),
                }
            )

        if first_clip is not None:
            media.append(
                {
                    "type": "first_clip",
                    "url": _validate_public_url(first_clip, "first_clip"),
                }
            )

        if last_frame is not None:
            media.append(
                {
                    "type": "last_frame",
                    "url": _validate_public_url(last_frame, "last_frame"),
                }
            )

        if audio_url is not None:
            media.append(
                {
                    "type": "driving_audio",
                    "url": _validate_public_url(audio_url, "audio_url"),
                }
            )

        if not media:
            raise ValueError("Wan2.7 requires at least one media input URL.")

        if not any(item["type"] in {"first_frame", "first_clip"} for item in media):
            raise ValueError("Wan2.7 requires either a first_frame or first_clip input.")

        return {
            "media": media,
            "images": images,
            "last_frame": last_frame,
            "audio_url": audio_url,
            "first_clip": first_clip,
        }
