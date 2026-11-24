"""Module for the KlingApi operator implementation."""

from typing import Any

from PIL import Image

from ._media import image_or_string_to_png_data_url
from .base_operator import BaseOperator


def _image_to_data_url(image_input: Image.Image | str) -> str:
    return image_or_string_to_png_data_url(image_input, load_existing_path=True)


class KlingApiOperator(BaseOperator):
    """Prepare prompts and image payloads for Kling API text-to-video and image-to-video requests."""

    def __init__(
        self,
        operation_types: list | None = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
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
        self, images: Image.Image | str | None = None, image_field: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        result: dict[str, Any] = {
            "image_payload": None,
            "images": None,
        }

        if images is None:
            return result

        if isinstance(images, str) and images.startswith(("http://", "https://")):
            value = images
        else:
            value = _image_to_data_url(images)
        result["image_payload"] = {
            "field": image_field or "image_url",
            "value": value,
        }
        result["images"] = images
        return result
