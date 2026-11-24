"""Module for the Wan2p6 operator implementation."""

from typing import Any

from PIL import Image

from ._media import pil_to_png_data_url
from .base_operator import BaseOperator


class Wan2p6Operator(BaseOperator):
    """Prepare prompts, encoded images and validated reference media for Wan2.6 requests."""

    def __init__(
        self,
        operation_types: list | None = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing", "reference_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "reference_prompt"]
        self.interaction_template_init()

    def process_image(self, image_input: Image.Image) -> str:
        """Process and normalize input image frames."""
        return pil_to_png_data_url(image_input)

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

    def _validate_reference_urls(
        self,
        reference_urls: list[str] | None,
    ) -> list[str] | None:
        if reference_urls is None:
            return None

        if not isinstance(reference_urls, list):
            raise TypeError(f"reference_urls must be a list, got {type(reference_urls)}")

        for item in reference_urls:
            if not isinstance(item, str):
                raise TypeError(f"reference_urls items must be strings, got {type(item)}")
            if not item.startswith(("http://", "https://")):
                raise ValueError("Wan2.6 reference_urls only support public HTTP/HTTPS URLs.")
        return reference_urls

    def process_perception(
        self,
        images: Image.Image | str | None = None,
        reference_urls: list[str] | None = None,
        audio_url: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        result: dict[str, Any] = {
            "encoded_image": None,
            "images": None,
            "reference_urls": self._validate_reference_urls(reference_urls),
            "audio_url": audio_url,
        }

        if audio_url is not None and not isinstance(audio_url, str):
            raise TypeError(f"audio_url must be a string, got {type(audio_url)}")

        if images is not None:
            if isinstance(images, Image.Image):
                result["encoded_image"] = self.process_image(images)
            elif isinstance(images, str):
                result["encoded_image"] = images
            else:
                raise TypeError(f"images must be PIL.Image or string, got {type(images)}")
            result["images"] = images

        return result
