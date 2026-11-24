"""Module for the Wan2p5 operator implementation."""

from typing import Any

from PIL import Image

from ._media import pil_to_png_data_url
from .base_operator import BaseOperator


class Wan2p5Operator(BaseOperator):
    """Prepare prompts and base64-encoded reference images for Wan2.5 requests.

    The operator only preprocesses inputs; it runs no model inference and makes no API calls.
    """

    def __init__(self, operation_types: list | None = None):
        """
        Initialize the Wan2.5 operator.

        Args:
            operation_types: Operation types handled by the operator.
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()

    def process_image(self, image_input: Image.Image) -> str:
        """
        Encode an image as a base64 PNG data URL.

        Args:
            image_input: Reference image.

        Returns:
            Base64 PNG data URL.
        """
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

    def process_perception(self, images: Image.Image | None = None, **kwargs) -> dict[str, Any]:
        """
        Prepare the optional reference image for a request.

        Args:
            images: Optional PIL reference image.
            **kwargs: Ignored.

        Returns:
            Dict with:
                - encoded_image: Base64 PNG data URL, or None without an image.
                - images: The reference image, or None.

        Raises:
            TypeError: ``images`` is not a PIL image.
        """
        result: dict[str, Any] = {"encoded_image": None, "images": None}

        if images is not None:
            if not isinstance(images, Image.Image):
                raise TypeError(f"images must be PIL.Image, got {type(images)}")
            result["encoded_image"] = self.process_image(images)
            result["images"] = images

        return result
