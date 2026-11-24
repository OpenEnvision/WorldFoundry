"""Module for the Veo3 operator implementation."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image

from ._media import pil_to_png_data_url
from .base_operator import BaseOperator


def _image_to_data_url(image_input: Image.Image) -> tuple[str, str]:
    """Compatibility helper for explicit callers; Veo generation uses PIL images."""
    mime_type = "image/png"
    return pil_to_png_data_url(image_input), mime_type


class Veo3Operator(BaseOperator):
    """Validate prompts and PIL images that are passed directly to the Gemini Veo 3 SDK.

    The operator only preprocesses inputs; it runs no model inference and makes no API calls.
    """

    def __init__(self, operation_types: list | None = None):
        """
        Initialize the Veo 3 operator.

        Args:
            operation_types: Operation types handled by the operator.
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()

    def process_image(self, image_input: Image.Image) -> tuple[str, str]:
        """Return a data URL for legacy explicit callers; inference skips this."""
        return _image_to_data_url(image_input)

    def build_user_content(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        resolution: str,
        duration_seconds: int,
        negative_prompt: str | None = None,
        seed: int | None = None,
        person_generation: str | None = None,
        reference_images: list[Image.Image] | None = None,
        image: Image.Image | None = None,  # None for text-to-video
        last_frame: Image.Image | None = None,
        enhance_prompt: bool | None = None,
        generate_audio: bool | None = None,
        fps: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retain the historical payload for explicit callers of this helper.

        Normal Veo inference uses native SDK fields and never invokes this method.
        """
        metadata = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
            "fps": fps,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps({"instruction": prompt, "metadata": metadata}, ensure_ascii=False),
            }
        ]

        def to_data_url(value: Image.Image) -> dict[str, Any]:
            url, _ = _image_to_data_url(value)
            return {"type": "image_url", "image_url": {"url": url}}

        if image is not None:
            content.append(to_data_url(image))
            if reference_images:
                content.extend(to_data_url(item) for item in reference_images)
            if last_frame is not None:
                content.append(to_data_url(last_frame))
        return content

    def get_interaction(self, prompt: str):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(prompt):
            self.current_interaction.append(prompt)

    def check_interaction(self, prompt: str) -> bool:
        """Validate the given interaction sequence or parameters."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")
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
        prompt: str,
        *,
        images: Image.Image | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: str | None = None,
        seed: int | None = None,
        last_frame: Image.Image | None = None,
        reference_images: list[Image.Image] | None = None,
        person_generation: str | None = None,
        enhance_prompt: bool | None = None,
        generate_audio: bool | None = None,
        fps: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Validate image inputs and collect the Veo generation settings.

        Args:
            prompt: Text prompt.
            images: Optional PIL first-frame image.
            aspect_ratio: Aspect ratio such as ``"16:9"``.
            resolution: Resolution such as ``"720p"``.
            duration_seconds: Video duration in seconds.
            seed: Random seed.
            last_frame: Optional PIL last-frame image.
            reference_images: Optional list of PIL reference images.
            person_generation: Optional person-generation setting.
            **kwargs: Ignored.

        Returns:
            Dict with ``images``, ``reference_images``, ``last_frame`` and ``veo_config``, the
            generation settings keyed by name.

        Raises:
            TypeError: An image input is not a PIL image.
        """
        if images is not None and not isinstance(images, Image.Image):
            raise TypeError(f"images must be PIL.Image, got {type(images)}")
        if last_frame is not None and not isinstance(last_frame, Image.Image):
            raise TypeError(f"last_frame must be PIL.Image, got {type(last_frame)}")
        if reference_images:
            for img in reference_images:
                if not isinstance(img, Image.Image):
                    raise TypeError(f"reference_images must be List[PIL.Image], got {type(img)}")

        return {
            "images": images,
            "reference_images": reference_images,
            "last_frame": last_frame,
            "veo_config": {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration_seconds": duration_seconds,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "person_generation": person_generation,
                "enhance_prompt": enhance_prompt,
                "generate_audio": generate_audio,
                "fps": fps,
            },
        }
