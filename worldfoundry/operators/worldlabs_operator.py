"""Module for the WorldLabs operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ._media import pil_to_png_base64
from .base_operator import BaseOperator


def _image_to_data_base64(image_input: Image.Image) -> str:
    return pil_to_png_base64(image_input)


class WorldLabsOperator(BaseOperator):
    """Build World Labs (Marble) world prompts from text, image, multi-image or video inputs."""

    def __init__(
        self,
        operation_types: list | None = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["prompt_processing", "image_processing", "video_processing"]
        super().__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multi_image_prompt", "video_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction is not None and not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string or None, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        now_interaction = self.current_interaction[-1] if self.current_interaction else ""
        if now_interaction is not None:
            self.interaction_history.append(now_interaction)
        return {"processed_prompt": now_interaction or ""}

    def _normalize_content(self, value: object) -> dict[str, Any]:
        if isinstance(value, dict):
            return value

        if isinstance(value, Image.Image):
            return {
                "source": "data_base64",
                "data_base64": _image_to_data_base64(value),
                "extension": "png",
            }

        if isinstance(value, str):
            path = Path(value)
            if value.startswith(("http://", "https://")):
                return {
                    "source": "uri",
                    "uri": value,
                }
            if path.exists():
                return {
                    "source": "local_file",
                    "path": str(path.resolve()),
                }
            raise ValueError("String image/video input must be a public URL or an existing local path.")

        raise TypeError(f"Unsupported content type: {type(value)}")

    def _build_multi_image_prompt(
        self,
        images: list[Image.Image | str | dict[str, Any]],
        image_azimuths: list[float] | None,
    ) -> list[dict[str, Any]]:
        if image_azimuths is not None and len(image_azimuths) != len(images):
            raise ValueError("image_azimuths length must match images length.")

        default_azimuths = [index * (360 / len(images)) for index in range(len(images))]
        azimuths = image_azimuths or default_azimuths

        return [
            {
                "azimuth": azimuths[index],
                "content": self._normalize_content(image),
            }
            for index, image in enumerate(images)
        ]

    def process_perception(
        self,
        images: Image.Image | str | dict[str, Any] | list[Image.Image | str | dict[str, Any]] | None = None,
        video: str | dict[str, Any] | None = None,
        prompt_type: str = "auto",
        image_azimuths: list[float] | None = None,
        disable_recaption: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        result: dict[str, Any] = {
            "world_prompt": None,
            "prompt_type": prompt_type,
            "images": images,
            "video": video,
        }

        if prompt_type == "auto":
            if video is not None:
                prompt_type = "video"
            elif isinstance(images, list):
                prompt_type = "multi-image"
            elif images is not None:
                prompt_type = "image"
            else:
                prompt_type = "text"

        prompt_data: dict[str, Any]
        if prompt_type == "text":
            prompt_data = {
                "type": "text",
            }
        elif prompt_type == "image":
            if images is None:
                raise ValueError("image prompt_type requires images input.")
            prompt_data = {
                "type": "image",
                "image_prompt": self._normalize_content(images),
            }
        elif prompt_type in {"multi-image", "multi_image"}:
            if not isinstance(images, list) or len(images) == 0:
                raise ValueError("multi-image prompt_type requires a non-empty images list.")
            prompt_data = {
                "type": "multi-image",
                "multi_image_prompt": self._build_multi_image_prompt(images, image_azimuths),
            }
        elif prompt_type == "video":
            if video is None:
                raise ValueError("video prompt_type requires video input.")
            prompt_data = {
                "type": "video",
                "video_prompt": self._normalize_content(video),
            }
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        # Text prompts carry no recaption flag.
        if prompt_type != "text" and disable_recaption is not None:
            prompt_data["disable_recaption"] = disable_recaption
        result["world_prompt"] = prompt_data
        result["prompt_type"] = prompt_type
        return result
