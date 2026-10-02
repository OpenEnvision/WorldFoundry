"""Input normalization for the WoW public pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from worldfoundry.core.media.types import VIDEO_EXTENSIONS
from worldfoundry.core.media.codecs.video import load_frames_from_video
from worldfoundry.core.utils import load_pil_image

from .base_operator import BaseOperator


def extract_first_frame(video_path: str | Path) -> Image.Image:
    """Decode the first video frame through the shared media utilities."""

    frames = load_frames_from_video(video_path, indices=(0,))
    if len(frames) != 1:
        raise RuntimeError(f"Cannot read video: {video_path}")
    return load_pil_image(frames[0])


def load_input_image(input_path: str | Path) -> Image.Image:
    """Load an image or a video's first frame."""

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return extract_first_frame(input_path)
    return load_pil_image(input_path)


class WoWOperator(BaseOperator):
    """Normalize WoW image/video observations and text interactions."""

    def __init__(self, operation_types=None) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = ["text_prompt", "image_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction: str) -> None:
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: str) -> bool:
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> dict[str, object]:
        """Process the recorded interactions and return the generated actions."""
        del kwargs
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {"processed_prompt": now_interaction}

    def process_perception(
        self,
        input_path: str | Path | Image.Image | None = None,
        **kwargs,
    ) -> dict[str, object]:
        """Process perception inputs like images, videos, and reference frames."""
        del kwargs
        if input_path is None:
            raise ValueError("input_path cannot be None")

        original_input_path = None
        if isinstance(input_path, Image.Image):
            input_image = load_pil_image(input_path)
        else:
            original_input_path = str(input_path)
            input_image = load_input_image(input_path)

        return {
            "input_image": input_image,
            "input_path": original_input_path,
        }
