"""Module for the Wan2p2 operator implementation."""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .base_operator import BaseOperator


def _load_input_image(input_path: str | Path | Image.Image) -> Image.Image:
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    img = Image.open(p).convert("RGB")
    return img


class Wan2p2Operator(BaseOperator):
    """Prepare prompts and input images for Wan2.2 TI2V.

    - process_interaction: returns the prompt, optionally extended by a prompt expander.
    - process_perception: loads the optional input image.
    - The operator creates no generation model and runs no inference.
    """

    def __init__(self, operation_types=None) -> None:
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super().__init__(operation_types=operation_types)

        self.interaction_template = ["text_prompt", "image_prompt"]
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

    def process_interaction(
        self,
        *,
        mode: str,
        images: Image.Image | None = None,
        # Prompt extension settings; they apply only to ti2v modes.
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "local_qwen",
        prompt_extend_model: str | None = None,
        prompt_extend_target_lang: str = "zh",
        # Device of the prompt expander, chosen by the caller instead of a fixed cuda:0.
        prompt_extend_device: int | str = 0,
        base_seed: int = -1,
        **kwargs,
    ) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]

        self.interaction_history.append(prompt)

        if "ti2v" in mode and use_prompt_extend:
            logging.info("Extending prompt ...")
            from ..base_models.diffusion_model.models.encoders.wan.prompt_expander import (
                DashScopePromptExpander,
                QwenPromptExpander,
            )

            if prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model,
                    mode=mode,
                    is_vl=images is not None,
                )
            elif prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    mode=mode,
                    is_vl=images is not None,
                    device=prompt_extend_device,
                )
            else:
                raise NotImplementedError(f"Unsupport prompt_extend_method: {prompt_extend_method}")

            prompt_output = prompt_expander(
                prompt,
                image=images,
                tar_lang=prompt_extend_target_lang,
                seed=base_seed,
            )
            if prompt_output.status is False:
                logging.info(f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
            else:
                prompt = prompt_output.prompt

            logging.info(f"Extended prompt: {prompt}")

        return {
            "processed_prompt": prompt,
        }

    def process_perception(
        self,
        *,
        input_path: str | Path | Image.Image | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Load the optional input image.

        ti2v accepts a missing reference image, in which case ``input_image`` is None;
        otherwise ``input_path`` is loaded as an RGB PIL image.
        """
        input_image = None if input_path is None else _load_input_image(input_path)

        return {
            "input_image": input_image,
        }
