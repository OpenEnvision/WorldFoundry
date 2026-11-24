"""Checkpoint-compatible prompt formatting for MolmoBot action inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import BaseConfig


@dataclass
class DataFormatter(BaseConfig):
    """The released MolmoBot prompt schema, restricted to inference prompts.

    Released checkpoints serialize the full upstream formatter configuration.
    The fields are retained so OmegaConf can load those checkpoints, while the
    executable path intentionally supports only the ``demo`` policy style used
    by MolmoBot action inference.
    """

    prompt_templates: str = "uber_model_v2"
    message_format: str = "qwen3"
    system_prompt: str = "demo_or_style_v2"
    always_start_with_space: bool = False
    default_inference_len: int | None = 65
    select_answer: str = "best"
    debug: bool = False
    image_last: bool = False
    format_message_list: str | None = None
    p_one_message: float = 0.0
    eval_system_prompt_mapping: dict[str, str] | None = None
    p_choice_content_in_mc: float = 1.0
    template_video_mc_questions: bool = True
    pointing_format: str = "html-v2"
    points_decimal_places: int = 1
    use_seperate_non_pointing_qa_style: bool = False
    timestamp_mode: str = "50-percent-seconds"
    output_timestamp_mode: str = "seconds"
    seconds_decimal_places: int = 1
    p_multi_point_all_image: float = 0.5
    use_seperate_count_without_pointing_style: bool = False
    sample_random_initial_point: bool = True

    @classmethod
    def update_legacy_settings(cls, config):
        return config

    def __call__(
        self,
        example: dict[str, Any],
        is_training: bool,
        for_inference: bool,
        rng: Any,
    ) -> tuple[list[str], dict[str, Any]]:
        del rng
        if is_training or not for_inference:
            raise ValueError("MolmoBot's in-tree DataFormatter supports inference only.")
        if example.get("style") != "demo":
            raise ValueError(f"MolmoBot action inference expects style='demo', got {example.get('style')!r}.")
        if self.prompt_templates not in {"uber_model", "uber_model_v2", "none"}:
            raise ValueError(f"Unsupported prompt_templates value: {self.prompt_templates!r}")
        if self.system_prompt not in {"demo_or_style", "demo_or_style_v2", "no_style", "none"}:
            raise ValueError(f"Unsupported system_prompt value: {self.system_prompt!r}")

        prompt = str(example.get("question") or example.get("prompt") or "")
        if self.image_last and ("image" in example or "video" in example):
            prompt += "<|image|>"
        if self.message_format == "qwen3":
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif self.message_format == "role":
            prompt = f"User: {prompt} Assistant:"
        elif self.message_format not in {"none", None}:
            raise ValueError(f"Unsupported message_format value: {self.message_format!r}")
        elif self.always_start_with_space:
            prompt = " " + prompt
        return [prompt], {}


__all__ = ["DataFormatter"]
