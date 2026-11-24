"""Module for the LAPA operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class LAPAOperator(EmbodiedActionOperator):
    """Operator for LAPA-style visual action models with latent action tokens."""

    MODEL_ID = "lapa"
    POLICY_FAMILY = "visual_action_model"
    ACTION_REPRESENTATION = "latent_action_tokens"
    OBSERVATION_LAYOUT = "video_or_frame_sequence"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": False,
        "image": True,
        "video": True,
        "actions": ["latent_action_tokens", "token_ids"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        conditioning_text = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "caption", "instruction", "task_instruction")
        )
        conditioning_text = "" if conditioning_text is None else str(conditioning_text)
        return {
            "prompt": conditioning_text,
            "task_instruction": conditioning_text,
            "prompt_channels": {
                "optional_text_condition": conditioning_text,
                "primary_condition": "visual_context",
            },
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        video_context = video or _first_present(kwargs, "video_context", "frame_sequence", "frames")
        observation = _compact_dict(
            {
                "frames": images,
                "video_context": video_context,
                "frame_stride": kwargs.get("frame_stride"),
                "context_length": kwargs.get("context_length"),
            }
        )
        return {
            "images": images,
            "video": video_context,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "lapa_observation": observation,
                "lapa_input_contract": {
                    "modalities": ["video_or_frames"],
                    "output_contract": "action_tokens",
                    "robot_control": False,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        tokens = _extract_action_signal(raw, "latent_action_tokens", "token_ids", "action_tokens")
        self.interaction_history.append(tokens)
        return {
            **self.action_payload(
                tokens,
                action_contract="latent_visual_action_tokens",
                control_mode="token_prediction",
                robot_control=False,
            ),
            "latent_action_tokens": tokens,
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "discrete_latent_tokens"},
            "policy_controls": {
                "tokenizer": "visual_action_tokenizer",
                "rollout_unit": "latent_action_token",
                "requires_robot_calibration": False,
            },
        }
