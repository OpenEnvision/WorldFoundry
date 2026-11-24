"""Module for the RT1 operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class RT1Operator(EmbodiedActionOperator):
    """Operator for RT-1-style discretized robotics transformer policies."""

    MODEL_ID = "rt-1"
    POLICY_FAMILY = "discretized_robotics_transformer_policy"
    ACTION_REPRESENTATION = "discretized_action_tokens"
    OBSERVATION_LAYOUT = "single_rgb_language_token_history"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["action_tokens", "tokenized_action"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "task_instruction")
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "tokenizer": kwargs.get("tokenizer", "rt1_action_tokenizer"),
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
        observation = _compact_dict(
            {
                "rgb": images,
                "history": kwargs.get("observation_history"),
                "previous_action_tokens": kwargs.get("previous_action_tokens"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "rt1_observation": observation,
                "rt1_input_contract": {
                    "modalities": ["rgb", "language"],
                    "action_vocabulary": "discretized_bins",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        tokens = _extract_action_signal(raw, "action_tokens", "tokenized_action", "discrete_action")
        self.interaction_history.append(tokens)
        return {
            **self.action_payload(
                tokens,
                action_contract="discretized_robot_action_tokens",
                control_mode="tokenized_control",
                token_bins=(raw.get("token_bins") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("token_bins"),
            ),
            "action_tokens": tokens,
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "discrete_action_tokens"},
            "policy_controls": {
                "decoder": "rt1_token_decoder",
                "uses_action_bins": True,
            },
        }
