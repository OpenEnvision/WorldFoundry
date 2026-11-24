"""Module for the StarVLA operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class StarVLAOperator(EmbodiedActionOperator):
    """Operator for StarVLA VLA and world-action-model variants."""

    MODEL_ID = "starvla"
    POLICY_FAMILY = "vision_language_action_and_world_action_model"
    ACTION_REPRESENTATION = "robot_or_world_action_sequence"
    OBSERVATION_LAYOUT = "visual_language_world_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": True,
        "world_state": True,
        "actions": ["robot_actions", "world_actions", "latent_actions"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(kwargs, "instruction", "task_instruction", "world_model_prompt")
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "track": kwargs.get("track", self.input_schema.get("track", "vla_or_wam")),
                "world_model_prompt": kwargs.get("world_model_prompt"),
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
        world_state = _first_present(kwargs, "world_state", "initial_world_state", "latent_world_state")
        observation = _compact_dict(
            {
                "rgb": images,
                "video_context": video or kwargs.get("video_context"),
                "world_state": world_state,
                "proprio": kwargs.get("proprio"),
            }
        )
        return {
            "images": images,
            "video": video or kwargs.get("video_context"),
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "starvla_observation": observation,
                "starvla_input_contract": {
                    "modalities": ["rgb", "video_context", "language", "world_state"],
                    "supports_world_action_modeling": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "robot_actions", "world_actions", "latent_actions")
        self.interaction_history.append(actions)
        track = raw.get("track") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="robot_or_world_action_sequence",
                control_mode=track or self.input_schema.get("track") or "vla_or_wam",
                supports_world_action_modeling=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "robot_or_world_action"},
            "policy_controls": {
                "supports_vla": True,
                "supports_visual_action_model": True,
                "supports_world_action_model": True,
            },
        }
