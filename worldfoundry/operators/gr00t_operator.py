"""Module for the GR00T operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class GR00TOperator(EmbodiedActionOperator):
    """Operator for GR00T-style embodiment-conditioned humanoid policies."""

    MODEL_ID = "gr00t"
    POLICY_FAMILY = "humanoid_foundation_policy"
    ACTION_REPRESENTATION = "embodiment_conditioned_joint_or_eef_action"
    OBSERVATION_LAYOUT = "multi_sensor_embodiment_state_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "proprio": True,
        "embodiment": True,
        "actions": ["whole_body_action", "joint_targets", "eef_targets"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "skill", "task_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "skill": kwargs.get("skill"),
                "embodiment": kwargs.get("embodiment"),
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
        embodiment = _first_present(kwargs, "embodiment", "robot_type", "embodiment_id")
        observation = _compact_dict(
            {
                "camera_views": images,
                "joint_state": _first_present(kwargs, "joint_state", "proprio", "humanoid_state"),
                "wrist_state": kwargs.get("wrist_state"),
                "embodiment": embodiment,
                "robot_description": kwargs.get("robot_description"),
                "joint_names": kwargs.get("joint_names"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "gr00t_observation": observation,
                "gr00t_input_contract": {
                    "modalities": ["rgb", "embodiment_state", "language"],
                    "requires_embodiment_descriptor": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "whole_body_action", "joint_targets", "eef_targets")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions,
                action_contract="embodiment_conditioned_control",
                control_mode="joint_or_eef_targets",
                embodiment_conditioned=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "embodiment_conditioned"},
            "policy_controls": {
                "conditioning": "embodiment_descriptor",
                "supports_humanoid": True,
                "supports_mobile_manipulation": True,
            },
        }
