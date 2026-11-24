"""Module for the OpenVLA operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class OpenVLAOperator(EmbodiedActionOperator):
    """Operator for OpenVLA-style autoregressive VLA policies."""

    MODEL_ID = "openvla"
    POLICY_FAMILY = "autoregressive_vision_language_action_policy"
    ACTION_REPRESENTATION = "continuous_7d_end_effector_delta"
    OBSERVATION_LAYOUT = "single_rgb_language_proprio"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "proprio": True,
        "actions": ["eef_delta", "gripper"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(kwargs, "instruction", "task_instruction", "language_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "policy_prompt_format": "plain_task_instruction",
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
        proprio = _first_present(kwargs, "proprio", "robot_state", "state_vector", "end_effector_pose")
        observation = _compact_dict(
            {
                "rgb": images,
                "proprio": proprio,
                "camera_names": kwargs.get("camera_names"),
                "timestep": kwargs.get("timestep"),
            }
        )
        extra_inputs = {
            **kwargs,
            "openvla_observation": observation,
            "openvla_input_contract": {
                "modalities": ["rgb", "language", "proprio"],
                "action_representation": self.ACTION_REPRESENTATION,
            },
        }
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": extra_inputs,
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "eef_delta", "robot_action")
        self.interaction_history.append(actions)
        action_space = raw.get("action_space") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="end_effector_delta_with_gripper",
                control_mode="eef_delta",
                action_horizon=1,
                requires_language=True,
            ),
            "action_space": action_space
            or self.input_schema.get("action_space")
            or {"kind": "continuous", "dimensions": 7},
            "policy_controls": {
                "decoder": "autoregressive_action_head",
                "normalization": "dataset_statistics",
            },
        }
