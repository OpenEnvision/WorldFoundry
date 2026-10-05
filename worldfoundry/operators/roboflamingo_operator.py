"""Module for the RoboFlamingo operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class RoboFlamingoOperator(EmbodiedActionOperator):
    """Operator for RoboFlamingo-style VLM policies with visual history."""

    MODEL_ID = "roboflamingo"
    POLICY_FAMILY = "flamingo_vlm_robot_policy"
    ACTION_REPRESENTATION = "continuous_end_effector_action"
    OBSERVATION_LAYOUT = "interleaved_visual_history_language_proprio"
    DEFAULT_INPUT_SCHEMA = {
        "prompt": True,
        "image": True,
        "video": True,
        "proprio": True,
        "actions": ["eef_action", "gripper_action"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "task_instruction")
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "multimodal_template": kwargs.get("multimodal_template", "flamingo_interleaved"),
            },
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        visual_history = _first_present(
            {"video": video, "image_sequence": kwargs.get("image_sequence"),
             "visual_history": kwargs.get("visual_history"), "images": images},
            "video", "image_sequence", "visual_history", "images",
        )
        observation = _compact_dict(
            {
                "visual_history": visual_history,
                "current_image": images,
                "gripper_image": _first_present(kwargs, "gripper_image", "wrist_image", "gripper"),
                "proprio": _first_present(kwargs, "proprio", "robot_state"),
                "gripper_state": kwargs.get("gripper_state"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "roboflamingo_observation": observation,
                "roboflamingo_input_contract": {
                    "modalities": ["interleaved_visual_history", "language", "proprio"],
                    "vlm_backbone": "flamingo_cross_attention",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "eef_action", "gripper_action", "robot_action")
        self.interaction_history.append(actions)
        return {
            **self.action_payload(
                actions,
                action_contract="continuous_eef_and_gripper_action",
                control_mode="vlm_policy_action_head",
                uses_visual_history=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous", "dimensions": 7},
            "policy_controls": {
                "vlm_backbone": "flamingo_cross_attention",
                "conditioning": "interleaved_visual_language",
            },
        }
