"""Module for the BeingH05 operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class BeingH05Operator(EmbodiedActionOperator):
    """Operator for Being-H0.5 cross-embodiment VLA policies."""

    MODEL_ID = "being-h05"
    POLICY_FAMILY = "beingh05_cross_embodiment_vla_policy"
    ACTION_REPRESENTATION = "unified_200d_action_space_with_robot_specific_slices"
    OBSERVATION_LAYOUT = "multi_view_rgb_state_language"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": False,
        "state": True,
        "actions": ["robot_action", "action_unified", "eef_delta", "gripper", "base_velocity"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(kwargs, "instruction", "task_instruction", "language_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        instruction_template = str(
            _first_present(kwargs, "instruction_template", "prompt_template")
            or self.input_schema.get("instruction_template")
            or "{task_description}"
        )
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "instruction_template": instruction_template,
                "policy_prompt_format": "beingh_task_description_template",
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
        del video
        top_view = _first_present(kwargs, "video.top_view", "top_view", "image_top", "rgb_top", "rgb")
        wrist_view = _first_present(kwargs, "video.wrist_view", "wrist_view", "image_wrist", "rgb_wrist")
        left_view = _first_present(kwargs, "video.left_view", "left_view", "image_left", "rgb_left")
        right_view = _first_present(kwargs, "video.right_view", "right_view", "image_right", "rgb_right")
        if images is not None:
            top_view = top_view if top_view is not None else images

        state = _compact_dict(
            {
                "eef_position": _first_present(kwargs, "state.eef_position", "eef_position", "proprio", "robot_state"),
                "eef_rotation": _first_present(kwargs, "state.eef_rotation", "eef_rotation"),
                "libero_gripper_position": _first_present(kwargs, "state.libero_gripper_position", "gripper_position"),
                "base_position": _first_present(kwargs, "state.base_position", "base_position"),
                "base_velocity": _first_present(kwargs, "state.base_velocity", "base_velocity"),
            }
        )
        observations = _compact_dict(
            {
                "video.top_view": top_view,
                "video.wrist_view": wrist_view,
                "video.left_view": left_view,
                "video.right_view": right_view,
                "state": state,
                "embodiment_tag": kwargs.get("embodiment_tag") or kwargs.get("embodiment"),
                "data_config_name": kwargs.get("data_config_name"),
                "dataset_name": kwargs.get("dataset_name"),
            }
        )
        return {
            "images": images,
            "video": None,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "being_h05_observation": observations,
                "being_h05_input_contract": {
                    "modalities": ["multi_view_rgb", "language", "robot_state"],
                    "data_configs": ["libero_nonorm", "robocasa_human"],
                    "action_space_dimensions": 200,
                },
            },
            "observation": observations,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_unified", "robot_action", "eef_delta", "gripper", "base_velocity")
        self.interaction_history.append(actions)
        action_space = raw.get("action_space") if isinstance(raw, Mapping) else None
        data_config_name = raw.get("data_config_name") if isinstance(raw, Mapping) else None
        dataset_name = raw.get("dataset_name") if isinstance(raw, Mapping) else None
        embodiment_tag = raw.get("embodiment_tag") if isinstance(raw, Mapping) else None
        metadata_variant = raw.get("metadata_variant") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="beingh_unified_action_chunk",
                control_mode="cross_embodiment_vla_policy",
                action_horizon=(raw.get("action_horizon") if isinstance(raw, Mapping) else None) or 8,
                supports_real_time_chunking=True,
            ),
            "action_space": action_space
            or self.input_schema.get("action_space")
            or {
                "kind": "unified_robot_action_space",
                "dimensions": 200,
                "robot_specific_slices": {
                    "libero": ["eef_position", "eef_rotation", "gripper_position"],
                    "robocasa": ["eef_position", "eef_rotation", "gripper_position", "base_velocity"],
                },
            },
            "policy_controls": {
                "data_config_name": data_config_name or self.input_schema.get("data_config_name") or "libero_nonorm",
                "dataset_name": dataset_name or self.input_schema.get("dataset_name") or "libero_posttrain",
                "embodiment_tag": embodiment_tag or self.input_schema.get("embodiment_tag") or "libero",
                "metadata_variant": metadata_variant or self.input_schema.get("metadata_variant"),
                "normalization": "checkpoint_experiment_cfg_metadata",
                "decoder": "flow_matching_chunk_head",
            },
        }
