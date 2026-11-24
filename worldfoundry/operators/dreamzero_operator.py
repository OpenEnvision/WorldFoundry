"""Module for the DreamZero operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class DreamZeroOperator(EmbodiedActionOperator):
    """Operator for NVIDIA DreamZero world-action policies."""

    MODEL_ID = "dreamzero"
    POLICY_FAMILY = "world_action_model_zero_shot_policy"
    ACTION_REPRESENTATION = "droid_joint_gripper_action_chunk_plus_predicted_video"
    OBSERVATION_LAYOUT = "droid_three_camera_video_language_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": True,
        "world_state": True,
        "actions": [
            "world_action",
            "robot_action",
            "dreamzero_action",
            "joint_position",
            "gripper_position",
            "video_prediction",
        ],
    }
    OFFICIAL_CAMERA_KEYS: ClassVar[dict[str, str]] = {
        "exterior_image_1_left": "video.exterior_image_1_left",
        "exterior_image_2_left": "video.exterior_image_2_left",
        "wrist_image_left": "video.wrist_image_left",
    }
    OFFICIAL_ROBOARENA_KEYS: ClassVar[dict[str, str]] = {
        "exterior_image_1_left": "observation/exterior_image_0_left",
        "exterior_image_2_left": "observation/exterior_image_1_left",
        "wrist_image_left": "observation/wrist_image_left",
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt
            if prompt not in (None, "")
            else _first_present(
                kwargs,
                "instruction",
                "task_instruction",
                "action_text",
                "annotation.language.action_text",
            )
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "annotation.language.action_text": instruction,
                "model_track": "wam",
                "embodiment": kwargs.get("embodiment", "oxe_droid"),
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
        camera_views = _compact_dict(
            {
                "exterior_image_1_left": _first_present(
                    kwargs,
                    "exterior_image_1_left",
                    "observation/exterior_image_0_left",
                    "video.exterior_image_1_left",
                ),
                "exterior_image_2_left": _first_present(
                    kwargs,
                    "exterior_image_2_left",
                    "observation/exterior_image_1_left",
                    "video.exterior_image_2_left",
                ),
                "wrist_image_left": _first_present(
                    kwargs,
                    "wrist_image_left",
                    "observation/wrist_image_left",
                    "video.wrist_image_left",
                ),
            }
        )
        state = _compact_dict(
            {
                "joint_position": _first_present(kwargs, "joint_position", "observation/joint_position", "proprio"),
                "cartesian_position": _first_present(kwargs, "cartesian_position", "observation/cartesian_position"),
                "gripper_position": _first_present(kwargs, "gripper_position", "observation/gripper_position"),
                "world_state": _first_present(kwargs, "world_state", "initial_world_state", "latent_world_state"),
            }
        )
        video_context = video or _first_present(kwargs, "video_context", "droid_video", "frame_sequence")
        observation = _compact_dict(
            {
                "image": images,
                "video_context": video_context,
                "camera_views": camera_views,
                "state": state,
                "session_id": kwargs.get("session_id"),
                "frame_schedule": kwargs.get("frame_schedule"),
                "debug_video_dir": kwargs.get("debug_video_dir"),
            }
        )
        return {
            "images": images,
            "video": video_context,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "dreamzero_observation": observation,
                "dreamzero_input_contract": {
                    "modalities": ["droid_three_camera_video", "language", "joint_state", "gripper_state"],
                    "server_protocol": "roboarena_websocket",
                    "image_resolution": [180, 320],
                    "camera_key_mapping": self.OFFICIAL_CAMERA_KEYS,
                    "roboarena_key_mapping": self.OFFICIAL_ROBOARENA_KEYS,
                    "emits_actions_and_predicted_video": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(
            raw,
            "dreamzero_actions",
            "world_actions",
            "robot_actions",
            "joint_position",
            "action_chunk",
            "video_prediction_actions",
        )
        self.interaction_history.append(actions)
        action_space = raw.get("action_space") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="droid_joint_position_plus_gripper_action_chunk",
                control_mode="world_action_model_policy",
                action_horizon=(raw.get("action_horizon") if isinstance(raw, Mapping) else None) or 24,
                supports_world_action_modeling=True,
                predicts_video=True,
            ),
            "action_space": action_space
            or self.input_schema.get("action_space")
            or {"kind": "joint_position_plus_gripper", "dimensions": 8, "horizon": 24},
            "policy_controls": {
                "policy_architecture": "wan_backbone_world_action_model",
                "server_protocol": "roboarena_websocket",
                "requires_distributed_server": True,
                "minimum_gpus_official": 2,
                "embodiment": "oxe_droid",
                "output_semantics": "action_chunk_with_server_saved_video_prediction",
            },
        }
