"""Module for the GigaBrain0 operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class GigaBrain0Operator(EmbodiedActionOperator):
    """Operator for GigaBrain-0 multi-view VLA policy inference."""

    MODEL_ID = "giga-brain-0"
    POLICY_FAMILY = "world_model_powered_vision_language_action_policy"
    ACTION_REPRESENTATION = "multi_embodiment_continuous_action_chunk_with_optional_subtask_and_2d_traj"
    OBSERVATION_LAYOUT = "multi_view_rgb_optional_depth_language_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": False,
        "state": True,
        "embodiment_id": True,
        "actions": ["robot_action", "action_chunk", "continuous_action", "subtask", "trajectory_2d"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = (
            prompt if prompt not in (None, "") else _first_present(kwargs, "instruction", "task", "task_instruction")
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "task": kwargs.get("task"),
                "subtask": kwargs.get("subtask"),
                "supports_subtask_prediction": True,
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
        rgb_views = _first_present(
            kwargs,
            "giga_brain_0_rgb_views",
            "rgb_views",
            "multi_view_rgb",
            "observation_images",
            "camera_images",
        )
        if rgb_views is None:
            rgb_views = images
        robot_state = _first_present(kwargs, "state", "robot_state", "observation_state", "proprio", "joint_state")
        embodiment_id = _first_present(kwargs, "embodiment_id", "robot_type", "embodiment")
        observation = _compact_dict(
            {
                "rgb_views": rgb_views,
                "robot_state": robot_state,
                "embodiment_id": embodiment_id,
                "camera_names": kwargs.get("camera_names"),
                "image_keys": kwargs.get(
                    "image_keys",
                    [
                        "observation.images.cam_high",
                        "observation.images.cam_left_wrist",
                        "observation.images.cam_right_wrist",
                    ],
                ),
                "depth_img_prefix_name": kwargs.get("depth_img_prefix_name"),
                "action_chunk": kwargs.get("action_chunk"),
                "original_action_dim": kwargs.get("original_action_dim"),
            }
        )
        return {
            "images": rgb_views,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "giga_brain_0_observation": observation,
                "giga_brain_0_input_contract": {
                    "modalities": ["multi_view_rgb", "optional_depth", "language", "robot_state"],
                    "official_image_keys": observation.get("image_keys"),
                    "embodiments": ["agilex_cobot_magic", "agibot_g1", "agibot_world"],
                    "supports_subtask_prediction": True,
                    "supports_2d_trajectory_output": True,
                    "requires_norm_stats": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(
            raw,
            "robot_action",
            "robot_actions",
            "action_chunk",
            "continuous_action",
            "pred_action",
            "subtask",
            "trajectory_2d",
        )
        self.interaction_history.append(actions)
        action_horizon = raw.get("action_horizon") if isinstance(raw, Mapping) else None
        autoregressive = raw.get("autoregressive_mode_only") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="multi_embodiment_action_chunk",
                control_mode="continuous_action_chunk",
                action_horizon=action_horizon or self.input_schema.get("action_horizon") or "model_default",
                supports_subtask_prediction=True,
                supports_2d_trajectory_output=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_robot_action_chunk", "dimensions": "original_action_dim"},
            "policy_controls": {
                "decoder": "fast_action_or_autoregressive_action_head",
                "autoregressive_mode_only": bool(autoregressive) if autoregressive is not None else "optional",
                "normalization": "norm_stats_json",
                "delta_mask": (raw.get("delta_mask") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("delta_mask"),
            },
        }
