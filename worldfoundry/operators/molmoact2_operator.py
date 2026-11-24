"""Module for the MolmoAct2 operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present

_CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "external_cam": ("external_cam", "scene_cam", "third_person_cam", "image"),
    "external_cam_2": ("external_cam_2", "exterior_2_cam", "scene_cam_2", "third_person_cam_2"),
    "wrist_cam": ("wrist_cam", "wrist_image"),
    "top_cam": ("top_cam", "front_camera_rgb", "front_cam"),
    "side_cam": ("side_cam", "side_camera_rgb"),
    "agentview_cam": ("agentview_cam", "agentview_rgb", "front_rgb", "agent_view"),
    "left_cam": ("left_cam", "left_camera_rgb"),
    "right_cam": ("right_cam", "right_camera_rgb"),
}


def _camera_mapping_from_kwargs(camera_keys: tuple[str, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for camera_key in camera_keys:
        for alias in _CAMERA_ALIASES.get(camera_key, (camera_key,)):
            if alias in kwargs and kwargs[alias] is not None:
                cameras[camera_key] = kwargs[alias]
                break
    return cameras


class MolmoAct2Operator(EmbodiedActionOperator):
    """Operator for MolmoAct2 action-reasoning VLA policies."""

    MODEL_ID = "molmoact2"
    POLICY_FAMILY = "flow_matching_action_reasoning_vla"
    ACTION_REPRESENTATION = "continuous_absolute_joint_pose_action_chunk"
    OBSERVATION_LAYOUT = "embodiment_specific_multiview_rgb_language_state"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": False,
        "state": True,
        "camera_keys": ["external_cam", "external_cam_2", "wrist_cam"],
        "norm_tag": "franka_droid",
        "actions": ["action_chunk", "robot_action", "actions"],
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
                "language_instruction",
                "task",
            )
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "normalize_language": bool(kwargs.get("normalize_language", True)),
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
        camera_keys = tuple(
            str(item) for item in kwargs.get("camera_keys") or self.input_schema.get("camera_keys") or ()
        )
        if camera_keys:
            cameras = _camera_mapping_from_kwargs(camera_keys, kwargs)
            if cameras and (images is None or len(camera_keys) > 1):
                images = cameras
        state = _first_present(kwargs, "state", "robot_state", "joint_state", "proprio", "joint_positions")
        embodiment = _first_present(kwargs, "embodiment", "robot_type", "variant_id", "variant")
        norm_tag = _first_present(kwargs, "norm_tag", "normalization_tag") or self.input_schema.get("norm_tag")
        observation = _compact_dict(
            {
                "images": images,
                "state": state,
                "camera_keys": list(camera_keys),
                "embodiment": embodiment,
                "norm_tag": norm_tag,
                "timestamp": kwargs.get("timestamp"),
                "sample_id": kwargs.get("sample_id"),
            }
        )
        extra_inputs = {
            **kwargs,
            "molmoact2_observation": observation,
            "molmoact2_input_contract": {
                "modalities": ["multiview_rgb", "language", "robot_state"],
                "action_representation": self.ACTION_REPRESENTATION,
                "policy_backend": "hf_predict_action",
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
        actions = _extract_action_signal(raw, "action_chunk", "robot_action")
        self.interaction_history.append(actions)
        horizon = raw.get("action_horizon") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="absolute_joint_pose_action_chunk",
                control_mode="flow_matching_continuous_actions",
                action_horizon=horizon or self.input_schema.get("action_horizon") or "model_default",
                chunked=True,
                requires_language=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_chunk", "dimensions": "embodiment_specific"},
            "policy_controls": {
                "sampler": "flow_matching",
                "normalization": "norm_stats_json",
                "depth_reasoning": "optional",
            },
        }


__all__ = ["MolmoAct2Operator"]
