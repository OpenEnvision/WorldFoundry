"""Module for the LingBotVA operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class LingBotVAOperator(EmbodiedActionOperator):
    """Operator for LingBot-VA video-action world-model and policy rollouts."""

    MODEL_ID = "lingbot-va"
    POLICY_FAMILY = "autoregressive_video_action_world_model"
    ACTION_REPRESENTATION = "continuous_robot_action_chunk_with_video_latents"
    OBSERVATION_LAYOUT = "multi_view_rgb_language_robot_state_video_context"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "instruction": True,
        "image": True,
        "video": True,
        "state": True,
        "actions": ["robot_actions", "action_chunk", "video_actions", "world_actions", "proprio"],
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
                "model_track": kwargs.get("track", self.input_schema.get("track", "va_vam_wam")),
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
        rgb_views = _first_present(kwargs, "lingbot_va_rgb_views", "rgb_views", "multi_view_rgb")
        if rgb_views is not None:
            images = rgb_views
        video_context = video or _first_present(kwargs, "video_context", "frame_sequence", "frames")
        robot_state = _first_present(kwargs, "robot_state", "state", "proprio", "joint_state")
        observation = _compact_dict(
            {
                "rgb_views": images,
                "video_context": video_context,
                "robot_state": robot_state,
                "camera_names": kwargs.get("camera_names"),
                "obs_cam_keys": kwargs.get("obs_cam_keys"),
                "frame_chunk_size": kwargs.get("frame_chunk_size"),
                "action_per_frame": kwargs.get("action_per_frame"),
            }
        )
        return {
            "images": images,
            "video": video_context,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "lingbot_va_observation": observation,
                "lingbot_va_input_contract": {
                    "modalities": ["multi_view_rgb", "video_context", "language", "robot_state"],
                    "official_modes": ["server", "i2va"],
                    "supports_libero": True,
                    "supports_robotwin": True,
                    "emits_actions_not_plain_video": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(
            raw, "robot_actions", "action_chunk", "video_actions", "world_actions", "state"
        )
        self.interaction_history.append(actions)
        action_horizon = raw.get("action_horizon") if isinstance(raw, Mapping) else None
        track = raw.get("track") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="interleaved_video_action_sequence",
                control_mode=track or self.input_schema.get("track") or "video_action_rollout",
                action_horizon=action_horizon or self.input_schema.get("action_horizon") or "model_default",
                supports_world_action_modeling=True,
                chunked=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_robot_action_chunk", "dimensions": "checkpoint_defined"},
            "policy_controls": {
                "sampler": "autoregressive_diffusion",
                "rollout_unit": "video_action_chunk",
                "supports_visual_prediction": True,
                "supports_robot_control": True,
                "runtime_modes": ["websocket_server", "i2va_generation"],
            },
        }
