"""Module for the OpenPI operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class OpenPIOperator(EmbodiedActionOperator):
    """Operator for OpenPI/pi0-style flow-matching action policies."""

    MODEL_ID = "openpi"
    POLICY_FAMILY = "flow_matching_vision_language_action_policy"
    ACTION_REPRESENTATION = "continuous_action_chunk"
    OBSERVATION_LAYOUT = "multi_view_rgb_language_proprio"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "proprio": True,
        "actions": ["action_chunk", "proprio"],
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
                "task_embedding_key": kwargs.get("task_embedding_key", "language"),
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
                "rgb_views": images,
                "proprio": _first_present(kwargs, "proprio", "robot_state", "joint_state"),
                "camera_names": kwargs.get("camera_names"),
                "observation_window": kwargs.get("observation_window"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "openpi_observation": observation,
                "openpi_input_contract": {
                    "modalities": ["multi_view_rgb", "language", "proprio"],
                    "sampler": "flow_matching",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_chunk", "trajectory")
        self.interaction_history.append(actions)
        horizon = raw.get("action_horizon") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="chunked_continuous_control",
                control_mode="flow_denoised_action_chunk",
                action_horizon=horizon or self.input_schema.get("action_horizon") or "model_default",
                chunked=True,
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_chunk"},
            "policy_controls": {
                "sampler": "flow_matching",
                "normalization": "dataset_statistics",
                "supports_bimanual": True,
            },
        }
