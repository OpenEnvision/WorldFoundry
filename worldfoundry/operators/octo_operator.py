"""Module for the Octo operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class OctoOperator(EmbodiedActionOperator):
    """Operator for Octo-style generalist robot transformer policies."""

    MODEL_ID = "octo"
    POLICY_FAMILY = "generalist_robot_transformer_policy"
    ACTION_REPRESENTATION = "task_conditioned_action_chunk"
    OBSERVATION_LAYOUT = "windowed_rgb_goal_language_proprio"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": False,
        "goal_image": True,
        "proprio": True,
        "actions": ["action_chunk", "action_sequence"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        task = prompt if prompt not in (None, "") else _first_present(kwargs, "task", "instruction", "task_instruction")
        task = "" if task is None else str(task)
        return {
            "prompt": task,
            "task_instruction": task,
            "prompt_channels": {
                "language_task": task,
                "task_embedding": kwargs.get("task_embedding"),
                "dataset_statistics_id": kwargs.get("dataset_statistics_id"),
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
                "observation_window": kwargs.get("observation_window") or images,
                "goal_image": kwargs.get("goal_image"),
                "proprio": _first_present(kwargs, "proprio", "proprio_history", "robot_state"),
                "timestep_pad_mask": kwargs.get("timestep_pad_mask"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "octo_observation": observation,
                "octo_input_contract": {
                    "modalities": ["windowed_rgb", "language_or_goal", "proprio"],
                    "supports_goal_conditioning": True,
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_chunk", "action_sequence")
        self.interaction_history.append(actions)
        horizon = raw.get("action_horizon") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract="task_conditioned_action_chunk",
                control_mode="chunked_transformer_policy",
                action_horizon=horizon or self.input_schema.get("action_horizon") or "model_default",
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_chunk"},
            "policy_controls": {
                "policy_architecture": "transformer_policy",
                "task_conditioning": ["language", "goal_image"],
                "normalization": "dataset_statistics",
            },
        }
