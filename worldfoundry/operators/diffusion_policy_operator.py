"""Module for the DiffusionPolicy operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class DiffusionPolicyOperator(EmbodiedActionOperator):
    """Operator for visuomotor diffusion-policy controllers."""

    MODEL_ID = "diffusion-policy"
    POLICY_FAMILY = "visuomotor_diffusion_policy"
    ACTION_REPRESENTATION = "denoised_action_trajectory"
    OBSERVATION_LAYOUT = "image_or_low_dim_history"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": False,
        "image": True,
        "video": True,
        "low_dim": True,
        "actions": ["trajectory", "action_sequence"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        task_label = prompt if prompt not in (None, "") else _first_present(kwargs, "task", "task_name")
        task_label = "" if task_label is None else str(task_label)
        return {
            "prompt": task_label,
            "task_instruction": task_label,
            "prompt_channels": {
                "task_label": task_label,
                "language_required": False,
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
                "image_history": video or images,
                "low_dim": _first_present(kwargs, "low_dim", "state", "proprio"),
                "observation_horizon": kwargs.get("observation_horizon")
                or self.input_schema.get("observation_horizon"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "diffusion_policy_observation": observation,
                "diffusion_policy_input_contract": {
                    "modalities": ["image_history_or_low_dim"],
                    "language_required": False,
                    "sampler": "action_diffusion",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        trajectory = _extract_action_signal(raw, "trajectory", "action_sequence", "actions")
        self.interaction_history.append(trajectory)
        return {
            **self.action_payload(
                trajectory,
                action_contract="denoised_action_trajectory",
                control_mode="diffusion_rollout",
                diffusion_steps=(raw.get("diffusion_steps") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("diffusion_steps"),
                action_horizon=(raw.get("action_horizon") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("action_horizon"),
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_trajectory"},
            "policy_controls": {
                "sampler": "denoising_diffusion",
                "uses_language": False,
                "predicts_action_sequence": True,
            },
        }
