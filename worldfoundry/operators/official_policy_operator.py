"""Module for the OfficialPolicy operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class OfficialPolicyOperator(EmbodiedActionOperator):
    """Generic operator contract for model-specific official VLA/action runtimes."""

    MODEL_ID = "official-policy"
    POLICY_FAMILY = "official_vla_action_policy"
    ACTION_REPRESENTATION = "model_native_action_trace"
    OBSERVATION_LAYOUT = "model_native_observation"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": True,
        "image": True,
        "video": True,
        "state": True,
        "actions": ["actions", "action_chunk", "robot_action", "policy_state"],
    }

    def _schema_value(self, key: str, default: Any = None) -> Any:
        return self.input_schema.get(key, default)

    def operator_metadata(self, **overrides: Any) -> dict[str, Any]:
        """Retrieve and return metadata for the operator."""
        model_id = str(self._schema_value("model_id", self.MODEL_ID))
        family = str(self._schema_value("policy_family", self.POLICY_FAMILY))
        representation = str(self._schema_value("action_representation", self.ACTION_REPRESENTATION))
        layout = str(self._schema_value("observation_layout", self.OBSERVATION_LAYOUT))
        return _compact_dict(
            {
                "model_id": model_id,
                "operator_kind": self.OPERATOR_KIND,
                "policy_family": family,
                "action_representation": representation,
                "observation_layout": layout,
                "interaction_template": list(self.interaction_template),
                **overrides,
            }
        )

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
            )
        )
        instruction = "" if instruction is None else str(instruction)
        return {
            "prompt": instruction,
            "task_instruction": instruction,
            "prompt_channels": {
                "language_instruction": instruction,
                "policy_prompt_format": self._schema_value("prompt_format", "model_native"),
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
        observation = kwargs.get("observation") if isinstance(kwargs.get("observation"), Mapping) else {}
        state = _first_present(kwargs, "state", "robot_state", "proprio", "joint_state")
        normalized = _compact_dict(
            {
                "image": images,
                "video": video,
                "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
                "state": state,
                "observation": observation,
                "camera_names": kwargs.get("camera_names"),
                "policy_state": kwargs.get("policy_state"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "official_policy_observation": normalized,
                "official_policy_input_contract": {
                    "modalities": self._schema_value("modalities", ["image", "language", "state"]),
                    "backend": self._schema_value("runtime_backend", "model_native"),
                },
            },
            "observation": normalized,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        actions = _extract_action_signal(raw, "action_chunk", "robot_action", "trajectory", "latent_action_tokens")
        self.interaction_history.append(actions)
        action_space = raw.get("action_space") if isinstance(raw, Mapping) else None
        return {
            **self.action_payload(
                actions,
                action_contract=str(self._schema_value("action_contract", "model_native_action_trace")),
                control_mode=str(self._schema_value("control_mode", "official_policy_inference")),
                action_horizon=(raw.get("action_horizon") if isinstance(raw, Mapping) else None)
                or self._schema_value("action_horizon", "model_default"),
            ),
            "action_space": action_space or self._schema_value("action_space") or {"kind": "model_native"},
            "policy_controls": {
                "runtime_backend": self._schema_value("runtime_backend", "model_native"),
                "normalization": self._schema_value("normalization", "model_native"),
            },
        }
