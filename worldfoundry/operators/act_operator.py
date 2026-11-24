"""Module for the ACT operator implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .embodied_action_operator import EmbodiedActionOperator, _compact_dict, _extract_action_signal, _first_present


class ACTOperator(EmbodiedActionOperator):
    """Operator for action-chunking transformer policies."""

    MODEL_ID = "act"
    POLICY_FAMILY = "action_chunking_transformer_policy"
    ACTION_REPRESENTATION = "chunked_action_sequence"
    OBSERVATION_LAYOUT = "rgb_proprio_action_chunk_context"
    DEFAULT_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "prompt": False,
        "image": True,
        "video": False,
        "proprio": True,
        "actions": ["action_chunk", "query_action_chunk"],
    }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        task_label = prompt if prompt not in (None, "") else _first_present(kwargs, "task", "task_name", "instruction")
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
                "rgb": images,
                "proprio": _first_present(kwargs, "proprio", "qpos", "joint_state"),
                "previous_action": kwargs.get("previous_action"),
                "query_frequency": kwargs.get("query_frequency") or self.input_schema.get("query_frequency"),
            }
        )
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": {
                **kwargs,
                "act_observation": observation,
                "act_input_contract": {
                    "modalities": ["rgb", "proprio"],
                    "action_head": "chunking_transformer",
                },
            },
            "observation": observation,
        }

    def process_interaction(self) -> dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        raw = self.current_interaction[-1] if self.current_interaction else None
        chunk = _extract_action_signal(raw, "action_chunk", "query_action_chunk", "actions")
        self.interaction_history.append(chunk)
        return {
            **self.action_payload(
                chunk,
                action_contract="chunked_action_sequence",
                control_mode="action_chunk_query",
                chunk_size=(raw.get("chunk_size") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("chunk_size"),
            ),
            "action_space": (raw.get("action_space") if isinstance(raw, Mapping) else None)
            or self.input_schema.get("action_space")
            or {"kind": "continuous_chunk"},
            "policy_controls": {
                "architecture": "action_chunking_transformer",
                "uses_cvae": True,
                "query_frequency": (raw.get("query_frequency") if isinstance(raw, Mapping) else None)
                or self.input_schema.get("query_frequency"),
            },
        }
