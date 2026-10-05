"""Module for the EmbodiedAction operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .base_operator import BaseOperator


def _is_empty_value(value: Any) -> bool:
    """Is empty value implementation."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _as_action_list(value: Any) -> list[Any]:
    """As action list implementation."""
    if _is_empty_value(value):
        return []
    if isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _compact_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compact dict implementation."""
    return {str(key): item for key, item in value.items() if not _is_empty_value(item)}


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    """First present implementation."""
    for key in keys:
        if key in mapping and not _is_empty_value(mapping[key]):
            return mapping[key]
    return None


def _extract_action_signal(value: Any, *keys: str) -> list[Any]:
    """Extract action signal implementation."""
    if isinstance(value, Mapping):
        action_value = _first_present(
            value,
            *keys,
            "actions",
            "action",
            "action_sequence",
            "action_chunk",
            "latent_action_tokens",
            "world_actions",
        )
        return _as_action_list(action_value)
    return _as_action_list(value)


class EmbodiedActionOperator(BaseOperator):
    """Base utilities for embodied model operators.

    Concrete VLA/VA/WAM operators intentionally live in model-specific modules.
    This base class only provides shared normalization helpers and the stable
    process_* method shape expected by the unified WorldFoundry pipeline.
    """

    MODEL_ID = "embodied-action-model"
    OPERATOR_KIND = "embodied_action"
    POLICY_FAMILY = "generic_embodied_policy"
    ACTION_REPRESENTATION = "generic_action"
    OBSERVATION_LAYOUT = "generic_observation"
    DEFAULT_INPUT_SCHEMA = {
        "prompt": True,
        "image": True,
        "video": True,
        "actions": ["robot_action", "latent_action", "world_action"],
    }

    def __init__(self, input_schema: Dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = list(self.input_schema.get("actions") or [])
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        _as_action_list(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        actions = _extract_action_signal(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(actions)
        return self.action_payload(actions)

    def operator_metadata(self, **overrides: Any) -> dict[str, Any]:
        """Retrieve and return metadata for the operator."""
        return _compact_dict(
            {
                "model_id": self.MODEL_ID,
                "operator_kind": self.OPERATOR_KIND,
                "policy_family": self.POLICY_FAMILY,
                "action_representation": self.ACTION_REPRESENTATION,
                "observation_layout": self.OBSERVATION_LAYOUT,
                "interaction_template": list(self.interaction_template),
                **overrides,
            }
        )

    def action_payload(self, actions: list[Any], **metadata: Any) -> Dict[str, Any]:
        """Format processed interactions into a model action payload."""
        return {
            "actions": actions,
            "operator_metadata": self.operator_metadata(**metadata),
        }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        instruction = kwargs.pop("instruction", None)
        text = prompt if prompt not in (None, "") else instruction
        return {
            "prompt": "" if text is None else str(text),
            "task_instruction": "" if text is None else str(text),
        }

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        extra_inputs = dict(kwargs)
        for key in ("observation", "observations", "video_context", "initial_world_state"):
            if key in extra_inputs:
                if key in {"video_context"} and video is None:
                    video = extra_inputs[key]
                if key in {"observation", "observations", "initial_world_state"} and images is None:
                    images = extra_inputs[key]
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": extra_inputs,
        }


__all__ = [
    "EmbodiedActionOperator",
    "_as_action_list",
    "_compact_dict",
    "_extract_action_signal",
    "_first_present",
    "_is_empty_value",
]


def __getattr__(name: str):
    """Resolve historical VLA imports via the package registry (OR-08).

    Older callers imported concrete operators from this module. The only
    class→module map is ``worldfoundry.operators._OPERATOR_MODULES``; this
    shim proxies there and warns so the extra registry cannot drift.
    """
    import warnings

    from worldfoundry.operators import _OPERATOR_MODULES
    from worldfoundry import operators as operators_pkg

    if name not in _OPERATOR_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"Importing {name!r} from {__name__!r} is deprecated; "
        f"import it from 'worldfoundry.operators' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(operators_pkg, name)
    globals()[name] = value
    return value
