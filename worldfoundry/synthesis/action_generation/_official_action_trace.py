from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def precomputed_actions(observation: Mapping[str, Any], action_context: Sequence[Any]) -> Any:
    value = first_present(
        observation,
        "actions",
        "action_chunk",
        "robot_action",
        "trajectory",
        "precomputed_actions",
    )
    if value not in (None, ""):
        return value
    if action_context:
        return list(action_context)
    return None


def checkpoint_gated_action_trace(
    *,
    model_id: str,
    instruction: str,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    official_entrypoint: str,
    required_port: str,
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    actions = precomputed_actions(observation, action_context)
    if actions is not None:
        return {
            "status": "completed_from_supplied_actions",
            "actions": actions,
            "instruction": instruction,
            "model_id": model_id,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "input_contract": dict(input_contract),
            "official_entrypoint": official_entrypoint,
        }

    return {
        "status": "blocked",
        "blocked_reason": required_port,
        "actions": [],
        "instruction": instruction,
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "input_contract": dict(input_contract),
        "official_entrypoint": official_entrypoint,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def unported_checkpoint_runtime(
    *,
    model_id: str,
    instruction: str,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    official_entrypoint: str,
    required_port: str,
    input_contract: Mapping[str, Any],
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(runtime_options or {})
    if _truthy(options.get("allow_supplied_actions_fallback")):
        actions = precomputed_actions(observation, action_context)
        if actions is not None:
            return {
                "status": "completed_from_supplied_actions",
                "actions": actions,
                "instruction": instruction,
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "device": device,
                "input_contract": dict(input_contract),
                "official_entrypoint": official_entrypoint,
            }

    details = {
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "official_entrypoint": official_entrypoint,
        "required_port": required_port,
        "input_contract": dict(input_contract),
    }
    raise RuntimeError(
        f"{model_id} has a Studio/catalog entry, but its checkpoint-backed in-tree "
        f"runtime is not implemented yet. Details: {details}"
    )
