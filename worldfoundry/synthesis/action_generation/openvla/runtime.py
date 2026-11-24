"""Closed-loop callable runtime entrypoint for in-tree OpenVLA."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.embodied.adapters.runtime_bridge import first_image
from worldfoundry.synthesis.action_generation.openvla.engine import (
    OpenVLARuntime,
    OpenVLARuntimeConfig,
    select_openvla_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

_RUNTIME_CACHE: dict[tuple[Any, ...], OpenVLARuntime] = {}


def _resolve_instruction(observation: Mapping[str, Any], instruction: str) -> str:
    prompt = instruction or str(observation.get("task_description") or observation.get("language_instruction") or "")
    if not prompt:
        raise ValueError("OpenVLA requires a language instruction")
    return prompt


def _resolve_image(observation: Mapping[str, Any], image: Any) -> Any:
    resolved = image if image is not None else first_image(observation)
    if resolved is None:
        raise ValueError("OpenVLA requires an agentview image in obs['images']")
    return resolved


def _runtime_for(config: OpenVLARuntimeConfig) -> OpenVLARuntime:
    key = (
        str(config.checkpoint_dir),
        config.unnorm_key,
        config.device,
        config.torch_dtype,
        config.attn_implementation,
        config.use_cache,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = OpenVLARuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime


def _action_from_artifact(artifact_path: Path) -> Any:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"OpenVLA artifact at {artifact_path} is not a JSON object")
    action = payload.get("action")
    if action is not None:
        return action
    actions = payload.get("actions")
    if isinstance(actions, list) and actions:
        return actions[0] if len(actions) == 1 else actions
    raise RuntimeError(f"OpenVLA artifact at {artifact_path} did not contain action/actions")


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str = "",
    device: str = "cuda",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one OpenVLA inference step for embodied closed-loop evaluation."""
    del action_context

    runtime_config_path = kwargs.get("runtime_config_path")
    defaults = load_vla_va_wam_runtime_config("openvla", runtime_config_path)
    merged = {**defaults, **kwargs, "device": device}

    unnorm_key = str(merged.get("unnorm_key") or "libero_spatial")
    profile = load_runtime_profile(
        "openvla",
        manifest_path=merged.get("manifest_path"),
        profile_path=merged.get("profile_path"),
        acquisition_root=merged.get("acquisition_root"),
        hf_models_root=merged.get("hf_models_root"),
    )
    checkpoint = select_openvla_checkpoint(
        checkpoint_dir=(
            checkpoint_path or merged.get("checkpoint_dir") or merged.get("checkpoint_path") or merged.get("ckpt_path")
        ),
        checkpoints=profile.checkpoints,
        unnorm_key=unnorm_key,
    )
    config = OpenVLARuntimeConfig(
        checkpoint_dir=checkpoint,
        unnorm_key=unnorm_key,
        device=str(merged.get("device") or device),
        torch_dtype=str(merged.get("torch_dtype") or "auto"),
        attn_implementation=str(merged.get("attn_implementation") or "eager"),
        use_cache=merged.get("use_cache"),
    )

    prompt = _resolve_instruction(observation, instruction)
    rgb_image = _resolve_image(observation, image)
    artifact_path = Path(tempfile.mkdtemp(prefix="wf-openvla-closed-loop-")) / "action.json"
    result = _runtime_for(config).predict_action(
        instruction=prompt,
        image=rgb_image,
        output_path=artifact_path,
        extra_metadata={"closed_loop": True, "model_id": "openvla"},
    )
    candidate_path = Path(
        str(result.get("artifact_path") or result.get("path") or result.get("output_path") or artifact_path)
    )
    if not candidate_path.is_file():
        raise RuntimeError(f"OpenVLA did not emit an action artifact at {candidate_path}")
    return {
        "actions": _action_from_artifact(candidate_path),
        "artifact_path": str(candidate_path),
        "status": str(result.get("status") or "success"),
        "model_id": "openvla",
    }


__all__ = ["predict_action"]
