from __future__ import annotations

from pathlib import Path
import os
import shutil
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_int,
    to_numpy_image,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_MODEL_CONFIG = load_vla_va_wam_runtime_config("mme-vla")
_DEFAULT_POLICY_CONFIG = str(_MODEL_CONFIG["default_policy_config"])
_DEFAULT_SEED = int(_MODEL_CONFIG["seed"])


def _stage_paligemma_tokenizer(location: str | Path | None = None) -> Path:
    """Atomically stage the local PaliGemma tokenizer for OpenPI inference."""

    from worldfoundry.core.io.paths import (
        checkpoint_root_path,
        resolve_local_hf_model_path,
        resolve_worldfoundry_path,
    )

    source = resolve_worldfoundry_path(str(location or _MODEL_CONFIG["paligemma_tokenizer_path"]))
    if not source.is_file() and location is None:
        source = resolve_local_hf_model_path(
            "google/paligemma-3b-pt-224",
            required_files=("tokenizer.model",),
        ) / "tokenizer.model"
    if not source.is_file():
        raise FileNotFoundError(f"MME-VLA PaliGemma tokenizer is missing: {source}")
    target = checkpoint_root_path("runtime_assets", "openpi", "big_vision", "tokenizer.model")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not target.is_symlink() and target.stat().st_size == source.stat().st_size:
        return target.resolve()

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _runtime_for(location: str, options: Mapping[str, Any]) -> Any:
    # Keep the JAX/OpenPI model graph out of module import.  Workspace discovery
    # imports every public runtime, while these dependencies are only required
    # when an MME-VLA checkpoint is actually loaded.
    from . import config as mme_config
    from . import policy_loader
    from ..openpi.modeling import tokenizer as openpi_tokenizer

    config_name = str(options.get("policy_config") or _DEFAULT_POLICY_CONFIG)
    key = (
        location,
        config_name,
        options.get("seed"),
        options.get("default_prompt"),
    )
    policy = _RUNTIME_CACHE.get(key)
    if policy is not None:
        return policy

    openpi_tokenizer.configure_local_tokenizer_assets(
        paligemma=str(_stage_paligemma_tokenizer(options.get("paligemma_tokenizer_path"))),
        fast="",
    )
    runtime_config = mme_config.get_config(config_name)
    policy = policy_loader.create_policy(
        runtime_config,
        Path(location),
        seed=option_int(options.get("seed"), _DEFAULT_SEED),
        default_prompt=options.get("default_prompt"),
    )
    _RUNTIME_CACHE[key] = policy
    return policy


def _policy_observation(observation: Mapping[str, Any], image: Any, instruction: str) -> dict[str, Any]:
    import numpy as np

    base_image = first_present(observation, "observation/image", "image", "base_image")
    wrist = first_present(observation, "observation/wrist_image", "wrist_image")
    if base_image is None and isinstance(image, Mapping):
        base_image = first_present(image, "observation/image", "image", "base_image")
        wrist = wrist if wrist is not None else first_present(image, "observation/wrist_image", "wrist_image")
    elif base_image is None:
        base_image = image

    payload: dict[str, Any] = {
        "observation/image": to_numpy_image(base_image),
        "observation/wrist_image": to_numpy_image(wrist),
        "observation/state": first_present(observation, "observation/state", "state", "robot_state", "proprio"),
        "prompt": first_present(observation, "prompt", "task_instruction", "instruction") or instruction,
    }
    if payload["observation/state"] is not None:
        payload["observation/state"] = np.asarray(payload["observation/state"], dtype=np.float32)
    for key in (
        "static_image_emb",
        "static_pos_emb",
        "static_state_emb",
        "static_mask",
        "recur_image_emb",
        "recur_pos_emb",
        "recur_state_emb",
        "recur_mask",
        "simple_subgoal",
        "grounded_subgoal",
    ):
        if key in observation and observation[key] is not None:
            payload[key] = observation[key]
    return payload


def _buffer_observation(observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    import numpy as np

    state = first_present(observation, "state", "states", "observation/state", "robot_state", "proprio")
    images = first_present(observation, "images")
    if images is not None and state is not None:
        return {
            "images": images,
            "state": state,
            "exec_start_idx": option_int(observation.get("exec_start_idx"), 0),
        }

    image = first_present(observation, "observation/image", "image", "base_image")
    image_np = to_numpy_image(image)
    if image_np is None or state is None:
        raise ValueError("MME-VLA history_observations require images/state or observation/image plus observation/state.")

    return {
        "images": np.asarray(image_np, dtype=np.uint8)[None, None, ...],
        "state": np.asarray(state, dtype=np.float32)[None, ...],
        "exec_start_idx": option_int(observation.get("exec_start_idx"), 0),
        "prompt": first_present(observation, "prompt", "task_instruction", "instruction") or instruction,
    }


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del action_context, device
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not location:
        raise ValueError("MME-VLA requires checkpoint_path pointing to a policy checkpoint step directory.")
    policy = _runtime_for(location, options)
    policy.reset()
    histories = observation.get("history_observations") or ()
    if isinstance(histories, Mapping):
        histories = (histories,)
    buffer_observations = tuple(histories) or (observation,)
    for history in buffer_observations:
        policy.add_buffer(_buffer_observation(history, instruction))
    obs = _policy_observation(observation, image, instruction)
    if obs.get("observation/image") is None or obs.get("observation/wrist_image") is None:
        raise ValueError("MME-VLA requires observation/image and observation/wrist_image inputs.")
    if obs.get("observation/state") is None:
        raise ValueError("MME-VLA requires observation/state input.")
    raw = policy.infer(obs)
    return completed_action_result(
        model_id="mme-vla",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device="jax",
        runtime="worldfoundry.mme_vla.native_in_process",
        metadata={
            "official_entrypoint": "worldfoundry.mme_vla.policy_loader:create_policy(...).infer",
            "policy_config": options.get("policy_config") or _DEFAULT_POLICY_CONFIG,
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
