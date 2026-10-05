from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    load_image,
    option_bool,
    option_float,
    option_int,
    runtime_options_cache_key,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}


def _checkpoint_location(location: str) -> str:
    path = Path(str(location)).expanduser()
    if not path.exists() or path.is_file():
        return str(location)
    checkpoint_dir = path / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        preferred = [item for item in candidates if item.name.lower().startswith(path.name.split("--")[-1].lower())]
        if len(preferred) == 1:
            return str(preferred[0])
    raise FileNotFoundError(
        f"CogACT checkpoint directory must contain exactly one checkpoints/*.pt file, got {len(candidates)} in {checkpoint_dir}"
    )


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    location = _checkpoint_location(location)
    cache_options = {
        key: value
        for key, value in options.items()
        if key not in {"hf_token", "huggingface_token", "token"}
    }
    key = (
        location,
        device,
        runtime_options_cache_key(cache_options),
    )
    model = _RUNTIME_CACHE.get(key)
    if model is not None:
        return model

    from .modeling.loader import load_vla

    model = load_vla(
        location,
        cache_dir=options.get("cache_dir"),
        tokenizer_ref=options.get("tokenizer_ref") or options.get("llm_backbone_ref"),
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        attn_implementation=str(options.get("attn_implementation") or "auto"),
        action_model_type=str(options["action_model_type"]) if options.get("action_model_type") else None,
        future_action_window_size=(
            option_int(options.get("future_action_window_size"), 15)
            if options.get("future_action_window_size") not in (None, "")
            else None
        ),
        past_action_window_size=(
            option_int(options.get("past_action_window_size"), 0)
            if options.get("past_action_window_size") not in (None, "")
            else None
        ),
        use_ema=option_bool(options.get("use_ema"), False),
        compile_action_model=option_bool(options.get("compile_action_model"), False),
    )
    _RUNTIME_CACHE[key] = model
    return model


def clear_runtime_cache() -> None:
    """Release cached CogACT policies between Workspace model sessions."""

    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


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
    del action_context
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "CogACT/CogACT-Base")
    policy = _runtime_for(location, device, options)
    selected = first_present(observation, "image", "full_image")
    if selected is None:
        selected = image
    selected_image = load_image(selected)
    if selected_image is None:
        raise ValueError("CogACT requires an image/full_image observation for direct inference.")
    requested_unnorm_key = first_present(observation, "unnorm_key")
    if requested_unnorm_key is None:
        requested_unnorm_key = options.get("unnorm_key")
    requested_cfg_scale = first_present(observation, "cfg_scale")
    if requested_cfg_scale is None:
        requested_cfg_scale = options.get("cfg_scale")
    requested_use_ddim = first_present(observation, "use_ddim")
    if requested_use_ddim is None:
        requested_use_ddim = options.get("use_ddim")
    requested_ddim_steps = first_present(observation, "num_ddim_steps")
    if requested_ddim_steps is None:
        requested_ddim_steps = options.get("num_ddim_steps")
    requested_seed = first_present(observation, "seed", "random_seed")
    if requested_seed is None:
        requested_seed = options.get("seed")

    raw_actions = policy.predict_action(
        selected_image,
        instruction,
        unnorm_key=str(requested_unnorm_key or "bridge_orig"),
        cfg_scale=option_float(requested_cfg_scale, 1.5),
        use_ddim=option_bool(requested_use_ddim, False),
        num_ddim_steps=option_int(requested_ddim_steps, 5),
        seed=option_int(requested_seed, 0) if requested_seed not in (None, "") else None,
    )
    actions, normalized_actions = raw_actions
    return completed_action_result(
        model_id="cogact",
        instruction=instruction,
        actions=actions.tolist(),
        raw_output={
            "actions": actions.tolist(),
            "normalized_actions": normalized_actions.tolist(),
        },
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.cogact.native_in_process",
        metadata={
            "official_entrypoint": "cogact.modeling.loader:load_vla(...).predict_action",
            "runtime_package": "worldfoundry.synthesis.action_generation.cogact.modeling",
            **dict(policy.last_inference_metadata),
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
