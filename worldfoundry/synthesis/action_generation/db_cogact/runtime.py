from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    load_json_if_present,
    option_bool,
    option_float,
    option_int,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _finalize_vision_tower(model: Any) -> None:
    vision_tower = model.model.mm_vision_tower
    if getattr(vision_tower, "_meta_initialized", False) and not vision_tower.is_loaded:
        # Mirrors CogActExp after Transformers 5 meta-device initialization:
        # checkpoint weights are loaded, but the image processor is non-parameter state.
        vision_tower.load_model()


def _normalization_stats(checkpoint_path: str, options: Mapping[str, Any]) -> dict[str, Any]:
    explicit = first_present(options, "norm_stats", "norm_stats_path", "normalization_stats_path")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if explicit is not None:
        loaded = load_json_if_present(str(explicit))
        if loaded is not None:
            if "norm_stats" in loaded:
                loaded = loaded["norm_stats"]
            return dict(loaded.get("default", loaded))
    loaded = load_json_if_present(Path(checkpoint_path) / "norm_stats.json")
    if loaded is not None:
        if "norm_stats" in loaded:
            loaded = loaded["norm_stats"]
        return dict(loaded.get("default", loaded))
    return {"min": -1, "max": 1}


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    camera_order = tuple(str(item) for item in options.get("camera_order") or ("front", "left_wrist", "right_wrist"))
    cpu_offload_enabled = option_bool(options.get("cpu_offload"), False)
    key = (location, device, camera_order, options.get("torch_dtype"), options.get("attention_backend"), cpu_offload_enabled)
    policy = _RUNTIME_CACHE.get(key)
    if policy is not None:
        return policy

    import torch
    from transformers import AutoTokenizer
    from transformers.modeling_utils import no_init_weights

    from worldfoundry.core.attention import resolve_transformers_attention_implementation
    from worldfoundry.core.model_loading.checkpoints import load_safetensors_into_model_streaming
    from worldfoundry.core.vram import skip_model_initialization

    from .modeling.architecture import CogACTForCausalLM, CogActConfig
    from .modeling.policy import CogACTPolicy

    dtype_name = str(options.get("torch_dtype") or "bfloat16").lower()
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported DB-CogACT torch_dtype {dtype_name!r}")
    device_text = str(device or "cpu")
    requested_cuda = device_text.startswith("cuda")
    if requested_cuda and not torch.cuda.is_available():
        device_text = "cpu"
    elif device_text == "cuda":
        device_text = "cuda:0"
    target_device = torch.device(device_text)

    config = CogActConfig.from_pretrained(location, local_files_only=True)
    config._attn_implementation = resolve_transformers_attention_implementation(
        preferred=options.get("attention_backend"),
        device=target_device,
    )
    # Build only parameter metadata, then assign each safetensors shard
    # directly on its final device. This avoids a 15+ GiB merged host state
    # dict and avoids loading on cuda:0 before copying to the requested GPU.
    with no_init_weights(), skip_model_initialization():
        model = CogACTForCausalLM(config)
    weight_report = load_safetensors_into_model_streaming(
        model,
        location,
        strict=True,
        device="cpu" if cpu_offload_enabled else str(target_device),
        assign=True,
    )
    model._worldfoundry_weight_report = weight_report
    _finalize_vision_tower(model)
    if cpu_offload_enabled:
        from accelerate import cpu_offload

        model = model.to(device="cpu", dtype=dtype)
        for module in model.modules():
            module._worldfoundry_execution_device = target_device
        cpu_offload(
            model,
            execution_device=target_device,
            offload_buffers=True,
        )
        model._worldfoundry_cpu_offload = True
    elif hasattr(model, "to"):
        model = model.to(device=target_device, dtype=dtype)
    if hasattr(model, "eval"):
        model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        location,
        trust_remote_code=option_bool(options.get("trust_remote_code"), False),
        local_files_only=True,
    )
    policy = CogACTPolicy(
        model=model,
        tokenizer=tokenizer,
        norm_stats=_normalization_stats(location, options),
        camera_order=list(camera_order),
    )
    _RUNTIME_CACHE[key] = policy
    return policy


def _policy_observation(observation: Mapping[str, Any], image: Any, instruction: str, options: Mapping[str, Any]) -> dict[str, Any]:
    camera_order = [str(item) for item in options.get("camera_order") or ("front", "left_wrist", "right_wrist")]
    payload: dict[str, Any] = {"prompt": first_present(observation, "prompt") or instruction}
    for key, value in observation.items():
        if key.startswith("image/") and value is not None:
            payload[key] = value
    if not any(key.startswith("image/") for key in payload):
        images = collect_images(observation, image, [*camera_order, "1", "2", "3"])
        for index, item in enumerate(images, start=1):
            payload[f"image/{index}"] = item
    state = first_present(observation, "state", "robot_state", "proprio")
    if state is not None:
        payload["state"] = state
    return payload


def _prepare_policy_observation(policy: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if not getattr(policy, "state_used", False):
        payload.pop("state", None)
    elif isinstance(payload.get("state"), list):
        # The official batch normalizer treats every Python list as a batch.
        # Preserve a single robot state vector as one array-valued sample.
        import numpy as np

        payload["state"] = np.asarray(payload["state"])
    return payload


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
    location = checkpoint_path or str(options.get("checkpoint_ref") or "Dexmal/libero-db-cogact")
    policy = _runtime_for(location, device, options)
    policy_obs = _prepare_policy_observation(
        policy,
        _policy_observation(observation, image, instruction, options),
    )
    if not any(key.startswith("image/") for key in policy_obs):
        raise ValueError("DB-CogACT requires at least one image slot for direct select_action inference.")
    from .modeling.types import SamplingConfig

    sampling = SamplingConfig(
        num_steps=option_int(options.get("num_steps"), 10),
        cfg_scale=option_float(options.get("cfg_scale"), 1.5),
        seed=options.get("seed"),
    )
    raw = policy.select_action(policy_obs, sampling)
    return completed_action_result(
        model_id="db-cogact",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.db_cogact.native_in_process",
        metadata={
            "official_entrypoint": "db_cogact.modeling.policy_base:BasePolicy.select_action",
            "action_mode": getattr(policy, "action_mode", None),
            "state_used": getattr(policy, "state_used", None),
            "cpu_offload": bool(getattr(policy.model, "_worldfoundry_cpu_offload", False)),
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
