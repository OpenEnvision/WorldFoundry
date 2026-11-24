from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from worldfoundry.synthesis.action_generation._official_action_trace import (
    checkpoint_gated_action_trace,
    precomputed_actions,
)
from worldfoundry.synthesis.action_generation.openvla_oft.inference import (
    OpenVLAOFTRuntime,
    OpenVLAOFTRuntimeConfig,
)

_RUNTIME_CACHE: dict[tuple[Any, ...], OpenVLAOFTRuntime] = {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_for(config: OpenVLAOFTRuntimeConfig) -> OpenVLAOFTRuntime:
    key = (
        config.checkpoint_location,
        config.device,
        config.torch_dtype,
        config.cache_dir,
        config.local_files_only,
        config.attn_implementation,
        config.unnorm_key,
        config.task_suite_name,
        config.use_l1_regression,
        config.use_diffusion,
        config.use_proprio,
        config.num_images_in_input,
        config.center_crop,
        config.num_diffusion_steps_train,
        config.num_diffusion_steps_inference,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = OpenVLAOFTRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime


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
    options = dict(runtime_options or {})
    if _as_bool(options.get("allow_supplied_actions_fallback"), False):
        supplied = precomputed_actions(observation, action_context)
        if supplied is not None:
            return checkpoint_gated_action_trace(
                model_id="openvla-oft",
                instruction=instruction,
                observation=observation,
                action_context=action_context,
                checkpoint_path=checkpoint_path,
                device=device,
                official_entrypoint=(
                    "experiments.robot.openvla_utils:get_vla/get_processor/"
                    "get_action_head/get_proprio_projector/get_vla_action"
                ),
                required_port="",
                input_contract={
                    "images": ["full_image", "wrist_image"],
                    "state": "8D LIBERO proprio vector",
                    "prompt": "task_description",
                    "action_output": "8x7 continuous action chunk",
                },
            )

    location = (
        checkpoint_path
        or str(options.get("checkpoint_ref") or "")
        or str(options.get("repo_id") or "")
        or "moojink/openvla-7b-oft-finetuned-libero-spatial"
    )
    if not location:
        return checkpoint_gated_action_trace(
            model_id="openvla-oft",
            instruction=instruction,
            observation=observation,
            action_context=action_context,
            checkpoint_path=checkpoint_path,
            device=device,
            official_entrypoint=(
                "experiments.robot.openvla_utils:get_vla/get_processor/"
                "get_action_head/get_proprio_projector/get_vla_action"
            ),
            required_port="OpenVLA-OFT requires a checkpoint_ref or checkpoint_path.",
            input_contract={
                "images": ["full_image", "wrist_image"],
                "state": "8D LIBERO proprio vector",
                "prompt": "task_description",
                "action_output": "8x7 continuous action chunk",
            },
        )

    config = OpenVLAOFTRuntimeConfig(
        checkpoint_location=location,
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        cache_dir=options.get("cache_dir"),
        local_files_only=_as_bool(options.get("local_files_only"), True),
        attn_implementation=str(options.get("attn_implementation") or "auto"),
        unnorm_key=str(options.get("unnorm_key") or "libero_spatial_no_noops"),
        task_suite_name=str(options.get("task_suite_name") or "libero_spatial"),
        use_l1_regression=_as_bool(options.get("use_l1_regression"), True),
        use_diffusion=_as_bool(options.get("use_diffusion"), False),
        use_proprio=_as_bool(options.get("use_proprio"), True),
        num_images_in_input=int(options.get("num_images_in_input") or 2),
        center_crop=_as_bool(options.get("center_crop"), True),
        num_diffusion_steps_train=int(options.get("num_diffusion_steps_train") or 50),
        num_diffusion_steps_inference=int(options.get("num_diffusion_steps_inference") or 50),
    )
    return _runtime_for(config).predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )
