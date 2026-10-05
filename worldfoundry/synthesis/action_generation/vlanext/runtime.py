from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    first_present,
    option_int,
    to_numpy_image,
)

_RUNTIME_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _local_hfd_ref(repo_id: str) -> str:
    slug = repo_id.replace("/", "--")
    hfd_root = hfd_root_path()
    for root in (hfd_root, hfd_root.parent / "hfd_models"):
        candidate = root / slug
        if (candidate / "config.json").is_file():
            return str(candidate)
    return repo_id


def _cfg(location: str, options: Mapping[str, Any]) -> Any:
    model_options = {
        "lmm_path": str(options.get("lmm_path") or _local_hfd_ref("Qwen/Qwen3-VL-2B-Instruct")),
    }
    if options.get("diffusion_steps") is not None:
        model_options["diffusion_steps"] = option_int(options.get("diffusion_steps"), 10)
    if options.get("scheduler_type") is not None:
        model_options["scheduler_type"] = str(options["scheduler_type"])
    if options.get("attention_backend") is not None:
        model_options["attention_backend"] = str(options["attention_backend"])
    return SimpleNamespace(eval=SimpleNamespace(finetuned_checkpoint=location), model=SimpleNamespace(**model_options))


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    from . import inference

    key = (
        location,
        device,
        options.get("diffusion_steps"),
        options.get("scheduler_type"),
        options.get("torch_dtype"),
        options.get("attention_backend"),
        options.get("lmm_path"),
    )
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached
    cfg = _cfg(location, options)
    model = inference.get_vla(
        cfg,
        device=device,
        torch_dtype=options.get("torch_dtype", "auto"),
    )
    if hasattr(model, "eval"):
        model = model.eval()
    processor = getattr(model, "processor", None)
    if processor is None:
        processor = inference.get_processor(cfg, checkpoint_config=model.checkpoint_config)
    cached = (cfg, model, processor)
    _RUNTIME_CACHE[key] = cached
    return cached


def _observation(observation: Mapping[str, Any], image: Any, instruction: str) -> dict[str, Any]:
    full_image = first_present(observation, "full_image", "image")
    if full_image is None and isinstance(image, Mapping):
        full_image = first_present(image, "full_image", "image")
    if full_image is None:
        full_image = image
    wrist = first_present(observation, "full_image_wrist", "wrist_image")
    if wrist is None and isinstance(image, Mapping):
        wrist = first_present(image, "full_image_wrist", "wrist_image")

    obs = dict(observation)
    obs["full_image"] = to_numpy_image(full_image)
    if wrist is not None:
        obs["full_image_wrist"] = to_numpy_image(wrist)
    if obs.get("image_history") is not None:
        obs["image_history"] = [to_numpy_image(item) for item in obs["image_history"]]
    if obs.get("image_history_wrist") is not None:
        obs["image_history_wrist"] = [to_numpy_image(item) for item in obs["image_history_wrist"]]
    obs.setdefault("task_description", instruction)
    return obs


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
    from . import inference

    del action_context
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "")
    if not location:
        raise ValueError("VLANeXt requires checkpoint_path pointing to a VLANeXt_*.pt checkpoint.")
    cfg, model, processor = _runtime_for(location, device, options)
    obs = _observation(observation, image, instruction)
    if obs.get("full_image") is None:
        raise ValueError("VLANeXt requires full_image or image input for direct inference.")
    raw = inference.get_vla_action(
        cfg,
        model,
        processor,
        obs,
        instruction,
        seed=options.get("seed"),
    )
    return completed_action_result(
        model_id="vlanext",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.vlanext.native_in_process",
        metadata={
            "official_entrypoint": "worldfoundry.vlanext.inference:get_vla_action",
            "task_suite_name": options.get("task_suite_name"),
            "strict_checkpoint": True,
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
