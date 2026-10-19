"""Shared Studio inference request validation and parameter assembly."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from worldfoundry.core.execution.inference import (
    InferenceArtifactSpec,
    InferenceCheckpointRef,
    InferenceFieldSpec,
    InferenceTaskProfile,
    InferenceVariantSpec,
)
from worldfoundry.runtime.inference_catalog import (
    ASSET_GATED_WORLD_RUNTIME_MODEL_IDS,
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    generic_model_inference_spec,
    get_model_inference_spec,
    model_inference_spec,
)
from worldfoundry.runtime.interactive_inference_catalog import interactive_task_for_variant
from worldfoundry.studio.inference.catalog import (
    COGVIDEOX_DEFAULT_VARIANT_ID,
    COGVIDEOX_STUDIO_PARENT_ID,
    SANA_DEFAULT_IMAGE_VARIANT_ID,
    CatalogEntry,
    cogvideox_runtime_model_id,
    find_entry,
    find_runtime_entry,
    lingbot_world_fast_load_kwargs,
    template_id_hint,
)
from worldfoundry.studio.inference.dispatch import DISPATCH_ONLY_CALL_KWARGS, DISPATCH_ONLY_LOAD_KWARGS


class InferenceRequestError(ValueError):
    """An inference request does not match its model contract."""


DEFAULT_SETTINGS: dict[str, Any] = {
    "auto_start_job": True,
    "device": os.getenv("WORLDFOUNDRY_STUDIO_DEVICE", "cuda"),
    "backend": "auto",
    "fps": 16,
    "num_frames": 81,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": -1,
    "attention_backend": "auto",
    "torch_compile": False,
    "cpu_offload": False,
}


RUNTIME_OPTION_LABELS = {
    "torch_compile": "Torch Compile",
    "cpu_offload": "CPU Offload",
    "vae_cpu_offload": "VAE Offload",
    "text_encoder_cpu_offload": "Text Encoder Offload",
    "fuse_qkv": "Fused QKV projections",
    "inplace_residual": "In-place residual",
    "static_cross_kv": "Static cross-attention KV cache",
    "fused_rope": "Fused RoPE",
}


RUNTIME_OPTION_ALIASES = {
    "torch_compile": (
        "torch_compile",
        "enable_torch_compile",
        "use_torch_compile",
        "compile",
    ),
    "cpu_offload": ("cpu_offload", "enable_offloading", "use_cpu_offload", "GPU_memory_mode"),
    "vae_cpu_offload": ("vae_cpu_offload", "offload_vae"),
    "text_encoder_cpu_offload": (
        "text_encoder_cpu_offload",
        "offload_t5",
        "offload_text_encoder_model",
    ),
    "fuse_qkv": ("fuse_qkv",),
    "inplace_residual": ("inplace_residual",),
    "static_cross_kv": ("static_cross_kv",),
    "fused_rope": ("fused_rope",),
}


RUNTIME_VALUE_OPTION_SPECS: dict[str, dict[str, Any]] = {
    "qkv_strategy": {
        "label": "QKV execution",
        "kind": "choice",
        "choices": ("auto", "packed", "split"),
        "default": "auto",
    },
    "qkv_split_threshold": {
        "label": "QKV split threshold",
        "kind": "integer",
        "minimum": 1,
        "default": 8192,
    },
    "rope_precision": {
        "label": "RoPE precision",
        "kind": "choice",
        "choices": ("fp32", "fp64"),
        "default": "fp32",
    },
    "rms_norm_precision": {
        "label": "RMSNorm precision",
        "kind": "choice",
        "choices": ("input", "fp32"),
        "default": "input",
    },
}


TORCH_COMPILE_ENV_MODELS = {"matrix-game-2"}


class JobCreateRequest(BaseModel):
    job_type: str = "inference"
    workload_type: str = ""
    model_id: str = ""
    variant_id: str = ""
    task_profile_id: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    input_path: str = ""
    model_ref: str = ""
    backend: str = "auto"
    endpoint: str = ""
    api_key: str = ""
    device: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    call_kwargs: dict[str, Any] = Field(default_factory=dict)
    load_kwargs: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = ""
    eval_mode: str = "existing-results"
    benchmark_id: str = ""
    requests_path: str = ""
    results_path: str = ""
    dataset_id: str = ""
    dataset_root: str = ""
    dataset_manifest: str = ""
    model_runner: str = ""
    model_zoo_manifest_dir: str = ""
    model_variant_id: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["artifact_count"])
    required_artifacts: list[str] = Field(default_factory=list)
    generation_cache_dir: str = ""
    generation_cache_mode: str = "off"
    resume: bool = False
    metric_batch_size: int = Field(default=32, ge=1)
    run_plan_path: str = ""
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    materialize_requests: bool = False
    limit: int | None = None


def _entry_workload(entry: CatalogEntry) -> str:
    if entry.module_path == "worldfoundry.pipelines.sana.pipeline_sana" and not entry.model_id.startswith(
        ("sana-video-", "longsana-video-", "sana-streaming-")
    ):
        return "image"
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    if task_type in {"t2v", "text-video", "text-to-video", "video-generation"}:
        return "t2v"
    if task_type in {"i2v", "image-video", "image-to-video"}:
        return "i2v"
    if task_type in {"video-to-video", "v2v"}:
        return "v2v"
    if task_type in {"video-to-audio", "v2a"}:
        return "v2a"
    if task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        return "image"
    template_id = template_id_hint(entry)
    if template_id == "video-to-video":
        return "v2v"
    if template_id == "conditioned-video":
        return "i2v"
    if template_id == "text-video":
        return "t2v"
    if template_id == "scene-3d":
        return "3d"
    if template_id == "depth-geometry":
        return "geometry"
    if template_id in {"embodied-policy", "visual-action"}:
        return "action"
    if template_id == "hosted-api":
        return "api"
    return "world"


def _entry_extra_variants(entry: CatalogEntry) -> tuple[InferenceVariantSpec, ...]:
    variants: list[InferenceVariantSpec] = []
    for raw_variant in entry.extra_variants:
        variant_id = str(raw_variant.get("variant_id") or "").strip()
        if not variant_id:
            continue
        checkpoints = tuple(
            InferenceCheckpointRef(
                role=str(raw_checkpoint.get("role") or "primary"),
                uri=str(raw_checkpoint.get("uri") or ""),
                required=bool(raw_checkpoint.get("required", True)),
                status=str(raw_checkpoint.get("status") or "unknown"),
            )
            for raw_checkpoint in raw_variant.get("checkpoints", ()) or ()
        )
        variants.append(
            InferenceVariantSpec(
                variant_id=variant_id,
                label=str(raw_variant.get("label") or variant_id),
                checkpoints=checkpoints,
                status=str(raw_variant.get("status") or "configured"),
                load_kwargs=dict(raw_variant.get("load_kwargs") or {}),
                call_kwargs=dict(raw_variant.get("call_kwargs") or {}),
                aliases=tuple(str(item) for item in raw_variant.get("aliases", ()) or ()),
                notes=tuple(str(item) for item in raw_variant.get("notes", ()) or ()),
            )
        )
    return tuple(variants)


def _entry_extra_variant_ids(entry: CatalogEntry) -> set[str]:
    return {str(raw_variant.get("variant_id") or "").strip() for raw_variant in entry.extra_variants}


def _append_task_inputs(
    spec: Any,
    *,
    model_id: str,
    fields: Sequence[InferenceFieldSpec],
):
    """Add model-specific UI fields without duplicating generic inputs."""

    if not fields:
        return spec
    patched_tasks = []
    for task in spec.tasks:
        existing = {(input_field.target, _param_key(input_field.field_id)) for input_field in task.inputs}
        merged = list(task.inputs)
        for input_field in fields:
            key = (input_field.target, _param_key(input_field.field_id))
            if key not in existing:
                merged.append(input_field)
                existing.add(key)
        patched_tasks.append(replace(task, inputs=tuple(merged)))
    return replace(spec, tasks=tuple(patched_tasks))


def _entry_inference_spec(entry: CatalogEntry):
    curated = get_model_inference_spec(entry.model_id)
    if curated is not None:
        return curated
    if entry.model_id in ASSET_GATED_WORLD_RUNTIME_MODEL_IDS:
        spec = generic_model_inference_spec(
            model_family_id=entry.model_id,
            display_name=entry.display_name,
            default_model_ref=entry.default_model_ref,
            default_load_kwargs=entry.default_load_kwargs,
            default_call_kwargs=entry.default_call_kwargs,
            supports_stream=entry.supports_stream,
            workload_type=_entry_workload(entry),
            supported_call_params=None,
        )
        return replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    inputs=tuple(
                        replace(input_field, default=entry.default_interactions or ("forward",))
                        if input_field.target == "params"
                        and _param_key(input_field.field_id)
                        in {"interactions", "interaction", "interaction_signal", "action"}
                        else input_field
                        for input_field in task.inputs
                    ),
                )
                for task in spec.tasks
            ),
        )
    supported_call_params = entry.input_params or (*entry.call_params, *entry.stream_params)
    if entry.family == "world_model" and not supported_call_params:
        supported_call_params = None
    spec = model_inference_spec(
        model_family_id=entry.model_id,
        display_name=entry.display_name,
        default_model_ref=entry.default_model_ref,
        default_load_kwargs=entry.default_load_kwargs,
        default_call_kwargs=entry.default_call_kwargs,
        supports_stream=entry.supports_stream,
        workload_type=_entry_workload(entry),
        supported_call_params=supported_call_params,
    )
    if entry.model_id in {"lagernvs", "stable-virtual-camera", "wonderjourney"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video Inference",
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if entry.model_id in {"dvlt", "lingbot-map"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="3D Reconstruction",
                    outputs=(
                        InferenceArtifactSpec("model", "generated_3d_asset", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    workload_type = _entry_workload(entry)
    if entry.model_id == "allegro_ti2v":
        spec = _append_task_inputs(
            spec,
            model_id=entry.model_id,
            fields=(
                InferenceFieldSpec(
                    "num_sampling_steps",
                    "Sampling Steps",
                    kind="integer",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("num_sampling_steps", 100),
                ),
                InferenceFieldSpec(
                    "guidance_scale",
                    "Guidance",
                    kind="number",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("guidance_scale", 8),
                ),
                InferenceFieldSpec(
                    "seed",
                    "Seed",
                    kind="integer",
                    target="load_kwargs",
                    default=entry.default_load_kwargs.get("seed", 1427329220),
                ),
            ),
        )
    if workload_type in {"v2v", "video-to-video"}:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video-to-Video Inference",
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if workload_type in {"t2v", "text-video", "text-to-video"} and entry.model_id != COGVIDEOX_STUDIO_PARENT_ID:
        spec = replace(
            spec,
            tasks=tuple(
                replace(
                    task,
                    label="Video Inference",
                    inputs=tuple(input_field for input_field in task.inputs if input_field.target != "input_path"),
                    outputs=(
                        InferenceArtifactSpec("video", "video", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if workload_type == "image" or task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        spec = replace(
            spec,
            default_task_id="image-generation",
            tasks=tuple(
                replace(
                    task,
                    task_id="image-generation",
                    label="Image Inference",
                    inputs=tuple(input_field for input_field in task.inputs if input_field.target != "input_path"),
                    outputs=(
                        InferenceArtifactSpec("image", "generated_image", required=True, preview=True),
                        InferenceArtifactSpec("manifest", "manifest", required=True),
                    ),
                )
                for task in spec.tasks
            ),
        )
    if entry.default_interactions:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for input_field in task.inputs:
                if (
                    input_field.target == "params"
                    and _param_key(input_field.field_id)
                    in {"interactions", "interaction", "interaction_signal", "action"}
                    and (input_field.default is None or input_field.default == "")
                ):
                    inputs.append(replace(input_field, default=entry.default_interactions))
                    changed = True
                else:
                    inputs.append(input_field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    if entry.default_prompt:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for input_field in task.inputs:
                if input_field.target == "prompt" and (input_field.default is None or input_field.default == ""):
                    inputs.append(replace(input_field, default=entry.default_prompt))
                    changed = True
                else:
                    inputs.append(input_field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    if entry.default_input_path:
        tasks = []
        for task in spec.tasks:
            inputs = []
            changed = False
            for input_field in task.inputs:
                if input_field.target == "input_path":
                    inputs.append(replace(input_field, default=entry.default_input_path))
                    changed = True
                else:
                    inputs.append(input_field)
            tasks.append(replace(task, inputs=tuple(inputs)) if changed else task)
        spec = replace(spec, tasks=tuple(tasks))
    extra_variants = _entry_extra_variants(entry)
    if not extra_variants:
        return spec
    if entry.model_id == "sana":
        return replace(
            spec,
            variants=extra_variants,
            default_variant_id=SANA_DEFAULT_IMAGE_VARIANT_ID,
        )
    if entry.model_id == COGVIDEOX_STUDIO_PARENT_ID:
        return replace(
            spec,
            variants=extra_variants,
            default_variant_id=COGVIDEOX_DEFAULT_VARIANT_ID,
        )
    existing_ids = {variant.variant_id for variant in spec.variants}
    merged = spec.variants + tuple(variant for variant in extra_variants if variant.variant_id not in existing_ids)
    return replace(spec, variants=merged)


def _entry_runtime_param_names(entry: CatalogEntry) -> set[str]:
    return set(entry.load_params) | set(entry.call_params) | set(entry.stream_params)


def _entry_runtime_options(entry: CatalogEntry) -> dict[str, dict[str, Any]]:
    names = _entry_runtime_param_names(entry)
    declared_defaults = {
        **dict(entry.default_call_kwargs),
        **dict(entry.default_load_kwargs),
    }
    options: dict[str, dict[str, Any]] = {}
    for key, aliases in RUNTIME_OPTION_ALIASES.items():
        matched = [alias for alias in aliases if alias in names]
        supported = bool(matched)
        if key == "torch_compile" and entry.model_id in TORCH_COMPILE_ENV_MODELS:
            supported = True
            matched.append("WORLDFOUNDRY_ENABLE_TORCH_COMPILE")
        default = next(
            (declared_defaults[alias] for alias in aliases if alias in declared_defaults),
            False,
        )
        options[key] = {
            "label": RUNTIME_OPTION_LABELS[key],
            "kind": "boolean",
            "supported": supported,
            "targets": matched,
            "default": bool(default),
        }
    for key, raw_spec in RUNTIME_VALUE_OPTION_SPECS.items():
        spec = dict(raw_spec)
        supported = key in names
        spec.update(
            {
                "supported": supported,
                "targets": [key] if supported else [],
                "default": declared_defaults.get(key, spec.get("default")),
            }
        )
        options[key] = spec
    return options


def _variant_model_ref(entry: CatalogEntry, variant: InferenceVariantSpec) -> str:
    if entry.model_id == "cosmos3":
        if variant.variant_id == "cosmos3-super":
            return variant.primary_checkpoint_uri or entry.default_model_ref
        return entry.default_model_ref or variant.primary_checkpoint_uri
    if entry.family == "world_model":
        return entry.default_model_ref
    if entry.model_id == LINGBOT_WORLD_MODEL_ID and variant.variant_id in {
        LINGBOT_VARIANT_BASE_CAM,
        LINGBOT_VARIANT_FAST,
    }:
        return entry.default_model_ref
    return variant.primary_checkpoint_uri or entry.default_model_ref


def _variant_load_kwargs(entry: CatalogEntry, variant: InferenceVariantSpec) -> dict[str, Any]:
    load_kwargs = dict(variant.load_kwargs)
    if entry.model_id == LINGBOT_WORLD_MODEL_ID:
        if variant.variant_id == LINGBOT_VARIANT_FAST:
            load_kwargs.update(lingbot_world_fast_load_kwargs())
            load_kwargs.setdefault("runtime_variant", "fast")
        elif variant.variant_id in {LINGBOT_VARIANT_BASE_CAM, LINGBOT_VARIANT_BASE_ACT_PREVIEW}:
            load_kwargs.update({"runtime_variant": None, "fast_model_path": None})
    return load_kwargs


def _resolve_inference_contract(
    entry: CatalogEntry,
    payload: JobCreateRequest,
) -> tuple[InferenceVariantSpec, InferenceTaskProfile, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = _entry_inference_spec(entry)
    try:
        if payload.variant_id:
            variant = spec.variant(payload.variant_id)
        else:
            try:
                variant = spec.variant(payload.model_id)
            except ValueError:
                variant = spec.variant()
        task = interactive_task_for_variant(spec, variant, spec.task(payload.task_profile_id))
    except ValueError as exc:
        raise InferenceRequestError(str(exc)) from exc

    model_ref = payload.model_ref or _variant_model_ref(entry, variant)
    load_kwargs = _variant_load_kwargs(entry, variant)
    for input_field in task.inputs:
        if input_field.target == "load_kwargs" and input_field.default is not None:
            # Task profiles may specialize a variant's loading policy.  For
            # example, Cosmos3 action inference intentionally skips the audio
            # tokenizer even though the same Nano checkpoint loads it for
            # sound-generation tasks.  Explicit request load_kwargs are merged
            # later and therefore remain the final override.
            load_kwargs[input_field.field_id] = input_field.default
    call_kwargs = {}
    if entry.model_id == "cosmos3":
        # Cosmos3 variants carry a complete T2V fallback, while the selected
        # task owns modality-specific values such as num_frames/output_type and
        # scheduler policy. Let the explicit task contract win for all variants.
        call_kwargs.update(dict(variant.call_kwargs))
        call_kwargs.update(dict(task.default_call_kwargs))
    else:
        if variant.variant_id not in _entry_extra_variant_ids(entry):
            call_kwargs.update(dict(task.default_call_kwargs))
        call_kwargs.update(dict(variant.call_kwargs))
    contract = {
        "model_family_id": entry.model_id,
        "variant_id": variant.variant_id,
        "task_profile_id": task.task_id,
        "variant": variant.to_dict(),
        "task": task.to_dict(),
    }
    return variant, task, model_ref, call_kwargs, load_kwargs, contract


def _preview_input_path_from_call_kwargs(call_kwargs: Mapping[str, Any]) -> str:
    for key in (
        "input_path",
        "image",
        "images",
        "image_path",
        "video",
        "video_path",
        "top_cam",
        "agentview_cam",
        "external_cam",
        "left_cam",
        "side_cam",
        "wrist_cam",
        "right_cam",
    ):
        value = call_kwargs.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, str) and item:
                    return item
        if isinstance(value, Mapping):
            for item in value.values():
                if isinstance(item, str) and item:
                    return item
    operator_kwargs = call_kwargs.get("operator_kwargs")
    if isinstance(operator_kwargs, Mapping):
        return _preview_input_path_from_call_kwargs(operator_kwargs)
    return ""


def _param_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def call_param_names(entry: CatalogEntry) -> set[str]:
    names = set(entry.call_params) | set(entry.stream_params) | set(entry.default_call_kwargs)
    if entry.model_id == LINGBOT_WORLD_MODEL_ID:
        names.add("sampling_steps")
    return names


def _load_param_names(entry: CatalogEntry) -> set[str]:
    return set(entry.load_params) | set(entry.default_load_kwargs)


def _hf_checkpoint_repo_identity(value: Any) -> tuple[str, str] | None:
    """Return an HF owner/repo identity only for recognizable repository refs."""

    text = str(value or "").strip().rstrip("/\\")
    if not text:
        return None
    normalized = text.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    for part in reversed(parts):
        if not part.startswith("models--"):
            continue
        owner, separator, repo = part.removeprefix("models--").partition("--")
        if separator and owner and repo:
            return owner, repo
    leaf = parts[-1] if parts else normalized
    owner, separator, repo = leaf.partition("--")
    if separator and owner and repo:
        return owner, repo
    if not normalized.startswith(("/", "./", "../")) and len(parts) == 2:
        owner, repo = parts
        if owner and repo:
            return owner, repo
    return None


def _checkpoint_refs_equivalent(first: Any, second: Any) -> bool:
    first_text = str(first or "").strip().rstrip("/\\")
    second_text = str(second or "").strip().rstrip("/\\")
    if not first_text or not second_text:
        return False
    if first_text == second_text:
        return True
    first_identity = _hf_checkpoint_repo_identity(first_text)
    return first_identity is not None and first_identity == _hf_checkpoint_repo_identity(second_text)


def _rebind_default_model_path(
    load_kwargs: dict[str, Any],
    *,
    model_ref: str,
    variant_load_kwargs: Mapping[str, Any],
    entry_default_load_kwargs: Mapping[str, Any],
) -> None:
    """Keep a catalog model_path aligned with an explicitly resolved model_ref."""

    if "model_path" in variant_load_kwargs:
        default_model_path = variant_load_kwargs["model_path"]
    else:
        default_model_path = entry_default_load_kwargs.get("model_path")
    current_model_path = load_kwargs.get("model_path")
    if str(current_model_path or "") != str(default_model_path or ""):
        return
    if _checkpoint_refs_equivalent(default_model_path, model_ref):
        load_kwargs["model_path"] = model_ref


def _supports_attention_backend(entry: CatalogEntry) -> bool:
    names = _load_param_names(entry) | call_param_names(entry)
    return "attention_backend" in names


def _supports_backend(entry: CatalogEntry, backend: str) -> bool:
    selected = (backend or "auto").strip()
    if selected == "auto":
        return True
    if selected == "from_pretrained":
        return entry.supports_from_pretrained
    if selected == "api_init":
        return entry.supports_api_init
    return False


def _actual_param_name(names: set[str], *aliases: str) -> str | None:
    lookup = {_param_key(name): name for name in names}
    for alias in aliases:
        match = lookup.get(_param_key(alias))
        if match:
            return match
    return None


def _supported_call_param(entry: CatalogEntry, *aliases: str) -> str | None:
    return _actual_param_name(call_param_names(entry), *aliases)


def _sync_input_path_to_call_kwargs(
    entry: CatalogEntry,
    *,
    task_type: str,
    input_path: str,
    call_kwargs: dict[str, Any],
) -> None:
    if not input_path:
        return
    workload = _entry_workload(entry).strip().lower().replace("_", "-")
    declared_task_type = str(task_type or entry.default_task_type or "").strip().lower().replace("_", "-")
    image_input_task_types = {
        "i2i",
        "image-to-image",
        "image-editing",
        "r2v",
        "reference-to-video",
    }
    if declared_task_type in image_input_task_types:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = [input_path] if target == "images" else input_path
        return
    video_input_task_types = {"v2v", "video-video", "video-to-video", "v2a", "video-to-audio"}
    if (
        workload in {"i2v", "image-video", "image-to-video"} and declared_task_type not in video_input_task_types
    ) or declared_task_type in {"i2v", "image-video", "image-to-video"}:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = input_path if target != "images" else [input_path]
        return
    if workload in {"action", "embodied", "embodied-policy", "robotics", "visual-action"}:
        target = _supported_call_param(entry, "image_path", "image", "images")
        if target:
            call_kwargs[target] = input_path if target != "images" else [input_path]
        return
    if workload in video_input_task_types or declared_task_type in video_input_task_types:
        target = _supported_call_param(entry, "video_path", "video", "videos")
        if target:
            call_kwargs[target] = input_path if target != "videos" else [input_path]
        return
    target = _supported_call_param(entry, "input_path")
    if target:
        call_kwargs[target] = input_path


def _param_key_aliases(key: str) -> tuple[str, ...]:
    normalized = _param_key(key)
    aliases = {
        "num_frames": ("num_frames", "frames", "video_length"),
        "frames": ("num_frames", "frames", "video_length"),
        "height": ("height", "user_height", "output_H", "resize_H", "image_height"),
        "width": ("width", "user_width", "output_W", "resize_W", "image_width"),
        "guidance_scale": ("guidance_scale", "cfg_scale", "scale"),
        "guidance": ("guidance_scale", "cfg_scale", "scale"),
        "seed": ("seed",),
        "fps": ("fps",),
        "num_inference_steps": ("num_inference_steps", "sampling_steps", "infer_steps", "num_steps", "steps"),
        "steps": ("num_inference_steps", "sampling_steps", "infer_steps", "num_steps", "steps"),
        "negative_prompt": ("negative_prompt",),
        "interactions": ("interactions", "interaction_signal", "interaction", "action"),
    }
    return aliases.get(normalized, (key,))


def _task_allowed_param_keys(task: InferenceTaskProfile) -> set[str]:
    allowed: set[str] = set()
    for input_field in task.inputs:
        if input_field.target != "params":
            continue
        field_key = _param_key(input_field.field_id)
        allowed.add(field_key)
        allowed.update(_param_key(alias) for alias in _param_key_aliases(field_key))
    return allowed


def _task_field_default(task: InferenceTaskProfile, *field_ids: str, target: str | None = None) -> Any:
    wanted = {_param_key(field_id) for field_id in field_ids}
    for input_field in task.inputs:
        if target is not None and input_field.target != target:
            continue
        if _param_key(input_field.field_id) in wanted and input_field.default is not None and input_field.default != "":
            return input_field.default
    return None


def _runtime_alias_names_for_supported_options(entry: CatalogEntry) -> set[str]:
    names: set[str] = set()
    for key, option in _entry_runtime_options(entry).items():
        if not option.get("supported"):
            continue
        names.add(key)
        names.update(str(alias) for alias in option.get("targets") or ())
        names.update(RUNTIME_OPTION_ALIASES.get(key, ()))
    return names


def _task_declared_kwargs(task: InferenceTaskProfile, target: str) -> set[str]:
    return {_param_key(input_field.field_id) for input_field in task.inputs if input_field.target == target}


def _validate_explicit_kwargs(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    call_names = call_param_names(entry)
    load_names = _load_param_names(entry)
    runtime_aliases = {_param_key(name) for name in _runtime_alias_names_for_supported_options(entry)}
    call_lookup = {_param_key(name) for name in call_names} | _task_declared_kwargs(task, "call_kwargs")
    call_lookup.update(_param_key(name) for name in task.default_call_kwargs)
    load_lookup = {_param_key(name) for name in load_names} | _task_declared_kwargs(task, "load_kwargs")
    call_lookup |= {_param_key(name) for name in DISPATCH_ONLY_CALL_KWARGS}
    load_lookup |= {_param_key(name) for name in DISPATCH_ONLY_LOAD_KWARGS}
    unsupported_call = [
        key
        for key in (payload.call_kwargs or {})
        if _param_key(key) not in call_lookup and _param_key(key) not in runtime_aliases
    ]
    unsupported_load = [
        key
        for key in (payload.load_kwargs or {})
        if _param_key(key) not in load_lookup and _param_key(key) not in runtime_aliases
    ]
    if unsupported_call or unsupported_load:
        details = []
        if unsupported_call:
            details.append(f"call_kwargs={', '.join(sorted(unsupported_call))}")
        if unsupported_load:
            details.append(f"load_kwargs={', '.join(sorted(unsupported_load))}")
        raise InferenceRequestError(
            f"{entry.model_id} does not declare these inference kwargs: {'; '.join(details)}",
        )


def _field_value_provided(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _validate_field_choice(
    entry: CatalogEntry,
    task: InferenceTaskProfile,
    field_id: str,
    choices: Sequence[str],
    value: Any,
) -> None:
    if not choices or not _field_value_provided(value):
        return
    allowed = {str(choice) for choice in choices}
    if str(value) in allowed:
        return
    raise InferenceRequestError(
        f"{entry.model_id}/{task.task_id} field {field_id} must be one of: {', '.join(str(choice) for choice in choices)}",
    )


def _task_choice_fields(task: InferenceTaskProfile, target: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    fields: dict[str, tuple[str, tuple[str, ...]]] = {}
    for input_field in task.inputs:
        if input_field.target != target or not input_field.choices:
            continue
        field_key = _param_key(input_field.field_id)
        for key in (field_key, *_param_key_aliases(field_key)):
            fields[_param_key(key)] = (input_field.field_id, input_field.choices)
    return fields


def _validate_task_field_choices(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    for target, values in (
        ("params", payload.params or {}),
        ("call_kwargs", payload.call_kwargs or {}),
        ("load_kwargs", payload.load_kwargs or {}),
    ):
        fields = _task_choice_fields(task, target)
        if not fields:
            continue
        for key, value in values.items():
            choice_field = fields.get(_param_key(key))
            if choice_field is None:
                continue
            field_id, choices = choice_field
            _validate_field_choice(entry, task, field_id, choices, value)

    direct_values = {
        "prompt": payload.prompt,
        "input_path": payload.input_path,
        "negative_prompt": payload.negative_prompt,
        "model_ref": payload.model_ref,
    }
    for input_field in task.inputs:
        if input_field.target not in direct_values:
            continue
        _validate_field_choice(
            entry, task, input_field.field_id, input_field.choices, direct_values[input_field.target]
        )


def _validate_inference_payload(entry: CatalogEntry, task: InferenceTaskProfile, payload: JobCreateRequest) -> None:
    params = dict(payload.params or {})
    _validate_runtime_options(entry, params)
    _validate_task_field_choices(entry, task, payload)
    backend = payload.backend or entry.default_backend or "auto"
    if not _supports_backend(entry, backend):
        raise InferenceRequestError(f"{entry.model_id} does not support backend={backend}")
    if (payload.endpoint or payload.api_key) and backend != "api_init":
        raise InferenceRequestError("endpoint and api_key are only used with backend=api_init")
    allowed_params = _task_allowed_param_keys(task)
    runtime_param_keys = {_param_key(key) for key in RUNTIME_OPTION_ALIASES}
    model_param_keys = {_param_key(key) for key in (call_param_names(entry) | _load_param_names(entry))}
    unsupported: list[str] = []
    for key, value in params.items():
        normalized = _param_key(key)
        if normalized in runtime_param_keys:
            continue
        if normalized == "attention_backend":
            if value in {"", None, "auto"} or _supports_attention_backend(entry):
                continue
            unsupported.append(key)
            continue
        if normalized not in allowed_params and normalized not in model_param_keys:
            unsupported.append(key)
    if unsupported:
        raise InferenceRequestError(
            f"{entry.model_id}/{task.task_id} does not use these params: {', '.join(sorted(unsupported))}",
        )
    _validate_explicit_kwargs(entry, task, payload)


def _is_missing_param_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _merge_common_params(
    entry: CatalogEntry,
    payload: JobCreateRequest,
    *,
    base_call_kwargs: dict[str, Any] | None = None,
    base_load_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(payload.params or {})
    explicit_call_kwargs = _normalize_explicit_kwargs(
        payload.call_kwargs or {},
        set(entry.call_params) | set(entry.stream_params),
    )
    call_kwargs = dict(base_call_kwargs or {})
    call_kwargs.update(explicit_call_kwargs)
    load_kwargs = dict(base_load_kwargs or {})
    load_kwargs.update(_normalize_explicit_kwargs(payload.load_kwargs or {}, set(entry.load_params)))

    def param_value(param_key: str, *aliases: str) -> Any:
        for key in (param_key, *aliases):
            value = params.get(key)
            if key in params and not _is_missing_param_value(value):
                return value
            normalized = _param_key(key)
            value = params.get(normalized)
            if normalized in params and not _is_missing_param_value(value):
                return value
        return None

    def set_call_from_param(param_key: str, *aliases: str) -> None:
        value = param_value(param_key, *aliases)
        if _is_missing_param_value(value):
            return
        target = _supported_call_param(entry, *aliases)
        if target and target not in explicit_call_kwargs:
            call_kwargs[target] = value

    def set_load_from_param(param_key: str, *aliases: str) -> None:
        value = param_value(param_key, *aliases)
        if _is_missing_param_value(value):
            return
        load_names = {_param_key(name): name for name in entry.load_params}
        for alias in aliases:
            target = load_names.get(_param_key(alias))
            if target and target not in load_kwargs:
                load_kwargs[target] = value
                return

    set_call_from_param("num_frames", "num_frames", "frames", "video_length")
    set_load_from_param("num_frames", "num_frames", "frames", "video_length")
    set_call_from_param("height", "height", "user_height", "output_H", "resize_H", "image_height")
    set_load_from_param("height", "height", "user_height", "output_H", "resize_H", "image_height")
    set_call_from_param("width", "width", "user_width", "output_W", "resize_W", "image_width")
    set_load_from_param("width", "width", "user_width", "output_W", "resize_W", "image_width")
    set_call_from_param("guidance_scale", "guidance_scale", "cfg_scale", "scale")
    set_load_from_param("guidance_scale", "guidance_scale", "cfg_scale", "scale")
    set_call_from_param("seed", "seed")
    set_load_from_param("seed", "seed")
    set_call_from_param("fps", "fps")
    set_load_from_param("fps", "fps", "frame_rate")
    set_call_from_param(
        "num_inference_steps", "num_inference_steps", "steps", "sampling_steps", "infer_steps", "num_steps"
    )
    set_load_from_param(
        "num_inference_steps", "num_inference_steps", "steps", "sampling_steps", "infer_steps", "num_steps"
    )
    if payload.negative_prompt:
        negative_key = _supported_call_param(entry, "negative_prompt")
        if negative_key:
            call_kwargs.setdefault(negative_key, payload.negative_prompt)

    attention_backend = param_value("attention_backend")
    if _supports_attention_backend(entry) and attention_backend not in {"", None, "auto"}:
        load_kwargs.setdefault("attention_backend", attention_backend)
        call_kwargs.setdefault("attention_backend", attention_backend)
    _apply_runtime_options(entry, params=params, load_kwargs=load_kwargs, call_kwargs=call_kwargs)

    call_names = {_param_key(name): name for name in (entry.call_params + entry.stream_params)}
    load_names = {_param_key(name): name for name in entry.load_params}
    for key, value in params.items():
        if _is_missing_param_value(value):
            continue
        normalized = _param_key(key)
        call_target = call_names.get(normalized)
        if call_target and call_target not in explicit_call_kwargs:
            call_kwargs.setdefault(call_target, value)
        load_target = load_names.get(normalized)
        if load_target and load_target not in load_kwargs:
            load_kwargs.setdefault(load_target, value)

    return call_kwargs, load_kwargs


def _normalize_explicit_kwargs(values: Mapping[str, Any], supported_names: set[str]) -> dict[str, Any]:
    lookup = {_param_key(name): name for name in supported_names}
    normalized: dict[str, Any] = {}
    for key, value in dict(values).items():
        normalized[lookup.get(_param_key(key), key)] = value
    return normalized


def _runtime_option_enabled(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _should_use_task_default_input_path(
    entry: CatalogEntry,
    payload: JobCreateRequest,
    task_type: str,
) -> bool:
    declared_task_type = str(task_type or entry.default_task_type or "").strip().lower().replace("_", "-")
    if declared_task_type in {"t2v", "text-video", "text-to-video"}:
        return False
    if declared_task_type in {
        "class-conditional-image-generation",
        "class-conditional-generation",
        "image-generation",
        "text-to-image",
        "t2i",
    }:
        return False
    if declared_task_type:
        return True
    workload = str(payload.workload_type or _entry_workload(entry) or "").strip().lower().replace("_", "-")
    if workload in {"t2v", "text-video", "text-to-video"}:
        return False
    return True


def _validate_runtime_options(entry: CatalogEntry, params: dict[str, Any]) -> None:
    options = _entry_runtime_options(entry)
    unsupported = [
        key
        for key in RUNTIME_OPTION_ALIASES
        if _runtime_option_enabled(params, key) and not options.get(key, {}).get("supported")
    ]
    if unsupported:
        labels = ", ".join(RUNTIME_OPTION_LABELS[key] for key in unsupported)
        raise InferenceRequestError(
            f"{entry.model_id} does not implement these runtime options: {labels}",
        )
    for key, spec in RUNTIME_VALUE_OPTION_SPECS.items():
        if key not in params:
            continue
        option = options.get(key, {})
        if not option.get("supported"):
            raise InferenceRequestError(
                f"{entry.model_id} does not implement runtime option {key}",
            )
        value = params[key]
        choices = tuple(str(choice) for choice in spec.get("choices", ()))
        if choices and str(value) not in choices:
            raise InferenceRequestError(
                f"{key} must be one of: {', '.join(choices)}",
            )
        if spec.get("kind") == "integer":
            if isinstance(value, bool):
                valid_integer = False
            else:
                try:
                    normalized = int(value)
                except (TypeError, ValueError):
                    valid_integer = False
                else:
                    minimum = int(spec.get("minimum", 0))
                    valid_integer = normalized >= minimum
            if not valid_integer:
                raise InferenceRequestError(
                    f"{key} must be an integer >= {spec.get('minimum', 0)}",
                )


def _set_runtime_option_value(target: dict[str, Any], alias: str, value: bool) -> None:
    if alias == "GPU_memory_mode":
        target.setdefault(alias, "model_cpu_offload" if value else "none")
    else:
        target.setdefault(alias, value)


def _apply_runtime_options(
    entry: CatalogEntry,
    *,
    params: dict[str, Any],
    load_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
) -> None:
    _validate_runtime_options(entry, params)
    load_names = set(entry.load_params)
    call_names = set(entry.call_params) | set(entry.stream_params)
    for key, aliases in RUNTIME_OPTION_ALIASES.items():
        if not _runtime_option_enabled(params, key):
            continue
        if key == "torch_compile" and entry.model_id in TORCH_COMPILE_ENV_MODELS:
            load_kwargs.setdefault("torch_compile", True)
            continue
        for alias in aliases:
            if alias in load_names:
                _set_runtime_option_value(load_kwargs, alias, True)
            if alias in call_names:
                _set_runtime_option_value(call_kwargs, alias, True)


def build_inference_run_kwargs(
    payload: JobCreateRequest,
    *,
    settings: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> tuple[CatalogEntry, dict[str, Any]]:
    defaults = DEFAULT_SETTINGS if settings is None else settings
    entry = find_entry(payload.model_id)
    variant, task, model_ref, variant_call_kwargs, variant_load_kwargs, contract = _resolve_inference_contract(
        entry, payload
    )
    runtime_id = cogvideox_runtime_model_id(entry.model_id, variant.variant_id)
    if runtime_id:
        entry = find_runtime_entry(runtime_id)
    if validate:
        _validate_inference_payload(entry, task, payload)
    call_kwargs, load_kwargs = _merge_common_params(
        entry,
        payload,
        base_call_kwargs=variant_call_kwargs,
        base_load_kwargs=variant_load_kwargs,
    )
    if variant.variant_id not in _entry_extra_variant_ids(entry):
        call_kwargs = {**entry.default_call_kwargs, **call_kwargs}
    load_kwargs = {**entry.default_load_kwargs, **load_kwargs}
    _rebind_default_model_path(
        load_kwargs,
        model_ref=model_ref,
        variant_load_kwargs=variant_load_kwargs,
        entry_default_load_kwargs=entry.default_load_kwargs,
    )
    params = dict(payload.params or {})
    interactions = params.get("interactions")
    if interactions is None:
        default_interactions = _task_field_default(
            task,
            "interactions",
            "interaction",
            "interaction_signal",
            "action",
            target="params",
        )
        if isinstance(default_interactions, (str, list, tuple)):
            interactions = default_interactions
    task_type = str(
        params.get("task_type")
        or call_kwargs.pop("task_type", "")
        or (task.task_id if payload.task_profile_id else "")
        or entry.default_task_type
        or ""
    )
    fps = int(params.get("fps") or call_kwargs.get("fps") or defaults.get("fps", DEFAULT_SETTINGS["fps"]))
    num_frames = int(
        params.get("num_frames")
        or params.get("frames")
        or call_kwargs.get("num_frames")
        or call_kwargs.get("frame_num")
        or call_kwargs.get("frames")
        or call_kwargs.get("video_length")
        or defaults.get("num_frames", DEFAULT_SETTINGS["num_frames"])
    )
    default_input_path = ""
    if _should_use_task_default_input_path(entry, payload, task_type):
        default_input_path = str(_task_field_default(task, "input_path", "image", "video", target="input_path") or "")
    input_path = payload.input_path or _preview_input_path_from_call_kwargs(call_kwargs) or default_input_path
    _sync_input_path_to_call_kwargs(entry, task_type=task_type, input_path=input_path, call_kwargs=call_kwargs)
    if entry.model_id == "matrix-game-1" and input_path:
        call_kwargs.setdefault("image_path", input_path)
    prompt = payload.prompt or str(_task_field_default(task, "prompt", target="prompt") or entry.default_prompt or "")
    return entry, dict(
        model_id=entry.model_id,
        action="run",
        prompt=prompt,
        input_path=input_path,
        image=None,
        video=None,
        last_frame=None,
        reference_files=None,
        interactions_text=json.dumps(interactions) if interactions is not None else "",
        camera_view_text=json.dumps(params.get("camera_view")) if params.get("camera_view") is not None else "",
        task_type=task_type,
        intrinsics_text=json.dumps(params.get("intrinsics")) if params.get("intrinsics") is not None else "",
        meta_path=str(params.get("meta_path") or ""),
        panorama_path=str(params.get("panorama_path") or ""),
        scene_name=str(params.get("scene_name") or ""),
        fps=fps,
        num_frames=num_frames,
        call_kwargs_text=json.dumps(call_kwargs),
        load_kwargs_text=json.dumps(load_kwargs),
        model_ref=model_ref,
        backend=payload.backend or entry.default_backend,
        endpoint=payload.endpoint or entry.default_endpoint,
        api_key=payload.api_key,
        device=payload.device or str(defaults.get("device") or DEFAULT_SETTINGS["device"]),
        infer_metadata=contract,
    )
