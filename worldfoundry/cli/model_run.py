"""Model-specific ``worldfoundry run <model>`` CLI contracts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import Any, Mapping, Sequence

_SKIPPED_CALL_FIELDS = {"output-path", "return-dict"}
_SKIPPED_LOAD_FIELDS = {"device", "model_id", "required_components"}
_FRIENDLY_ALIASES = {
    "prompt",
    "negative-prompt",
    "input-path",
    "image",
    "images",
    "video",
    "audio",
    "state",
    "actions",
    "task-type",
    "interactions",
    "frames",
    "fps",
    "height",
    "width",
    "steps",
    "guidance-scale",
    "seed",
}


class ModelRunSchemaError(ValueError):
    """Raised when a model or requested task/variant has no discoverable CLI contract."""


@dataclass(frozen=True)
class ModelRunField:
    """One typed model-specific option and its resolved configuration target."""

    option: str
    dest: str
    scope: str
    key_path: tuple[str, ...]
    kind: str = "string"
    default: Any = None
    choices: tuple[Any, ...] = ()
    required: bool = False
    description: str = ""
    label: str = ""
    input_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "scope": self.scope,
            "key_path": list(self.key_path),
            "kind": self.kind,
            "default": self.default,
            "choices": list(self.choices),
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True)
class ModelRunSchema:
    """Resolved model identity, task, variant, and typed CLI fields."""

    requested_model_id: str
    model_id: str
    display_name: str
    description: str
    variant_id: str
    catalog_variant_id: str | None
    task_id: str
    task_choices: tuple[str, ...]
    fields: tuple[ModelRunField, ...]
    output_artifacts: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    runnable: bool = True
    runner_entry_kind: str = "runnable_runner"
    integration_status: str = "integrated"
    source_status: str = "unknown"
    runner_target: str = ""
    runtime_status: str = ""
    blocked_reason: str = ""
    schema_source: str = "studio_inference_contract"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model_id": self.requested_model_id,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "description": self.description,
            "variant_id": self.variant_id,
            "catalog_variant_id": self.catalog_variant_id,
            "task_id": self.task_id,
            "task_choices": list(self.task_choices),
            "fields": [field.to_dict() for field in self.fields],
            "output_artifacts": [dict(item) for item in self.output_artifacts],
            "notes": list(self.notes),
            "runnable": self.runnable,
            "runner_entry_kind": self.runner_entry_kind,
            "integration_status": self.integration_status,
            "source_status": self.source_status,
            "runner_target": self.runner_target,
            "runtime_status": self.runtime_status,
            "blocked_reason": self.blocked_reason,
            "schema_source": self.schema_source,
        }


@dataclass(frozen=True)
class ResolvedModelRunOptions:
    """Values collected from model-specific parser actions."""

    model_parameters: Mapping[str, Any]
    model_runtime: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    inputs: Mapping[str, Any]
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_parameters": dict(self.model_parameters),
            "model_runtime": dict(self.model_runtime),
            "generation_defaults": dict(self.generation_defaults),
            "inputs": dict(self.inputs),
            "task_id": self.task_id,
        }


def _normalise(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _workload_type(category: str, call_params: Sequence[str]) -> str:
    normalized = _normalise(category)
    params = {_normalise(item) for item in call_params}
    if "depth" in normalized or "geometry" in normalized:
        return "geometry"
    if "3d" in normalized or "scene" in normalized:
        return "3d"
    if "action" in normalized or "policy" in normalized:
        return "action"
    if params & {"images", "image", "input-image"}:
        return "i2v"
    return "video"


def _field_kind(name: str, default: Any = None) -> str:
    normalized = _normalise(name)
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int) and not isinstance(default, bool):
        return "integer"
    if isinstance(default, float):
        return "number"
    if isinstance(default, (list, tuple, dict)):
        return "json"
    if normalized.endswith(("-path", "-dir", "-root")) or normalized in {
        "model-path",
        "model-ref",
        "checkpoint",
    }:
        return "path"
    if isinstance(default, str) and default:
        lowered = default.strip().lower()
        if lowered in {"true", "false"}:
            return "boolean"
        integer_name = normalized in {"frames", "steps", "fps", "seed", "height", "width"} or normalized.startswith(
            ("num-", "max-", "min-")
        ) or normalized.endswith(("-frames", "-steps", "-fps", "-seed", "-height", "-width", "-num"))
        if integer_name:
            try:
                int(default)
            except ValueError:
                return "string"
            return "integer"
        number_name = any(
            token in normalized for token in ("scale", "guidance", "threshold", "ratio", "distance")
        )
        if number_name:
            try:
                float(default)
            except ValueError:
                return "string"
            return "number"
        return "string"
    if normalized.startswith(("is-", "use-", "enable-", "disable-", "save-", "report-")):
        return "boolean"
    if normalized in {"i2v", "return-dict", "static-scene", "low-vram", "lazy", "tiled"}:
        return "boolean"
    if normalized.endswith(("-fsdp", "-offload")):
        return "boolean"
    if normalized in {"frames", "steps", "fps", "seed", "height", "width"}:
        return "integer"
    if normalized in {"ulysses-size", "ring-size", "nproc-per-node", "torchrun-nproc"}:
        return "integer"
    if normalized.startswith(("num-", "max-", "min-")) or normalized.endswith(
        ("-frames", "-steps", "-fps", "-seed", "-height", "-width", "-num")
    ):
        return "integer"
    if any(token in normalized for token in ("scale", "guidance", "threshold", "ratio", "distance")):
        return "number"
    return "string"


def _flatten_mapping(value: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> tuple[tuple[tuple[str, ...], Any], ...]:
    rows: list[tuple[tuple[str, ...], Any]] = []
    for key, item in value.items():
        path = (*prefix, str(key))
        if isinstance(item, Mapping):
            rows.extend(_flatten_mapping(item, path))
        else:
            rows.append((path, item))
    return tuple(rows)


def _mapping_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _call_key(field_id: str, call_params: Sequence[str]) -> str:
    normalized = _normalise(field_id)
    by_normalized = {_normalise(item): str(item) for item in call_params}
    if normalized in by_normalized:
        return by_normalized[normalized]
    aliases = {
        "frames": ("num_frames", "num_output_frames", "video_length", "frame_num"),
        "steps": ("num_inference_steps", "sampling_steps", "infer_steps", "num_steps"),
        "guidance-scale": ("guidance_scale", "cfg_scale", "guidance"),
        "input-path": ("input_path", "image_path", "video_path", "data_path"),
    }
    for candidate in aliases.get(normalized, ()):
        if _normalise(candidate) in by_normalized:
            return by_normalized[_normalise(candidate)]
    return normalized.replace("-", "_")


def _default_for_call_key(defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    normalized = _normalise(key)
    for default_key, value in defaults.items():
        if _normalise(default_key) == normalized:
            return value
    return fallback


def _input_key(field_id: str, call_params: Sequence[str]) -> str:
    normalized = _normalise(field_id)
    if normalized == "prompt":
        return "prompt"
    if "video" in normalized:
        return "video"
    if "image" in normalized:
        return "image"
    params = {_normalise(item) for item in call_params}
    if params & {"image", "images", "image-path"}:
        return "image"
    if params & {"video", "videos", "video-path"}:
        return "video"
    return "input"


def _field_description(label: str, description: str, *, scope: str, display_name: str) -> str:
    if description:
        return description
    if scope == "input":
        return f"{label} used as the input for direct {display_name} inference."
    if scope == "load":
        return f"{label} used while loading the {display_name} pipeline."
    if scope == "runtime":
        return f"{label} used by the WorldFoundry runtime."
    return f"{label} passed to the pipeline for every generated sample."


def _suggestions(model_id: str) -> str:
    ids: list[str] = []
    try:
        from worldfoundry.studio.catalog import discover_catalog

        ids.extend(entry.model_id for entry in discover_catalog())
    except Exception:
        pass
    try:
        from worldfoundry.evaluation.models.catalog.zoo_registry import load_model_zoo_registry
        from worldfoundry.evaluation.utils import MODEL_ZOO_DIR

        ids.extend(entry.model_id for entry in load_model_zoo_registry(MODEL_ZOO_DIR).list())
    except Exception:
        pass
    matches = get_close_matches(_normalise(model_id), tuple(dict.fromkeys(ids)), n=5, cutoff=0.35)
    return f" Did you mean: {', '.join(matches)}?" if matches else ""


def _dedupe_text(values: Sequence[Any]) -> tuple[str, ...]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _normalise(text)
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return tuple(rows)


def _text_items(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _runtime_profile_id(value: Any) -> str:
    text = str(value or "").strip()
    prefix = "runtime-profile:"
    return text[len(prefix) :] if text.lower().startswith(prefix) else text


@lru_cache(maxsize=1)
def _runtime_profile_paths_by_stem() -> Mapping[str, tuple[Any, ...]]:
    try:
        from worldfoundry.evaluation.models.runtime.profiles import DEFAULT_RUNTIME_PROFILES_ROOT
    except Exception:
        return {}
    paths: dict[str, list[Any]] = {}
    for path in DEFAULT_RUNTIME_PROFILES_ROOT.rglob("*.y*ml"):
        if path.is_file():
            paths.setdefault(_normalise(path.stem), []).append(path)
    return {key: tuple(value) for key, value in paths.items()}


def _load_catalog_runtime_profile(entry: Any, variant: Any | None) -> tuple[Any | None, str]:
    """Load one explicit runtime profile without synthesizing the entire catalog."""
    candidates = _dedupe_text(
        (
            _runtime_profile_id(getattr(variant, "runtime_profile", None)),
            _runtime_profile_id(getattr(entry, "runtime_profile", None)),
            getattr(variant, "variant_id", None),
            getattr(entry, "model_id", None),
        )
    )
    try:
        from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile_manifests
    except Exception:
        return None, candidates[0] if candidates else ""
    paths_by_stem = _runtime_profile_paths_by_stem()
    for candidate in candidates:
        for path in paths_by_stem.get(_normalise(candidate), ()):
            try:
                profiles = load_runtime_profile_manifests(path)
            except Exception:
                continue
            exact = next(
                (
                    profile
                    for profile in profiles
                    if _normalise(profile.model_id) == _normalise(candidate)
                    or _normalise(getattr(profile, "execution", {}).get("profile_id", ""))
                    == _normalise(candidate)
                ),
                None,
            )
            if exact is not None:
                return exact, candidate
            if len(profiles) == 1:
                return profiles[0], candidate
    return None, candidates[0] if candidates else ""


def _catalog_execution_metadata(entry: Any | None, variant: Any | None) -> dict[str, Any]:
    if entry is None:
        return {
            "runnable": True,
            "runner_entry_kind": "runnable_runner",
            "integration_status": "integrated",
            "source_status": "unknown",
            "runner_target": "",
            "blocked_reason": "",
        }

    runner_target = str(
        (getattr(variant, "runner_target", None) if variant is not None else None)
        or getattr(entry, "runner_target", None)
        or ""
    )
    integration_status = str(
        getattr(variant, "integration_status", None)
        if variant is not None
        else getattr(entry, "integration_status", "planned")
    )
    if variant is None:
        runner_entry_kind = str(getattr(entry, "runner_entry_kind", "listed_only"))
        runnable = bool(getattr(entry, "is_runnable_runner_entry", False))
    elif runner_target and integration_status == "integrated":
        runner_entry_kind = "runnable_runner"
        runnable = True
    elif runner_target:
        runner_entry_kind = "runner_candidate"
        runnable = False
    else:
        runner_entry_kind = "listed_only"
        runnable = False

    blocked_reason = ""
    if not runnable:
        subject = f"variant {variant.variant_id!r}" if variant is not None else f"model {entry.model_id!r}"
        if runner_entry_kind == "listed_only":
            blocked_reason = f"{subject} has no registered runner target"
        else:
            blocked_reason = (
                f"{subject} has a runner candidate, but its integration status is "
                f"{integration_status!r} rather than 'integrated'"
            )
        notes = tuple(getattr(variant, "notes", ())) if variant is not None else tuple(getattr(entry, "notes", ()))
        explicit_reason = next(
            (str(note).split(":", 1)[-1].strip() for note in notes if str(note).lower().startswith("blocked:")),
            "",
        )
        if explicit_reason:
            blocked_reason = f"{blocked_reason}: {explicit_reason}"

    return {
        "runnable": runnable,
        "runner_entry_kind": runner_entry_kind,
        "integration_status": integration_status,
        "source_status": str(getattr(entry, "source_status", "unknown")),
        "runner_target": runner_target,
        "blocked_reason": blocked_reason,
    }


def _checkpoint_value(entry: Any, variant: Any | None, profile: Any | None) -> Any:
    refs = (
        *(tuple(getattr(variant, "checkpoint_refs", ())) if variant is not None else ()),
        *tuple(getattr(entry, "checkpoint_refs", ())),
        getattr(entry, "checkpoint", None),
    )
    for ref in refs:
        if ref is None:
            continue
        value = getattr(ref, "hf_repo_id", None) or getattr(ref, "path", None)
        if value:
            return str(value)
    if profile is not None:
        for checkpoint in tuple(getattr(profile, "checkpoints", ())):
            if not isinstance(checkpoint, Mapping):
                continue
            for key in ("repo_id", "hf_repo_id", "local_dir", "path", "uri"):
                value = checkpoint.get(key)
                if value:
                    return str(value)
    return None


def _profile_inputs(profile: Any | None) -> tuple[tuple[str, Any, bool], ...]:
    schema = getattr(profile, "input_schema", {}) if profile is not None else {}
    if not isinstance(schema, Mapping):
        return ()
    required = schema.get("required")
    optional = schema.get("optional")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        rows = [(str(key), True, True) for key in required]
        if isinstance(optional, Sequence) and not isinstance(optional, (str, bytes)):
            rows.extend((str(key), "optional", False) for key in optional)
        return tuple(rows)
    return tuple((str(key), value, False) for key, value in schema.items())


def _fallback_input_fields(entry: Any, profile: Any | None, display_name: str) -> list[ModelRunField]:
    """Translate runtime-profile inputs into conservative direct-run CLI fields."""
    fixed_metadata = {
        "model-id",
        "policy-family",
        "action-representation",
        "artifact-kind",
        "output-path",
    }
    rows = list(_profile_inputs(profile))
    enabled_modalities = [
        _normalise(name)
        for name, value, _required in rows
        if _normalise(name) in {"prompt", "instruction", "image", "video", "audio", "state"}
        and value not in (False, None, "false", "disabled")
    ]
    singleton_modality = enabled_modalities[0] if len(enabled_modalities) == 1 else ""
    fields: list[ModelRunField] = []
    seen: set[str] = set()
    for raw_name, raw_contract, explicit_required in rows:
        option_name = _normalise(raw_name)
        if option_name in fixed_metadata or raw_contract in (False, None, "false", "disabled"):
            continue
        option_name = {
            "input-image": "image",
            "input-video": "video",
            "instruction": "prompt",
            "caption": "prompt",
            "text": "prompt",
        }.get(option_name, option_name)
        option = f"--pipeline.{option_name}"
        if option in seen:
            continue
        seen.add(option)
        key = option_name.replace("-", "_")
        scope = "call" if option_name in {"actions", "action", "interactions", "task-type"} else "input"
        kind = "string"
        choices: tuple[Any, ...] = ()
        if option_name in {"image", "video", "audio", "input-path", "ref-image-path"}:
            kind = "path"
        elif option_name in {"images", "state", "actions", "action", "interactions", "controls", "proprio"}:
            kind = "json"
        elif isinstance(raw_contract, Mapping):
            kind = "json"
        elif isinstance(raw_contract, Sequence) and not isinstance(raw_contract, (str, bytes)):
            if option_name == "task-type":
                choices = tuple(raw_contract)
            else:
                kind = "json"
        required = explicit_required or (
            option_name == singleton_modality and raw_contract is True
        )
        label = option_name.replace("-", " ").title()
        fields.append(
            ModelRunField(
                option=option,
                dest=f"model_run__fallback__{option_name.replace('-', '_')}",
                scope=scope,
                key_path=(key,),
                kind=kind,
                choices=choices,
                required=required,
                description=_field_description(label, "", scope=scope, display_name=display_name),
                label=label,
                input_key=key,
            )
        )

    task_text = " ".join(str(item).lower() for item in getattr(entry, "tasks", ()))
    if not fields:
        if any(token in task_text for token in ("image", "geometry", "reconstruction", "depth", "3d")):
            fields.append(
                ModelRunField(
                    option="--pipeline.image",
                    dest="model_run__fallback__image",
                    scope="input",
                    key_path=("image",),
                    kind="path",
                    required=True,
                    description=f"Image or image-directory input for direct {display_name} inference.",
                    label="Image",
                    input_key="image",
                )
            )
        elif any(token in task_text for token in ("generation", "world", "text", "reasoning")):
            fields.append(
                ModelRunField(
                    option="--pipeline.prompt",
                    dest="model_run__fallback__prompt",
                    scope="input",
                    key_path=("prompt",),
                    kind="string",
                    required=True,
                    description=f"Text prompt for direct {display_name} inference.",
                    label="Prompt",
                    input_key="prompt",
                )
            )
    return fields


def _fallback_generation_fields(entry: Any, profile: Any | None, seen: set[str]) -> list[ModelRunField]:
    task_text = " ".join(
        str(item).lower()
        for item in (*tuple(getattr(entry, "tasks", ())), *tuple(getattr(entry, "output_artifacts", ())))
    )
    artifact_kind = str(getattr(profile, "artifact_kind", "") or "").lower()
    combined = f"{task_text} {artifact_kind}"
    if any(token in combined for token in ("action_trace", "robot_policy", "vla_policy", "world_action_model")):
        return []
    if any(token in task_text for token in ("geometry", "reconstruction", "depth")):
        return []
    if not any(token in combined for token in ("generation", "video", "image", "world")):
        return []
    known_defaults: dict[str, Any] = {}
    for source in (getattr(profile, "execution", {}), getattr(profile, "output", {})):
        if isinstance(source, Mapping):
            for name in ("seed", "num_frames", "fps", "height", "width", "num_inference_steps", "guidance_scale"):
                if name in source:
                    known_defaults[name] = source[name]
    specs = (
        ("seed", "seed", "integer", "RNG seed for deterministic generation."),
        ("frames", "num_frames", "integer", "Number of frames to generate."),
        ("fps", "fps", "integer", "Output video frame rate."),
        ("height", "height", "integer", "Output height in pixels."),
        ("width", "width", "integer", "Output width in pixels."),
        ("steps", "num_inference_steps", "integer", "Number of inference or sampling steps."),
        ("guidance-scale", "guidance_scale", "number", "Classifier-free guidance scale."),
    )
    fields: list[ModelRunField] = []
    for option_name, key, kind, description in specs:
        option = f"--pipeline.{option_name}"
        if option in seen:
            continue
        seen.add(option)
        fields.append(
            ModelRunField(
                option=option,
                dest=f"model_run__fallback__generation__{option_name.replace('-', '_')}",
                scope="call",
                key_path=(key,),
                kind=kind,
                default=known_defaults.get(key),
                description=description,
                label=option_name.replace("-", " ").title(),
            )
        )
    return fields


def _fallback_model_run_schema(
    *,
    requested_model_id: str,
    entry: Any,
    variant: Any | None,
    task_id: str | None,
) -> ModelRunSchema:
    profile, profile_id = _load_catalog_runtime_profile(entry, variant)
    display_name = str(getattr(entry, "name", None) or entry.model_id)
    task_choices = _dedupe_text(
        (
            getattr(variant, "task", None),
            *tuple(getattr(entry, "tasks", ())),
            getattr(profile, "task_family", None),
        )
    ) or ("inference",)
    selected_task = task_choices[0]
    if task_id:
        match = next((item for item in task_choices if _normalise(item) == _normalise(task_id)), None)
        if match is None:
            raise ModelRunSchemaError(
                f"unknown task profile {task_id!r} for {entry.model_id!r}; choose one of: {', '.join(task_choices)}"
            )
        selected_task = match

    fields = _fallback_input_fields(entry, profile, display_name)
    seen = {field.option for field in fields}
    fields.extend(_fallback_generation_fields(entry, profile, seen))
    checkpoint = _checkpoint_value(entry, variant, profile)
    if checkpoint is not None:
        fields.append(
            ModelRunField(
                option="--pipeline.load.model-path",
                dest="model_run__fallback__load__model_path",
                scope="load",
                key_path=("model_path",),
                kind="path",
                default=checkpoint,
                description=f"Checkpoint path or repository used to load {display_name}.",
                label="Model Path",
            )
        )
    if profile_id:
        fields.append(
            ModelRunField(
                option="--runtime.profile",
                dest="model_run__fallback__runtime__profile",
                scope="runtime",
                key_path=("runtime_profile",),
                kind="string",
                default=profile_id,
                description="Runtime profile used for artifact and lifecycle configuration.",
                label="Runtime Profile",
            )
        )
    fields.append(
        ModelRunField(
            option="--runtime.device",
            dest="model_run__fallback__runtime__device",
            scope="runtime",
            key_path=("device",),
            kind="string",
            default="cuda",
            description="PyTorch device used to load and execute the pipeline.",
            label="Device",
        )
    )

    task_text = " ".join(str(item).lower() for item in getattr(entry, "tasks", ()))
    output_kinds = tuple(getattr(entry, "output_artifacts", ()))
    if not output_kinds and any(token in task_text for token in ("robot", "policy", "vla", "action")):
        output_kinds = ("action_trace",)
    if not output_kinds and any(token in task_text for token in ("geometry", "reconstruction", "3d")):
        output_kinds = ("generated_3d_asset",)
    if not output_kinds and getattr(profile, "artifact_kind", None):
        output_kinds = (getattr(profile, "artifact_kind"),)
    outputs = tuple(
        {
            "artifact_id": str(kind),
            "kind": str(kind),
            "required": False,
            "preview": str(kind) in {"generated_image", "generated_video", "generated_world"},
            "description": f"{display_name} {str(kind).replace('_', ' ')} output.",
        }
        for kind in output_kinds
    )
    status = _catalog_execution_metadata(entry, variant)
    notes = _dedupe_text(
        (
            *tuple(getattr(entry, "notes", ())),
            *(tuple(getattr(variant, "notes", ())) if variant is not None else ()),
            *(tuple(getattr(profile, "notes", ())) if profile is not None else ()),
        )
    )
    task_summary = ", ".join(task_choices[:4])
    return ModelRunSchema(
        requested_model_id=requested_model_id,
        model_id=entry.model_id,
        display_name=display_name,
        description=f"Model Zoo runtime contract for {task_summary}.",
        variant_id=getattr(variant, "variant_id", None) or "default",
        catalog_variant_id=getattr(variant, "variant_id", None),
        task_id=selected_task,
        task_choices=task_choices,
        fields=tuple(fields),
        output_artifacts=outputs,
        notes=notes,
        **status,
        runtime_status=str(getattr(profile, "runtime_status", "") or ""),
        schema_source="model_zoo_runtime_fallback",
    )


@lru_cache(maxsize=512)
def load_model_run_schema(
    model_id: str,
    variant_id: str | None = None,
    task_id: str | None = None,
) -> ModelRunSchema:
    """Resolve one model into a Studio contract or Model Zoo/runtime fallback."""
    from worldfoundry.core.inference import model_inference_spec
    from worldfoundry.evaluation.models.catalog.schema import select_default_variant
    from worldfoundry.evaluation.models.catalog.zoo_registry import load_model_zoo_registry
    from worldfoundry.evaluation.utils import MODEL_ZOO_DIR
    from worldfoundry.studio.catalog import find_entry

    canonical_model_id = model_id
    catalog_variant_id: str | None = None
    catalog_aliases: tuple[str, ...] = ()
    selected_catalog_variant = None
    catalog_registry = load_model_zoo_registry(MODEL_ZOO_DIR)
    try:
        catalog_entry = catalog_registry.get(model_id)
    except (KeyError, TypeError, ValueError):
        catalog_entry = None
    if catalog_entry is not None:
        canonical_model_id = catalog_entry.model_id
        catalog_aliases = tuple(catalog_entry.aliases)

    entry = None
    for candidate in (model_id, canonical_model_id, *catalog_aliases):
        try:
            entry = find_entry(candidate)
        except KeyError:
            continue
        else:
            break
    if catalog_entry is None and entry is not None:
        for candidate in (entry.model_id, *entry.aliases):
            try:
                catalog_entry = catalog_registry.get(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            else:
                break
        if catalog_entry is None:
            compact_studio_id = "".join(character for character in entry.model_id if character.isalnum()).lower()
            compact_matches = [
                item
                for item in catalog_registry.list()
                if "".join(character for character in item.model_id if character.isalnum()).lower()
                == compact_studio_id
            ]
            if len(compact_matches) == 1:
                catalog_entry = compact_matches[0]
        if catalog_entry is not None:
            canonical_model_id = catalog_entry.model_id
            catalog_aliases = tuple(catalog_entry.aliases)

    if catalog_entry is not None:
        requested_variants = _dedupe_text(
            (variant_id,)
            if variant_id is not None
            else (
                model_id if _normalise(model_id) != _normalise(catalog_entry.model_id) else "",
                *(
                    tuple(entry.aliases)
                    if entry is not None
                    and _normalise(model_id) != _normalise(catalog_entry.model_id)
                    else ()
                ),
            )
        )
        selected_catalog_variant = next(
            (
                item
                for item in catalog_entry.variants
                if any(
                    _normalise(item.variant_id) == _normalise(requested_variant)
                    for requested_variant in requested_variants
                )
            ),
            None,
        )
        if selected_catalog_variant is not None:
            catalog_variant_id = selected_catalog_variant.variant_id
        elif variant_id is None and not catalog_entry.runner_target:
            selected_catalog_variant = select_default_variant(
                catalog_entry,
                allow_runner_target_fallback=True,
            )
            if selected_catalog_variant is not None:
                catalog_variant_id = selected_catalog_variant.variant_id

    if entry is None:
        if catalog_entry is None:
            raise ModelRunSchemaError(f"unknown model {model_id!r}.{_suggestions(model_id)}")
        if variant_id and selected_catalog_variant is None:
            choices = tuple(item.variant_id for item in catalog_entry.variants)
            suffix = f"; choose one of: {', '.join(choices)}" if choices else ""
            raise ModelRunSchemaError(
                f"unknown model variant {variant_id!r} for {catalog_entry.model_id!r}{suffix}"
            )
        return _fallback_model_run_schema(
            requested_model_id=model_id,
            entry=catalog_entry,
            variant=selected_catalog_variant,
            task_id=task_id,
        )

    spec = model_inference_spec(
        model_family_id=entry.model_id,
        display_name=entry.display_name,
        default_model_ref=entry.default_model_ref,
        default_load_kwargs=entry.default_load_kwargs,
        default_call_kwargs=entry.default_call_kwargs,
        supports_stream=entry.supports_stream,
        workload_type=_workload_type(entry.category, entry.call_params),
        supported_call_params=entry.call_params,
    )
    try:
        variant = spec.variant(None if catalog_variant_id else variant_id)
    except ValueError as exc:
        raise ModelRunSchemaError(str(exc)) from exc
    try:
        task = spec.task(task_id)
    except ValueError as exc:
        raise ModelRunSchemaError(str(exc)) from exc

    call_defaults = {
        **dict(entry.default_call_kwargs or {}),
        **dict(task.default_call_kwargs or {}),
        **dict(variant.call_kwargs or {}),
    }
    load_defaults = {
        **dict(entry.default_load_kwargs or {}),
        **dict(variant.load_kwargs or {}),
    }
    if entry.default_model_ref and not any(
        key in load_defaults for key in ("model_path", "model_ref", "pretrained_model_path")
    ):
        load_defaults["model_path"] = entry.default_model_ref

    fields: list[ModelRunField] = []
    seen_options: set[str] = set()
    for input_field in task.inputs:
        field_id = _normalise(input_field.field_id)
        if field_id in _SKIPPED_CALL_FIELDS:
            continue
        if (
            field_id == "input-path"
            and "t2v" in _normalise(entry.model_id)
            and "i2v" not in _normalise(entry.model_id)
        ):
            continue
        target = _normalise(input_field.target)
        scope = "input" if target in {"prompt", "input-path"} else "call"
        key = _call_key(field_id, entry.call_params)
        default = _default_for_call_key(call_defaults, key, input_field.default)
        if field_id == "prompt" and default in (None, ""):
            default = entry.default_prompt or None
        if scope == "input" and field_id != "prompt" and entry.default_input_path:
            default = entry.default_input_path
        inferred_kind = _field_kind(key, default)
        kind = input_field.kind
        if _normalise(kind) == "string" and inferred_kind != "string":
            kind = inferred_kind
        option_name = field_id
        option = f"--pipeline.{option_name}"
        if option in seen_options:
            continue
        seen_options.add(option)
        fields.append(
            ModelRunField(
                option=option,
                dest=f"model_run__call__{option_name.replace('-', '_')}",
                scope=scope,
                key_path=(key,),
                kind=kind or inferred_kind,
                default=default,
                choices=tuple(input_field.choices or ()),
                required=bool(input_field.required),
                description=_field_description(
                    input_field.label,
                    input_field.description,
                    scope=scope,
                    display_name=entry.display_name,
                ),
                label=input_field.label,
                input_key=_input_key(field_id, entry.call_params),
            )
        )

    for path, default in _flatten_mapping(load_defaults):
        option_name = ".".join(_normalise(part) for part in path)
        option = f"--pipeline.load.{option_name}"
        if option in seen_options:
            continue
        seen_options.add(option)
        label = " ".join(part.replace("_", " ").title() for part in path)
        fields.append(
            ModelRunField(
                option=option,
                dest="model_run__load__" + "__".join(path),
                scope="load",
                key_path=tuple(path),
                kind=_field_kind(path[-1], default),
                default=default,
                description=_field_description(
                    label,
                    "",
                    scope="load",
                    display_name=entry.display_name,
                ),
                label=label,
            )
        )

    flattened_load_leaves = {path[-1] for path, _ in _flatten_mapping(load_defaults)}
    call_parameter_names = {str(item) for item in entry.call_params}
    for raw_name in entry.load_params:
        if (
            raw_name in _SKIPPED_LOAD_FIELDS
            or raw_name in flattened_load_leaves
            or raw_name in call_parameter_names
        ):
            continue
        option_name = _normalise(raw_name)
        option = f"--pipeline.load.{option_name}"
        if option in seen_options:
            continue
        seen_options.add(option)
        label = raw_name.replace("_", " ").title()
        fields.append(
            ModelRunField(
                option=option,
                dest=f"model_run__load__{raw_name}",
                scope="load",
                key_path=(raw_name,),
                kind=_field_kind(raw_name),
                description=_field_description(label, "", scope="load", display_name=entry.display_name),
                label=label,
            )
        )

    fields.append(
        ModelRunField(
            option="--runtime.device",
            dest="model_run__runtime__device",
            scope="runtime",
            key_path=("device",),
            kind="string",
            default=str(_mapping_value(load_defaults, ("device",)) or "cuda"),
            description="PyTorch device used to load and execute the pipeline.",
            label="Device",
        )
    )

    outputs = tuple(output.to_dict() for output in task.outputs)
    profile, _profile_id = (
        _load_catalog_runtime_profile(catalog_entry, selected_catalog_variant)
        if catalog_entry is not None
        else (None, "")
    )
    status = _catalog_execution_metadata(catalog_entry, selected_catalog_variant)
    notes = _dedupe_text(
        (
            *_text_items(spec.notes),
            *_text_items(entry.notes),
            *(_text_items(getattr(catalog_entry, "notes", ())) if catalog_entry is not None else ()),
            *(_text_items(getattr(selected_catalog_variant, "notes", ())) if selected_catalog_variant is not None else ()),
        )
    )
    return ModelRunSchema(
        requested_model_id=model_id,
        model_id=canonical_model_id,
        display_name=entry.display_name,
        description=task.description or entry.summary,
        variant_id=catalog_variant_id or variant.variant_id,
        catalog_variant_id=catalog_variant_id,
        task_id=task.task_id,
        task_choices=tuple(item.task_id for item in spec.tasks),
        fields=tuple(fields),
        output_artifacts=outputs,
        notes=notes,
        **status,
        runtime_status=str(getattr(profile, "runtime_status", "") or ""),
        schema_source="studio_inference_contract",
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected JSON: {exc.msg}") from exc


def _argument_type(kind: str):
    normalized = _normalise(kind)
    if normalized in {"integer", "int"}:
        return int
    if normalized in {"number", "float"}:
        return float
    if normalized in {"boolean", "bool"}:
        return _parse_bool
    if normalized == "json":
        return _parse_json
    return str


def _metavar(kind: str) -> str:
    normalized = _normalise(kind)
    return {
        "integer": "INT",
        "int": "INT",
        "number": "FLOAT",
        "float": "FLOAT",
        "boolean": "{True,False}",
        "bool": "{True,False}",
        "path": "PATH",
        "json": "JSON",
    }.get(normalized, "STR")


def register_model_run_arguments(parser: argparse.ArgumentParser, schema: ModelRunSchema) -> None:
    """Register typed model fields on the existing ``run`` subparser."""
    existing_options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    status_parts = [
        schema.runner_entry_kind,
        f"integration={schema.integration_status}",
        f"source={schema.source_status}",
        f"schema={schema.schema_source}",
    ]
    if schema.runtime_status:
        status_parts.append(f"runtime={schema.runtime_status}")
    status_description = "; ".join(status_parts) + "."
    if schema.blocked_reason:
        status_description += f" Execution blocked: {schema.blocked_reason}."
    status_group = parser.add_argument_group("model status", status_description)
    status_group.add_argument(
        "--model-status",
        action="store_true",
        dest="print_config",
        help="Print the resolved model contract and execution readiness without loading weights.",
    )
    pipeline_group = parser.add_argument_group(
        "pipeline options",
        schema.description or f"Generation configuration for {schema.display_name}.",
    )
    if schema.task_choices:
        pipeline_group.add_argument(
            "--pipeline.task-profile",
            dest="model_run_task",
            choices=schema.task_choices,
            default=schema.task_id,
            help="Inference task profile used for direct model execution.",
        )
    load_group = parser.add_argument_group(
        "pipeline.load options",
        f"Checkpoint and component configuration used to instantiate {schema.display_name}.",
    )
    runtime_group = parser.add_argument_group(
        "runtime options",
        "Device and process settings owned by the WorldFoundry runner.",
    )

    for field in schema.fields:
        group = load_group if field.scope == "load" else runtime_group if field.scope == "runtime" else pipeline_group
        option_strings = [field.option]
        short_name = "--" + field.option.removeprefix("--pipeline.")
        if (
            field.scope in {"input", "call"}
            and field.option.removeprefix("--pipeline.") in _FRIENDLY_ALIASES
            and short_name not in existing_options
        ):
            option_strings.append(short_name)
            existing_options.add(short_name)
        if field.scope == "load" and len(field.key_path) > 1:
            flat_option = f"--pipeline.{_normalise(field.key_path[-1])}"
            if flat_option not in existing_options:
                option_strings.append(flat_option)
                existing_options.add(flat_option)
        kwargs: dict[str, Any] = {
            "dest": field.dest,
            "type": _argument_type(field.kind),
            "default": field.default,
            "help": field.description,
            "metavar": _metavar(field.kind),
        }
        if field.choices:
            converter = kwargs["type"]
            kwargs["choices"] = tuple(converter(choice) for choice in field.choices)
            kwargs.pop("metavar", None)
        group.add_argument(*option_strings, **kwargs)
        existing_options.update(option_strings)


def resolve_model_run_options(args: argparse.Namespace) -> ResolvedModelRunOptions:
    """Collect dynamic parser values into load/runtime/call/input mappings."""
    schema: ModelRunSchema | None = getattr(args, "model_run_schema", None)
    if schema is None:
        return ResolvedModelRunOptions({}, {}, {}, {}, "")
    model_parameters: dict[str, Any] = {}
    model_runtime: dict[str, Any] = {}
    generation_defaults: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for field in schema.fields:
        value = getattr(args, field.dest, None)
        if value is None:
            continue
        if field.scope == "load":
            _set_nested(model_parameters, field.key_path, value)
        elif field.scope == "runtime":
            _set_nested(model_runtime, field.key_path, value)
        elif field.scope == "input":
            inputs[field.input_key or field.key_path[-1]] = value
        else:
            _set_nested(generation_defaults, field.key_path, value)
    return ResolvedModelRunOptions(
        model_parameters=model_parameters,
        model_runtime=model_runtime,
        generation_defaults=generation_defaults,
        inputs=inputs,
        task_id=str(getattr(args, "model_run_task", None) or schema.task_id),
    )


def model_id_from_run_argv(argv: Sequence[str]) -> str | None:
    """Return a model reference from positional or legacy singular run syntax."""
    if not argv or argv[0] != "run":
        return None
    if len(argv) > 1 and argv[1] and not argv[1].startswith("-"):
        return argv[1]
    values: list[str] = []
    for index, item in enumerate(argv[1:]):
        if item in {"--model", "--model-id"} and index + 2 < len(argv):
            values.append(argv[index + 2])
        elif item.startswith("--model=") or item.startswith("--model-id="):
            values.append(item.split("=", 1)[1])
    unique = tuple(dict.fromkeys(value for value in values if value))
    return unique[0] if len(unique) == 1 else None


def option_from_argv(argv: Sequence[str], option: str) -> str | None:
    """Read one ``--option value`` or ``--option=value`` token without parsing."""
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None
