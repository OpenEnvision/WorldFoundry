"""Pipeline loading, target resolution, and ``from_pretrained`` dispatch.

Resolves :class:`PipelineRoute` and :class:`PipelineRunnerSpec` from
:class:`WorldModelConfig`, dynamically imports pipeline classes, and
invokes their ``from_pretrained`` method with signature-aware kwargs.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.api import WorldModelConfig

from .bindings import PipelineRoute, first_text, resolve_pipeline_route, runtime_profile_id

GENERATION_DEFAULTS_PARAMETER = "_worldfoundry_generation_defaults"


# ── Route resolution ────────────────────────────────────────────────

def pipeline_route_from_config(
    config: WorldModelConfig,
    parameters: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> PipelineRoute:
    """Resolve the pipeline route (target, binding, source) from model config, parameters, and runtime maps.

    Searches for ``runtime_profile``, ``pipeline_binding``, and
    ``pipeline_target`` values across explicit overrides, resolved config
    metadata, and finally manifest defaults.

    Raises:
        ValueError: If a binding is declared but cannot be resolved, or if no
            target or binding is found at all.
    """
    manifest_metadata = dict(config.manifest.metadata or {}) if config.manifest is not None else {}
    profile_value = first_text(
        parameters.get("runtime_profile"),
        runtime.get("runtime_profile"),
        config.metadata.get("resolved_runtime_profile"),
        config.metadata.get("runtime_profile"),
        config.variant,
        manifest_metadata.get("default_runtime_profile"),
        manifest_metadata.get("runtime_profile"),
    )
    binding_id = first_text(
        parameters.get("pipeline_binding"),
        runtime.get("pipeline_binding"),
        config.metadata.get("pipeline_binding"),
        manifest_metadata.get("default_pipeline_binding"),
        manifest_metadata.get("pipeline_binding"),
    )
    binding_root = first_text(
        parameters.get("pipeline_bindings_root"),
        runtime.get("pipeline_bindings_root"),
        config.metadata.get("pipeline_bindings_root"),
        manifest_metadata.get("pipeline_bindings_root"),
    )
    alias_root = first_text(
        parameters.get("pipeline_aliases_root"),
        runtime.get("pipeline_aliases_root"),
        config.metadata.get("pipeline_aliases_root"),
        manifest_metadata.get("pipeline_aliases_root"),
    )
    route = resolve_pipeline_route(
        model_id=first_text(parameters.get("model_id"), config.model_id),
        pipeline_target=first_text(
            parameters.get("pipeline_target"),
            runtime.get("pipeline_target"),
            config.metadata.get("pipeline_target"),
            manifest_metadata.get("default_pipeline_target"),
            manifest_metadata.get("pipeline_target"),
        ),
        pipeline_binding=binding_id,
        runtime_profile=profile_value,
        binding_root=binding_root or None,
        alias_root=alias_root or None,
    )
    if route is not None:
        return route
    if binding_id:
        raise ValueError(
            f"{config.model_id!r} declares pipeline_binding {binding_id!r}, but no matching binding was found."
        )
    raise ValueError(
        f"{config.model_id!r} requires a pipeline_target or a resolvable pipeline_binding in the model-zoo entry, "
        "variant, runtime, runner parameters, or data/models/bindings/pipelines."
    )


# ── Runner spec dataclass ───────────────────────────────────────────

@dataclass(frozen=True)
class PipelineRunnerSpec:
    """Resolved, side-effect-free inputs needed to construct a pipeline runner.

    Attributes:
        model_id: Canonical identifier of the model.
        pipeline_target: ``module:Class`` string used to dynamically import the pipeline.
        runtime_profile_id: Normalized runtime profile for variant selection.
        model_path: Primary model path — may be a mapping of component paths or a single string.
        fallback_model_path: Secondary model path tried when the primary path is not accepted by
            the pipeline's ``from_pretrained`` signature.
        required_components: Optional mapping of component names to paths/IDs.
        device: Target device string (e.g. ``"cuda"``).
        output_dir: Optional directory for pipeline output artifacts.
        generation_defaults: Defaults merged into each generation request.
    """

    model_id: str
    pipeline_target: str
    runtime_profile_id: str
    model_path: Any
    fallback_model_path: Any
    required_components: Mapping[str, Any] | None
    device: str
    output_dir: Path | None = None
    generation_defaults: Mapping[str, Any] | None = None


# ── Dynamic import & HFD cache resolution ───────────────────────────

def import_pipeline_target(target: str) -> Any:
    """Import a dynamic pipeline target class by ``module:Class`` string.

    Args:
        target: A string in ``"module.path:ClassName"`` format.

    Returns:
        The resolved class or attribute object.

    Raises:
        ValueError: If ``target`` does not follow the ``module:Class`` syntax.
    """
    module_name, separator, attr_name = target.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"pipeline_target must use 'module:Class' syntax: {target!r}")
    module = importlib.import_module(module_name)
    # Traverse dotted attribute chains (e.g. "Class.Inner").
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_hfd_cached_repo(reference_path: Any, repo_id: str) -> str:
    """Prefer a sibling cached HFD repo when a model path is already under an HFD cache.

    If ``repo_id`` is already a valid local path, return it directly.
    Otherwise, search the parent directories of ``reference_path`` for an
    ``hfd`` cache folder and look for a sibling directory named
    ``<owner>--<repo>`` (the HFD cache naming convention).

    Args:
        reference_path: An existing path or mapping whose values are paths, used as
            the search anchor.
        repo_id: A HuggingFace-style repo ID (``"owner/repo"``) or local path.

    Returns:
        The resolved local path string, or the original ``repo_id`` if no cache
        sibling is found.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        return repo_id

    candidate_path = Path(repo_id).expanduser()
    if candidate_path.exists():
        return str(candidate_path)

    reference_values: list[str] = []
    if isinstance(reference_path, Mapping):
        reference_values.extend(str(value) for value in reference_path.values() if isinstance(value, str))
    elif isinstance(reference_path, str):
        reference_values.append(reference_path)

    repo_dir_name = repo_id.replace("/", "--")
    for value in reference_values:
        path = Path(value).expanduser()
        if not path.exists():
            continue
        if path.is_file():
            path = path.parent
        for current in (path, *path.parents):
            if current.name == "hfd":
                sibling = current / repo_dir_name
                if sibling.exists():
                    return str(sibling)
                break
            if current.parent.name == "hfd":
                sibling = current.parent / repo_dir_name
                if sibling.exists():
                    return str(sibling)
                break

    return repo_id


# ── Model-path entry helpers ────────────────────────────────────────

def resolve_model_path_entry(model_path: Any, key: str) -> Any:
    """Resolve a required model-path entry from either a single path or a mapping.

    When ``model_path`` is a mapping, look up ``key`` and raise if missing.
    Otherwise return ``model_path`` directly (single-path case).

    Raises:
        KeyError: If ``key`` is absent from a mapping ``model_path``.
    """
    if isinstance(model_path, Mapping):
        if key not in model_path:
            raise KeyError(f"Expected key '{key}' in model_path dict, but only found: {list(model_path.keys())}")
        return model_path[key]
    return model_path


def resolve_optional_model_path_entry(model_path: Any, key: str, default: str) -> str:
    """Resolve an optional path entry and prefer sibling HFD cache directories for repo defaults."""
    value = model_path.get(key, default) if isinstance(model_path, Mapping) else default
    return resolve_hfd_cached_repo(model_path, value)


def load_named_pipeline(model_path: Any, device: str, pipeline_cls: Any, model_id: str) -> Any:
    """Load a named WorldFoundry pipeline with the standard ``from_pretrained`` contract.

    If ``model_path`` is a mapping, its entries are merged into the options
    dict; otherwise the path is set as ``repo_root``.
    """
    options = {"model_id": model_id}
    if isinstance(model_path, Mapping):
        options.update(model_path)
        # NOTE: preserve the explicit model_id over any merged value.
        options.setdefault("model_id", model_id)
    elif model_path:
        options["repo_root"] = str(model_path)
    return pipeline_cls.from_pretrained(model_path=options, device=device, model_id=model_id)


# ── Spec building ───────────────────────────────────────────────────

def pipeline_target_from_config(
    config: WorldModelConfig,
    parameters: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    """Extract and resolve the pipeline target class string from config."""
    pipeline_target, _, _ = pipeline_route_from_config(config, parameters, runtime)
    return pipeline_target


def build_pipeline_runner_spec(config: WorldModelConfig) -> PipelineRunnerSpec:
    """Construct a :class:`PipelineRunnerSpec` representing all resolved parameters for pipeline creation.

    Merges model path, runtime profile, and device settings from
    ``config.parameters``, ``config.runtime``, and manifest metadata.
    """
    parameters = dict(config.parameters or {})
    runtime = dict(config.runtime or {})
    generation_defaults = parameters.pop(GENERATION_DEFAULTS_PARAMETER, None)
    if generation_defaults is not None and not isinstance(generation_defaults, Mapping):
        raise TypeError(f"{GENERATION_DEFAULTS_PARAMETER} must be a mapping when provided.")
    model_id = str(parameters.get("model_id") or config.model_id)
    manifest_metadata = dict(config.manifest.metadata or {}) if config.manifest is not None else {}
    # NOTE: runtime_profile is resolved across parameters, runtime, manifest, config metadata, and model_id.
    resolved_runtime_profile_id = runtime_profile_id(
        first_text(
            parameters.get("runtime_profile"),
            runtime.get("runtime_profile"),
            config.metadata.get("resolved_runtime_profile"),
            config.metadata.get("runtime_profile"),
            config.variant,
            manifest_metadata.get("default_runtime_profile"),
            manifest_metadata.get("runtime_profile"),
            model_id,
        )
    )
    pipeline_target = pipeline_target_from_config(config, parameters, runtime)
    required_components = parameters.get("required_components", runtime.get("required_components"))
    if required_components is not None and not isinstance(required_components, Mapping):
        raise TypeError("required_components must be a mapping when provided.")

    # Build a model_path dict that includes all parameters plus runtime extras.
    model_path = {**parameters, "model_id": model_id}
    if config.variant:
        model_path.setdefault("variant_id", config.variant)
    model_path.setdefault("profile_id", resolved_runtime_profile_id)
    model_path.setdefault("runtime_profile", resolved_runtime_profile_id)
    # NOTE: pipeline_target is consumed at import time, not passed to from_pretrained.
    model_path.pop("pipeline_target", None)
    for key in ("profile_path", "manifest_path", "acquisition_root", "hf_models_root"):
        if key in runtime and key not in model_path:
            model_path[key] = runtime[key]

    # Resolve a fallback model path for pipelines that expect a single string.
    fallback_model_path = (
        first_text(
            parameters.get("model_path"),
            runtime.get("model_path"),
            parameters.get("pretrained_model_path"),
            runtime.get("pretrained_model_path"),
            parameters.get("repo_id"),
            parameters.get("repo_root"),
        )
        or model_path
    )
    output_dir = Path(runtime["output_dir"]) if runtime.get("output_dir") else None
    return PipelineRunnerSpec(
        model_id=model_id,
        pipeline_target=pipeline_target,
        runtime_profile_id=resolved_runtime_profile_id,
        model_path=model_path,
        fallback_model_path=fallback_model_path,
        required_components=dict(required_components) if isinstance(required_components, Mapping) else None,
        device=str(runtime.get("device", parameters.get("device", "cuda"))),
        output_dir=output_dir,
        generation_defaults=(dict(generation_defaults) if isinstance(generation_defaults, Mapping) else None),
    )


# ── from_pretrained dispatch ────────────────────────────────────────

def call_pipeline_from_pretrained(
    pipeline_cls: Any,
    *,
    model_path: Any,
    fallback_model_path: Any,
    required_components: Mapping[str, Any] | None,
    device: str,
    model_id: str,
) -> Any:
    """Invoke the ``from_pretrained`` class method with signature inspection and fallbacks.

    Inspects the callable's signature to determine which calling convention
    to use:

    1. **Unified kwargs** — if the signature accepts all four unified keys
       (``model_path``, ``required_components``, ``device``, ``model_id``),
       or has ``**kwargs`` with no explicit positional/keyword params.
    2. **Subset kwargs** — passes only the keys that exist in the signature,
       mapping ``fallback_model_path`` to ``pretrained_model_path`` or
       ``model_path`` to ``config`` when those names appear instead.
    3. **Single positional** — if the first required positional parameter
       has a conventional name like ``model_path``, ``path``, or ``repo_id``,
       pass ``fallback_model_path`` as a positional argument.
    4. **No parameters** — call with no arguments.

    Args:
        pipeline_cls: The pipeline class with a ``from_pretrained`` method.
        model_path: Primary model path or options mapping.
        fallback_model_path: Secondary path for legacy signatures.
        required_components: Optional component requirements mapping.
        device: Target device string.
        model_id: Canonical model identifier.

    Returns:
        The instantiated pipeline object.

    Raises:
        TypeError: If ``pipeline_cls`` has no callable ``from_pretrained``.
    """
    loader = getattr(pipeline_cls, "from_pretrained", None)
    if not callable(loader):
        raise TypeError(f"{pipeline_cls!r} does not define a callable from_pretrained().")

    # ── Unified kwargs convention (preferred) ─────────────────────
    unified_kwargs = {
        "model_path": model_path,
        "required_components": required_components,
        "device": device,
        "model_id": model_id,
    }

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        # Cannot inspect — fall back to passing all unified kwargs.
        return loader(**unified_kwargs)

    parameters = signature.parameters
    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    required_positional = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    explicit_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    # ── Check whether the unified kwargs can be used directly ─────
    supports_unified = all(name in parameters for name in unified_kwargs) or (
        has_var_kwargs and not explicit_parameters
    )
    if supports_unified and (
        not required_positional or all(parameter.name in unified_kwargs for parameter in required_positional)
    ):
        return loader(**unified_kwargs)

    # ── Subset kwargs: pass only names that exist in the signature ─
    kwargs: dict[str, Any] = {}
    # NOTE: when using subset kwargs, model_path is replaced with
    # fallback_model_path because legacy signatures expect a single string.
    for name, value in (
        ("model_path", fallback_model_path),
        ("required_components", required_components),
        ("device", device),
        ("model_id", model_id),
    ):
        if name in parameters:
            kwargs[name] = value

    # Map to legacy parameter names when the signature uses them.
    if "pretrained_model_path" in parameters:
        kwargs["pretrained_model_path"] = fallback_model_path
    if "config" in parameters:
        kwargs["config"] = model_path

    if kwargs:
        return loader(**kwargs)

    # ── Single positional argument fallback ───────────────────────
    if required_positional:
        first = required_positional[0]
        # NOTE: conventional first-arg names receive the fallback string.
        if first.name in {"model_path", "pretrained_model_path", "path", "repo_id", "checkpoint"}:
            return loader(fallback_model_path)
        return loader(model_path)

    # No parameters at all — call with no arguments.
    return loader()


# ── Pipeline loading ────────────────────────────────────────────────

def load_pipeline_from_spec(spec: PipelineRunnerSpec) -> Any:
    """Import and instantiate a pipeline based on a resolved :class:`PipelineRunnerSpec`.

    Ensures the inference infrastructure is installed, imports the target
    class, and dispatches to :func:`call_pipeline_from_pretrained`.
    """
    from worldfoundry.core import install_worldfoundry_inference_infra

    install_worldfoundry_inference_infra()
    pipeline_cls = import_pipeline_target(spec.pipeline_target)
    return call_pipeline_from_pretrained(
        pipeline_cls,
        model_path=spec.model_path,
        fallback_model_path=spec.fallback_model_path,
        required_components=spec.required_components,
        device=spec.device,
        model_id=spec.model_id,
    )


def load_pipeline_from_config(config: WorldModelConfig) -> tuple[PipelineRunnerSpec, Any]:
    """Resolve specification and load a pipeline instance directly from :class:`WorldModelConfig`.

    Returns both the :class:`PipelineRunnerSpec` and the loaded pipeline
    object so callers can inspect the spec while using the pipeline.
    """
    from worldfoundry.core import install_worldfoundry_inference_infra

    install_worldfoundry_inference_infra()
    spec = build_pipeline_runner_spec(config)
    return spec, load_pipeline_from_spec(spec)


__all__ = [
    "GENERATION_DEFAULTS_PARAMETER",
    "PipelineRunnerSpec",
    "build_pipeline_runner_spec",
    "call_pipeline_from_pretrained",
    "first_text",
    "import_pipeline_target",
    "load_named_pipeline",
    "load_pipeline_from_config",
    "load_pipeline_from_spec",
    "pipeline_route_from_config",
    "pipeline_target_from_config",
    "resolve_hfd_cached_repo",
    "resolve_model_path_entry",
    "resolve_optional_model_path_entry",
]
