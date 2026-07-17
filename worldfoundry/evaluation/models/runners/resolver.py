"""Resolver: maps model IDs and runner targets to fully instantiated runner objects.

Provides two resolution paths:

- **Direct runner targets**: Build a :class:`WorldModelConfig` from ``model_id``
  and ``runner`` strings, then instantiate via the registry.
- **Model-zoo catalog entries**: Resolve a zoo entry (with optional variant
  selection) to a :class:`ResolvedModelZooConfig`, then instantiate the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.runners.registry import default_model_runner_registry
from worldfoundry.evaluation.models.import_target import load_attr
from worldfoundry.evaluation.models.runners.builtins import (
    list_builtin_runtime_runners,
)
from worldfoundry.evaluation.models.pipelines.bindings import (
    first_text,
    resolve_pipeline_route,
    runtime_profile_execution_metadata,
)


class ModelResolutionError(ValueError):
    """Raised when a model id or runner target cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedWorldModel:
    """Resolved world-model runner ready for execution.

    Attributes:
        model_id: The canonical model identifier.
        runner: Instantiated runner object.
        source: Provenance — ``"runner_target"`` or ``"model_zoo"``.
        runner_target: ``module:Class`` path used to resolve the runner class.
        diagnostics: Optional structured metadata about the resolution process.
    """

    model_id: str
    runner: Any
    source: str
    runner_target: str | None = None
    diagnostics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the resolved model to a JSON-friendly dictionary."""
        return {
            "model_id": self.model_id,
            "source": self.source,
            "runner_target": self.runner_target,
            "runner_class": f"{self.runner.__class__.__module__}:{self.runner.__class__.__qualname__}",
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True)
class ResolvedModelZooConfig:
    """Resolved configuration and diagnostics for a model-zoo catalog entry.

    Attributes:
        model_id: The canonical model identifier from the zoo entry.
        config: Fully resolved :class:`WorldModelConfig` ready for runner creation.
        runner_target: ``module:Class`` path used to resolve the runner class.
        diagnostics: Structured metadata describing the resolution and variant
            selection process.
    """

    model_id: str
    config: WorldModelConfig
    runner_target: str
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the resolved zoo config to a JSON-friendly dictionary."""
        return {
            "model_id": self.model_id,
            "runner_target": self.runner_target,
            "config": self.config.to_dict(),
            "diagnostics": dict(self.diagnostics or {}),
        }


def import_object(target: str) -> Any:
    """Import a dynamic runner target by ``module:attribute`` string.

    Args:
        target: A dotted ``module:attribute`` string (e.g.
            ``"some_pkg.runners:MyRunner"``).

    Raises:
        ModelResolutionError: If the target does not use the required
            ``module:attribute`` syntax.
    """
    module_name, separator, attr_name = target.partition(":")
    if not separator or not module_name or not attr_name:
        raise ModelResolutionError(f"runner target must use 'module:attribute' syntax: {target!r}")
    return load_attr(module_name, attr_name)


def list_builtin_model_runners() -> tuple[Any, ...]:
    """Return the list of all registered built-in runtime runners."""
    return list_builtin_runtime_runners()


def _config_from_input(
    model_id: str | None = None,
    *,
    runner: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    config: WorldModelConfig | Mapping[str, Any] | None = None,
) -> WorldModelConfig:
    """Build and resolve a :class:`WorldModelConfig` from flexible keyword or config inputs.

    When ``config`` is already a :class:`WorldModelConfig` or a mapping, it is
    returned/coerced directly.  Otherwise ``model_id`` and ``runner`` are
    required to construct a fresh config.

    Args:
        model_id: Canonical model identifier (required when ``config`` is ``None``).
        runner: ``module:Class`` runner target (required when ``config`` is ``None``).
        parameters: Model parameters forwarded to :class:`WorldModelConfig`.
        runtime: Runtime overrides forwarded to :class:`WorldModelConfig`.
        config: An existing config object or mapping that takes precedence.

    Raises:
        ModelResolutionError: If neither a ``config`` nor both ``model_id``
            and ``runner`` are provided.
    """
    if isinstance(config, WorldModelConfig):
        return config
    if isinstance(config, Mapping):
        return WorldModelConfig.from_dict(config)
    if not model_id:
        raise ModelResolutionError("model_id is required unless a WorldModelConfig is provided")
    if not runner:
        raise ModelResolutionError(
            f"model {model_id!r} does not declare a runner target; "
            "run zoo model-show/model-download --check-local first or pass a WorldModelConfig with runner='module:Class'"
        )
    return WorldModelConfig(
        model_id=model_id,
        runner=runner,
        parameters=parameters or {},
        runtime=runtime or {},
    )


def resolve_world_model_runner(
    model_id: str | None = None,
    *,
    runner: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    config: WorldModelConfig | Mapping[str, Any] | None = None,
) -> ResolvedWorldModel:
    """Resolve and instantiate a world-model runner instance from inputs or a config.

    Args:
        model_id: Canonical model identifier (required unless ``config`` is given).
        runner: ``module:Class`` runner target (required unless ``config`` is given).
        parameters: Model parameters forwarded to :class:`WorldModelConfig`.
        runtime: Runtime overrides forwarded to :class:`WorldModelConfig`.
        config: An existing :class:`WorldModelConfig` or mapping that takes
            precedence over keyword arguments.

    Returns:
        A :class:`ResolvedWorldModel` with the instantiated runner and provenance.

    Raises:
        ModelResolutionError: If the runner target cannot be resolved or imported.
    """
    model_config = _config_from_input(
        model_id,
        runner=runner,
        parameters=parameters,
        runtime=runtime,
        config=config,
    )
    runner_target = model_config.runner
    try:
        instance = default_model_runner_registry().create(model_config)
    except (KeyError, ValueError, ModuleNotFoundError, AttributeError) as exc:
        raise ModelResolutionError(str(exc)) from exc
    return ResolvedWorldModel(
        model_id=model_config.model_id,
        runner=instance,
        source="runner_target",
        runner_target=runner_target,
    )


def resolve_model_zoo_config(
    model_id: str,
    *,
    manifest_dir: str | Path,
    variant_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> ResolvedModelZooConfig:
    """Resolve the execution config and diagnostics for a model-zoo catalog entry and variant.

    Loads the zoo registry from ``manifest_dir``, selects an entry by
    ``model_id``, and optionally overrides the runner/pipeline targets using
    a specific ``variant_id``.  Pipeline bindings and runtime profiles are
    resolved into the resulting :class:`WorldModelConfig`.

    Args:
        model_id: The model-zoo catalog entry identifier.
        manifest_dir: Directory containing the model-zoo registry manifest.
        variant_id: Optional variant identifier; when ``None``, a default
            variant is selected if the entry does not declare a direct
            ``runner_target``.
        parameters: Additional model parameters merged into the config.
        runtime: Runtime overrides merged into the config.

    Returns:
        A :class:`ResolvedModelZooConfig` with the resolved config, runner
        target, and diagnostics.

    Raises:
        ModelResolutionError: If the model ID is not found, the variant does
            not exist, or no ``runner_target`` can be resolved.
    """
    from worldfoundry.evaluation.models.catalog.manifest import model_zoo_entry_to_world_model_manifest
    from worldfoundry.evaluation.models.catalog.schema import select_default_variant
    from worldfoundry.evaluation.models.catalog.zoo_registry import load_model_zoo_registry

    # ── Load zoo entry and resolve variant ───────────────────────────────
    entry = load_model_zoo_registry(manifest_dir).get(model_id)
    if not variant_id and model_id != entry.model_id and any(
        item.variant_id == model_id for item in entry.variants
    ):
        variant_id = model_id
    selected_variant = None
    runner_target = entry.runner_target
    pipeline_target = entry.pipeline_target
    pipeline_binding = entry.pipeline_binding

    # ── Select variant overrides ─────────────────────────────────────────
    if variant_id:
        selected_variant = next((item for item in entry.variants if item.variant_id == variant_id), None)
        if selected_variant is None:
            raise ModelResolutionError(f"model {entry.model_id!r} has no variant {variant_id!r}")
        runner_target = selected_variant.runner_target or runner_target
        pipeline_target = selected_variant.pipeline_target or pipeline_target
        pipeline_binding = selected_variant.pipeline_binding or pipeline_binding
    elif not runner_target:
        # NOTE: When no variant_id is specified and no runner_target on the
        # entry, fall back to a default variant that may carry a runner target.
        selected_variant = select_default_variant(entry, allow_runner_target_fallback=True)
        if selected_variant is not None:
            runner_target = selected_variant.runner_target
            pipeline_target = selected_variant.pipeline_target or pipeline_target
            pipeline_binding = selected_variant.pipeline_binding or pipeline_binding

    # ── Validate that a runner_target was resolved ───────────────────────
    if not runner_target:
        if entry.runner_entry_kind == "listed_only":
            raise ModelResolutionError(
                f"model-zoo entry {entry.model_id!r} is listed_only and has no runner_target; "
                "it cannot be run in-process until a runnable runner entry is added"
            )
        raise ModelResolutionError(
            f"model-zoo entry {entry.model_id!r} is {entry.runner_entry_kind} "
            "but has no runner_target on the selected entry or variant"
        )

    # ── Assemble resolved parameters and pipeline route ──────────────────
    resolved_variant_id = selected_variant.variant_id if selected_variant is not None else (variant_id or "")
    resolved_parameters = dict(parameters or {})
    if pipeline_target and "pipeline_target" not in resolved_parameters:
        resolved_parameters["pipeline_target"] = pipeline_target
    if pipeline_binding and "pipeline_binding" not in resolved_parameters:
        resolved_parameters["pipeline_binding"] = pipeline_binding
    runtime_profile = selected_variant.runtime_profile if selected_variant is not None else entry.runtime_profile
    profile_metadata = runtime_profile_execution_metadata(runtime_profile)
    runtime_options = dict(runtime or {})
    parameter_options = dict(parameters or {})
    resolved_route = resolve_pipeline_route(
        model_id=entry.model_id,
        pipeline_target=pipeline_target,
        pipeline_binding=pipeline_binding,
        profile_metadata=profile_metadata,
        binding_root=first_text(
            parameter_options.get("pipeline_bindings_root"),
            runtime_options.get("pipeline_bindings_root"),
        )
        or None,
        alias_root=first_text(
            parameter_options.get("pipeline_aliases_root"),
            runtime_options.get("pipeline_aliases_root"),
        )
        or None,
    )
    if resolved_route is not None:
        resolved_pipeline_target, route_pipeline_binding, resolved_pipeline_route_source = resolved_route
    else:
        resolved_pipeline_target, route_pipeline_binding, resolved_pipeline_route_source = None, "", None
    resolved_pipeline_binding = route_pipeline_binding or pipeline_binding or profile_metadata.get("pipeline_binding")

    # ── Build the final WorldModelConfig ─────────────────────────────────
    config = WorldModelConfig(
        model_id=entry.model_id,
        runner=runner_target,
        variant=resolved_variant_id,
        parameters=resolved_parameters,
        runtime=runtime or {},
        manifest=model_zoo_entry_to_world_model_manifest(entry),
        metadata={
            "source": "model_zoo",
            "manifest_dir": str(Path(manifest_dir)),
            "integration_status": entry.integration_status,
            "runner_entry_kind": entry.runner_entry_kind,
            "verification_status": entry.verification_status,
            "source_status": entry.source_status,
            "variant_id": resolved_variant_id,
            "variant_integration_status": (
                selected_variant.integration_status if selected_variant is not None else None
            ),
            **profile_metadata,
            "runtime_profile": runtime_profile,
            "pipeline_target": resolved_pipeline_target,
            "pipeline_binding": resolved_pipeline_binding,
            "pipeline_route_source": resolved_pipeline_route_source,
        },
    )
    diagnostics = {
        "manifest_dir": str(Path(manifest_dir)),
        "entry_id": entry.model_id,
        "variant_id": resolved_variant_id,
        "entry_integration_status": entry.integration_status,
        "entry_runner_entry_kind": entry.runner_entry_kind,
        "entry_verification_status": entry.verification_status,
        "variant_integration_status": (
            selected_variant.integration_status if selected_variant is not None else None
        ),
        "variant_runner_entry_kind": (
            selected_variant.runner_entry_kind if selected_variant is not None else None
        ),
        "variant_verification_status": (
            selected_variant.verification_status if selected_variant is not None else None
        ),
        "runtime_profile": runtime_profile,
        **profile_metadata,
        "pipeline_target": resolved_pipeline_target,
        "pipeline_binding": resolved_pipeline_binding,
        "pipeline_route_source": resolved_pipeline_route_source,
    }
    return ResolvedModelZooConfig(
        model_id=entry.model_id,
        config=config,
        runner_target=runner_target,
        diagnostics=diagnostics,
    )


def resolve_model_zoo_runner(
    model_id: str,
    *,
    manifest_dir: str | Path,
    variant_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> ResolvedWorldModel:
    """Resolve, configure, and instantiate a world-model runner for a model-zoo catalog entry.

    Combines :func:`resolve_model_zoo_config` and
    :func:`resolve_world_model_runner` into a single convenience call.

    Args:
        model_id: The model-zoo catalog entry identifier.
        manifest_dir: Directory containing the model-zoo registry manifest.
        variant_id: Optional variant identifier for runner/pipeline overrides.
        parameters: Additional model parameters merged into the config.
        runtime: Runtime overrides merged into the config.

    Returns:
        A :class:`ResolvedWorldModel` with provenance ``"model_zoo"`` and
        diagnostics inherited from the zoo resolution step.

    Raises:
        ModelResolutionError: If the entry cannot be resolved or instantiated.
    """
    resolved_config = resolve_model_zoo_config(
        model_id,
        manifest_dir=manifest_dir,
        variant_id=variant_id,
        parameters=parameters,
        runtime=runtime,
    )
    resolved = resolve_world_model_runner(config=resolved_config.config)
    return ResolvedWorldModel(
        model_id=resolved.model_id,
        runner=resolved.runner,
        source="model_zoo",
        runner_target=resolved.runner_target,
        diagnostics=resolved_config.diagnostics,
    )


__all__ = [
    "ModelResolutionError",
    "ResolvedModelZooConfig",
    "ResolvedWorldModel",
    "import_object",
    "list_builtin_model_runners",
    "resolve_model_zoo_config",
    "resolve_model_zoo_runner",
    "resolve_world_model_runner",
]
