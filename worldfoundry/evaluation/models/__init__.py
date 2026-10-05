"""Lazy public facade for model catalog, runner, and runtime helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES: dict[str, str] = {
    "SourceFamilyMaps": "worldfoundry.evaluation.models.catalog.manifest",
    "SourceModelMap": "worldfoundry.evaluation.models.catalog.manifest",
    "ModelRunnerRegistry": "worldfoundry.evaluation.models.runners.registry",
    "ModelRunnerRegistryEntry": "worldfoundry.evaluation.models.runners.registry",
    "ModelRunnerRegistryIssue": "worldfoundry.evaluation.models.runners.registry",
    "ModelRunnerRegistryReport": "worldfoundry.evaluation.models.runners.registry",
    "ModelResolutionError": "worldfoundry.evaluation.models.runners.resolver",
    "ModelDefinition": "worldfoundry.evaluation.models.catalog.registry",
    "ModelRegistry": "worldfoundry.evaluation.models.catalog.registry",
    "BUILTIN_RUNTIME_RUNNERS": "worldfoundry.evaluation.models.runners.builtins",
    "BuiltinRuntimeRunnerEntry": "worldfoundry.evaluation.models.runners.builtins",
    "ResolvedModelZooConfig": "worldfoundry.evaluation.models.runners.resolver",
    "ResolvedWorldModel": "worldfoundry.evaluation.models.runners.resolver",
    "PipelineLifecycleContext": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "PipelineRuntimeProfile": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineInvocationProtocol": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineLifecycle": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineProtocol": "worldfoundry.evaluation.models.pipelines.lifecycle",
    "WorldFoundryPipelineRunner": "worldfoundry.evaluation.models.runners.pipeline",
    "build_model_manifest_registry": "worldfoundry.evaluation.models.catalog.manifest",
    "build_model_manifests": "worldfoundry.evaluation.models.catalog.manifest",
    "builtin_model_runner_registry": "worldfoundry.evaluation.models.runners.registry",
    "default_model_runner_registry": "worldfoundry.evaluation.models.runners.registry",
    "clear_model_registry_cache": "worldfoundry.evaluation.models.catalog.registry",
    "discover_model_registry": "worldfoundry.evaluation.models.catalog.registry",
    "get_builtin_runtime_runner_class": "worldfoundry.evaluation.models.runners.builtins",
    "import_object": "worldfoundry.evaluation.models.runners.resolver",
    "list_builtin_model_runners": "worldfoundry.evaluation.models.runners.resolver",
    "list_model_runner_registry_entries": "worldfoundry.evaluation.models.runners.registry",
    "list_builtin_runtime_runners": "worldfoundry.evaluation.models.runners.builtins",
    "model_manifests_from_source_maps": "worldfoundry.evaluation.models.catalog.manifest",
    "model_runner_registry_report": "worldfoundry.evaluation.models.runners.registry",
    "model_runner_registry_snapshot": "worldfoundry.evaluation.models.runners.registry",
    "resolve_model_zoo_config": "worldfoundry.evaluation.models.runners.resolver",
    "resolve_model_zoo_runner": "worldfoundry.evaluation.models.runners.resolver",
    "resolve_world_model_runner": "worldfoundry.evaluation.models.runners.resolver",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
