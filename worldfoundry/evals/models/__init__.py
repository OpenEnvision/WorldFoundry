"""Lazy public facade for model catalog, runner, and runtime helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "ModelRunnerRegistry": "worldfoundry.evals.models.runners.registry",
    "ModelRunnerRegistryEntry": "worldfoundry.evals.models.runners.registry",
    "ModelResolutionError": "worldfoundry.evals.models.runners.resolver",
    "BUILTIN_RUNTIME_RUNNERS": "worldfoundry.evals.models.runners.builtins",
    "BuiltinRuntimeRunnerEntry": "worldfoundry.evals.models.runners.builtins",
    "ResolvedModelZooConfig": "worldfoundry.evals.models.runners.resolver",
    "ResolvedWorldModel": "worldfoundry.evals.models.runners.resolver",
    "PipelineLifecycleContext": "worldfoundry.evals.models.pipelines.lifecycle",
    "PipelineRuntimeProfile": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineInvocationProtocol": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineLifecycle": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineProtocol": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineRunner": "worldfoundry.evals.models.runners.pipeline",
    "builtin_model_runner_registry": "worldfoundry.evals.models.runners.registry",
    "default_model_runner_registry": "worldfoundry.evals.models.runners.registry",
    "get_builtin_runtime_runner_class": "worldfoundry.evals.models.runners.builtins",
    "import_object": "worldfoundry.evals.models.runners.resolver",
    "list_builtin_model_runners": "worldfoundry.evals.models.runners.resolver",
    "list_model_runner_registry_entries": "worldfoundry.evals.models.runners.registry",
    "list_builtin_runtime_runners": "worldfoundry.evals.models.runners.builtins",
    "model_runner_registry_snapshot": "worldfoundry.evals.models.runners.registry",
    "resolve_model_zoo_config": "worldfoundry.evals.models.runners.resolver",
    "resolve_model_zoo_runner": "worldfoundry.evals.models.runners.resolver",
    "resolve_world_model_runner": "worldfoundry.evals.models.runners.resolver",
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
