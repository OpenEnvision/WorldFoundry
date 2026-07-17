"""WorldFoundry evaluation: one execution stack, two typed sides.

Runs always go through :mod:`worldfoundry.evaluation.runner` — ``execute_evaluate_run`` /
``run_model_benchmark`` / suites are the programmatic entrypoints. The top-level
:mod:`worldfoundry.cli` package wraps the same flows for command-line use.

**Benchmark side** (what to evaluate): task, dataset, catalog, and benchmark-zoo metadata live under
:mod:`~worldfoundry.evaluation.tasks`.

**Model side** (:mod:`~worldfoundry.evaluation.models`): world-model catalog resolution — turning catalog
``runner_target`` strings into implementations consumed by :mod:`~worldfoundry.evaluation.runner`.
:mod:`~worldfoundry.evaluation.models.pipelines` resolves data-backed pipeline bindings and hosts the
runner lifecycle bridge used by formal model-benchmark runs. Runtime environment helpers live under
:mod:`worldfoundry.runtime`; local open-eval CLI checks live under :mod:`worldfoundry.cli.local_open_eval`;
low-level evaluation path/manifest/IO helpers live under
:mod:`~worldfoundry.evaluation.utils`.

Shortcuts below mirror the main runner and resolver APIs for ``import worldfoundry.evaluation as ev``.
"""

from __future__ import annotations

from importlib import import_module

_PRIMARY_EXPORT_MAP = {
    "execute_evaluate_run": (".runner", "execute_evaluate_run"),
    "run_worldfoundry": (".framework", "run_worldfoundry"),
    "run_model_benchmark": (".runner", "run_model_benchmark"),
    "resolve_model_zoo_runner": (".models", "resolve_model_zoo_runner"),
}

_PUBLIC_SUBMODULES = (
    "api",
    "framework",
    "models",
    "reporting",
    "runner",
    "tasks",
    "utils",
)

_DEFERRED_EXPORTS = (
    "ModelDefinition",
    "ModelRegistry",
    "TaskRegistry",
    "discover_model_registry",
    "load_benchmark_zoo_registry",
)

__all__ = [
    *_PUBLIC_SUBMODULES,
    *_DEFERRED_EXPORTS,
    *_PRIMARY_EXPORT_MAP,
]

_DEFERRED_EXPORT_MAP = {
    "TaskRegistry": (".tasks", "TaskRegistry"),
    "load_benchmark_zoo_registry": (".tasks", "load_benchmark_zoo_registry"),
    "ModelDefinition": (".models", "ModelDefinition"),
    "ModelRegistry": (".models", "ModelRegistry"),
    "discover_model_registry": (".models", "discover_model_registry"),
}


def __getattr__(name: str):
    """Lazily import and resolve submodules and shortcut exports."""
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in _PRIMARY_EXPORT_MAP:
        module_name, attr_name = _PRIMARY_EXPORT_MAP[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name not in _DEFERRED_EXPORTS:
        raise AttributeError(name)

    module_name, attr_name = _DEFERRED_EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the list of all exported and public attributes."""
    return sorted({*globals(), *__all__})
