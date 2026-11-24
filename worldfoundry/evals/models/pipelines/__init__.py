"""Lazy public facade for pipeline runner contracts and data-backed bindings.

Each public name is mapped to the submodule that defines it.  Accessing any
exported symbol triggers a lazy ``import_module`` so that heavyweight
sub-modules (YAML parsing, pipeline imports) are only loaded when actually
needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ── Lazy import map ─────────────────────────────────────────────────────
# Maps each exported symbol name to the dotted module path that defines it.
_EXPORT_MODULES: dict[str, str] = {
    "ACTION_CONTROL_KEYS": "worldfoundry.evals.models.pipelines.invocation",
    "CONSUMED_INPUT_KEYS": "worldfoundry.evals.models.pipelines.invocation",
    "IMAGE_INPUT_KEYS": "worldfoundry.evals.models.pipelines.invocation",
    "PipelineInvocation": "worldfoundry.evals.models.pipelines.invocation",
    "PipelineLifecycleContext": "worldfoundry.evals.models.pipelines.lifecycle",
    "PipelineResultContext": "worldfoundry.evals.models.pipelines.results",
    "PipelineRunnerSpec": "worldfoundry.evals.models.pipelines.loading",
    "PipelineRuntimeProfile": "worldfoundry.evals.models.pipelines.lifecycle",
    "TEXT_INPUT_KEYS": "worldfoundry.evals.models.pipelines.invocation",
    "VIDEO_INPUT_KEYS": "worldfoundry.evals.models.pipelines.invocation",
    "WorldFoundryPipelineInvocationProtocol": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineLifecycle": "worldfoundry.evals.models.pipelines.lifecycle",
    "WorldFoundryPipelineProtocol": "worldfoundry.evals.models.pipelines.lifecycle",
    "build_pipeline_invocation": "worldfoundry.evals.models.pipelines.invocation",
    "build_pipeline_runner_spec": "worldfoundry.evals.models.pipelines.loading",
    "call_pipeline_from_pretrained": "worldfoundry.evals.models.pipelines.loading",
    "failed_generation_result": "worldfoundry.evals.models.pipelines.results",
    "generation_result_from_pipeline": "worldfoundry.evals.models.pipelines.results",
    "import_pipeline_target": "worldfoundry.evals.models.pipelines.loading",
    "invoke_pipeline": "worldfoundry.evals.models.pipelines.invocation",
    "load_named_pipeline": "worldfoundry.evals.models.pipelines.loading",
    "load_pipeline_from_config": "worldfoundry.evals.models.pipelines.loading",
    "load_pipeline_from_spec": "worldfoundry.evals.models.pipelines.loading",
    "pipeline_metadata": "worldfoundry.evals.models.pipelines.results",
    "pipeline_result_error": "worldfoundry.evals.models.pipelines.results",
    "pipeline_result_status": "worldfoundry.evals.models.pipelines.results",
    "pipeline_route_from_config": "worldfoundry.evals.models.pipelines.loading",
    "pipeline_target_from_config": "worldfoundry.evals.models.pipelines.loading",
    "request_output_dir": "worldfoundry.evals.models.pipelines.invocation",
    "resolve_hfd_cached_repo": "worldfoundry.evals.models.pipelines.loading",
    "resolve_model_path_entry": "worldfoundry.evals.models.pipelines.loading",
    "resolve_optional_model_path_entry": "worldfoundry.evals.models.pipelines.loading",
    "sample_output_path": "worldfoundry.evals.models.pipelines.invocation",
}

__all__ = sorted(_EXPORT_MODULES)


# ── Lazy attribute resolution ──────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Lazily import and resolve submodules and shortcut exports on access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the list of all exported attributes."""
    return sorted({*globals(), *__all__})
