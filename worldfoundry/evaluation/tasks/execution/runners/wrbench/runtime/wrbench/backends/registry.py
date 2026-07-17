"""Backend registry and resolution."""

from __future__ import annotations

from wrbench.backends.base import DryRunBackend, GenerationBackend
from wrbench.backends.worldfoundry import WorldFoundryPipelineBackend
from wrbench.registry import canonical_model_key
from wrbench.runtime import RuntimeConfig, load_runtime_config


def resolve_backend(
    model: str,
    *,
    runtime: RuntimeConfig | None = None,
    backend_name: str | None = None,
) -> GenerationBackend:
    """Return the backend selected by explicit config and model support."""

    key = canonical_model_key(model)
    if backend_name == "dry_run":
        return DryRunBackend()
    if backend_name in (None, "", "worldfoundry", "worldfoundry_pipeline", "in_tree"):
        runtime = runtime if runtime is not None else load_runtime_config()
        return WorldFoundryPipelineBackend(runtime)
    if backend_name == "local_subprocess":
        raise ValueError(
            "WRBench's duplicate local_subprocess model launcher is disabled in WorldFoundry; "
            "use the worldfoundry_pipeline backend"
        )
    raise ValueError(f"Unknown backend {backend_name!r}")


def list_backends(model: str, *, runtime: RuntimeConfig | None = None) -> list[tuple[str, bool, str]]:
    """Return ``(name, available, reason)`` tuples for *model*."""

    key = canonical_model_key(model)
    runtime = runtime if runtime is not None else load_runtime_config()
    rows: list[tuple[str, bool, str]] = []
    dry = DryRunBackend()
    ok, msg = dry.available()
    rows.append((dry.name, ok, msg))
    in_tree = WorldFoundryPipelineBackend(runtime)
    ok, msg = in_tree.available_for(key)
    rows.append((in_tree.name, ok, msg))
    return rows
