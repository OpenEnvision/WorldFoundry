"""Load local runtime paths for real generation backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRuntime:
    """Per-model execution paths resolved from ``wrbench.runtime.json``."""

    key: str
    python_bin: str
    repo_root: str | None
    gpu_id: int
    model_path: str | None = None
    extra_paths: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    models: dict[str, ModelRuntime]

    def model(self, key: str) -> ModelRuntime | None:
        return self.models.get(key)


class RuntimeConfigError(ValueError):
    """Raised when ``wrbench.runtime.json`` is present but invalid."""


def _resolve_runtime_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"runtime config not found: {path}")
        return path
    return None


def _load_runtime_payload(resolved: Path) -> dict[str, Any]:
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeConfigError(f"{resolved}: runtime config root must be a JSON object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise RuntimeConfigError(f"{resolved}: required integer field 'schema_version' is missing or invalid")
    return payload


def _require_mapping(node: dict[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = node.get(field)
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{context}: required object field '{field}' is missing or invalid")
    return value


def _optional_mapping(node: dict[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = node.get(field)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{context}: optional object field '{field}' must be an object when present")
    return value


def _require_str(node: dict[str, Any], field: str, *, context: str) -> str:
    value = node.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"{context}: required string field '{field}' is missing or invalid")
    return value


def _require_int(node: dict[str, Any], field: str, *, context: str) -> int:
    value = node.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeConfigError(f"{context}: required integer field '{field}' is missing or invalid")
    return int(value)


def load_runtime_config(path: Path | None = None) -> RuntimeConfig | None:
    resolved = _resolve_runtime_path(path)
    if resolved is None:
        return None
    payload = _load_runtime_payload(resolved)
    model_nodes = _require_mapping(payload, "models", context=str(resolved))
    models: dict[str, ModelRuntime] = {}
    for key, node in sorted(model_nodes.items()):
        if not isinstance(node, dict):
            raise RuntimeConfigError(f"{resolved}: models.{key} must be an object")
        context = f"{resolved}: models.{key}"
        models[str(key)] = ModelRuntime(
            key=str(key),
            python_bin=_require_str(node, "python_bin", context=context),
            # Retained only for legacy config compatibility. Model source is
            # always taken from WorldFoundry's bundled in-tree integration.
            repo_root=(str(node["repo_root"]) if node.get("repo_root") else None),
            model_path=node.get("model_path"),
            gpu_id=_require_int(node, "gpu_id", context=context),
            extra_paths={str(k): str(v) for k, v in _optional_mapping(node, "extra_paths", context=context).items()},
            env={str(k): str(v) for k, v in _optional_mapping(node, "env", context=context).items()},
        )
    return RuntimeConfig(
        schema_version=int(payload["schema_version"]),
        models=models,
    )
