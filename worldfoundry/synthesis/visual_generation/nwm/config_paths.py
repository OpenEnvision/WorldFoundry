from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.core.io.paths import resolve_data_path


def runtime_config_root() -> Path:
    override = os.environ.get("WORLDFOUNDRY_NWM_CONFIG_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return resolve_data_path("models", "runtime", "configs", "nwm").resolve()


def runtime_config_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    if parts and parts[0] == "config":
        candidate = Path(*parts[1:])
    return (runtime_config_root() / candidate).resolve()


def load_runtime_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = runtime_config_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"NWM config must contain a mapping: {resolved}")
    return payload


__all__ = ["load_runtime_yaml", "runtime_config_path", "runtime_config_root"]
