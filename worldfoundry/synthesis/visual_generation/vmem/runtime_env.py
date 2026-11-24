"""Resolve VMem configuration/data roots independently of installed model code."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.io.paths import package_data_path

DEFAULT_VMEM_RUNTIME_ROOT = Path(__file__).resolve().parent / "vmem_runtime"


def _runtime_root_candidates() -> list[Path]:
    """
    Generates the supported VMem configuration/data root candidates.

    The candidates are ordered by priority:
    1. `VMEM_RUNTIME_ROOT` environment variable.
    2. The in-tree runtime beside this file (`DEFAULT_VMEM_RUNTIME_ROOT`).

    Returns:
        list[Path]: A deduped list of absolute paths that are potential VMem configuration/data roots.
    """
    candidates: list[Path] = []

    # Prioritize environment variable 'VMEM_RUNTIME_ROOT' if set.
    env_root = os.environ.get("VMEM_RUNTIME_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(DEFAULT_VMEM_RUNTIME_ROOT)

    # Deduplicate and resolve all candidate paths to ensure uniqueness and absolute representation.
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


def runtime_root(override: str | Path | None = None) -> Path:
    """
    Selects a configuration/data root, never a Python source directory.

    If an `override` path is provided, it is used directly. Otherwise, it iterates through
    the supported candidates and returns the first one that exists and is a directory. If
    no valid directory is found, it falls back to the first candidate path, which might not
    exist.

    Args:
        override (Optional[str | Path]): An optional path to explicitly set as the runtime root.

    Returns:
        Path: The determined VMem runtime root directory.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    # Iterate through candidate paths and return the first one that is an existing directory.
    for candidate in _runtime_root_candidates():
        if candidate.is_dir():
            return candidate
    # If no existing directory is found among candidates, return the first candidate path.
    return _runtime_root_candidates()[0]


def default_config_path(runtime_override: str | Path | None = None) -> Path:
    """
    Finds configs/inference/inference.yaml in a data root or uses the packaged config.

    Args:
        runtime_override (Optional[str | Path]): An optional path to explicitly set the runtime root,
                                                 which will then be used to find the config.

    Returns:
        Path: The absolute path to the default inference.yaml configuration file.
    """
    if runtime_override is not None or os.environ.get("VMEM_RUNTIME_ROOT"):
        custom = runtime_root(runtime_override) / "configs" / "inference" / "inference.yaml"
        if custom.is_file():
            return custom
    return package_data_path("models", "runtime", "configs", "vmem", "inference/inference.yaml")


__all__ = [
    "DEFAULT_VMEM_RUNTIME_ROOT",
    "default_config_path",
    "runtime_root",
]
