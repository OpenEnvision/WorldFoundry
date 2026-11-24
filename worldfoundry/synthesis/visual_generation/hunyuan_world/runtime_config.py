"""Utilities for managing runtime configurations for "Hunyuan World" evaluations.

This module provides functions to locate, load, and apply default settings
from checked-in YAML configuration files specific to various Hunyuan World
runtimes. It includes logic for expanding special placeholders within
configuration values, such as paths resolved via `worldfoundry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.core.io.manifests import load_manifest
from worldfoundry.core.io.paths import package_data_path, resolve_worldfoundry_path

# Root directory where Hunyuan World runtime configuration YAML files are stored.
HUNYUAN_WORLD_RUNTIME_CONFIG_ROOT = package_data_path("models", "runtime", "configs", "hunyuan_world")


def hunyuan_world_runtime_config_path(runtime_name: str) -> Path:
    """Return the checked-in YAML path for one Hunyuan World runtime.

    Args:
        runtime_name: The name of the Hunyuan World runtime (e.g., "default", "small").

    Returns:
        The file system path to the runtime's configuration YAML file.
    """
    return HUNYUAN_WORLD_RUNTIME_CONFIG_ROOT / f"{runtime_name}.yaml"


def load_hunyuan_world_runtime_defaults(runtime_name: str) -> dict[str, Any]:
    """Load user-editable default configuration values for a Hunyuan World runtime.

    This function reads the specified runtime's configuration YAML file,
    extracts the 'defaults' section, and expands any special placeholders
    (like `"${WORLDFOUNDRY_..."`) within those values.

    Args:
        runtime_name: The name of the Hunyuan World runtime.

    Returns:
        A dictionary containing the default settings for the runtime, with
        environment-like variables expanded.

    Raises:
        ValueError: If the config file content or its 'defaults' section is not a mapping.
    """
    path = hunyuan_world_runtime_config_path(runtime_name)
    payload = load_manifest(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Hunyuan World runtime config must be a mapping: {path}")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Hunyuan World runtime config defaults must be a mapping: {path}")
    # Ensure all keys are strings and recursively expand values.
    return {str(key): _expand_config_value(value) for key, value in defaults.items()}


def _expand_config_value(value: Any) -> Any:
    """Recursively expands special placeholders in a configuration value.

    This function is primarily used to resolve `"${WORLDFOUNDRY_..."` string
    placeholders into actual file system paths, enabling flexible configuration.
    It handles strings, dictionaries, and lists by recursively expanding their
    contents.

    Args:
        value: The configuration value to expand. Can be a string, dict, list,
               or any other primitive type.

    Returns:
        The expanded configuration value.
    """
    if isinstance(value, str) and "${WORLDFOUNDRY_" in value:
        # Resolve special worldfoundry path placeholders within strings.
        return str(resolve_worldfoundry_path(value))
    if isinstance(value, dict):
        # Recursively expand values in dictionaries.
        return {str(key): _expand_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        # Recursively expand items in lists.
        return [_expand_config_value(item) for item in value]
    return value


def apply_hunyuan_world_argparse_defaults(parser: Any, runtime_name: str) -> Any:
    """Apply checked-in YAML defaults to an argparse parser.

    This function loads the default configuration settings for a specified
    Hunyuan World runtime and sets them as the default values for the
    corresponding arguments in the provided `argparse` parser. This allows
    command-line arguments to have pre-defined values from the configuration
    file, which can still be overridden by explicit command-line input.

    Args:
        parser: An `argparse.ArgumentParser` instance to which defaults will be applied.
        runtime_name: The name of the Hunyuan World runtime whose defaults should be applied.

    Returns:
        The modified `argparse.ArgumentParser` instance with defaults set.
    """
    parser.set_defaults(**load_hunyuan_world_runtime_defaults(runtime_name))
    return parser
