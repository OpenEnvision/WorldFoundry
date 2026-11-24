"""
This module provides utilities for loading and processing model runtime configurations,
including VLA/VA/WAM configurations and variant-specific settings from YAML files.
It handles path resolution and normalization for configuration lookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.core.io.paths import resolve_data_path


def _read_yaml_object(path: Path) -> dict[str, Any]:
    """
    Reads a YAML file from the given path and returns its content as a dictionary.

    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary representing the content of the YAML file.

    Raises:
        TypeError: If the root of the YAML file is not a mapping (e.g., a list or a scalar).
    """
    # Load the YAML content; 'or {}' ensures that if the file is empty or contains only whitespace,
    # payload defaults to an empty dictionary rather than None.
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Ensure the loaded YAML content is a mapping (dictionary-like object).
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected YAML object in runtime config: {path}")
    return dict(payload)


def load_vla_va_wam_runtime_config(model_id: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Loads a user-editable VLA/VA/WAM runtime configuration for a specified model ID.

    The configuration is loaded from a YAML file located within the `worldfoundry` data package.
    If `config_path` is provided, it can specify a custom path, which can be absolute
    or relative to the default VLA/VA/WAM runtime config directory.

    Args:
        model_id: The identifier of the model for which to load the configuration.
        config_path: An optional custom path to the configuration file.
                     Can be a string or Path object. If None, the default path
                     `models/runtime/configs/vla_va_wam/{model_id}.yaml` is used.

    Returns:
        A dictionary containing the loaded runtime configuration.
    """
    if config_path is None:
        # Construct the default path for the model's runtime config within the data package.
        path = resolve_data_path("models", "runtime", "configs", "vla_va_wam", f"{model_id}.yaml")
    else:
        # Resolve user-provided config_path, expanding user home directory if '~' is used.
        path = Path(config_path).expanduser()
        # If the provided path is relative, prepend the default VLA/VA/WAM config directory.
        if not path.is_absolute():
            path = resolve_data_path("models", "runtime", "configs", "vla_va_wam") / path
    # Resolve the final path to its canonical form before reading the YAML file.
    return _read_yaml_object(path.resolve())


def variant_defaults(config: Mapping[str, Any], variant_id: Any) -> dict[str, Any]:
    """
    Retrieves the configuration block for a specific variant, embodiment, or alias
    from a larger configuration dictionary.

    It performs case-insensitive, hyphen-normalized matching against variant names,
    embodiment identifiers, and listed aliases.

    Args:
        config: The full configuration dictionary, expected to potentially contain a "variants" key.
        variant_id: The identifier (string or any type convertible to string) to match
                    against variant names, embodiments, or aliases.

    Returns:
        A dictionary containing the configuration for the matching variant.
        Returns an empty dictionary if no matching variant is found or if the 'variants'
        block is missing/malformed.
    """
    variants = config.get("variants")
    # If the 'variants' key is not present or not a mapping, return an empty dictionary.
    if not isinstance(variants, Mapping):
        return {}

    # Normalize the requested variant ID for case-insensitive and hyphen-insensitive matching.
    # Empty string if variant_id is None or evaluates to false.
    requested = str(variant_id or "").strip().lower().replace("_", "-")
    if not requested:
        return {}

    # Iterate through each defined variant in the configuration.
    for name, payload in variants.items():
        # Ensure the variant's payload is a mapping before processing.
        if not isinstance(payload, Mapping):
            continue

        # Collect all potential identifiers for the current variant: its name, embodiment, and aliases.
        candidates = [name, payload.get("embodiment"), *(payload.get("aliases") or ())]
        # Normalize all candidate identifiers to a set for efficient matching.
        # This includes stripping whitespace, converting to lowercase, and replacing underscores with hyphens.
        normalized = {str(item).strip().lower().replace("_", "-") for item in candidates if item}

        # If the normalized requested ID matches any of the normalized candidates for this variant,
        # return its configuration payload.
        if requested in normalized:
            return dict(payload)

    # Return an empty dictionary if no matching variant is found after checking all options.
    return {}
