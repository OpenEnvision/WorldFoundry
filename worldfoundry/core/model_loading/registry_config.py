"""Data-backed model-loader registries.

Checkpoint identity (key-hash, Hugging Face architecture, patch hash)
is mapped to a loader class by YAML under the package data tree.
:func:`load_model_loader_registry` validates the file and resolves
class names through a caller-supplied mapping into a frozen
:class:`ModelLoaderRegistry`.

The registry does not download weights or instantiate models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io import load_serialized, resolve_data_path

# ──────────────────────────────────────────────────────────────────────────
# Frozen routing table — hashes / architectures / patches → loader classes
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelLoaderRegistry:
    """Validated routing data used to choose a model loader from checkpoint identity.

    Args:
        model_loader_configs: Single-file hash rules and their resolved model classes.
        huggingface_model_loader_configs: Architecture-to-library routing rules.
        patch_model_loader_configs: Patch/adaptor hash rules and extra kwargs.
        preset_models_on_huggingface: Named Hugging Face model presets.
        preset_models_on_modelscope: Named ModelScope model presets.
        preset_model_ids: Stable preset identifiers.
        preset_model_websites: Human-facing model source labels.
    """

    model_loader_configs: list[tuple[Any, str, list[str], list[type], str]]
    huggingface_model_loader_configs: list[tuple[str, str, str, Any]]
    patch_model_loader_configs: list[tuple[str, list[str], list[type], dict[str, Any]]]
    preset_models_on_huggingface: dict[str, Any]
    preset_models_on_modelscope: dict[str, Any]
    preset_model_ids: tuple[str, ...]
    preset_model_websites: tuple[str, ...]


# ──────────────────────────────────────────────────────────────────────────
# YAML load — validate lists/mappings; resolve class names through the caller map
# ──────────────────────────────────────────────────────────────────────────


def load_model_loader_registry(
    config_path: str | Path,
    model_classes: Mapping[str, type],
) -> ModelLoaderRegistry:
    """Load checkpoint-hash model routing rules from a package data YAML file."""

    payload = load_serialized(_resolve_registry_config_path(config_path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Model loader registry must be a mapping: {config_path}")
    return ModelLoaderRegistry(
        model_loader_configs=_single_file_loader_configs(payload, model_classes, config_path),
        huggingface_model_loader_configs=_huggingface_loader_configs(payload),
        patch_model_loader_configs=_patch_loader_configs(payload, model_classes, config_path),
        preset_models_on_huggingface=dict(payload.get("preset_models_on_huggingface") or {}),
        preset_models_on_modelscope=dict(payload.get("preset_models_on_modelscope") or {}),
        preset_model_ids=tuple(str(item) for item in payload.get("preset_model_ids") or ()),
        preset_model_websites=tuple(
            str(item) for item in payload.get("preset_model_websites") or ("ModelScope", "HuggingFace")
        ),
    )


def _resolve_registry_config_path(config_path: str | Path) -> Path:
    """Absolute paths stay as-is; relative paths resolve under the package data tree."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return resolve_data_path(*path.parts)


def _single_file_loader_configs(
    payload: Mapping[str, Any],
    model_classes: Mapping[str, type],
    source: str | Path,
) -> list[tuple[Any, str, list[str], list[type], str]]:
    """Parse single-file hash rules; unknown ``model_classes`` names fail closed."""
    configs = []
    for entry in _entries(payload, "model_loader_configs", source):
        model_names = _string_list(entry, "model_names", source)
        class_names = _string_list(entry, "model_classes", source)
        configs.append(
            (
                entry.get("keys_hash"),
                str(entry["keys_hash_with_shape"]),
                model_names,
                [_resolve_model_class(name, model_classes, source) for name in class_names],
                str(entry["model_resource"]),
            )
        )
    return configs


def _huggingface_loader_configs(payload: Mapping[str, Any]) -> list[tuple[str, str, str, Any]]:
    """Architecture → Hugging Face library routing; missing section is an empty list."""
    configs = []
    for entry in payload.get("huggingface_model_loader_configs") or ():
        configs.append(
            (
                str(entry["architecture"]),
                str(entry["huggingface_lib"]),
                str(entry["model_name"]),
                entry.get("redirected_architecture"),
            )
        )
    return configs


def _patch_loader_configs(
    payload: Mapping[str, Any],
    model_classes: Mapping[str, type],
    source: str | Path,
) -> list[tuple[str, list[str], list[type], dict[str, Any]]]:
    """Parse patch/adaptor hash rules; accept singular or plural name/class keys."""
    configs = []
    for entry in _entries(payload, "patch_model_loader_configs", source):
        model_names = (
            _string_list(entry, "model_names", source) if "model_names" in entry else [str(entry["model_name"])]
        )
        class_names = (
            _string_list(entry, "model_classes", source) if "model_classes" in entry else [str(entry["model_class"])]
        )
        configs.append(
            (
                str(entry["keys_hash_with_shape"]),
                model_names,
                [_resolve_model_class(name, model_classes, source) for name in class_names],
                dict(entry.get("extra_kwargs") or {}),
            )
        )
    return configs


def _entries(payload: Mapping[str, Any], key: str, source: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Require *key* to be a list of mappings; ``None`` is treated as empty."""
    value = payload.get(key, [])
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list in {source}")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{key} entries must be mappings in {source}")
    return tuple(value)


def _string_list(entry: Mapping[str, Any], key: str, source: str | Path) -> list[str]:
    """Coerce a required list field to strings; a non-list is a config error."""
    value = entry.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list in {source}")
    return [str(item) for item in value]


def _resolve_model_class(name: str, model_classes: Mapping[str, type], source: str | Path) -> type:
    """Look up *name* in the caller-supplied map; list known keys on failure."""
    try:
        return model_classes[name]
    except KeyError as exc:
        known = ", ".join(sorted(model_classes))
        raise ValueError(f"unknown model class {name!r} in {source}; expected one of: {known}") from exc


__all__ = ["ModelLoaderRegistry", "load_model_loader_registry"]
