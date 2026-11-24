"""Small, dependency-free configuration objects for the G0Plus runtime."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class AttrConfig(dict[str, Any]):
    """Recursively expose declarative mappings as attributes."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        super().__init__((str(key), _convert(item)) for key, item in value.items())

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _convert(value: Any) -> Any:
    if isinstance(value, AttrConfig):
        return value
    if isinstance(value, Mapping):
        return AttrConfig(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def as_config(value: Mapping[str, Any]) -> AttrConfig:
    return AttrConfig(deepcopy(dict(value)))


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> AttrConfig:
    """Recursively merge one mixture override into the shared joint config."""

    result: dict[str, Any] = deepcopy(dict(base))

    def update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            current = target.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested = deepcopy(dict(current))
                update(nested, value)
                target[key] = nested
            else:
                target[key] = deepcopy(value)

    update(result, override)
    return as_config(result)


__all__ = ["AttrConfig", "as_config", "merge_config"]
