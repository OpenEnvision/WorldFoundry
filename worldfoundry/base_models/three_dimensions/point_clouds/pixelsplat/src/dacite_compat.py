"""Small dataclass loader fallback for pixelSplat's inference-only config path."""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints


class Config:
    """Subset of :mod:`dacite` configuration used by the bundled runtime."""

    def __init__(self, *, type_hooks=None, cast=None):
        self.type_hooks = dict(type_hooks or {})
        self.cast = tuple(cast or ())


def _convert(value: Any, annotation: Any, config: Config) -> Any:
    if annotation is Any:
        return value
    hook = config.type_hooks.get(annotation)
    if hook is not None:
        return hook(value)

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if value not in arguments:
            raise ValueError(f"Expected one of {arguments!r}, got {value!r}")
        return value
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in arguments:
            return None
        errors = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _convert(value, candidate, config)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        raise TypeError(f"Could not map {value!r} to {annotation!r}: {errors!r}")
    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected a mapping for {annotation.__name__}, got {type(value).__name__}")
        return from_dict(annotation, value, config)
    if origin is list:
        item_type = arguments[0] if arguments else Any
        return [_convert(item, item_type, config) for item in value]
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_convert(item, arguments[0], config) for item in value)
        if arguments:
            return tuple(_convert(item, item_type, config) for item, item_type in zip(value, arguments))
        return tuple(value)
    if origin is dict:
        key_type, value_type = arguments or (Any, Any)
        return {
            _convert(key, key_type, config): _convert(item, value_type, config)
            for key, item in value.items()
        }
    if annotation in config.cast:
        return annotation(value)
    if isinstance(annotation, type) and not isinstance(value, annotation):
        raise TypeError(f"Expected {annotation.__name__}, got {type(value).__name__}")
    return value


def from_dict(data_class, data, config=None):
    """Instantiate a nested dataclass while ignoring unrelated Hydra keys."""

    if not is_dataclass(data_class):
        raise TypeError(f"Expected a dataclass type, got {data_class!r}")
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping, got {type(data).__name__}")
    config = config or Config()
    annotations = get_type_hints(data_class)
    values = {}
    for field in fields(data_class):
        if field.name in data:
            values[field.name] = _convert(data[field.name], annotations.get(field.name, Any), config)
        elif field.default is MISSING and field.default_factory is MISSING:
            raise TypeError(f"Missing required field {field.name!r} for {data_class.__name__}")
    return data_class(**values)


__all__ = ["Config", "from_dict"]
