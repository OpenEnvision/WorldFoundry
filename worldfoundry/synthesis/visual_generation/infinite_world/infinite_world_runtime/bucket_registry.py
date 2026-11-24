from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml

_CONFIG_RESOURCE = "models/runtime/configs/infinite_world/buckets.yaml"


@lru_cache(maxsize=1)
def load_bucket_configs() -> dict[str, dict[str, Any]]:
    config_path = files("worldfoundry.data").joinpath(_CONFIG_RESOURCE)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Infinite World bucket config must be a mapping: {_CONFIG_RESOURCE}")
    bucket_configs = payload.get("bucket_configs")
    if not isinstance(bucket_configs, dict):
        raise ValueError(f"Infinite World bucket config is missing 'bucket_configs': {_CONFIG_RESOURCE}")
    return {str(name): dict(config) for name, config in bucket_configs.items()}


def get_bucket_config(name: str) -> dict[str, Any]:
    configs = load_bucket_configs()
    try:
        return configs[name]
    except KeyError as exc:
        known = ", ".join(sorted(configs))
        raise ValueError(f"unknown Infinite World bucket config {name!r}; expected one of: {known}") from exc


globals().update(load_bucket_configs())

__all__ = [
    "get_bucket_config",
    "load_bucket_configs",
    *sorted(load_bucket_configs()),
]
