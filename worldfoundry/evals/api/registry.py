"""Metric-spec registry plus the shared registry primitives it builds on."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from worldfoundry.core.registry import (
    AliasRegistryStore,
    DuplicateRegistryKeyError,
    RegistryError,
    SpecRegistry,
    UnknownRegistryKeyError,
    lookup_key,
    metadata_aliases,
    require_text,
)
from worldfoundry.core.world_model_manifest import (
    ModelManifestRegistry,
    ModelRegistry,
    WorldModelManifest,
)

from .metrics import MetricSpec

ModelManifest = WorldModelManifest


def _coerce_metric_spec(value: MetricSpec | Mapping[str, Any]) -> MetricSpec:
    """Coerce a MetricSpec or mapping to a MetricSpec instance."""
    if isinstance(value, MetricSpec):
        return value
    if isinstance(value, Mapping):
        return MetricSpec.from_dict(value)
    raise TypeError(f"expected MetricSpec or mapping, got {type(value).__name__}")


def _metric_key(metric: MetricSpec) -> str:
    """Retrieve the primary ID of a MetricSpec."""
    return metric.id


def _metric_aliases(metric: MetricSpec) -> tuple[str, ...]:
    """Compute all aliases for a MetricSpec."""
    aliases: list[str] = list(getattr(metric, "aliases", ()))
    aliases.extend(metadata_aliases(metric))
    return tuple(aliases)


class MetricSpecRegistry(SpecRegistry[MetricSpec]):
    """Registry for metric specifications."""

    def __init__(self, metrics: Iterable[MetricSpec | Mapping[str, Any]] = ()) -> None:
        super().__init__(kind="metric spec", key_fn=_metric_key, alias_fn=_metric_aliases)
        for metric in metrics:
            self.register(metric)

    def register(self, item: MetricSpec | Mapping[str, Any]) -> MetricSpec:
        return super().register(_coerce_metric_spec(item))


MetricRegistry = MetricSpecRegistry


__all__ = [
    "AliasRegistryStore",
    "DuplicateRegistryKeyError",
    "MetricRegistry",
    "MetricSpec",
    "MetricSpecRegistry",
    "ModelManifest",
    "ModelManifestRegistry",
    "ModelRegistry",
    "RegistryError",
    "UnknownRegistryKeyError",
    "WorldModelManifest",
    "lookup_key",
    "require_text",
]
