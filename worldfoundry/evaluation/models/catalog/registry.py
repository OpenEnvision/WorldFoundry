"""Model registry built from catalog zoo entries with alias-based lookups.

Exposes :class:`ModelDefinition` (the lightweight registration record),
:class:`ModelRegistry` (an in-memory index with family/alias queries), and
:func:`discover_model_registry` (a cached discovery function that merges
all zoo entries into a single registry).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from ...api.registry import lookup_key
from ..pipelines.bindings import resolve_pipeline_route
from .schema import ModelZooEntry
from .zoo_registry import load_model_zoo_registry


# ── Dataclass definition ────────────────────────────────────

@dataclass(frozen=True)
class ModelDefinition:
    """Core registration details for an integrated model definition.

    Attributes:
        model_type: Primary model identifier (same as :attr:`ModelZooEntry.model_id`).
        family: Task-family grouping derived from the zoo entry.
        has_loader: Whether a pipeline import target is available.
        has_infer: Whether both a pipeline and a runnable runner exist.
        pipeline_target: Dotted ``module:qualname`` import path for the pipeline.
        runner_target: Dotted ``module:qualname`` import path for the runner.
        aliases: Alternative identifiers for registry lookup.
    """

    model_type: str
    family: str
    has_loader: bool
    has_infer: bool
    pipeline_target: str = ""
    runner_target: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Convert the model definition to a dictionary."""
        return {
            "model_type": self.model_type,
            "family": self.family,
            "has_loader": self.has_loader,
            "has_infer": self.has_infer,
            "pipeline_target": self.pipeline_target,
            "runner_target": self.runner_target,
            "aliases": list(self.aliases),
        }


# ── Registry class ────────────────────────────────────────────

class ModelRegistry:
    """Registry managing collections of :class:`ModelDefinition` with alias lookups.

    Builds an internal index keyed by normalised model type and alias strings
    so that :meth:`get` resolves any identifier to its canonical definition.
    """

    def __init__(self, definitions: Iterable[ModelDefinition]) -> None:
        """Initialize the ModelRegistry with a set of model definitions."""
        entries = sorted(definitions, key=lambda item: (item.family, item.model_type))
        self._definitions = tuple(entries)
        index: dict[str, ModelDefinition] = {}
        for item in self._definitions:
            index[lookup_key(item.model_type, "model registry key")] = item
            for alias in item.aliases:
                index.setdefault(lookup_key(alias, "model registry alias"), item)
        self._index = index

    def families(self) -> list[str]:
        """Get all unique family names present in the registry."""
        return sorted({item.family for item in self._definitions})

    def list(self, family: str | None = None) -> list[ModelDefinition]:
        """List registered model definitions, optionally filtered by family."""
        if family is None:
            return list(self._definitions)
        return [item for item in self._definitions if item.family == family]

    def get(self, model_type: str) -> ModelDefinition:
        """Retrieve a ModelDefinition by its primary type or alias."""
        key = lookup_key(model_type, "model registry key")
        if key not in self._index:
            raise KeyError(f"Unknown model_type '{model_type}'.")
        return self._index[key]


# ── Entry-to-definition helpers ──────────────────────────────

def _family_from_entry(entry: ModelZooEntry) -> str:
    """Extract a task family name from a model-zoo entry."""
    if entry.tasks:
        return entry.tasks[0]
    return "model_zoo"


def _definition_from_entry(entry: ModelZooEntry, aliases: Iterable[str] = ()) -> ModelDefinition | None:
    """Build a ModelDefinition from a ModelZooEntry if runnable or has a pipeline route."""
    route = resolve_pipeline_route(
        model_id=entry.model_id,
        pipeline_target=entry.pipeline_target,
        pipeline_binding=entry.pipeline_binding,
    )
    pipeline_target = route[0] if route is not None else ""
    has_pipeline_target = bool(pipeline_target and ":" in pipeline_target)
    has_runner_target = entry.is_runnable_runner_entry
    if not (has_pipeline_target or has_runner_target):
        return None
    return ModelDefinition(
        model_type=entry.model_id,
        family=_family_from_entry(entry),
        has_loader=has_pipeline_target,
        has_infer=has_pipeline_target and has_runner_target,
        pipeline_target=pipeline_target or "",
        runner_target=entry.runner_target or "",
        aliases=tuple(aliases),
    )


def _catalog_definitions() -> tuple[ModelDefinition, ...]:
    """Retrieve all model definitions constructed from the model zoo registry."""
    definitions: list[ModelDefinition] = []
    registry = load_model_zoo_registry()
    for entry in registry.list():
        definition = _definition_from_entry(entry, registry.aliases_for(entry.model_id))
        if definition is not None:
            definitions.append(definition)
    return tuple(definitions)


# ── Cached discovery ─────────────────────────────────────────

@lru_cache(maxsize=1)
def discover_model_registry() -> ModelRegistry:
    """Discover, construct, and return the cached global ModelRegistry.

    ``ModelRegistry`` has no public ``register`` path.  Call
    :func:`clear_model_registry_cache` after catalog edits in a long-lived
    process.
    """
    definitions_by_id: dict[str, ModelDefinition] = {}
    for definition in _catalog_definitions():
        existing = definitions_by_id.get(definition.model_type)
        if existing is None or (not existing.has_loader and definition.has_loader):
            definitions_by_id[definition.model_type] = definition
    return ModelRegistry(definitions_by_id.values())


def clear_model_registry_cache() -> None:
    """Clear the cached :func:`discover_model_registry` instance."""
    discover_model_registry.cache_clear()


__all__ = [
    "ModelDefinition",
    "ModelRegistry",
    "clear_model_registry_cache",
    "discover_model_registry",
]
