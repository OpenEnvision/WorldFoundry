"""In-memory model-zoo registry with alias lookups, variant filtering, and caching.

Provides :class:`ModelZooRegistry` for loading, indexing, and querying
model-zoo manifest data, plus helper functions for hydrating pipeline routes
and de-duplicating entries across multiple YAML sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ...api import WorldModelManifest
from ...api.registry import AliasRegistryStore, lookup_key
from ...utils import MODEL_ZOO_DIR, load_manifest, manifest_paths
from .manifest import model_zoo_entries_to_world_model_manifests
from .schema import ModelVariantSpec, ModelZooEntry, iter_model_zoo_payloads

LOGGER = logging.getLogger(__name__)

# ── Custom exceptions ────────────────────────────────────────

class DuplicateModelZooKeyError(ValueError):
    """Raised when a model-zoo key or alias resolves ambiguously."""


class UnknownModelZooKeyError(KeyError):
    """Raised when a model-zoo lookup cannot be resolved."""


# ── Variant record ────────────────────────────────────────────

@dataclass(frozen=True)
class ModelZooVariantRecord:
    """A variant-level readiness record tied back to its parent model entry.

    Attributes:
        entry: The parent :class:`ModelZooEntry` this variant belongs to.
        variant: The :class:`ModelVariantSpec` describing this variant.
    """

    entry: ModelZooEntry
    variant: ModelVariantSpec

    @property
    def model_id(self) -> str:
        """Get the parent model ID."""
        return self.entry.model_id

    @property
    def variant_id(self) -> str:
        """Get the variant-level ID."""
        return self.variant.variant_id


def default_model_zoo_dir() -> Path:
    """Return the default path to the model-zoo directory."""
    return MODEL_ZOO_DIR


def is_model_catalog_metadata_manifest(path: str | Path) -> bool:
    """Return True for catalog metadata files that are not model entries."""
    return Path(path).name.startswith("_")


# ── Entry helpers ────────────────────────────────────────────

def _normalise_key(value: str) -> str:
    """Normalize a registry key string for case-insensitive lookup."""
    return lookup_key(value, "model-zoo lookup key")


def _entry_tasks(entry: ModelZooEntry) -> tuple[str, ...]:
    """Retrieve all distinct tasks defined in a model entry and its variants."""
    tasks = list(entry.tasks)
    for variant in entry.variants:
        if variant.task and variant.task not in tasks:
            tasks.append(variant.task)
    return tuple(tasks)


def _entry_aliases(
    entry: ModelZooEntry,
    *,
    excluded_hf_repo_aliases: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Compile unambiguous names, repositories, and variant IDs for an entry."""
    aliases: list[str] = []
    for value in (
        *entry.aliases,
        entry.name,
        *(variant.variant_id for variant in entry.variants),
    ):
        if value and value != entry.model_id and value not in aliases:
            aliases.append(value)
    if entry.hf_repo_id:
        repo_key = _normalise_key(entry.hf_repo_id)
        if repo_key not in excluded_hf_repo_aliases and entry.hf_repo_id not in aliases:
            aliases.append(entry.hf_repo_id)
    return tuple(aliases)


def _entry_mapping_priority(data: Mapping[str, Any]) -> int:
    """Determine the mapping schema priority based on the schema version."""
    return 1 if str(data.get("schema_version") or "") == "2" else 0


def _load_entries_with_priority(path: str | Path) -> tuple[tuple[int, ModelZooEntry], ...]:
    """Load model entries along with their schema priorities from a file.

    Schema validation errors are re-raised with the manifest path prepended so
    a broken file inside a large zoo directory is directly locatable.
    """
    resolved = Path(path)
    payload = load_manifest(resolved)
    entries: list[tuple[int, ModelZooEntry]] = []
    for item in iter_model_zoo_payloads(payload):
        try:
            entries.append((_entry_mapping_priority(item), ModelZooEntry.from_dict(item)))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{resolved}: {exc}") from exc
    return tuple(entries)


# ── Pipeline route hydration ─────────────────────────────────

def _resolve_entry_pipeline_route(entry: ModelZooEntry) -> tuple[str, str] | None:
    """Resolve the pipeline import target and binding for a main entry."""
    if not (entry.runner_target or entry.pipeline_target or entry.pipeline_binding):
        return None
    # NOTE: deferred import to avoid circular dependency at module load time.
    from worldfoundry.evaluation.models.pipelines.bindings import resolve_pipeline_route

    route = resolve_pipeline_route(
        model_id=entry.model_id,
        pipeline_target=entry.pipeline_target,
        pipeline_binding=entry.pipeline_binding,
        runtime_profile=None,
        include_plugins=False,
    )
    if route is None:
        return None
    pipeline_target, pipeline_binding, _source = route
    return pipeline_target, pipeline_binding


def _resolve_variant_pipeline_route(entry: ModelZooEntry, variant: ModelVariantSpec) -> tuple[str, str] | None:
    """Resolve the pipeline import target and binding for a specific model variant."""
    if not (variant.runner_target or variant.pipeline_target or variant.pipeline_binding):
        return None
    # NOTE: deferred import to avoid circular dependency at module load time.
    from worldfoundry.evaluation.models.pipelines.bindings import resolve_pipeline_route

    route = resolve_pipeline_route(
        model_id=variant.variant_id,
        pipeline_target=variant.pipeline_target,
        pipeline_binding=variant.pipeline_binding,
        runtime_profile=None,
        include_plugins=False,
    )
    if route is None:
        route = resolve_pipeline_route(
            model_id=entry.model_id,
            pipeline_target=variant.pipeline_target,
            pipeline_binding=variant.pipeline_binding or entry.pipeline_binding,
            runtime_profile=None,
            include_plugins=False,
        )
    if route is None:
        return None
    pipeline_target, pipeline_binding, _source = route
    return pipeline_target, pipeline_binding


def _hydrate_pipeline_routes(entry: ModelZooEntry) -> ModelZooEntry:
    """Enrich a model entry and its variants with fully resolved pipeline routes."""
    updates: dict[str, Any] = {}
    route = _resolve_entry_pipeline_route(entry)
    if route is not None:
        pipeline_target, pipeline_binding = route
        if not entry.pipeline_target:
            updates["pipeline_target"] = pipeline_target
        if pipeline_binding and not entry.pipeline_binding:
            updates["pipeline_binding"] = pipeline_binding

    variants: list[ModelVariantSpec] = []
    variant_changed = False
    for variant in entry.variants:
        variant_route = _resolve_variant_pipeline_route(entry, variant)
        if variant_route is None:
            variants.append(variant)
            continue
        pipeline_target, pipeline_binding = variant_route
        variant_updates: dict[str, Any] = {}
        if not variant.pipeline_target:
            variant_updates["pipeline_target"] = pipeline_target
        if pipeline_binding and not variant.pipeline_binding:
            variant_updates["pipeline_binding"] = pipeline_binding
        if variant_updates:
            variants.append(replace(variant, **variant_updates))
            variant_changed = True
        else:
            variants.append(variant)
    if variant_changed:
        updates["variants"] = tuple(variants)

    return replace(entry, **updates) if updates else entry


# ── Entry de-duplication ──────────────────────────────────────

def _dedupe_entries_by_target_priority(paths: Iterable[str | Path]) -> tuple[ModelZooEntry, ...]:
    """De-duplicate model entries from multiple paths, prioritizing higher schema versions.

    Hydrates pipeline routes before de-duplication, then keeps the entry with
    the highest priority for each ``model_id``, preserving first-seen order.
    A higher schema-version priority silently supersedes (that is the
    documented v2-wins mechanism); a duplicate at the *same* priority keeps the
    first-seen entry and logs a warning with both file paths so accidental
    copies stay visible.
    """
    selected: dict[str, tuple[int, int, ModelZooEntry, str]] = {}
    sequence = 0
    for path in paths:
        path_text = str(path)
        for priority, entry in _load_entries_with_priority(path):
            entry = _hydrate_pipeline_routes(entry)
            existing = selected.get(entry.model_id)
            if existing is None:
                selected[entry.model_id] = (priority, sequence, entry, path_text)
                sequence += 1
            elif priority > existing[0]:
                selected[entry.model_id] = (priority, existing[1], entry, path_text)
            elif priority == existing[0]:
                LOGGER.warning(
                    "Duplicate model_id %r at equal schema priority: keeping entry from %s, ignoring duplicate in %s.",
                    entry.model_id,
                    existing[3],
                    path_text,
                )
    return tuple(item[2] for item in sorted(selected.values(), key=lambda value: value[1]))


# ── Registry class ────────────────────────────────────────────

class ModelZooRegistry:
    """In-memory registry for model-zoo manifest metadata with alias lookups.

    Supports registration, key resolution (model ID or alias), and filtering
    by integration status, runner entry kind, source status, or task.
    """

    def __init__(self, entries: Iterable[ModelZooEntry] = ()) -> None:
        """Initialize the ModelZooRegistry with optional model entries."""
        self._frozen = False
        self._store = AliasRegistryStore[ModelZooEntry](
            item_name=lambda entry: entry.model_id,
            duplicate_error=DuplicateModelZooKeyError,
            unknown_error=lambda key: UnknownModelZooKeyError(f"unknown model-zoo entry: {key!r}"),
            field_name="model-zoo lookup key",
        )
        self._ambiguous_hf_repo_aliases: set[str] = set()
        for entry in entries:
            self.register(entry)

    def __contains__(self, key: object) -> bool:
        """Check if a key or alias is present in the registry."""
        if not isinstance(key, str):
            return False
        try:
            self.resolve_key(key)
        except (UnknownModelZooKeyError, ValueError):
            return False
        return True

    def __iter__(self) -> Iterator[ModelZooEntry]:
        """Iterate over all model entries in the registry."""
        return iter(self.list())

    def __len__(self) -> int:
        """Get the total number of registered entries."""
        return len(self._store)

    @classmethod
    def from_directory(cls, path: str | Path | None = None, *, pattern: str | None = None) -> "ModelZooRegistry":
        """Load and build a registry from files inside a model-zoo directory."""
        root = Path(path) if path is not None else default_model_zoo_dir()
        paths = sorted(root.glob(pattern)) if pattern else manifest_paths(root)
        return cls(
            _dedupe_entries_by_target_priority(
                path for path in paths if path.is_file() and not is_model_catalog_metadata_manifest(path)
            )
        )

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ModelZooRegistry":
        """Load and build a registry from specific manifest file paths."""
        return cls(_dedupe_entries_by_target_priority(paths))

    def register(self, entry: ModelZooEntry) -> ModelZooEntry:
        """Register a single model entry and its aliases in the registry."""
        if self._frozen:
            raise RuntimeError(
                "cannot register on a cached ModelZooRegistry; "
                "construct a new instance or call clear_model_zoo_registry_cache()"
            )
        if not isinstance(entry, ModelZooEntry):
            raise TypeError(f"expected ModelZooEntry, got {type(entry).__name__}")
        if entry.hf_repo_id:
            repo_key = _normalise_key(entry.hf_repo_id)
            if repo_key not in self._ambiguous_hf_repo_aliases:
                try:
                    repo_owner = self._store.get(entry.hf_repo_id)
                except UnknownModelZooKeyError:
                    repo_owner = None
                if (
                    repo_owner is not None
                    and repo_owner.model_id != entry.model_id
                    and repo_owner.hf_repo_id
                    and _normalise_key(repo_owner.hf_repo_id) == repo_key
                ):
                    # One HF repository may intentionally host multiple immutable
                    # model recipes (for example first- and third-person weights).
                    # A shared repository identifier is not a resolvable model alias.
                    self._store.discard_alias(entry.hf_repo_id)
                    self._ambiguous_hf_repo_aliases.add(repo_key)
        registered, alias, existing = self._store.register_with_conflict(
            entry.model_id,
            _entry_aliases(
                entry,
                excluded_hf_repo_aliases=frozenset(self._ambiguous_hf_repo_aliases),
            ),
            entry,
        )
        if existing is not None:
            raise DuplicateModelZooKeyError(
                f"duplicate model-zoo alias {_normalise_key(str(alias))!r}: "
                f"{entry.model_id!r} conflicts with {existing.model_id!r}"
            )
        return registered

    def resolve_key(self, key: str) -> str:
        """Resolve a model ID or alias to its primary registered model ID."""
        return self._store.resolve_key(key)

    def get(self, key: str) -> ModelZooEntry:
        """Get a registered model entry by model ID or alias."""
        return self._store.get(key)

    def by_model_id(self, key: str) -> ModelZooEntry:
        """Retrieve a model entry by its exact model ID or alias."""
        return self.get(key)

    def list(self) -> list[ModelZooEntry]:
        """List all unique registered model entries."""
        return self._store.list()

    def keys(self) -> list[str]:
        """Get all primary registered model IDs."""
        return self._store.keys(lambda entry: entry.model_id)

    def aliases_for(self, key: str) -> tuple[str, ...]:
        """Get all resolved aliases for a given model ID."""
        entry = self.get(key)
        return _entry_aliases(
            entry,
            excluded_hf_repo_aliases=frozenset(self._ambiguous_hf_repo_aliases),
        )

    def by_integration_status(self, status: str) -> tuple[ModelZooEntry, ...]:
        """Filter entries by their integration status."""
        lookup = _normalise_key(status)
        return tuple(entry for entry in self.list() if entry.integration_status.casefold() == lookup)

    def find_by_integration_status(self, status: str) -> tuple[ModelZooEntry, ...]:
        """Find model entries by integration status."""
        return self.by_integration_status(status)

    def variants_by_integration_status(self, status: str) -> tuple[ModelZooVariantRecord, ...]:
        """Get all variants matching a specific integration status."""
        lookup = _normalise_key(status)
        return tuple(
            ModelZooVariantRecord(entry=entry, variant=variant)
            for entry in self.list()
            for variant in entry.variants
            if variant.integration_status.casefold() == lookup
        )

    def integrated_variants(self) -> tuple[ModelZooVariantRecord, ...]:
        """Get all variants that have been fully integrated."""
        return self.variants_by_integration_status("integrated")

    def by_runner_entry_kind(self, kind: str) -> tuple[ModelZooEntry, ...]:
        """Filter model entries by their runner entry kind."""
        lookup = _normalise_key(kind)
        return tuple(entry for entry in self.list() if entry.runner_entry_kind.casefold() == lookup)

    def listed_only_entries(self) -> tuple[ModelZooEntry, ...]:
        """Get all entries that are only listed without runnable targets."""
        return self.by_runner_entry_kind("listed_only")

    def runner_candidate_entries(self) -> tuple[ModelZooEntry, ...]:
        """Get all runner candidate model entries."""
        return self.by_runner_entry_kind("runner_candidate")

    def runnable_runner_entries(self) -> tuple[ModelZooEntry, ...]:
        """Get all model entries that have integrated and runnable runners."""
        return self.by_runner_entry_kind("runnable_runner")

    def runnable_variants(self) -> tuple[ModelZooVariantRecord, ...]:
        """Get all variants that are fully runnable."""
        return tuple(
            ModelZooVariantRecord(entry=entry, variant=variant)
            for entry in self.list()
            for variant in entry.variants
            if variant.is_runnable_runner_entry
        )

    def by_source_status(self, status: str) -> tuple[ModelZooEntry, ...]:
        """Filter model entries by their source status."""
        lookup = _normalise_key(status)
        return tuple(entry for entry in self.list() if entry.source_status.casefold() == lookup)

    def find_by_source_status(self, status: str) -> tuple[ModelZooEntry, ...]:
        """Find model entries by their source status."""
        return self.by_source_status(status)

    def by_task(self, task: str) -> tuple[ModelZooEntry, ...]:
        """Filter model entries by a specific task they support."""
        lookup = _normalise_key(task)
        return tuple(
            entry
            for entry in self.list()
            if any(item.casefold() == lookup for item in _entry_tasks(entry))
        )

    def find_by_task(self, task: str) -> tuple[ModelZooEntry, ...]:
        """Find model entries by a specific task."""
        return self.by_task(task)

    def to_world_model_manifests(self) -> tuple[WorldModelManifest, ...]:
        """Convert all registered entries to the public WorldModelManifest representations."""
        return model_zoo_entries_to_world_model_manifests(self.list())

    def to_manifests(self) -> tuple[WorldModelManifest, ...]:
        """Convert all registered entries to manifests."""
        return self.to_world_model_manifests()


# ── Cached registry loader ────────────────────────────────────

@lru_cache(maxsize=32)
def _load_model_zoo_registry_cached(resolved_root: str) -> ModelZooRegistry:
    """Load the model zoo registry into memory and cache the registry instance."""
    registry = ModelZooRegistry.from_directory(Path(resolved_root))
    registry._frozen = True
    return registry


def load_model_zoo_registry(path: str | Path | None = None) -> ModelZooRegistry:
    """Load and cache the ModelZooRegistry from a directory path.

    The returned object is the process-wide cached instance and is frozen:
    ``register`` raises so callers cannot pollute later lookups.  Overlay
    extra entries on ``ModelZooRegistry(load_model_zoo_registry(...).list())``.
    Call :func:`clear_model_zoo_registry_cache` after editing YAML in a
    long-lived process.
    """
    root = Path(path) if path is not None else default_model_zoo_dir()
    return _load_model_zoo_registry_cached(str(root.resolve()))


def clear_model_zoo_registry_cache() -> None:
    """Clear the cached ModelZooRegistry instances."""
    _load_model_zoo_registry_cached.cache_clear()
