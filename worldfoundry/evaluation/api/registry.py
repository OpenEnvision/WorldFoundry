"""Registries and store implementations for WorldFoundry models and metrics."""

from __future__ import annotations

from collections.abc import Callable as CollectionsCallable
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, TypeVar

from .metrics import MetricSpec
from .world_model_manifest import WorldModelManifest

ItemT = TypeVar("ItemT")


def require_text(value: str, field_name: str) -> str:
    """Validate and trim a non-empty text value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def lookup_key(value: str, field_name: str = "registry key") -> str:
    """Normalize a registry lookup key."""
    return require_text(value, field_name).casefold()


class AliasRegistryStore(Generic[ItemT]):
    """Ordered item store with normalized canonical keys and aliases."""

    def __init__(
        self,
        *,
        item_name: CollectionsCallable[[ItemT], str],
        duplicate_error: CollectionsCallable[[str], Exception],
        unknown_error: CollectionsCallable[[str], Exception],
        field_name: str,
    ) -> None:
        self._item_name = item_name
        self._duplicate_error = duplicate_error
        self._unknown_error = unknown_error
        self._field_name = field_name
        self._entries: dict[str, ItemT] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def register(self, canonical: str, aliases: Iterable[str], item: ItemT) -> ItemT:
        registered, alias, existing = self.register_with_conflict(canonical, aliases, item)
        if existing is not None:
            raise self._duplicate_error(
                f"{alias!r}: {self._item_name(item)!r} conflicts with {self._item_name(existing)!r}"
            )
        return registered

    def register_with_conflict(
        self,
        canonical: str,
        aliases: Iterable[str],
        item: ItemT,
    ) -> tuple[ItemT, str | None, ItemT | None]:
        canonical_key = lookup_key(canonical, self._field_name)
        if canonical_key in self._entries or canonical_key in self._aliases:
            raise self._duplicate_error(canonical)

        seen = {canonical_key}
        for alias in aliases:
            alias_key = lookup_key(alias, self._field_name)
            if alias_key in seen:
                continue
            existing = self._entries.get(alias_key) or self._entries.get(self._aliases.get(alias_key, ""))
            if existing is not None:
                return item, alias, existing
            seen.add(alias_key)

        self._entries[canonical_key] = item
        self._order.append(canonical_key)
        for alias_key in seen - {canonical_key}:
            self._aliases[alias_key] = canonical_key
        return item, None, None

    def discard_alias(self, alias: str) -> bool:
        """Remove one alias without ever removing a canonical entry."""

        alias_key = lookup_key(alias, self._field_name)
        return self._aliases.pop(alias_key, None) is not None

    def resolve_key(self, key: str) -> str:
        normalized = lookup_key(key, self._field_name)
        if normalized in self._entries:
            return normalized
        if normalized in self._aliases:
            return self._aliases[normalized]
        raise self._unknown_error(key)

    def get(self, key: str) -> ItemT:
        return self._entries[self.resolve_key(key)]

    def list(self) -> list[ItemT]:
        return [self._entries[key] for key in self._order]

    def keys(self, key_name: CollectionsCallable[[ItemT], str]) -> list[str]:
        return [key_name(self._entries[key]) for key in self._order]


class RegistryError(ValueError):
    """Base error for registry validation failures."""


class DuplicateRegistryKeyError(RegistryError):
    """Raised when a registry name or alias is already registered."""


class UnknownRegistryKeyError(KeyError):
    """Raised when a registry lookup cannot be resolved."""


def _string_tuple(values: Iterable[str] | str | None, field_name: str) -> tuple[str, ...]:
    """Convert a string or iterable of strings into a tuple of non-empty strings."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)

    result = []
    for value in values:
        result.append(require_text(value, field_name))
    return tuple(result)


ModelManifest = WorldModelManifest
SpecT = TypeVar("SpecT", WorldModelManifest, MetricSpec)
KeyFunc = Callable[[SpecT], str]
AliasFunc = Callable[[SpecT], Iterable[str]]


def _coerce_model_manifest(value: WorldModelManifest | Mapping[str, Any]) -> WorldModelManifest:
    """Coerce a WorldModelManifest or mapping to a WorldModelManifest instance."""
    if isinstance(value, WorldModelManifest):
        return value
    if isinstance(value, Mapping):
        return WorldModelManifest.from_dict(value)
    raise TypeError(f"expected WorldModelManifest or mapping, got {type(value).__name__}")


def _coerce_metric_spec(value: MetricSpec | Mapping[str, Any]) -> MetricSpec:
    """Coerce a MetricSpec or mapping to a MetricSpec instance."""
    if isinstance(value, MetricSpec):
        return value
    if isinstance(value, Mapping):
        return MetricSpec.from_dict(value)
    raise TypeError(f"expected MetricSpec or mapping, got {type(value).__name__}")


def _metadata_aliases(item: WorldModelManifest | MetricSpec) -> tuple[str, ...]:
    """Extract alias strings from the metadata field of an item."""
    metadata = getattr(item, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return ()

    aliases: list[str] = []
    for key in ("alias", "aliases"):
        if key in metadata:
            aliases.extend(_string_tuple(metadata[key], "alias"))
    return tuple(aliases)


def _model_key(manifest: WorldModelManifest) -> str:
    """Retrieve the primary model ID of a WorldModelManifest."""
    return manifest.model_id


def _model_aliases(manifest: WorldModelManifest) -> tuple[str, ...]:
    """Compute all aliases and names for a WorldModelManifest."""
    aliases = list(getattr(manifest, "aliases", ()))
    aliases.extend(_metadata_aliases(manifest))
    # Compare with the same casefold normalization used at registration; a
    # name differing from model_id only in case is the same key, and adding it
    # as an alias would self-collide (EF-02).
    if manifest.name and lookup_key(manifest.name) != lookup_key(manifest.model_id):
        aliases.append(manifest.name)
    return tuple(aliases)


def _metric_key(metric: MetricSpec) -> str:
    """Retrieve the primary ID of a MetricSpec."""
    return metric.id


def _metric_aliases(metric: MetricSpec) -> tuple[str, ...]:
    """Compute all aliases for a MetricSpec."""
    aliases = list(getattr(metric, "aliases", ()))
    aliases.extend(_metadata_aliases(metric))
    return tuple(aliases)


class _Registry(Generic[SpecT]):
    """Typed registry facade over :class:`AliasRegistryStore`.

    Public ``register(item)`` keeps the EF-01/EF-02 message contract
    (self-collision vs cross-item) and ``_key_fn`` / ``_alias_fn``
    customization; storage, lookup, and insertion order are delegated to
    the shared store so the two implementations cannot drift (EF-03).
    """

    def __init__(
        self,
        items: Iterable[SpecT] = (),
        *,
        kind: str,
        key_fn: KeyFunc[SpecT],
        alias_fn: AliasFunc[SpecT],
    ) -> None:
        self._kind = kind
        self._key_fn = key_fn
        self._alias_fn = alias_fn
        self._store = AliasRegistryStore[SpecT](
            item_name=key_fn,
            duplicate_error=DuplicateRegistryKeyError,
            unknown_error=lambda key: UnknownRegistryKeyError(f"unknown {kind}: {key!r}"),
            field_name="registry key",
        )

        for item in items:
            self.register(item)

    @property
    def _items(self) -> dict[str, SpecT]:
        return self._store._entries

    @property
    def _aliases(self) -> dict[str, str]:
        return self._store._aliases

    @property
    def _order(self) -> list[str]:
        return self._store._order

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self.resolve_key(key)
        except UnknownRegistryKeyError:
            return False
        return True

    def __iter__(self) -> Iterator[SpecT]:
        return iter(self.list())

    def __len__(self) -> int:
        return len(self._store)

    def register(self, item: SpecT) -> SpecT:
        item_name = self._key_fn(item)
        canonical_key = lookup_key(item_name)
        # Repeats *within* the item's own alias list collapse silently
        # (mirrors AliasRegistryStore.register_with_conflict).  An alias that
        # equals the item's own canonical key stays an error: upstream
        # builders deliberately leave duplicate source keys for the registry
        # to reject — but report it as a self-collision, not a cross-item
        # conflict (EF-02).  The store itself would skip that alias, so the
        # check must stay on this wrapper.
        aliases = tuple(dict.fromkeys(lookup_key(alias) for alias in self._alias_fn(item)))
        for alias_key in aliases:
            if alias_key == canonical_key:
                raise DuplicateRegistryKeyError(
                    f"duplicate {self._kind} key {alias_key!r} in {item_name!r}: "
                    "an alias duplicates the item's own canonical key "
                    "(redundant alias or duplicate source keys)"
                )
        self._validate_new_keys(canonical_key, aliases, item_name)
        return self._store.register(item_name, aliases, item)

    def get(self, key: str) -> SpecT:
        return self._store.get(key)

    def list(self) -> list[SpecT]:
        return self._store.list()

    def keys(self) -> list[str]:
        return self._store.keys(self._key_fn)

    def resolve_key(self, key: str) -> str:
        return self._store.resolve_key(key)

    def resolve_name(self, key: str) -> str:
        return self._key_fn(self.get(key))

    def _validate_new_keys(
        self,
        canonical_key: str,
        aliases: tuple[str, ...],
        item_name: str,
    ) -> None:
        """Reject keys that collide with a *different* already-registered item."""
        for key in (canonical_key, *aliases):
            existing_name = self._existing_name_for_key(key)
            if existing_name is not None:
                raise DuplicateRegistryKeyError(
                    f"duplicate {self._kind} key {key!r}: "
                    f"{item_name!r} conflicts with {existing_name!r}"
                )

    def _existing_name_for_key(self, key: str) -> str | None:
        # Use _key_fn: MetricSpec has no .name attribute (EF-01).
        if key in self._items:
            return self._key_fn(self._items[key])
        if key in self._aliases:
            return self._key_fn(self._items[self._aliases[key]])
        return None


class ModelManifestRegistry(_Registry[WorldModelManifest]):
    """Registry for world-model manifests."""

    def __init__(self, manifests: Iterable[WorldModelManifest | Mapping[str, Any]] = ()) -> None:
        super().__init__(kind="model manifest", key_fn=_model_key, alias_fn=_model_aliases)
        for manifest in manifests:
            self.register(manifest)

    def register(self, item: WorldModelManifest | Mapping[str, Any]) -> WorldModelManifest:
        return super().register(_coerce_model_manifest(item))


class MetricSpecRegistry(_Registry[MetricSpec]):
    """Registry for metric specifications."""

    def __init__(self, metrics: Iterable[MetricSpec | Mapping[str, Any]] = ()) -> None:
        super().__init__(kind="metric spec", key_fn=_metric_key, alias_fn=_metric_aliases)
        for metric in metrics:
            self.register(metric)

    def register(self, item: MetricSpec | Mapping[str, Any]) -> MetricSpec:
        return super().register(_coerce_metric_spec(item))


ModelRegistry = ModelManifestRegistry
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
