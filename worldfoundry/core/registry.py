"""Typed Registry helpers shared across WorldFoundry subsystems.

Keys are ``casefold``-normalized so user-facing spellings stay case-insensitive.
Registration is serialized by ``_register_lock``: import-time writes are already
serialized by the import lock, but runtime registration from worker threads must
not race the duplicate-key check against the write. Lookups resolve a canonical
key first, then an alias; unknown keys raise :class:`UnknownRegistryKeyError`
with a close-match hint.

:class:`AliasRegistryStore` keeps insertion order instead, and :class:`SpecRegistry`
builds the model-manifest and metric-spec registries on top of it.
"""

from __future__ import annotations

import builtins
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Generic, TypeVar

ItemT = TypeVar("ItemT")
SpecT = TypeVar("SpecT")


class RegistryError(ValueError):
    """Base class for registry definition errors."""


class DuplicateRegistryKeyError(RegistryError):
    """Raised when a key or alias maps to multiple different entries."""


class UnknownRegistryKeyError(KeyError):
    """Raised when a registry lookup cannot be resolved."""


def normalize_registry_key(value: str, *, field_name: str = "registry key") -> str:
    """Strip and ``casefold`` a user-facing Registry key for case-insensitive lookup."""

    text = str(value or "").strip().casefold()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


@dataclass(frozen=True)
class RegistryItem(Generic[ItemT]):
    """One registered item plus its public aliases."""

    key: str
    value: ItemT
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze aliases/metadata as plain tuples and dicts so callers cannot mutate them."""
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases))
        object.__setattr__(self, "metadata", dict(self.metadata))


class TypedRegistry(Generic[ItemT]):
    """Deterministic keyed registry with alias support.

    Registration methods (mutate state):
        - ``register(key, value, ...)`` — add one item and its aliases.

    Lookup methods (read state):
        - ``get(key)`` / ``get_item(key)`` — resolve a key or alias.
        - ``keys`` / ``values`` / ``items`` / ``aliases`` — enumerate entries.

    Raises:
        DuplicateRegistryKeyError: On conflicting keys or aliases.
        UnknownRegistryKeyError: When a lookup cannot be resolved.
    """

    def __init__(self, items: Iterable[RegistryItem[ItemT]] = ()) -> None:
        """Seed the registry; each seed item is run through :meth:`register` so aliases collide loudly."""
        self._items: dict[str, RegistryItem[ItemT]] = {}
        self._aliases: dict[str, str] = {}
        # Registration usually happens at import time (already serialized by
        # the import lock), but runtime registration from worker threads must
        # not race the duplicate check against the write.
        self._register_lock = threading.Lock()
        for item in items:
            self.register(item.key, item.value, aliases=item.aliases, metadata=item.metadata)

    def register(
        self,
        key: str,
        value: ItemT,
        *,
        aliases: Iterable[str] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> RegistryItem[ItemT]:
        """Register an item and return the normalized registry record."""

        normalized = normalize_registry_key(key)
        # Hold the lock across the duplicate check and the write so two threads
        # cannot both observe "absent" and then insert the same key.
        with self._register_lock:
            if normalized in self._items:
                raise DuplicateRegistryKeyError(f"duplicate registry key: {key!r}")
            if normalized in self._aliases:
                owner = self._aliases[normalized]
                raise DuplicateRegistryKeyError(f"registry key {key!r} conflicts with alias for {owner!r}")

            alias_tuple = tuple(str(alias) for alias in aliases)
            item = RegistryItem(key=str(key), value=value, aliases=alias_tuple, metadata=dict(metadata or {}))
            alias_lookup: dict[str, str] = {}
            for alias in item.aliases:
                alias_key = normalize_registry_key(alias, field_name="registry alias")
                if alias_key == normalized:
                    continue
                if alias_key in self._items:
                    raise DuplicateRegistryKeyError(f"registry alias {alias!r} conflicts with an existing key")
                if alias_key in self._aliases:
                    owner = self._aliases[alias_key]
                    raise DuplicateRegistryKeyError(f"duplicate registry alias {alias!r}; already owned by {owner!r}")
                alias_lookup[alias_key] = normalized

            self._items[normalized] = item
            self._aliases.update(alias_lookup)
        return item

    def _unknown_key_error(self, key: str, normalized: str) -> UnknownRegistryKeyError:
        """Build an unknown-key error; include a close-match hint when the spelling is near a known key."""
        candidates = list(self._items) + list(self._aliases)
        close = get_close_matches(normalized, candidates, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        return UnknownRegistryKeyError(f"unknown registry key: {key!r}{hint}")

    def get(self, key: str) -> ItemT:
        """Resolve a key or alias to the registered value."""

        return self.get_item(key).value

    def get_item(self, key: str) -> RegistryItem[ItemT]:
        """Resolve a key or alias to the full registry item."""

        normalized = normalize_registry_key(key)
        item = self._items.get(normalized)
        if item is None:
            owner = self._aliases.get(normalized)
            item = self._items.get(owner or "")
        if item is None:
            raise self._unknown_key_error(key, normalized)
        return item

    def keys(self) -> tuple[str, ...]:
        """Return canonical keys in deterministic order."""

        return tuple(item.key for _, item in sorted(self._items.items(), key=lambda pair: pair[1].key))

    def aliases(self) -> Mapping[str, str]:
        """Return normalized alias to normalized canonical key mapping."""

        return dict(sorted(self._aliases.items()))

    def items(self) -> tuple[RegistryItem[ItemT], ...]:
        """Return registry items sorted by their public key."""

        return tuple(item for _, item in sorted(self._items.items(), key=lambda pair: pair[1].key))

    def values(self) -> tuple[ItemT, ...]:
        """Return registered values sorted by public key."""

        return tuple(item.value for item in self.items())

    def __contains__(self, key: object) -> bool:
        """True when *key* is a registered canonical name or alias; non-strings are never members."""
        if not isinstance(key, str):
            return False
        try:
            normalized = normalize_registry_key(key)
        except ValueError:
            return False
        return normalized in self._items or normalized in self._aliases

    def __iter__(self) -> Iterator[RegistryItem[ItemT]]:
        """Iterate items in public-key order, matching :meth:`items`."""
        return iter(self.items())

    def __len__(self) -> int:
        """Number of canonical entries; aliases do not count."""
        return len(self._items)


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
        item_name: Callable[[ItemT], str],
        duplicate_error: Callable[[str], Exception],
        unknown_error: Callable[[str], Exception],
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

    def list(self) -> builtins.list[ItemT]:
        return [self._entries[key] for key in self._order]

    def keys(self, key_name: Callable[[ItemT], str]) -> builtins.list[str]:
        return [key_name(self._entries[key]) for key in self._order]


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


def metadata_aliases(item: Any) -> tuple[str, ...]:
    """Extract alias strings from the metadata field of an item."""
    metadata = getattr(item, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return ()

    aliases: list[str] = []
    for key in ("alias", "aliases"):
        if key in metadata:
            aliases.extend(_string_tuple(metadata[key], "alias"))
    return tuple(aliases)


class SpecRegistry(Generic[SpecT]):
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
        key_fn: Callable[[SpecT], str],
        alias_fn: Callable[[SpecT], Iterable[str]],
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
    def _order(self) -> builtins.list[str]:
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

    def list(self) -> builtins.list[SpecT]:
        return self._store.list()

    def keys(self) -> builtins.list[str]:
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
                    f"duplicate {self._kind} key {key!r}: {item_name!r} conflicts with {existing_name!r}"
                )

    def _existing_name_for_key(self, key: str) -> str | None:
        # Use _key_fn: MetricSpec has no .name attribute (EF-01).
        if key in self._items:
            return self._key_fn(self._items[key])
        if key in self._aliases:
            return self._key_fn(self._items[self._aliases[key]])
        return None


__all__ = [
    "AliasRegistryStore",
    "DuplicateRegistryKeyError",
    "RegistryError",
    "RegistryItem",
    "SpecRegistry",
    "TypedRegistry",
    "UnknownRegistryKeyError",
    "lookup_key",
    "metadata_aliases",
    "normalize_registry_key",
    "require_text",
]
