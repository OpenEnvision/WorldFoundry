"""Typed Registry helpers shared across WorldFoundry subsystems.

Keys are ``casefold``-normalized so user-facing spellings stay case-insensitive.
Registration is serialized by ``_register_lock``: import-time writes are already
serialized by the import lock, but runtime registration from worker threads must
not race the duplicate-key check against the write. Lookups resolve a canonical
key first, then an alias; unknown keys raise :class:`UnknownRegistryKeyError`
with a close-match hint.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Generic, Iterable, Iterator, Mapping, TypeVar

ItemT = TypeVar("ItemT")


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


__all__ = [
    "DuplicateRegistryKeyError",
    "RegistryError",
    "RegistryItem",
    "TypedRegistry",
    "UnknownRegistryKeyError",
    "normalize_registry_key",
]
