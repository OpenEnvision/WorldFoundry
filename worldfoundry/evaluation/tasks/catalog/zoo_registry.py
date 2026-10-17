"""Manages a registry of benchmark-zoo entries, providing discovery and lookup capabilities.

This module allows for loading, querying, and managing `BenchmarkZooEntry` objects,
which describe various benchmarks available in the `worldfoundry` ecosystem. It supports
loading entries from YAML/JSON catalog files, resolving aliases, and filtering by
various metadata fields like domain, modality, and integration status.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from worldfoundry.evaluation.api.registry import AliasRegistryStore, lookup_key, require_text
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import iter_benchmark_catalog_manifest_paths
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR

from .schema import INTEGRATION_STATUSES, BenchmarkZooEntry


class BenchmarkZooRegistryError(ValueError):
    """Base error for benchmark-zoo registry validation failures."""


class DuplicateBenchmarkZooKeyError(BenchmarkZooRegistryError):
    """Raised when a benchmark id or alias is already registered, indicating a naming conflict."""


class UnknownBenchmarkZooKeyError(KeyError):
    """Raised when a benchmark ID or alias lookup cannot be resolved in the registry."""


def _require_text(value: str, field_name: str) -> str:
    """Validates and requires that a given value represents a valid non-empty string.

    Args:
        value: Input text.
        field_name: The name of the field for formatting error messages.

    Returns:
        The validated text string.
    """
    return require_text(value, field_name)


def _lookup_key(value: str) -> str:
    """Standardizes a query key string by converting it to lowercase and stripping whitespace.

    Args:
        value: Input key to look up.

    Returns:
        A standardized lookup key string.
    """
    return lookup_key(value, "benchmark-zoo key")


def _coerce_entry(value: BenchmarkZooEntry | Mapping[str, Any]) -> BenchmarkZooEntry:
    """Coerces various raw input formats into a typed BenchmarkZooEntry instance.

    This function handles `BenchmarkZooEntry` objects directly, dictionaries, and
    objects with a `to_dict` method, converting them into a consistent `BenchmarkZooEntry`.

    Args:
        value: A `BenchmarkZooEntry`, a dictionary representation, or an object
               with a `to_dict` method.

    Returns:
        A `BenchmarkZooEntry` instance.

    Raises:
        TypeError: If value is of an unsupported type that cannot be coerced.
    """
    if isinstance(value, BenchmarkZooEntry):
        return value
    if isinstance(value, Mapping):
        return BenchmarkZooEntry.from_dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return BenchmarkZooEntry.from_dict(to_dict())
    raise TypeError(f"expected BenchmarkZooEntry or mapping, got {type(value).__name__}")


class BenchmarkZooRegistry:
    """An in-memory registry for querying and resolving auto-discovered benchmark-zoo entries.

    This registry stores `BenchmarkZooEntry` objects, allowing them to be retrieved
    by their canonical ID or registered aliases. It supports loading from catalog files
    and filtering entries based on various metadata.
    """

    def __init__(self, entries: Iterable[BenchmarkZooEntry | Mapping[str, Any]] = ()) -> None:
        """Initializes the registry, optionally pre-populating with benchmark-zoo entries.

        Args:
            entries: An iterable of `BenchmarkZooEntry` objects or their dictionary
                     representations to pre-populate the registry.
        """
        # Initialize the underlying AliasRegistryStore for managing benchmark IDs and aliases.
        self._store = AliasRegistryStore[BenchmarkZooEntry](
            item_name=lambda entry: entry.benchmark_id,
            duplicate_error=DuplicateBenchmarkZooKeyError,
            unknown_error=lambda key: UnknownBenchmarkZooKeyError(f"unknown benchmark-zoo entry: {key!r}"),
            field_name="benchmark-zoo key",
        )

        for entry in entries:
            self.register(entry)

    def __contains__(self, key: object) -> bool:
        """Checks if a benchmark ID or alias exists in this registry.

        Args:
            key: The benchmark ID or alias to check.

        Returns:
            True if the key resolves to a registered benchmark, False otherwise.
        """
        if not isinstance(key, str):
            return False
        try:
            # Attempt to resolve the key to see if it exists.
            self.resolve_key(key)
        except (UnknownBenchmarkZooKeyError, ValueError, TypeError):
            # Catch expected errors for unknown keys or invalid input types.
            return False
        return True

    def __iter__(self) -> Iterator[BenchmarkZooEntry]:
        """Iterates over all registered BenchmarkZooEntry objects in registration order."""
        return iter(self.list())

    def __len__(self) -> int:
        """Returns the total number of unique registered benchmarks (excluding aliases)."""
        return len(self._store)

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "BenchmarkZooRegistry":
        """Factory method to construct a registry from catalog shards under a target path.

        If no path is provided, it defaults to the standard benchmark zoo directory.

        Args:
            path: Directory containing catalog shards (YAML/JSON manifest files).
                  Defaults to `default_benchmark_zoo_dir()`.

        Returns:
            An initialized `BenchmarkZooRegistry` containing entries from the specified path.
        """
        root = Path(path) if path is not None else default_benchmark_zoo_dir()
        return cls.from_paths(_manifest_paths(root))

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "BenchmarkZooRegistry":
        """Factory method to construct a registry from multiple catalog shard file paths.

        This method loads entries from a list of specified YAML/JSON manifest files.

        Args:
            paths: File paths pointing to catalog YAML/JSON manifests.

        Returns:
            An initialized `BenchmarkZooRegistry` containing entries from all specified paths.
        """
        entries: list[BenchmarkZooEntry] = []
        for path in paths:
            # Import here to avoid circular dependencies.
            from .schema import load_entries

            entries.extend(load_entries(path))
        return cls(entries)

    def register(self, entry: BenchmarkZooEntry | Mapping[str, Any]) -> BenchmarkZooEntry:
        """Registers a new benchmark-zoo entry into this registry.

        The entry can be provided as a `BenchmarkZooEntry` object or a dictionary.
        It will check for and raise an error on duplicate benchmark IDs or aliases.

        Args:
            entry: A `BenchmarkZooEntry` object or its dictionary mapping representation.

        Returns:
            The registered `BenchmarkZooEntry` instance.

        Raises:
            DuplicateBenchmarkZooKeyError: If a name collision occurs with an existing
                                          benchmark ID or alias.
        """
        benchmark = _coerce_entry(entry)
        # Attempt to register the benchmark and its aliases, checking for conflicts.
        registered, alias, existing = self._store.register_with_conflict(
            benchmark.benchmark_id,
            self._entry_aliases(benchmark),
            benchmark,
        )
        # If a conflict is detected, raise a specific error.
        if existing is not None:
            raise DuplicateBenchmarkZooKeyError(
                f"duplicate benchmark-zoo alias {alias!r}: "
                f"{benchmark.benchmark_id!r} conflicts with {existing.benchmark_id!r}"
            )
        return registered

    def resolve_key(self, key: str) -> str:
        """Resolves a given alias or input key to its canonical benchmark ID.

        Args:
            key: Benchmark ID or alias to resolve.

        Returns:
            The canonical benchmark ID string.

        Raises:
            UnknownBenchmarkZooKeyError: If the key cannot be resolved to a known benchmark ID.
        """
        return self._store.resolve_key(key)

    def get(self, key: str) -> BenchmarkZooEntry:
        """Retrieves a single BenchmarkZooEntry by its canonical ID or registered alias.

        Args:
            key: Benchmark ID or alias.

        Returns:
            The matching `BenchmarkZooEntry` object.

        Raises:
            UnknownBenchmarkZooKeyError: If the key does not match any registered benchmark.
        """
        return self._store.get(key)

    def list(self) -> list[BenchmarkZooEntry]:
        """Lists all registered BenchmarkZooEntry objects.

        Returns:
            A list of `BenchmarkZooEntry` objects in registration order.
        """
        return self._store.list()

    def keys(self) -> list[str]:
        """Lists all canonical registered benchmark ID keys.

        Returns:
            A list of canonical benchmark ID strings.
        """
        return self._store.keys(lambda entry: entry.benchmark_id)

    def aliases_for(self, key: str) -> tuple[str, ...]:
        """Retrieves all registered alias names for a benchmark.

        This includes aliases explicitly defined and the benchmark's `name` if
        it's different from its `benchmark_id`.

        Args:
            key: Benchmark ID or alias.

        Returns:
            A tuple of alias strings representing that benchmark.
        """
        return self._entry_aliases(self.get(key))

    def by_domain(self, domain: str) -> list[BenchmarkZooEntry]:
        """Filters registered benchmarks by domain.

        The search is case-insensitive and matches against the `domains` tuple of each entry.

        Args:
            domain: Target domain string to filter by.

        Returns:
            A list of matching `BenchmarkZooEntry` objects.
        """
        return self._find_in_tuple("domains", domain)

    def by_modality(self, modality: str) -> list[BenchmarkZooEntry]:
        """Filters registered benchmarks by modality.

        The search is case-insensitive and matches against the `modalities` tuple of each entry.

        Args:
            modality: Target modality string to filter by.

        Returns:
            A list of matching `BenchmarkZooEntry` objects.
        """
        return self._find_in_tuple("modalities", modality)

    def by_tag(self, tag: str) -> list[BenchmarkZooEntry]:
        """Filters registered benchmarks by tag.

        The search is case-insensitive and matches against the `tags` tuple of each entry.

        Args:
            tag: Target tag string to filter by.

        Returns:
            A list of matching `BenchmarkZooEntry` objects.
        """
        return self._find_in_tuple("tags", tag)

    def by_integration_status(self, status: str) -> list[BenchmarkZooEntry]:
        """Filters registered benchmarks by their integration status.

        The search is case-insensitive. Valid statuses are defined in `INTEGRATION_STATUSES`.

        Args:
            status: Expected integration status (e.g., 'integrated' or 'planned').

        Returns:
            A list of matching `BenchmarkZooEntry` objects.

        Raises:
            ValueError: If status is not a recognized integration status.
        """
        status_key = _require_text(status, "integration status").casefold()
        if status_key not in INTEGRATION_STATUSES:
            allowed = ", ".join(sorted(INTEGRATION_STATUSES))
            raise ValueError(f"integration status must be one of: {allowed}. Got {status!r}.")
        return [entry for entry in self if entry.integration_status == status_key]

    def query(
        self,
        *,
        domain: str | None = None,
        modality: str | None = None,
        tag: str | None = None,
        integration_status: str | None = None,
    ) -> list[BenchmarkZooEntry]:
        """Performs a multi-criteria query across all registered benchmark-zoo entries.

        All provided criteria are combined with an AND logic, meaning an entry must match
        all specified filters to be included in the results. Filtering is case-insensitive.

        Args:
            domain: Optional domain filter string.
            modality: Optional modality filter string.
            tag: Optional tag filter string.
            integration_status: Optional integration status filter string.

        Returns:
            A list of `BenchmarkZooEntry` objects matching all criteria.

        Raises:
            ValueError: If an invalid integration status is provided.
        """
        matches = self.list()
        # Apply domain filter if specified.
        if domain is not None:
            domain_key = _lookup_key(domain)
            matches = [entry for entry in matches if any(item.casefold() == domain_key for item in entry.domains)]
        # Apply modality filter if specified.
        if modality is not None:
            modality_key = _lookup_key(modality)
            matches = [entry for entry in matches if any(item.casefold() == modality_key for item in entry.modalities)]
        # Apply tag filter if specified.
        if tag is not None:
            tag_key = _lookup_key(tag)
            matches = [entry for entry in matches if any(item.casefold() == tag_key for item in entry.tags)]
        # Apply integration status filter if specified, validating the status first.
        if integration_status is not None:
            status_key = _require_text(integration_status, "integration status").casefold()
            if status_key not in INTEGRATION_STATUSES:
                allowed = ", ".join(sorted(INTEGRATION_STATUSES))
                raise ValueError(f"integration status must be one of: {allowed}. Got {integration_status!r}.")
            matches = [entry for entry in matches if entry.integration_status == status_key]
        return matches

    def _find_in_tuple(self, field_name: str, value: str) -> list[BenchmarkZooEntry]:
        """Helper method to filter benchmarks based on a string value present in a tuple field.

        Args:
            field_name: The name of the tuple field on `BenchmarkZooEntry` (e.g., 'domains', 'tags').
            value: The string value to search for within the field's tuple, case-insensitively.

        Returns:
            A list of matching `BenchmarkZooEntry` objects.
        """
        lookup = _lookup_key(value)
        matches = []
        for entry in self:
            values = getattr(entry, field_name)
            if any(item.casefold() == lookup for item in values):
                matches.append(entry)
        return matches

    @staticmethod
    def _entry_aliases(entry: BenchmarkZooEntry) -> tuple[str, ...]:
        """Generates a tuple of effective aliases for a given BenchmarkZooEntry.

        This includes explicit aliases and the benchmark's name if it differs from its ID,
        ensuring uniqueness and excluding the canonical ID itself.

        Args:
            entry: The BenchmarkZooEntry for which to find aliases.

        Returns:
            A tuple of unique alias strings.
        """
        aliases: list[str] = []
        # Include explicit aliases and the entry's name if it's not the canonical ID
        # and not already added to the list of aliases.
        for value in (*entry.aliases, entry.name):
            if value and value != entry.benchmark_id and value not in aliases:
                aliases.append(value)
        return tuple(aliases)


def default_benchmark_zoo_dir() -> Path:
    """Returns the default directory path for the benchmark zoo.

    This directory is typically where predefined benchmark catalog manifests are stored.

    Returns:
        The default `BENCHMARK_ZOO_DIR` Path object.
    """
    return BENCHMARK_ZOO_DIR


def _manifest_paths(root: Path) -> tuple[Path, ...]:
    """Resolves all manifest paths under the given directory or a specific file."""
    return iter_benchmark_catalog_manifest_paths(root)


@lru_cache(maxsize=32)
def _load_benchmark_zoo_registry_cached(resolved_root: str) -> BenchmarkZooRegistry:
    """An LRU cached helper to load and construct a BenchmarkZooRegistry from a path string.

    This function is decorated with `lru_cache` to store recently created registry instances,
    improving performance for repeated requests to the same path.

    Args:
        resolved_root: The resolved string representation of the root path for the registry.

    Returns:
        A `BenchmarkZooRegistry` instance loaded from the specified path.
    """
    return BenchmarkZooRegistry.from_path(Path(resolved_root))


def load_benchmark_zoo_registry(path: str | Path | None = None) -> BenchmarkZooRegistry:
    """Loads and returns a BenchmarkZooRegistry, caching the instance under its resolved root path.

    If a registry for the given (resolved) path has been loaded recently, the cached
    instance is returned. Otherwise, a new registry is created and cached.

    Args:
        path: Path to the target catalog directory or file. Defaults to `default_benchmark_zoo_dir()`.

    Returns:
        A cached or newly loaded `BenchmarkZooRegistry` instance.
    """
    root = Path(path) if path is not None else default_benchmark_zoo_dir()
    # Resolve the path to get a canonical representation for caching.
    return _load_benchmark_zoo_registry_cached(str(root.resolve()))


def clear_benchmark_zoo_registry_cache() -> None:
    """Clears the cached registry instances to force reloading of manifests from disk.

    Calling this function will ensure that the next call to `load_benchmark_zoo_registry`
    will rebuild the registry from scratch, reflecting any changes in the underlying manifest files.
    """
    _load_benchmark_zoo_registry_cached.cache_clear()
