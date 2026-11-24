"""Discover and load benchmark-zoo catalog manifests, suite inventory, and runtime profiles."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.evals.utils import (
    BENCHMARK_RUNTIME_PROFILE_DIR,
    BENCHMARK_ZOO_DIR,
    load_manifest,
    manifest_paths,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_EMBODIED_CATALOG_DIR = BENCHMARK_ZOO_DIR / "embodied"
DEFAULT_VIDEO_CATALOG_DIR = BENCHMARK_ZOO_DIR / "video"
DEFAULT_CATALOG_SHARD_DIRS = (DEFAULT_EMBODIED_CATALOG_DIR, DEFAULT_VIDEO_CATALOG_DIR)
CATALOG_MANIFEST_FILENAME = "_manifest.yaml"

EXCLUDED_BENCHMARK_IDS = frozenset({"met3r", "psivg"})

DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR = BENCHMARK_RUNTIME_PROFILE_DIR / "official"
DEFAULT_BENCHMARK_RUNTIME_PROFILE_PATH = DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR
DEFAULT_BENCHMARK_RUNTIME_DEFAULTS_PATH = BENCHMARK_RUNTIME_PROFILE_DIR / "defaults.yaml"


def is_catalog_metadata_manifest(path: Path) -> bool:
    """Return True when a YAML file stores shard metadata instead of one benchmark entry."""
    return path.name == CATALOG_MANIFEST_FILENAME


def resolve_benchmark_catalog_root(path: str | Path | None = None) -> Path:
    """Resolve the benchmark catalog root directory."""
    return Path(path or BENCHMARK_ZOO_DIR)


def is_default_benchmark_catalog_root(root: Path) -> bool:
    """Return True when *root* is the bundled benchmark catalog directory."""
    try:
        return root.resolve() == BENCHMARK_ZOO_DIR.resolve()
    except OSError:
        return root == BENCHMARK_ZOO_DIR


def iter_benchmark_catalog_manifest_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """Return benchmark entry manifest paths under the catalog root."""
    resolved_root = resolve_benchmark_catalog_root(root)
    if resolved_root.is_file():
        return (resolved_root,)

    paths: list[Path] = []
    if is_default_benchmark_catalog_root(resolved_root):
        for shard_dir in DEFAULT_CATALOG_SHARD_DIRS:
            if shard_dir.is_dir():
                paths.extend(
                    sorted(
                        candidate
                        for candidate in shard_dir.glob("*.yaml")
                        if candidate.is_file() and not is_catalog_metadata_manifest(candidate)
                    )
                )
        return tuple(paths)

    return tuple(
        sorted(candidate for candidate in manifest_paths(resolved_root) if not is_catalog_metadata_manifest(candidate))
    )


def _entry_benchmark_id(entry: Mapping[str, Any]) -> str | None:
    """Extract the benchmark id from a catalog entry, accepting both key spellings."""
    benchmark_id = entry.get("benchmark_id") or entry.get("id")
    if isinstance(benchmark_id, str) and benchmark_id.strip():
        return benchmark_id.strip()
    return None


@dataclass(frozen=True)
class CatalogShard:
    """One parsed catalog manifest file: its raw entry mappings, or the error that stopped parsing it."""

    path: Path
    entries: tuple[Mapping[str, Any], ...] = ()
    error: Exception | None = None


def _root_key(root: str | Path | None) -> str:
    return str(resolve_benchmark_catalog_root(root).expanduser().resolve())


@lru_cache(maxsize=16)
def _catalog_shards(root_key: str) -> tuple[CatalogShard, ...]:
    """Parse every catalog manifest under *root_key* once; all catalog views derive from this."""
    from .schema import iter_benchmark_zoo_payloads

    shards: list[CatalogShard] = []
    for path in iter_benchmark_catalog_manifest_paths(root_key):
        try:
            entries = tuple(iter_benchmark_zoo_payloads(load_manifest(path)))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            shards.append(CatalogShard(path, error=exc))
            continue
        shards.append(CatalogShard(path, entries))
    return tuple(shards)


def readable_catalog_shards(root: str | Path | None = None) -> tuple[CatalogShard, ...]:
    """Return the parseable shards under *root*, logging each malformed shard that is skipped.

    A silently dropped shard would surface much later as a confusing "benchmark not found",
    so every skip is logged with the shard path.
    """
    readable: list[CatalogShard] = []
    for shard in _catalog_shards(_root_key(root)):
        if shard.error is not None:
            _LOGGER.warning("skipping malformed benchmark catalog shard %s: %s", shard.path, shard.error)
            continue
        readable.append(shard)
    return tuple(readable)


def _strict_catalog_shards(root: str | Path | None) -> tuple[CatalogShard, ...]:
    shards = _catalog_shards(_root_key(root))
    for shard in shards:
        if shard.error is not None:
            raise shard.error
    return shards


@lru_cache(maxsize=16)
def _catalog_benchmark_path_index(directory: str) -> dict[str, Path]:
    """Map benchmark ids to concrete catalog manifest paths under *directory*."""
    index: dict[str, Path] = {}
    for shard in readable_catalog_shards(directory):
        for entry in shard.entries:
            benchmark_id = _entry_benchmark_id(entry)
            if benchmark_id:
                index.setdefault(benchmark_id, shard.path)
    return index


def resolve_benchmark_manifest_path(value: str | Path, benchmark_id: str | None = None) -> Path:
    """Turn a manifest file path or a catalog directory into a concrete YAML path."""
    path = Path(value)
    if path.suffix not in {".yaml", ".yml"}:
        if benchmark_id:
            cache_key = str(path if path.is_absolute() else Path.cwd() / path)
            indexed = _catalog_benchmark_path_index(cache_key).get(benchmark_id)
            if indexed is not None:
                return indexed
        candidates = list(iter_benchmark_catalog_manifest_paths(path))
        if candidates:
            return candidates[0]
    return path


@lru_cache(maxsize=8)
def benchmark_catalog_ids(root: str) -> tuple[str, ...]:
    """Return benchmark ids declared in catalog manifests."""
    ids = {
        benchmark_id
        for shard in _strict_catalog_shards(root)
        for entry in shard.entries
        if (benchmark_id := _entry_benchmark_id(entry))
    }
    return tuple(sorted(ids))


@lru_cache(maxsize=16)
def _load_benchmark_catalog_entries_cached(root_key: str) -> tuple[Any, ...]:
    """Build benchmark-zoo entries from the parsed catalog shards under *root_key*."""
    from .schema import BenchmarkZooEntry

    return tuple(
        BenchmarkZooEntry.from_dict(item) for shard in _strict_catalog_shards(root_key) for item in shard.entries
    )


def load_benchmark_catalog_entries(root: str | Path | None = None) -> tuple[Any, ...]:
    """Load all benchmark-zoo entries declared under the catalog root."""
    return _load_benchmark_catalog_entries_cached(_root_key(root))


def load_benchmark_catalog_shard_entries(shard: str, root: str | Path | None = None) -> tuple[Any, ...]:
    """Load benchmark-zoo entries from one catalog shard directory."""
    return load_benchmark_catalog_entries(resolve_benchmark_catalog_root(root) / shard)


def clear_benchmark_catalog_cache() -> None:
    """Drop every cached catalog view so the next lookup re-reads the manifests from disk."""
    from .dispatch import clear_catalog_dispatch_cache
    from .zoo_registry import _load_benchmark_zoo_registry_cached

    _catalog_shards.cache_clear()
    _catalog_benchmark_path_index.cache_clear()
    benchmark_catalog_ids.cache_clear()
    _load_benchmark_catalog_entries_cached.cache_clear()
    _load_benchmark_zoo_registry_cached.cache_clear()
    clear_catalog_dispatch_cache()


def formal_benchmark_ids(catalog_dir: str | Path | None = None) -> list[str]:
    """Return benchmark ids declared in benchmark-zoo catalog manifests."""
    root = Path(catalog_dir or BENCHMARK_ZOO_DIR)
    return sorted(
        benchmark_id
        for benchmark_id in benchmark_catalog_ids(str(root))
        if benchmark_id.lower() not in EXCLUDED_BENCHMARK_IDS
    )


def resolve_benchmark_runtime_profile_root(path: str | Path | None = None) -> Path:
    """Resolve a runtime profile root path, file, or directory."""
    return Path(path or DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR)


def benchmark_runtime_defaults_path(root: str | Path | None = None) -> Path:
    """Return the shared defaults manifest path for benchmark runtime profiles."""
    return resolve_benchmark_runtime_profile_root(root).parent / "defaults.yaml"


def iter_benchmark_runtime_profile_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """List per-benchmark runtime profile manifest paths."""
    resolved_root = resolve_benchmark_runtime_profile_root(root)
    if resolved_root.is_file():
        return (resolved_root,)
    profiles_dir = resolved_root if resolved_root.name == "official" else resolved_root / "official"
    if profiles_dir.is_dir():
        return tuple(sorted(path for path in profiles_dir.glob("*.yaml") if path.is_file()))
    if resolved_root.is_dir():
        return tuple(sorted(path for path in resolved_root.glob("*.yaml") if path.is_file()))
    return ()


def load_benchmark_runtime_profile_entry(path: Path) -> dict[str, Any]:
    """Load one benchmark runtime profile mapping from a manifest file."""
    payload = load_manifest(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark runtime profile must be a mapping: {path}")
    if isinstance(payload.get("profile"), Mapping):
        return dict(payload["profile"])
    profiles = payload.get("profiles")
    if isinstance(profiles, list):
        if len(profiles) != 1 or not isinstance(profiles[0], Mapping):
            raise ValueError(f"benchmark runtime profile file must contain exactly one profile: {path}")
        return dict(profiles[0])
    if payload.get("id"):
        return dict(payload)
    raise ValueError(f"benchmark runtime profile file is missing profile metadata: {path}")


def load_benchmark_runtime_profile_defaults(root: str | Path | None = None) -> dict[str, Any]:
    """Load shared defaults for benchmark runtime profiles."""
    defaults_path = benchmark_runtime_defaults_path(root)
    if not defaults_path.is_file():
        resolved_root = resolve_benchmark_runtime_profile_root(root)
        if resolved_root.is_file():
            payload = load_manifest(resolved_root)
            if isinstance(payload, Mapping):
                defaults = payload.get("defaults")
                if isinstance(defaults, Mapping):
                    return dict(defaults)
        return {}
    payload = load_manifest(defaults_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark runtime defaults must be a mapping: {defaults_path}")
    if isinstance(payload.get("defaults"), Mapping):
        return dict(payload["defaults"])
    return dict(payload)


@lru_cache(maxsize=8)
def _load_profiles_from_root(root: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    resolved_root = resolve_benchmark_runtime_profile_root(root)
    defaults = load_benchmark_runtime_profile_defaults(resolved_root)
    if resolved_root.is_file():
        payload = load_manifest(resolved_root)
        if isinstance(payload, Mapping) and isinstance(payload.get("profiles"), list):
            profiles = tuple(dict(item) for item in payload["profiles"] if isinstance(item, Mapping))
            return defaults, profiles
        profiles = (load_benchmark_runtime_profile_entry(resolved_root),)
        return defaults, profiles
    profiles = tuple(
        load_benchmark_runtime_profile_entry(path) for path in iter_benchmark_runtime_profile_paths(resolved_root)
    )
    return defaults, profiles


def load_benchmark_runtime_profiles(root: str | Path | None = None) -> dict[str, Any]:
    """Load benchmark runtime defaults and profile entries."""
    defaults, profiles = _load_profiles_from_root(str(resolve_benchmark_runtime_profile_root(root)))
    return {"defaults": defaults, "profiles": list(profiles)}


def benchmark_runtime_profiles_by_id(root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Map benchmark id to its runtime profile entry."""
    payload = load_benchmark_runtime_profiles(root)
    profiles: dict[str, dict[str, Any]] = {}
    for profile in payload.get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        benchmark_ids = profile.get("benchmark_ids") or (profile.get("id"),)
        for benchmark_id in benchmark_ids:
            if benchmark_id:
                profiles.setdefault(str(benchmark_id), dict(profile))
    return profiles
