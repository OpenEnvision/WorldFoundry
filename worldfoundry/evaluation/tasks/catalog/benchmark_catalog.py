"""Discover and load benchmark-zoo catalog manifests, suite inventory, and runtime profiles."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.evaluation.utils import (
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
_LEGACY_CATALOG_SHARDS = ("embodied_world_benchmarks.yaml", "video_world_benchmarks.yaml")

FORMAL_BENCHMARK_INVENTORY_SUITE_ID = "benchmark-inventory-catalog"
EXCLUDED_BENCHMARK_IDS = frozenset({"met3r", "psivg"})

DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR = BENCHMARK_RUNTIME_PROFILE_DIR / "official"
DEFAULT_BENCHMARK_RUNTIME_PROFILE_PATH = DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR
DEFAULT_BENCHMARK_RUNTIME_DEFAULTS_PATH = BENCHMARK_RUNTIME_PROFILE_DIR / "defaults.yaml"
_LEGACY_RUNTIME_PROFILE_MONOLITH_PATH = BENCHMARK_RUNTIME_PROFILE_DIR / "benchmark_zoo_official_eval.yaml"


class BenchmarkSuiteError(ValueError):
    """Raised when a removed named suite preset is requested."""


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
        if paths:
            return tuple(paths)
        paths = [
            resolved_root / name
            for name in _LEGACY_CATALOG_SHARDS
            if (resolved_root / name).is_file()
        ]
        return tuple(paths)

    return tuple(
        sorted(
            candidate
            for candidate in manifest_paths(resolved_root)
            if not is_catalog_metadata_manifest(candidate)
        )
    )


def _entry_benchmark_id(entry: Mapping[str, Any]) -> str | None:
    """Extract the benchmark id from a catalog entry, accepting both key spellings."""
    benchmark_id = entry.get("benchmark_id") or entry.get("id")
    if isinstance(benchmark_id, str) and benchmark_id.strip():
        return benchmark_id.strip()
    return None


@lru_cache(maxsize=16)
def _catalog_benchmark_path_index(directory: str) -> dict[str, Path]:
    """Map benchmark ids to concrete catalog manifest paths under *directory*."""
    from .schema import iter_benchmark_zoo_payloads

    index: dict[str, Path] = {}
    for candidate in iter_benchmark_catalog_manifest_paths(directory):
        try:
            payload = load_manifest(candidate)
            entries = iter_benchmark_zoo_payloads(payload)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            # Skip malformed shards so callers can fall back gracefully, but say
            # so: a silently dropped shard surfaces much later as a confusing
            # "benchmark not found" from downstream callers.
            _LOGGER.warning("skipping malformed benchmark catalog shard %s: %s", candidate, exc)
            continue
        for entry in entries:
            benchmark_id = _entry_benchmark_id(entry)
            if benchmark_id:
                index.setdefault(benchmark_id, candidate)
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


def load_benchmark_catalog_metadata(root: str | Path | None = None) -> dict[str, Any]:
    """Load shard metadata such as scope and verification policy."""
    resolved_root = resolve_benchmark_catalog_root(root)
    metadata: dict[str, Any] = {}
    if not is_default_benchmark_catalog_root(resolved_root):
        return metadata
    for shard_dir in DEFAULT_CATALOG_SHARD_DIRS:
        manifest_path = shard_dir / CATALOG_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        payload = load_manifest(manifest_path)
        if isinstance(payload, Mapping):
            metadata[shard_dir.name] = dict(payload)
    return metadata


@lru_cache(maxsize=8)
def benchmark_catalog_ids(root: str) -> tuple[str, ...]:
    """Return benchmark ids declared in catalog manifests."""
    from .schema import iter_benchmark_zoo_payloads

    ids: set[str] = set()
    for path in iter_benchmark_catalog_manifest_paths(root):
        payload = load_manifest(path)
        for entry in iter_benchmark_zoo_payloads(payload):
            # Same dual-key extraction as the path index: a shard declaring only
            # benchmark_id: must not vanish from the id universe (ET-18).
            benchmark_id = _entry_benchmark_id(entry)
            if benchmark_id:
                ids.add(benchmark_id)
    return tuple(sorted(ids))


@lru_cache(maxsize=8)
def _load_benchmark_catalog_entries_cached(root_key: str) -> tuple[Any, ...]:
    """Load all benchmark-zoo entries declared under the catalog root."""
    from .schema import load_entries

    entries = []
    for path in iter_benchmark_catalog_manifest_paths(root_key):
        entries.extend(load_entries(path))
    return tuple(entries)


def load_benchmark_catalog_entries(root: str | Path | None = None) -> tuple[Any, ...]:
    """Load all benchmark-zoo entries declared under the catalog root."""
    return _load_benchmark_catalog_entries_cached(str(resolve_benchmark_catalog_root(root)))


def clear_benchmark_catalog_cache() -> None:
    """Clear cached benchmark catalog path indexes and loaded entries."""
    from .dispatch import clear_catalog_dispatch_cache

    _catalog_benchmark_path_index.cache_clear()
    benchmark_catalog_ids.cache_clear()
    _load_benchmark_catalog_entries_cached.cache_clear()
    clear_catalog_dispatch_cache()


def load_benchmark_catalog_shard_entries(shard: str, root: str | Path | None = None) -> tuple[Any, ...]:
    """Load benchmark-zoo entries from one catalog shard directory."""
    from .schema import load_entries

    shard_root = resolve_benchmark_catalog_root(root) / shard
    entries = []
    for path in iter_benchmark_catalog_manifest_paths(shard_root):
        entries.extend(load_entries(path))
    return tuple(entries)


def formal_benchmark_ids(catalog_dir: str | Path | None = None) -> list[str]:
    """Return benchmark ids declared in benchmark-zoo catalog manifests."""
    root = Path(catalog_dir or BENCHMARK_ZOO_DIR)
    return sorted(
        benchmark_id
        for benchmark_id in benchmark_catalog_ids(str(root))
        if benchmark_id.lower() not in EXCLUDED_BENCHMARK_IDS
    )


def benchmark_ids_for_suite(suite_id: str, suites_path: str | Path | None = None) -> list[str]:
    """Resolve benchmark ids for legacy inventory suite aliases."""
    del suites_path
    normalized = str(suite_id).strip().lower().replace("_", "-")
    if normalized in {
        FORMAL_BENCHMARK_INVENTORY_SUITE_ID,
        "benchmark-inventory",
        "docs-inventory",
    }:
        return formal_benchmark_ids()
    raise BenchmarkSuiteError(
        "Named benchmark suite presets were removed; pass explicit --benchmark-id values "
        f"or use --all-benchmarks instead of --suite {suite_id!r}."
    )


def resolve_benchmark_runtime_profile_root(path: str | Path | None = None) -> Path:
    """Resolve a runtime profile root path, file, or directory."""
    candidate = Path(path or DEFAULT_BENCHMARK_RUNTIME_PROFILES_DIR)
    if candidate.exists():
        return candidate
    if _LEGACY_RUNTIME_PROFILE_MONOLITH_PATH.exists() and path is None:
        return _LEGACY_RUNTIME_PROFILE_MONOLITH_PATH
    return candidate


def benchmark_runtime_defaults_path(root: str | Path | None = None) -> Path:
    """Return the shared defaults manifest path for benchmark runtime profiles."""
    resolved_root = resolve_benchmark_runtime_profile_root(root)
    if resolved_root.is_file():
        return resolved_root.parent / "defaults.yaml"
    return resolved_root.parent / "defaults.yaml"


def iter_benchmark_runtime_profile_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """List per-benchmark runtime profile manifest paths."""
    resolved_root = resolve_benchmark_runtime_profile_root(root)
    if resolved_root.is_file():
        payload = load_manifest(resolved_root)
        if isinstance(payload, Mapping) and payload.get("profiles"):
            return (resolved_root,)
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


def benchmark_runtime_profile_benchmark_ids(root: str | Path | None = None) -> set[str]:
    """Return benchmark ids declared in runtime profile manifests."""
    ids: set[str] = set()
    for profile in load_benchmark_runtime_profiles(root).get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        for benchmark_id in profile.get("benchmark_ids") or ():
            ids.add(str(benchmark_id))
    return ids


def iter_benchmark_runtime_profile_source_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    """Return manifest files scanned for runner-script references."""
    resolved_root = resolve_benchmark_runtime_profile_root(root)
    paths = list(iter_benchmark_runtime_profile_paths(resolved_root))
    defaults_path = benchmark_runtime_defaults_path(resolved_root)
    if defaults_path.is_file():
        paths.insert(0, defaults_path)
    if not paths and resolved_root.is_file():
        paths = [resolved_root]
    return tuple(paths)
