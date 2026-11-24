"""Locating datasets on disk from environment variables, data roots, and local asset manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evals.utils import load_manifest
from worldfoundry.runtime.assets import (
    expand_worldfoundry_path,
    iter_manifest_asset_items,
    load_local_asset_manifest,
    resolve_asset_manifest_path,
)
from worldfoundry.runtime.env import resolve_hf_cache_dir

from ..catalog.schema import BenchmarkDatasetRef
from .local_status import check_local_dataset
from .refs import DatasetRef, JsonValue, _first_dataset_ref, _plain, normalize_hf_dataset_id

_LOCAL_DATA_ROOT_ENV = "WORLDFOUNDRY_BENCHMARK_DATA_ROOT"


_LOCAL_MANIFEST_ENV = (
    "WORLDFOUNDRY_BENCHMARK_DATA_MANIFEST"  # legacy dataset-only manifest; prefer LOCAL_ASSET_MANIFEST
)


@dataclass(frozen=True)
class DatasetLocation:
    """Represents the resolved physical disk path where a dataset is located on the system.

    Attributes:
        hf_dataset_id: Normalized Hugging Face repository ID.
        path: Resolved absolute Path.
        ready: Readiness indicator.
        source: Source of resolution (e.g. 'env', 'manifest', 'hf_cache').
        status: Status indicator ('ready', 'not_found', etc.).
        reason: Optional diagnostic reason.
        env_var: Environment variable name that overrode this lookup, if any.
        manifest_path: Manifest file Path that mapped this dataset, if any.
        cache_dataset_dir: Cache directory Path checked, if any.
        snapshot_dir: Snapshot directory Path checked, if any.
        checked_paths: Candidate paths that were examined during resolution.
    """

    hf_dataset_id: str | None
    path: Path | None
    ready: bool
    source: str
    status: str = "not_found"
    reason: str | None = None
    env_var: str | None = None
    manifest_path: Path | None = None
    cache_dataset_dir: Path | None = None
    snapshot_dir: Path | None = None
    checked_paths: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        """Check if the dataset was successfully located.

        Returns:
            True if located and ready, False otherwise.
        """
        return self.ready

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this location object to a plain dictionary.

        Returns:
            Plain dictionary of fields.
        """
        payload = _plain(asdict(self))
        payload["ok"] = self.ok
        return payload


def dataset_location_env_var(dataset_id: str) -> str:
    """Return the per-dataset local path environment variable name.

    Args:
        dataset_id: Hugging Face dataset id such as ``org/name``.
    """

    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        raise ValueError("dataset_id must be a non-empty Hugging Face dataset id")
    suffix = "".join(char.upper() if char.isalnum() else "_" for char in normalized).strip("_")
    while "__" in suffix:
        suffix = suffix.replace("__", "_")
    return f"WORLDFOUNDRY_DATASET_{suffix}_PATH"


def _existing_path(path: Path) -> Path | None:
    """Return the Path object if it exists on disk, otherwise None.

    Args:
        path: Input Path to check.

    Returns:
        The existing Path, or None.
    """
    return path if path.exists() else None


def _dataset_root_candidates(root: Path, dataset_id: str) -> tuple[Path, ...]:
    """Generate all candidate local folder paths for a dataset under a given root.

    Args:
        root: Root folder Path.
        dataset_id: Normalised dataset ID.

    Returns:
        Tuple of candidate folder Paths.
    """
    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        return ()
    org, name = normalized.split("/", 1)
    return (
        root / org / name,
        root / normalized.replace("/", "--"),
        root / f"datasets--{normalized.replace('/', '--')}",
    )


def _manifest_dataset_items(payload: Any) -> tuple[Mapping[str, Any], ...]:
    """Parse and normalize dataset mapping items from a location manifest payload.

    Args:
        payload: Decoded manifest dictionary.

    Returns:
        Tuple of normalized item dict mappings.

    Raises:
        ValueError: If payload format is not a mapping.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("dataset location manifest must be a YAML mapping")
    for key in ("datasets", "dataset_locations", "local_datasets"):
        value = payload.get(key)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
        if isinstance(value, Mapping):
            return tuple(
                {"hf_dataset_id": dataset_id, **item}
                if isinstance(item, Mapping)
                else {"hf_dataset_id": dataset_id, "path": item}
                for dataset_id, item in value.items()
            )
    return tuple(
        {"hf_dataset_id": dataset_id, **item}
        if isinstance(item, Mapping)
        else {"hf_dataset_id": dataset_id, "path": item}
        for dataset_id, item in payload.items()
    )


def _read_location_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load and parse dataset items from a location manifest file path.

    Args:
        path: Path to the location manifest YAML/JSON.

    Returns:
        Tuple of parsed dataset mappings.
    """
    payload = load_manifest(path)
    return _manifest_dataset_items(payload)


def _location_from_existing_path(
    *,
    dataset_id: str,
    path: Path,
    source: str,
    env_var: str | None = None,
    manifest_path: Path | None = None,
    checked_paths: tuple[Path, ...] = (),
) -> DatasetLocation:
    """Build a resolved DatasetLocation object for a path known to exist.

    Args:
        dataset_id: Mapped dataset ID.
        path: Confirmed local folder Path.
        source: Source tracing label ('env', 'manifest', etc.).
        env_var: Override variable name, if any.
        manifest_path: Manifest file path, if any.
        checked_paths: Visited paths during discovery.

    Returns:
        The generated DatasetLocation.
    """
    return DatasetLocation(
        hf_dataset_id=dataset_id,
        path=path,
        ready=True,
        source=source,
        status="ready",
        env_var=env_var,
        manifest_path=manifest_path,
        checked_paths=checked_paths or (path,),
    )


def _resolve_manifest_location(
    *,
    dataset_id: str,
    manifest_path: Path,
) -> DatasetLocation | None:
    """Resolve a dataset location using a local location manifest file.

    Args:
        dataset_id: Target dataset repository ID.
        manifest_path: Location manifest file Path.

    Returns:
        Resolved DatasetLocation, or None if not listed.
    """
    if not manifest_path.is_file():
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=None,
            ready=False,
            source="manifest",
            status="not_found",
            reason="dataset location manifest does not exist",
            manifest_path=manifest_path,
            checked_paths=(manifest_path,),
        )
    for item in _read_location_manifest(manifest_path):
        item_id = normalize_hf_dataset_id(item.get("hf_dataset_id") or item.get("repo_id") or item.get("id"))
        if item_id != dataset_id:
            continue
        raw_path = item.get("path") or item.get("local_path") or item.get("root")
        if raw_path is None:
            return DatasetLocation(
                hf_dataset_id=dataset_id,
                path=None,
                ready=False,
                source="manifest",
                status="missing_path",
                reason="manifest entry does not include path/local_path/root",
                manifest_path=manifest_path,
            )
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.exists():
            return _location_from_existing_path(
                dataset_id=dataset_id,
                path=path,
                source="manifest",
                manifest_path=manifest_path,
            )
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=path,
            ready=False,
            source="manifest",
            status="not_found",
            reason="manifest path does not exist",
            manifest_path=manifest_path,
            checked_paths=(path,),
        )
    return None


def _resolve_local_assets_manifest_location(
    *,
    dataset_id: str,
    manifest_path: Path,
    env: Mapping[str, str],
) -> DatasetLocation | None:
    """Resolve a dataset path from a unified local asset manifest."""

    if not manifest_path.is_file():
        return None
    try:
        manifest = load_local_asset_manifest(manifest_path, env=env)
    except (OSError, ValueError):
        return None
    for _, item in iter_manifest_asset_items(manifest):
        if str(item.get("kind") or "") != "dataset":
            continue
        item_id = normalize_hf_dataset_id(item.get("hf_dataset_id") or item.get("repo_id") or item.get("id"))
        if item_id != dataset_id:
            continue
        raw_path = item.get("path") or item.get("local_path") or item.get("root")
        if raw_path is None:
            env_name = item.get("env")
            if isinstance(env_name, str) and env_name.strip():
                raw_path = env.get(env_name.strip())
        if raw_path is None:
            return DatasetLocation(
                hf_dataset_id=dataset_id,
                path=None,
                ready=False,
                source="local_assets_manifest",
                status="missing_path",
                reason="local asset manifest dataset entry does not include path/local_path/root/env",
                manifest_path=manifest_path,
            )
        path = expand_worldfoundry_path(str(raw_path), env)
        if path.exists():
            return _location_from_existing_path(
                dataset_id=dataset_id,
                path=path,
                source="local_assets_manifest",
                manifest_path=manifest_path,
            )
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=path,
            ready=False,
            source="local_assets_manifest",
            status="not_found",
            reason="local asset manifest path does not exist",
            manifest_path=manifest_path,
            checked_paths=(path,),
        )
    return None


def _manifest_lookup_candidates(
    *,
    manifest_path: str | Path | None,
    env: Mapping[str, str],
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path).expanduser())
    legacy = env.get(_LOCAL_MANIFEST_ENV)
    if legacy:
        candidates.append(Path(legacy).expanduser())
    local_assets = resolve_asset_manifest_path(env=env)
    if local_assets not in candidates:
        candidates.append(local_assets)
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def locate_local_dataset(
    ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
    *,
    data_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> DatasetLocation:
    """Locate a previously downloaded benchmark dataset without network access.

    Lookup order is: per-dataset environment override, local asset/data
    manifests, an explicit or configured direct data root, then Hugging Face
    cache snapshots.

    Args:
        ref: Dataset reference, manifest-like mapping, or Hugging Face dataset id.
        data_root: Optional directory with ``org/name``, ``org--name``, or HF cache-style subfolders.
        manifest_path: Optional YAML manifest mapping dataset ids to local paths. When omitted, falls back to ``WORLDFOUNDRY_LOCAL_ASSET_MANIFEST`` (preferred) or legacy ``WORLDFOUNDRY_BENCHMARK_DATA_MANIFEST``.
        cache_dir: Optional Hugging Face cache root to inspect for ready snapshots.
        env: Environment mapping; defaults to ``os.environ``.
    """

    parsed = _first_dataset_ref(ref)
    if parsed.not_applicable:
        return DatasetLocation(
            hf_dataset_id=parsed.hf_dataset_id,
            path=None,
            ready=True,
            source="not_applicable",
            status="not_applicable",
            reason=parsed.reason or "dataset not applicable",
        )
    if not parsed.hf_dataset_id:
        return DatasetLocation(
            hf_dataset_id=None,
            path=None,
            ready=False,
            source="none",
            status="missing_dataset_id",
            reason="dataset ref does not include an hf_dataset_id",
        )

    dataset_id = parsed.hf_dataset_id
    environ = os.environ if env is None else env
    env_var = dataset_location_env_var(dataset_id)
    env_path = environ.get(env_var)
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return _location_from_existing_path(dataset_id=dataset_id, path=path, source="env", env_var=env_var)
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=path,
            ready=False,
            source="env",
            status="not_found",
            reason=f"{env_var} points to a path that does not exist",
            env_var=env_var,
            checked_paths=(path,),
        )

    for candidate in _manifest_lookup_candidates(manifest_path=manifest_path, env=environ):
        manifest_location = _resolve_manifest_location(
            dataset_id=dataset_id,
            manifest_path=candidate,
        )
        if manifest_location is not None:
            return manifest_location
        assets_location = _resolve_local_assets_manifest_location(
            dataset_id=dataset_id,
            manifest_path=candidate,
            env=environ,
        )
        if assets_location is not None:
            return assets_location

    root_value = data_root or environ.get(_LOCAL_DATA_ROOT_ENV)
    if root_value:
        candidates = _dataset_root_candidates(Path(root_value).expanduser(), dataset_id)
        existing = next((_existing_path(path) for path in candidates if path.exists()), None)
        if existing is not None:
            return _location_from_existing_path(
                dataset_id=dataset_id,
                path=existing,
                source="data_root",
                checked_paths=candidates,
            )
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=None,
            ready=False,
            source="data_root",
            status="not_found",
            reason="dataset was not found under data root",
            checked_paths=candidates,
        )

    cache_value = cache_dir or resolve_hf_cache_dir(environ)
    if cache_value:
        local_status = check_local_dataset(parsed, cache_value)
        if local_status.ready and local_status.snapshot_dirs:
            snapshot_dir = local_status.snapshot_dirs[0]
            return DatasetLocation(
                hf_dataset_id=dataset_id,
                path=snapshot_dir,
                ready=True,
                source="hf_cache",
                status="ready",
                cache_dataset_dir=local_status.cache_dataset_dir,
                snapshot_dir=snapshot_dir,
                checked_paths=local_status.snapshot_dirs,
            )
        return DatasetLocation(
            hf_dataset_id=dataset_id,
            path=None,
            ready=False,
            source="hf_cache",
            status=local_status.status,
            reason=local_status.reason,
            cache_dataset_dir=local_status.cache_dataset_dir,
            checked_paths=local_status.snapshot_dirs,
        )

    return DatasetLocation(
        hf_dataset_id=dataset_id,
        path=None,
        ready=False,
        source="none",
        status="not_configured",
        reason=f"set {env_var}, WORLDFOUNDRY_LOCAL_ASSET_MANIFEST, {_LOCAL_DATA_ROOT_ENV}, or a Hugging Face cache root",
    )


__all__ = [
    "DatasetLocation",
    "dataset_location_env_var",
    "locate_local_dataset",
]
