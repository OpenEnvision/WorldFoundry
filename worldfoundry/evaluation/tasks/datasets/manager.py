"""Benchmark dataset references, access checks, and local cache discovery."""

from __future__ import annotations

import json
import os
import shutil
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path
from worldfoundry.runtime.env import resolve_hf_cache_dir
from worldfoundry.runtime.assets import (
    expand_worldfoundry_path,
    iter_manifest_asset_items,
    load_local_asset_manifest,
    resolve_asset_manifest_path,
)
from worldfoundry.evaluation.utils import load_manifest

from ..catalog.schema import BenchmarkDatasetRef, BenchmarkZooEntry


JsonValue = Any

_FALSEY_ACCESS_VALUES = frozenset({None, False, "false", "False", "none", "None", "", 0})
_HF_DATASET_KEYS = ("huggingface_datasets", "hf_datasets", "datasets")
_PAYLOAD_LIST_KEYS = ("benchmarks", "entries", "benchmark_zoo", "manifests")
_LOCAL_DATA_ROOT_ENV = "WORLDFOUNDRY_BENCHMARK_DATA_ROOT"
_LOCAL_MANIFEST_ENV = "WORLDFOUNDRY_BENCHMARK_DATA_MANIFEST"  # legacy dataset-only manifest; prefer LOCAL_ASSET_MANIFEST
_OPEN_LICENSES = frozenset(
    {
        "afl-3.0",
        "agpl-3.0",
        "apache-2.0",
        "artistic-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "bsl-1.0",
        "cc-by-2.0",
        "cc-by-2.5",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "cdla-permissive-1.0",
        "gpl-2.0",
        "gpl-3.0",
        "isc",
        "lgpl-2.1",
        "lgpl-3.0",
        "mit",
        "mpl-2.0",
        "odc-by",
        "postgresql",
        "unlicense",
        "wtfpl",
        "zlib",
    }
)
_LICENSE_REVIEW_MARKERS = (
    "custom",
    "non-commercial",
    "noncommercial",
    "proprietary",
    "research",
    "unknown",
)
_DIRECT_DATASET_FILE_COUNT_LIMIT = 10_000
_DIRECT_DATASET_INCOMPLETE_LIMIT = 20
_DIRECT_DATASET_METADATA_FILE_LIMIT = 200
_DIRECT_DATASET_METADATA_DIR_LIMIT = 256
_IGNORED_DIRECT_DATASET_SIBLING_NAMES = frozenset({".DS_Store"})
_IGNORED_DIRECT_DATASET_SIBLING_PARTS = frozenset({"__MACOSX"})


def _plain(value: JsonValue) -> JsonValue:
    """Normalize and convert nested structure Path objects to strings.

    Args:
        value: Input JSON-like data.

    Returns:
        JSON-serializable data with Paths stringified.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _optional_str(value: Any) -> str | None:
    """Safely coerce a value to string, returning None if empty or None.

    Args:
        value: Input value.

    Returns:
        The coerced string or None.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any input list, string, or iterable to a tuple of strings.

    Args:
        value: Input iterable, list, or single value.

    Returns:
        Deduplicated and coerced tuple of non-empty strings.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _items(value: Any) -> tuple[Any, ...]:
    """Safely convert single item or sequence to a tuple of items.

    Args:
        value: Any value or sequence.

    Returns:
        Tuple of items.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _access_value_is_truthy(value: Any) -> bool:
    """Evaluate whether an access/gate flag is truthy or restricted.

    Args:
        value: Input gate flag value.

    Returns:
        True if truthy, False otherwise.
    """
    return value not in _FALSEY_ACCESS_VALUES


def normalize_hf_dataset_id(value: Any) -> str | None:
    """Return a Hugging Face dataset repo id from a repo id or dataset URL.

    Args:
        value: Dataset ID or Hugging Face URL.

    Returns:
        Normalized dataset repo ID or None.
    """

    text = _optional_str(value)
    if text is None:
        return None

    marker = "huggingface.co/datasets/"
    if marker not in text:
        if text.startswith("datasets/"):
            return text.removeprefix("datasets/").strip("/") or None
        return text

    suffix = text.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = [part for part in suffix.split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def is_commit_like_revision(value: str | None) -> bool:
    """Determine whether a given revision string looks like a git commit SHA hash.

    Args:
        value: Revision string.

    Returns:
        True if it is a commit hash, False otherwise.
    """
    if not value:
        return False
    return len(value) >= 7 and all(char in "0123456789abcdefABCDEF" for char in value)


def hf_cache_dataset_dir(cache_dir: str | Path, dataset_id: str) -> Path:
    """Resolve the default Hugging Face cache folder name for a given dataset ID.

    Args:
        cache_dir: Root of HF cache.
        dataset_id: Target dataset repository ID.

    Returns:
        Path of the cache directory.
    """
    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        raise ValueError("dataset_id must be a non-empty Hugging Face dataset id")
    return Path(cache_dir) / f"datasets--{normalized.replace('/', '--')}"


def direct_dataset_dir_candidates(cache_dir: str | Path, dataset_id: str) -> tuple[Path, ...]:
    """Retrieve all direct folder candidates where a dataset might reside in direct layout.

    Args:
        cache_dir: Root cache directory.
        dataset_id: Target dataset ID.

    Returns:
        Tuple of candidate directory Paths.
    """
    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        raise ValueError("dataset_id must be a non-empty Hugging Face dataset id")
    root = Path(cache_dir)
    org, name = normalized.split("/", 1)
    candidate_roots = [root]
    nested_hfd_root = root / "hfd_datasets"
    if root.name != "hfd_datasets":
        candidate_roots.append(nested_hfd_root)

    candidates: list[Path] = []
    for candidate_root in candidate_roots:
        candidates.extend(
            (
                candidate_root / normalized.replace("/", "--"),
                candidate_root / normalized.replace("/", "__"),
                candidate_root / org / name,
            )
        )
    return tuple(dict.fromkeys(candidates))


def _dataset_ref_path_candidate(path: str | None) -> Path | None:
    """Resolve an explicit relative or absolute path from dataset reference.

    Args:
        path: Path string from reference.

    Returns:
        Resolved Path object or None.
    """
    if not path:
        return None
    resolved = resolve_worldfoundry_path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved


def direct_dataset_dir_candidates_for_ref(cache_dir: str | Path, ref: "DatasetRef") -> tuple[Path, ...]:
    """Return direct-layout candidates, preferring an explicit manifest path."""

    if not ref.hf_dataset_id:
        return ()
    candidates: list[Path] = []
    explicit_path = _dataset_ref_path_candidate(ref.path)
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(direct_dataset_dir_candidates(cache_dir, ref.hf_dataset_id))
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class DatasetRef:
    """Represents a unified, immutable reference to a benchmark dataset.

    This reference wraps various ways to retrieve or reference a dataset, including
    Hugging Face repository IDs, local paths, split constraints, licensing rules,
    and authorization prerequisites.
    """
    hf_dataset_id: str | None = None
    revision: str | None = None
    license: str | None = None
    private: bool | None = None
    gated: JsonValue | None = None
    split: str | None = None
    path: str | None = None
    not_applicable: bool = False
    reason: str | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()
    source: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerces, standardizes, and validates input fields during initialization."""
        object.__setattr__(self, "hf_dataset_id", normalize_hf_dataset_id(self.hf_dataset_id))
        object.__setattr__(self, "revision", _optional_str(self.revision))
        object.__setattr__(self, "license", _optional_str(self.license))
        if self.private is not None:
            object.__setattr__(self, "private", bool(self.private))
        object.__setattr__(self, "split", _optional_str(self.split))
        object.__setattr__(self, "path", _optional_str(self.path))
        object.__setattr__(self, "not_applicable", bool(self.not_applicable))
        object.__setattr__(self, "reason", _optional_str(self.reason))
        # Determine if authentication is required based on explicit overrides, privacy flags, or gated tags.
        object.__setattr__(
            self,
            "requires_auth",
            bool(self.requires_auth) or self.private is True or _access_value_is_truthy(self.gated),
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))
        object.__setattr__(self, "source", _optional_str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))


    @classmethod
    def from_benchmark_ref(cls, ref: BenchmarkDatasetRef, *, source: str | None = None) -> "DatasetRef":
        """Construct a DatasetRef from a catalog BenchmarkDatasetRef.

        Args:
            ref: Input BenchmarkDatasetRef catalog structure.
            source: Label indicating origin of this reference.

        Returns:
            The mapped DatasetRef.
        """
        return cls(
            hf_dataset_id=ref.hf_dataset_id,
            revision=ref.revision,
            license=ref.license,
            private=ref.private,
            gated=ref.gated,
            split=ref.split,
            path=ref.path,
            not_applicable=ref.not_applicable,
            reason=ref.reason,
            requires_auth=ref.requires_auth,
            notes=ref.notes,
            source=source,
            metadata=ref.to_dict(),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str | None = None) -> "DatasetRef":
        """Parse a DatasetRef from an arbitrary raw key-value mapping.

        Args:
            data: Raw dictionary mapping.
            source: Source trace label.

        Returns:
            The parsed DatasetRef.
        """
        reason = data.get("reason")
        notes = _tuple_of_str(data.get("notes"))
        if not reason and data.get("not_applicable") and notes:
            reason = notes[0]
        return cls(
            hf_dataset_id=data.get("hf_dataset_id") or data.get("repo_id") or data.get("id") or data.get("url"),
            revision=data.get("revision") or data.get("sha") or data.get("commit"),
            license=data.get("license"),
            private=data.get("private"),
            gated=data.get("gated"),
            split=data.get("split"),
            path=data.get("path"),
            not_applicable=bool(data.get("not_applicable", False)),
            reason=reason,
            requires_auth=bool(data.get("requires_auth", False)),
            notes=notes,
            source=source,
            metadata=dict(data),
        )

    def to_benchmark_ref(self) -> BenchmarkDatasetRef:
        """Convert this DatasetRef back into a catalog BenchmarkDatasetRef.

        Returns:
            The catalog BenchmarkDatasetRef structure.
        """
        return BenchmarkDatasetRef(
            hf_dataset_id=self.hf_dataset_id,
            revision=self.revision,
            license=self.license,
            private=self.private,
            gated=self.gated,
            split=self.split,
            path=self.path,
            not_applicable=self.not_applicable,
            reason=self.reason,
            requires_auth=self.requires_auth,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this reference to a plain, JSON-serializable dictionary.

        Returns:
            Plain dictionary of reference fields.
        """
        return _plain(asdict(self))


@dataclass(frozen=True)
class DatasetAccessIssue:
    """Represents a specific compliance, licensing, or security issue found in a dataset reference.

    Attributes:
        code: Stable string identifier for the type of issue.
        severity: Severity indicator (e.g. 'error', 'restricted', 'warning').
        message: Explanatory diagnostic message.
        field: Reference field name implicated in this issue.
        value: Offending value, if any.
    """
    code: str
    severity: str
    message: str
    field: str | None = None
    value: JsonValue | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this issue to a plain dictionary.

        Returns:
            Dictionary of issue fields.
        """
        return _plain(asdict(self))


@dataclass(frozen=True)
class DatasetAccessReport:
    """Represents a security and accessibility audit report for a dataset reference.

    Attributes:
        hf_dataset_id: Normalised Hugging Face repository ID.
        access_status: Accessibility category ('public', 'restricted', etc.).
        license_status: License type ('open', 'review_required', etc.).
        requires_auth: Whether access needs Hugging Face authorization credentials.
        issues: Tuple of detected compliance/access issues.
    """
    hf_dataset_id: str | None
    access_status: str
    license_status: str
    requires_auth: bool = False
    issues: tuple[DatasetAccessIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Check if the dataset is freely accessible with an open license.

        Returns:
            True if fully public and openly licensed, False otherwise.
        """
        return self.access_status == "public" and self.license_status == "open"

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this report to a plain dictionary.

        Returns:
            Dictionary of report fields.
        """
        payload = _plain(asdict(self))
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class DatasetLocalStatus:
    """Represents the real-time presence and integrity status of a dataset on the local system.

    Attributes:
        hf_dataset_id: Hugging Face dataset repository ID.
        cache_dataset_dir: Directory path of the Hugging Face cache.
        direct_dataset_dir: Directory path under a direct file layout (non-cache).
        local_layout: Detected layout string ('hf_cache', 'direct_hfd', 'missing').
        expected_revision: Expected Git revision SHA/tag.
        ref_name: Checked reference label (e.g. 'main').
        referenced_snapshot: Active commit SHA resolved for the ref_name.
        revision_matches: Whether the local revision matches expectation.
        snapshot_dirs: Tuple of snapshot Paths checked under HF cache.
        file_count: Number of completed files discovered.
        expected_file_count: Total expected file count.
        missing_file_count: Count of missing files compared to expectations.
        direct_file_count: File count under direct layout directory.
        direct_file_count_capped: Whether the direct file count scan was capped.
        direct_revision: Direct layout revision string.
        direct_revision_matches: Whether direct layout revision matches expectation.
        direct_ready: Whether direct layout files are complete.
        incomplete_files: Tuple of incomplete/partial files under HF cache.
        direct_incomplete_files: Tuple of incomplete files under direct layout.
        broken_links: Tuple of broken symlinks paths.
        ready: Ultimate readiness boolean indicator.
        status: Detailed status label ('ready', 'not_found', etc.).
        reason: Optional descriptive reason for non-ready states.
    """
    hf_dataset_id: str | None
    cache_dataset_dir: Path | None
    direct_dataset_dir: Path | None = None
    local_layout: str = "missing"
    expected_revision: str | None = None
    ref_name: str | None = None
    referenced_snapshot: str | None = None
    revision_matches: bool = True
    snapshot_dirs: tuple[Path, ...] = ()
    file_count: int = 0
    expected_file_count: int | None = None
    missing_file_count: int = 0
    direct_file_count: int = 0
    direct_file_count_capped: bool = False
    direct_revision: str | None = None
    direct_revision_matches: bool = True
    direct_ready: bool = False
    incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    direct_incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    broken_links: tuple[str, ...] = ()
    ready: bool = False
    status: str = "not_ready"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """Check if the local dataset is fully ready for evaluation.

        Returns:
            True if ready, False otherwise.
        """
        return self.ready

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this local status to a plain dictionary.

        Returns:
            Dictionary of status fields.
        """
        payload = _plain(asdict(self))
        payload["ok"] = self.ok
        return payload


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


@dataclass(frozen=True)
class DatasetDownloadPlan:
    """Represents a cohesive downloading execution plan for one or more datasets.

    Attributes:
        refs: Tuple of mapped DatasetRef specs involved.
        commands: Tuple of tuple command arguments to run.
        cache_dir: Root cache directory where datasets should download.
        access_reports: Accessibility and compliance reports per dataset.
        local_checks: Local readiness status structures, if checked.
        metadata: Optional dictionary detailing overall plan statistics.
    """
    refs: tuple[DatasetRef, ...]
    commands: tuple[tuple[str, ...], ...]
    cache_dir: Path
    access_reports: tuple[DatasetAccessReport, ...] = ()
    local_checks: tuple[DatasetLocalStatus, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Get the unique Hugging Face dataset repository IDs involved in this plan.

        Returns:
            Tuple of dataset ID strings.
        """
        dataset_ids: list[str] = []
        for ref in self.refs:
            if ref.hf_dataset_id and ref.hf_dataset_id not in dataset_ids:
                dataset_ids.append(ref.hf_dataset_id)
        return tuple(dataset_ids)

    @property
    def requires_auth(self) -> bool:
        """Evaluate if any of the datasets in the plan requires credentials/authentication.

        Returns:
            True if any dataset is restricted/gated, False otherwise.
        """
        return any(report.requires_auth for report in self.access_reports)

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this download plan to a plain dictionary.

        Returns:
            Plain dictionary representation.
        """
        payload = _plain(asdict(self))
        payload["dataset_ids"] = list(self.dataset_ids)
        payload["requires_auth"] = self.requires_auth
        return payload


def _dedupe_refs(refs: Iterable[DatasetRef]) -> tuple[DatasetRef, ...]:
    """Deduplicate dataset references based on unique properties to prevent redundant scans.

    Args:
        refs: Iterable collection of DatasetRef objects.

    Returns:
        Deduplicated tuple of DatasetRef objects.
    """
    deduped: list[DatasetRef] = []
    seen: set[tuple[str | None, str | None, str | None, str | None, bool, str | None]] = set()
    for ref in refs:
        if not ref.hf_dataset_id and not ref.not_applicable:
            continue
        key = (ref.hf_dataset_id, ref.revision, ref.split, ref.path, ref.not_applicable, ref.reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _dedupe_download_refs(refs: Iterable[DatasetRef]) -> tuple[DatasetRef, ...]:
    """Deduplicate references strictly for downloading execution.

    Keeps only commit-specific references over branch-agnostic references if a conflict exists.

    Args:
        refs: Iterable collection of DatasetRef objects.

    Returns:
        Deduplicated tuple of download-specific DatasetRef objects.
    """
    refs_tuple = tuple(refs)
    ids_with_revision = {ref.hf_dataset_id for ref in refs_tuple if ref.hf_dataset_id and ref.revision}
    deduped: list[DatasetRef] = []
    seen: set[tuple[str | None, str | None]] = set()
    for ref in refs_tuple:
        if ref.hf_dataset_id in ids_with_revision and not ref.revision:
            continue
        key = (ref.hf_dataset_id, ref.revision)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _mapping_has_manifest_shape(data: Mapping[str, Any]) -> bool:
    """Evaluate if a mapping matches the expected keys of a complete dataset manifest.

    Args:
        data: Mapping object to inspect.

    Returns:
        True if it has manifest keys, False otherwise.
    """
    return any(
        key in data
        for key in (
            "benchmark_id",
            "dataset",
            "dataset_refs",
            "official_sources",
            "source_provenance",
            "integration",
            "runner",
            "metrics",
            "source_status",
        )
    )


def _mapping_has_direct_ref_shape(data: Mapping[str, Any]) -> bool:
    """Evaluate if a mapping directly maps to a DatasetRef spec rather than an entire manifest.

    Args:
        data: Mapping object to inspect.

    Returns:
        True if it represents a dataset reference, False otherwise.
    """
    if "hf_dataset_id" in data or "repo_id" in data or "url" in data or "not_applicable" in data:
        return True
    return "id" in data and not _mapping_has_manifest_shape(data)


def _parse_official_sources(data: Mapping[str, Any], refs: list[DatasetRef], *, source_prefix: str = "official_sources") -> None:
    """Parse official repository sources inside a manifest dictionary into a list.

    Args:
        data: Manifest mapping containing source entries.
        refs: Output list to append parsed DatasetRefs to.
        source_prefix: Prefix trace string.
    """
    for key in _HF_DATASET_KEYS:
        values = data.get(key)
        for item in _items(values):
            if isinstance(item, Mapping):
                refs.append(DatasetRef.from_mapping(item, source=f"{source_prefix}.{key}"))
            elif item is not None:
                refs.append(DatasetRef(hf_dataset_id=item, source=f"{source_prefix}.{key}", metadata={"raw": item}))


def _parse_mapping_refs(data: Mapping[str, Any]) -> tuple[DatasetRef, ...]:
    """Parse mapping refs from a dataset mapping.

    Args:
        data: Input dictionary structure.

    Returns:
        Tuple of parsed DatasetRefs.
    """
    for key in _PAYLOAD_LIST_KEYS:
        values = data.get(key)
        if isinstance(values, list):
            nested: list[DatasetRef] = []
            for item in values:
                nested.extend(parse_dataset_refs(item))
            return _dedupe_refs(nested)

    refs: list[DatasetRef] = []

    if "dataset_refs" in data:
        for item in data.get("dataset_refs") or ():
            refs.extend(parse_dataset_refs(item))

    data_refs = data.get("data_refs")
    if isinstance(data_refs, Mapping):
        refs.extend(parse_dataset_refs(data_refs))

    dataset = data.get("dataset")
    if isinstance(dataset, Mapping):
        if _mapping_has_direct_ref_shape(dataset):
            refs.append(DatasetRef.from_mapping(dataset, source="dataset"))
    elif dataset is not None:
        refs.append(DatasetRef(hf_dataset_id=dataset, source="dataset", metadata={"raw": dataset}))

    official_sources = data.get("official_sources")
    if isinstance(official_sources, Mapping):
        _parse_official_sources(official_sources, refs)

    source_provenance = data.get("source_provenance")
    if isinstance(source_provenance, Mapping):
        _parse_official_sources(source_provenance, refs, source_prefix="source_provenance")

    if _mapping_has_direct_ref_shape(data):
        ref_source = "manifest" if _mapping_has_manifest_shape(data) else None
        direct_payload: dict[str, Any] = {}
        for key in (
            "hf_dataset_id",
            "repo_id",
            "id",
            "url",
            "revision",
            "sha",
            "commit",
            "license",
            "private",
            "gated",
            "split",
            "path",
            "not_applicable",
            "reason",
            "requires_auth",
            "notes",
            "expected_file_count",
            "expected_sibling_count",
            "sibling_count",
            "siblings",
        ):
            if key in data:
                direct_payload[key] = data[key]
        if direct_payload:
            refs.append(DatasetRef.from_mapping(direct_payload, source=ref_source))

    return _dedupe_refs(refs)


def parse_dataset_refs(source: Any) -> tuple[DatasetRef, ...]:
    """Parse dataset references from catalog, zoo, or manifest sources.

    Args:
        source: BenchmarkDatasetRef, BenchmarkZooEntry, manifest dict, or other input.

    Returns:
        Tuple of parsed and deduplicated DatasetRef objects.

    Raises:
        TypeError: If the source format is not supported.
    """

    if source is None:
        return ()
    if isinstance(source, DatasetRef):
        return (source,) if source.hf_dataset_id or source.not_applicable else ()
    if isinstance(source, BenchmarkDatasetRef):
        return _dedupe_refs((DatasetRef.from_benchmark_ref(source),))
    if isinstance(source, BenchmarkZooEntry):
        refs = [DatasetRef.from_benchmark_ref(ref, source="dataset_refs") for ref in source.dataset_refs]
        if source.dataset.not_applicable or (source.dataset.hf_dataset_id and not refs):
            refs.append(DatasetRef.from_benchmark_ref(source.dataset, source="dataset"))
        return _dedupe_refs(refs)
    if isinstance(source, Mapping):
        return _parse_mapping_refs(source)
    if isinstance(source, (list, tuple)):
        refs: list[DatasetRef] = []
        for item in source:
            refs.extend(parse_dataset_refs(item))
        return _dedupe_refs(refs)
    if hasattr(source, "data") and isinstance(getattr(source, "data"), Mapping):
        return parse_dataset_refs(getattr(source, "data"))
    if isinstance(source, str):
        return _dedupe_refs((DatasetRef(hf_dataset_id=source),))
    raise TypeError(f"cannot parse dataset refs from {type(source).__name__}")


def _first_dataset_ref(source: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str) -> DatasetRef:
    """Retrieve the first parsed DatasetRef from a source, or an empty fallback ref.

    Args:
        source: Any dataset reference source.

    Returns:
        The first resolved DatasetRef.
    """
    refs = parse_dataset_refs(source)
    return refs[0] if refs else DatasetRef()


def filter_dataset_refs(refs: Iterable[DatasetRef], dataset_ids: Iterable[str] | None = None) -> tuple[DatasetRef, ...]:
    """Filter dataset references down to a specific subset of Hugging Face repository IDs.

    Args:
        refs: Collection of DatasetRefs.
        dataset_ids: Optional collection of dataset IDs to keep.

    Returns:
        Filtered tuple of DatasetRefs.
    """
    requested = tuple(normalize_hf_dataset_id(dataset_id) for dataset_id in dataset_ids or ())
    requested_set = {dataset_id for dataset_id in requested if dataset_id}
    if not requested_set:
        return _dedupe_refs(refs)
    return _dedupe_refs(ref for ref in refs if ref.hf_dataset_id in requested_set)


def _normalize_license(value: str | None) -> str | None:
    """Clean and normalize a raw license identifier string.

    Args:
        value: Raw license string.

    Returns:
        Normalized lowercase hyphenated license ID, or None.
    """
    text = _optional_str(value)
    if text is None:
        return None
    return text.casefold().replace("_", "-").replace(" ", "-")


def classify_dataset_access(ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any]) -> DatasetAccessReport:
    """Classify dataset access rules, privacy settings, and license requirements.

    Args:
        ref: Dataset reference source.

    Returns:
        The generated DatasetAccessReport.
    """
    parsed = _first_dataset_ref(ref)
    if parsed.not_applicable:
        return DatasetAccessReport(
            hf_dataset_id=parsed.hf_dataset_id,
            access_status="not_applicable",
            license_status="not_applicable",
            requires_auth=False,
            issues=(),
        )

    issues: list[DatasetAccessIssue] = []
    if not parsed.hf_dataset_id:
        issues.append(
            DatasetAccessIssue(
                code="missing_hf_dataset_id",
                severity="error",
                message="dataset ref does not include an hf_dataset_id",
                field="hf_dataset_id",
            )
        )

    if parsed.private is True:
        issues.append(
            DatasetAccessIssue(
                code="private_dataset",
                severity="restricted",
                message="dataset is marked private",
                field="private",
                value=parsed.private,
            )
        )
    if _access_value_is_truthy(parsed.gated):
        issues.append(
            DatasetAccessIssue(
                code="gated_dataset",
                severity="restricted",
                message="dataset is marked gated and may require access approval",
                field="gated",
                value=parsed.gated,
            )
        )
    if parsed.requires_auth and not any(issue.code in {"private_dataset", "gated_dataset"} for issue in issues):
        issues.append(
            DatasetAccessIssue(
                code="requires_auth",
                severity="restricted",
                message="dataset requires Hugging Face authentication",
                field="requires_auth",
                value=True,
            )
        )

    license_value = _normalize_license(parsed.license)
    if license_value is None:
        license_status = "missing"
        issues.append(
            DatasetAccessIssue(
                code="missing_license",
                severity="warning",
                message="dataset ref does not declare a license",
                field="license",
            )
        )
    elif license_value in _OPEN_LICENSES:
        license_status = "open"
    elif license_value in {"other", "unknown"} or "cc-by-nc" in license_value or any(
        marker in license_value for marker in _LICENSE_REVIEW_MARKERS
    ):
        license_status = "review_required"
        issues.append(
            DatasetAccessIssue(
                code="license_review_required",
                severity="warning",
                message="dataset license should be reviewed before redistribution or automated use",
                field="license",
                value=parsed.license,
            )
        )
    else:
        license_status = "review_required"
        issues.append(
            DatasetAccessIssue(
                code="unrecognized_license",
                severity="warning",
                message="dataset license is not in the known open-license allowlist",
                field="license",
                value=parsed.license,
            )
        )

    if not parsed.hf_dataset_id:
        access_status = "missing_dataset_id"
    elif parsed.requires_auth:
        access_status = "restricted"
    else:
        access_status = "public"

    return DatasetAccessReport(
        hf_dataset_id=parsed.hf_dataset_id,
        access_status=access_status,
        license_status=license_status,
        requires_auth=parsed.requires_auth,
        issues=tuple(issues),
    )


def find_hf_downloader() -> tuple[str, ...]:
    """Find the available Hugging Face command line downloader executable on the system PATH.

    Returns:
        Tuple containing the executable name/path and argument (e.g. ('hf', 'download')).

    Raises:
        FileNotFoundError: If neither 'hf' nor 'huggingface-cli' is present.
    """
    hf = shutil.which("hf")
    if hf:
        return (hf, "download")
    huggingface_cli = shutil.which("huggingface-cli")
    if huggingface_cli:
        return (huggingface_cli, "download")
    raise FileNotFoundError("neither 'hf' nor 'huggingface-cli' was found on PATH")


def build_hf_download_command(
    ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
    cache_dir: str | Path,
    downloader: Sequence[str] = ("hf", "download"),
) -> tuple[str, ...]:
    """Construct the command string to run the Hugging Face CLI downloader for a dataset ref.

    Args:
        ref: Dataset reference.
        cache_dir: Destination cache folder.
        downloader: Downloader executable sequence.

    Returns:
        Command arguments tuple.

    Raises:
        ValueError: If the dataset reference is missing its hf_dataset_id.
    """
    parsed = _first_dataset_ref(ref)
    if not parsed.hf_dataset_id:
        raise ValueError("cannot build a Hugging Face download command without hf_dataset_id")
    command = [*downloader, parsed.hf_dataset_id, "--repo-type", "dataset", "--cache-dir", str(Path(cache_dir))]
    if parsed.revision:
        command.extend(("--revision", parsed.revision))
    return tuple(command)


def _read_ref(path: Path) -> str | None:
    """Read the content of a Git reference pointer file (e.g. refs/heads/main).

    Args:
        path: Path to the reference file.

    Returns:
        Content string, or None.
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _status_from_local_check(
    *,
    dataset_dir: Path,
    snapshot_dirs: tuple[Path, ...],
    revision_matches: bool,
    file_count: int,
    expected_file_count: int | None = None,
    incomplete_files: tuple[Mapping[str, JsonValue], ...],
    broken_links: tuple[str, ...],
) -> str:
    """Determine the combined local readiness status based on individual validation results.

    Args:
        dataset_dir: Checked cache folder path.
        snapshot_dirs: Snapshot directories checked.
        revision_matches: Revision validation result.
        file_count: Discovered file count.
        expected_file_count: Total expected file count.
        incomplete_files: Incomplete files detected.
        broken_links: Broken links detected.

    Returns:
        Status string (e.g. 'ready', 'not_found', 'incomplete_files', etc.).
    """
    if not dataset_dir.exists():
        return "not_found"
    if incomplete_files:
        return "incomplete_files"
    if broken_links:
        return "broken_links"
    if not revision_matches:
        return "revision_mismatch"
    if not snapshot_dirs or not all(path.exists() for path in snapshot_dirs):
        return "missing_snapshot"
    if file_count <= 0:
        return "empty_snapshot"
    if expected_file_count is not None and file_count < expected_file_count:
        return "incomplete_snapshot"
    return "ready"


def _shallow_incomplete_files(root: Path, *, limit: int = _DIRECT_DATASET_INCOMPLETE_LIMIT) -> tuple[Mapping[str, JsonValue], ...]:
    """Perform a shallow fast scan to detect any temporary download markers.

    Args:
        root: Root directory to search.
        limit: Capped size of result collection.

    Returns:
        Tuple of incomplete file info mappings.
    """
    incomplete_files: list[Mapping[str, JsonValue]] = []
    seen: set[Path] = set()
    pending = deque([root])
    while pending and len(incomplete_files) < limit:
        directory = pending.popleft()
        if len(incomplete_files) >= limit or not directory.is_dir() or directory in seen:
            continue
        seen.add(directory)
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if len(incomplete_files) >= limit:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                except OSError:
                    continue
                if not entry.name.endswith((".incomplete", ".aria2")):
                    continue
                path = Path(entry.path)
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                incomplete_files.append({"path": str(path), "size_bytes": path.stat().st_size})
    return tuple(incomplete_files)


def _snapshot_scan(snapshot_dir: Path) -> tuple[int, tuple[str, ...]]:
    """Scan snapshot folder recursively to count files and check for broken symlinks.

    Args:
        snapshot_dir: Snapshot directory Path.

    Returns:
        Tuple of (file_count_int, broken_links_tuple_of_strings).
    """
    if not snapshot_dir.is_dir():
        return 0, ()
    file_count = 0
    broken_links: list[str] = []
    pending = deque([snapshot_dir])
    seen: set[Path] = set()
    while pending:
        directory = pending.popleft()
        if directory in seen:
            continue
        seen.add(directory)
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_symlink():
                        if not path.exists():
                            broken_links.append(str(path))
                        else:
                            file_count += 1
                        continue
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                except OSError:
                    continue
    return file_count, tuple(broken_links)


def _expected_file_count_from_ref(ref: DatasetRef) -> int | None:
    """Retrieve expected dataset file count from metadata fields or siblings list.

    Args:
        ref: DatasetRef specification object.

    Returns:
        The expected file count, or None if unknown.
    """
    for source in (ref.metadata,):
        if not isinstance(source, Mapping):
            continue
        for key in ("expected_file_count", "expected_sibling_count", "sibling_count"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit() and int(value) > 0:
                return int(value)
        siblings = source.get("siblings")
        if isinstance(siblings, Sequence) and not isinstance(siblings, (str, bytes)):
            return len(siblings)
    return None


def _direct_dataset_scan(
    dataset_dir: Path,
    *,
    file_limit: int | None = None,
    incomplete_limit: int | None = None,
) -> tuple[int, bool, tuple[Mapping[str, JsonValue], ...]]:
    """Perform a comprehensive check on files under a direct dataset folder layout.

    Checks presence of expected siblings, identifies missing files, and detects incomplete downloads.

    Args:
        dataset_dir: Direct dataset layout directory Path.
        file_limit: Maximum files to scan.
        incomplete_limit: Maximum incomplete files to collect.

    Returns:
        Tuple of (file_count, file_count_capped, incomplete_files).
    """
    if not dataset_dir.is_dir():
        return 0, False, ()
    file_limit = _DIRECT_DATASET_FILE_COUNT_LIMIT if file_limit is None else file_limit
    incomplete_limit = _DIRECT_DATASET_INCOMPLETE_LIMIT if incomplete_limit is None else incomplete_limit

    def add_incomplete(path: Path, *, kind: str | None = None, relative_path: str | None = None) -> None:
        if len(incomplete_files) >= incomplete_limit:
            return
        item: dict[str, JsonValue] = {"path": str(path)}
        if path.exists():
            item["size_bytes"] = path.stat().st_size
        if kind:
            item["kind"] = kind
        if relative_path:
            item["relative_path"] = relative_path
        incomplete_files.append(item)

    file_count = 0
    file_count_capped = False
    incomplete_files: list[Mapping[str, JsonValue]] = []
    resume_expected_files = _direct_dataset_resume_expected_files(dataset_dir)

    if resume_expected_files:
        for relative_path in resume_expected_files:
            path = dataset_dir / relative_path
            if path.is_file():
                if file_count < file_limit:
                    file_count += 1
                else:
                    file_count_capped = True
            else:
                add_incomplete(path, kind="missing_expected_file", relative_path=relative_path)
                if len(incomplete_files) >= incomplete_limit:
                    break
        return file_count, file_count_capped, tuple(incomplete_files)

    expected_files = _direct_dataset_expected_files(dataset_dir)

    if expected_files:
        expected_to_check: list[str] = []
        seen_expected: set[str] = set()
        if expected_files and len(expected_files) > file_limit:
            file_count_capped = True
        for relative_path in expected_files[:file_limit]:
            expected_to_check.append(relative_path)
            seen_expected.add(relative_path)

        for relative_path in expected_to_check:
            path = dataset_dir / relative_path
            if path.is_file():
                if file_count < file_limit:
                    file_count += 1
                else:
                    file_count_capped = True
            else:
                add_incomplete(path, kind="missing_expected_file", relative_path=relative_path)
                if len(incomplete_files) >= incomplete_limit:
                    break

        return file_count, file_count_capped, tuple(incomplete_files)

    try:
        entries = os.scandir(dataset_dir)
    except OSError:
        return 0, False, ()
    with entries:
        for entry in entries:
            if entry.name in {".hfd", ".cache", ".git"}:
                continue
            path = Path(entry.path)
            if entry.name.endswith((".incomplete", ".aria2")):
                add_incomplete(path)
                continue
            try:
                if entry.is_file(follow_symlinks=False) or entry.is_dir(follow_symlinks=False):
                    file_count += 1
            except OSError:
                continue
            if file_count >= file_limit:
                file_count_capped = True
                add_incomplete(dataset_dir, kind="direct_hfd_scan_truncated")
                break
    return file_count, file_count_capped, tuple(incomplete_files)


def _safe_direct_dataset_relative_path(value: Any) -> str | None:
    """Validate and normalize a direct layout relative path string.

    Filters out absolute paths, parent directory traversals, and ignored names.

    Args:
        value: Input path value.

    Returns:
        Normalized relative path string or None.
    """
    text = _optional_str(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.name in _IGNORED_DIRECT_DATASET_SIBLING_NAMES or any(
        part in _IGNORED_DIRECT_DATASET_SIBLING_PARTS for part in path.parts
    ):
        return None
    return path.as_posix()


def _direct_dataset_expected_files(dataset_dir: Path) -> tuple[str, ...]:
    """Retrieve expected direct dataset file names listed in HuggingFace metadata.

    Args:
        dataset_dir: Root dataset directory Path.

    Returns:
        Tuple of expected relative file paths.
    """
    metadata_path = dataset_dir / ".hfd" / "repo_metadata.json"
    if not metadata_path.is_file():
        return ()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        return ()

    expected: list[str] = []
    seen: set[str] = set()
    for item in siblings:
        if isinstance(item, Mapping):
            value = item.get("rfilename")
        else:
            value = item
        normalized = _safe_direct_dataset_relative_path(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        expected.append(normalized)
    return tuple(expected)


def _direct_dataset_resume_expected_files(dataset_dir: Path) -> tuple[str, ...]:
    """Return relative paths from an HFD aria2 resume list."""

    urls_path = dataset_dir / ".hfd" / "aria2c_urls.txt"
    if not urls_path.is_file():
        return ()
    try:
        lines = urls_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    expected: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = line.strip()
        if not text.startswith(("http://", "https://")) or "/resolve/" not in text:
            continue
        suffix = text.split("#", 1)[0].split("?", 1)[0].split("/resolve/", 1)[1]
        parts = suffix.split("/", 1)
        if len(parts) != 2:
            continue
        normalized = _safe_direct_dataset_relative_path(unquote(parts[1]))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        expected.append(normalized)
    return tuple(expected)


def direct_dataset_expected_files(dataset_dir: str | Path) -> tuple[str, ...]:
    """Return expected relative files listed by an HFD direct-layout metadata file."""

    return _direct_dataset_expected_files(Path(dataset_dir))


def _direct_dataset_revision(dataset_dir: Path) -> str | None:
    local_dir_metadata = dataset_dir / ".cache" / "huggingface" / "download"
    local_dir_revision = _direct_dataset_download_metadata_revision(local_dir_metadata)
    if local_dir_revision:
        return local_dir_revision

    metadata_path = dataset_dir / ".hfd" / "repo_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        revision = payload.get("sha") or payload.get("revision") or payload.get("commit")
        if isinstance(revision, str) and revision.strip():
            return revision.strip()

    return None


def _direct_dataset_download_metadata_revision(local_dir_metadata: Path) -> str | None:
    """Read a bounded sample of Hugging Face download metadata revisions.

    Args:
        local_dir_metadata: Direct-layout ``.cache/huggingface/download`` directory.
    """

    if not local_dir_metadata.is_dir():
        return None
    revisions: set[str] = set()
    pending = deque([local_dir_metadata])
    seen_dirs = 0
    seen_files = 0
    while pending and seen_dirs < _DIRECT_DATASET_METADATA_DIR_LIMIT and seen_files < _DIRECT_DATASET_METADATA_FILE_LIMIT:
        directory = pending.popleft()
        seen_dirs += 1
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if seen_files >= _DIRECT_DATASET_METADATA_FILE_LIMIT:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".metadata"):
                        continue
                except OSError:
                    continue
                path = Path(entry.path)
                seen_files += 1
                try:
                    first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
                except (OSError, IndexError, UnicodeDecodeError):
                    continue
                if first_line:
                    revisions.add(first_line)
                if len(revisions) > 1:
                    return None
    return next(iter(revisions)) if len(revisions) == 1 else None


def _direct_revision_matches(revision: str | None, direct_revision: str | None) -> bool:
    if revision and is_commit_like_revision(revision):
        return direct_revision == revision
    if revision:
        return bool(direct_revision)
    return True


def _direct_dataset_status(
    *,
    dataset_dir: Path,
    revision_matches: bool,
    file_count: int,
    incomplete_files: tuple[Mapping[str, JsonValue], ...],
) -> str:
    if not dataset_dir.exists():
        return "not_found"
    if incomplete_files:
        return "direct_hfd_incomplete_files"
    if file_count <= 0:
        return "direct_hfd_empty"
    if not revision_matches:
        return "direct_hfd_revision_mismatch"
    return "ready"


def check_local_dataset(
    ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
    cache_dir: str | Path,
    expected_revision: str | None = None,
) -> DatasetLocalStatus:
    """Verifies the presence and completeness of a dataset in the local Hugging Face cache.

    Analyzes whether files are complete, matches commit revision hashes, detects partial/corrupted
    downloads (e.g. unfinished `.hfd` tasks or empty cache locks), and computes statistics.

    Args:
        ref: The dataset reference to check (ID, ref object, or manifest dictionary).
        cache_dir: The root directory containing Hugging Face caches.
        expected_revision: An optional commit hash or branch name to enforce.

    Returns:
        DatasetLocalStatus: Object capturing readiness state, checked folder path, and error diagnoses.
    """
    parsed = _first_dataset_ref(ref)

    revision = expected_revision or parsed.revision
    if parsed.not_applicable:
        return DatasetLocalStatus(
            hf_dataset_id=parsed.hf_dataset_id,
            cache_dataset_dir=None,
            expected_revision=revision,
            ready=True,
            status="not_applicable",
            reason=parsed.reason or "dataset not applicable",
        )
    if not parsed.hf_dataset_id:
        return DatasetLocalStatus(
            hf_dataset_id=None,
            cache_dataset_dir=None,
            expected_revision=revision,
            revision_matches=False,
            ready=False,
            status="missing_dataset_id",
            reason="dataset ref does not include an hf_dataset_id",
        )

    dataset_dir = hf_cache_dataset_dir(cache_dir, parsed.hf_dataset_id)
    refs_dir = dataset_dir / "refs"
    snapshots_dir = dataset_dir / "snapshots"
    ref_name = revision if revision and not is_commit_like_revision(revision) else "main"
    referenced_snapshot = _read_ref(refs_dir / ref_name)

    snapshot_dirs: list[Path] = []
    if revision and is_commit_like_revision(revision):
        snapshot_dirs.append(snapshots_dir / revision)
    elif revision and referenced_snapshot:
        snapshot_dirs.append(snapshots_dir / referenced_snapshot)
    elif referenced_snapshot:
        snapshot_dirs.append(snapshots_dir / referenced_snapshot)
    elif snapshots_dir.is_dir():
        snapshot_dirs.extend(path for path in sorted(snapshots_dir.iterdir()) if path.is_dir())

    incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    if dataset_dir.exists():
        incomplete_files = _shallow_incomplete_files(dataset_dir)

    broken_links: list[str] = []
    file_count = 0
    for snapshot_dir in snapshot_dirs:
        snapshot_file_count, snapshot_broken_links = _snapshot_scan(snapshot_dir)
        file_count += snapshot_file_count
        broken_links.extend(snapshot_broken_links)

    if revision and is_commit_like_revision(revision):
        revision_matches = (snapshots_dir / revision).is_dir() or referenced_snapshot == revision
    elif revision:
        revision_matches = bool(referenced_snapshot) and (snapshots_dir / referenced_snapshot).is_dir()
    else:
        revision_matches = True

    snapshot_tuple = tuple(snapshot_dirs)
    broken_link_tuple = tuple(broken_links)
    expected_file_count = _expected_file_count_from_ref(parsed)
    missing_file_count = max((expected_file_count or 0) - file_count, 0)
    status = _status_from_local_check(
        dataset_dir=dataset_dir,
        snapshot_dirs=snapshot_tuple,
        revision_matches=revision_matches,
        file_count=file_count,
        expected_file_count=expected_file_count,
        incomplete_files=incomplete_files,
        broken_links=broken_link_tuple,
    )
    hf_ready = status == "ready"
    if hf_ready:
        return DatasetLocalStatus(
            hf_dataset_id=parsed.hf_dataset_id,
            cache_dataset_dir=dataset_dir,
            direct_dataset_dir=None,
            local_layout="hf_cache",
            expected_revision=revision,
            ref_name=ref_name,
            referenced_snapshot=referenced_snapshot,
            revision_matches=revision_matches,
            snapshot_dirs=snapshot_tuple,
            file_count=file_count,
            expected_file_count=expected_file_count,
            missing_file_count=missing_file_count,
            direct_file_count=0,
            direct_file_count_capped=False,
            direct_revision=None,
            direct_revision_matches=False,
            direct_ready=False,
            incomplete_files=incomplete_files,
            direct_incomplete_files=(),
            broken_links=broken_link_tuple,
            ready=True,
            status=status,
        )
    direct_candidates = direct_dataset_dir_candidates_for_ref(cache_dir, parsed)
    direct_states: list[tuple[Path, int, bool, tuple[Mapping[str, JsonValue], ...], str | None, bool, str]] = []
    for candidate in direct_candidates:
        direct_file_count, direct_file_count_capped, direct_incomplete_files = _direct_dataset_scan(candidate)
        direct_revision = _direct_dataset_revision(candidate)
        direct_revision_match = _direct_revision_matches(revision, direct_revision)
        direct_status = _direct_dataset_status(
            dataset_dir=candidate,
            revision_matches=direct_revision_match,
            file_count=direct_file_count,
            incomplete_files=direct_incomplete_files,
        )
        direct_states.append(
            (
                candidate,
                direct_file_count,
                direct_file_count_capped,
                direct_incomplete_files,
                direct_revision,
                direct_revision_match,
                direct_status,
            )
        )
        if direct_status == "ready":
            break
    direct_state = next((item for item in direct_states if item[6] == "ready"), None)
    if direct_state is None:
        direct_state = next((item for item in direct_states if item[0].exists()), direct_states[0])
    (
        direct_dir,
        direct_file_count,
        direct_file_count_capped,
        direct_incomplete_files,
        direct_revision,
        direct_revision_match,
        direct_status,
    ) = direct_state
    direct_ready = direct_status == "ready"
    ready = hf_ready or direct_ready
    combined_status = "ready" if ready else (direct_status if status == "not_found" else status)
    if hf_ready:
        local_layout = "hf_cache"
    elif direct_dir.exists():
        local_layout = "direct_hfd"
    elif status != "not_found":
        local_layout = "hf_cache"
    else:
        local_layout = "missing"
    return DatasetLocalStatus(
        hf_dataset_id=parsed.hf_dataset_id,
        cache_dataset_dir=dataset_dir,
        direct_dataset_dir=direct_dir,
        local_layout=local_layout,
        expected_revision=revision,
        ref_name=ref_name,
        referenced_snapshot=referenced_snapshot,
        revision_matches=revision_matches or direct_revision_match,
        snapshot_dirs=snapshot_tuple,
        file_count=file_count or direct_file_count,
        expected_file_count=expected_file_count,
        missing_file_count=missing_file_count,
        direct_file_count=direct_file_count,
        direct_file_count_capped=direct_file_count_capped,
        direct_revision=direct_revision,
        direct_revision_matches=direct_revision_match,
        direct_ready=direct_ready,
        incomplete_files=incomplete_files,
        direct_incomplete_files=direct_incomplete_files,
        broken_links=broken_link_tuple,
        ready=ready,
        status=combined_status,
    )


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
                {"hf_dataset_id": dataset_id, **item} if isinstance(item, Mapping) else {"hf_dataset_id": dataset_id, "path": item}
                for dataset_id, item in value.items()
            )
    return tuple(
        {"hf_dataset_id": dataset_id, **item} if isinstance(item, Mapping) else {"hf_dataset_id": dataset_id, "path": item}
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


def _metadata_for_plan(
    *,
    source: Any,
    all_refs: tuple[DatasetRef, ...],
    selected_refs: tuple[DatasetRef, ...],
    requested_dataset_ids: tuple[str, ...],
    metadata_mode: str,
) -> dict[str, JsonValue]:
    """Generate plan statistics and diagnostics for a planned download execution.

    Args:
        source: BenchmarkZooEntry or mapping representing dataset source.
        all_refs: Parsed DatasetRefs.
        selected_refs: Selected DatasetRefs.
        requested_dataset_ids: Target dataset repo IDs requested.
        metadata_mode: 'filtered' or 'full'.

    Returns:
        Structured plan metadata dictionary.
    """
    if metadata_mode not in {"filtered", "full"}:
        raise ValueError("metadata_mode must be one of: filtered, full")

    benchmark_id = None
    if isinstance(source, BenchmarkZooEntry):
        benchmark_id = source.benchmark_id
    elif isinstance(source, Mapping):
        value = source.get("benchmark_id") or source.get("id")
        benchmark_id = str(value) if value is not None else None

    selected_dataset_ids = {ref.hf_dataset_id for ref in selected_refs}
    metadata: dict[str, JsonValue] = {
        "metadata_mode": metadata_mode,
        "benchmark_id": benchmark_id,
        "filtered": bool(requested_dataset_ids),
        "requested_dataset_ids": list(requested_dataset_ids),
        "hf_dataset_ids": [ref.hf_dataset_id for ref in selected_refs if ref.hf_dataset_id],
        "dataset_refs": [ref.to_dict() for ref in selected_refs],
        "missing_requested_dataset_ids": [
            dataset_id
            for dataset_id in requested_dataset_ids
            if dataset_id not in selected_dataset_ids
        ],
    }
    not_applicable = next((ref for ref in selected_refs or all_refs if ref.not_applicable), None)
    if not_applicable is not None:
        metadata["dataset_not_applicable"] = True
        metadata["reason"] = not_applicable.reason

    if metadata_mode == "full":
        metadata["available_hf_dataset_ids"] = [ref.hf_dataset_id for ref in all_refs if ref.hf_dataset_id]
        metadata["available_dataset_refs"] = [ref.to_dict() for ref in all_refs]

    return metadata


class DatasetManager:
    """Reusable management facade for benchmark datasets and Hugging Face cache repositories.

    This class provides unified access points to:
    - Parse and filter dataset reference structures.
    - Resolve cache directories.
    - Check local dataset completeness and file layouts.
    - Locate physical assets on disk (local roots, custom manifests, or Hugging Face cache).
    - Generate automated CLI download plans and commands.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        downloader: Sequence[str] = ("hf", "download"),
    ) -> None:
        """Initializes the manager with a target cache directory and a download utility prefix.

        Args:
            cache_dir: The cache directory Path.
            downloader: Executable name/args for downloader utility.
        """
        self.cache_dir = Path(cache_dir)
        self.downloader = tuple(downloader)

    def parse_refs(self, source: Any) -> tuple[DatasetRef, ...]:
        """Parses arbitrary dataset definitions (manifests, mappings, refs) into clean DatasetRef objects.

        Args:
            source: Source dataset spec or container.

        Returns:
            Tuple of DatasetRef objects.
        """
        return parse_dataset_refs(source)

    def filter_refs(
        self,
        refs: Iterable[DatasetRef],
        dataset_ids: Iterable[str] | None = None,
    ) -> tuple[DatasetRef, ...]:
        """Filters dataset references down to a specific set of active/requested dataset IDs.

        Args:
            refs: Dataset references collection.
            dataset_ids: Dataset repository IDs.

        Returns:
            Filtered tuple of DatasetRefs.
        """
        return filter_dataset_refs(refs, dataset_ids)

    def cache_dataset_dir(self, dataset_id: str) -> Path:
        """Resolves the expected local Hugging Face snapshot cache path for a given dataset ID.

        Args:
            dataset_id: Target dataset ID.

        Returns:
            HuggingFace snapshot cache path.
        """
        return hf_cache_dataset_dir(self.cache_dir, dataset_id)

    def build_download_command(self, ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str) -> tuple[str, ...]:
        """Constructs an executable CLI command array to retrieve the target dataset from Hugging Face.

        Args:
            ref: DatasetRef target spec.

        Returns:
            Command arguments tuple.
        """
        return build_hf_download_command(ref, self.cache_dir, self.downloader)

    def classify_access(self, ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any]) -> DatasetAccessReport:
        """Classifies security and license accessibility for a dataset (e.g. public vs gated vs custom license).

        Args:
            ref: Target dataset reference.

        Returns:
            Accessibility report.
        """
        return classify_dataset_access(ref)

    def check_local(
        self,
        ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
        *,
        expected_revision: str | None = None,
    ) -> DatasetLocalStatus:
        """Scans local cache and verifies dataset completeness, files count, and revision hashes.

        Args:
            ref: Dataset target spec.
            expected_revision: Optional expected revision string.

        Returns:
            Readiness status structure.
        """
        return check_local_dataset(ref, self.cache_dir, expected_revision)

    def locate_local(
        self,
        ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
        *,
        data_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> DatasetLocation:
        """Determines the physical directory path where a dataset resides.

        Queries paths in the following prioritized order:
        1. High-priority local manifest rules.
        2. Absolute data roots.
        3. Environment variable overrides.
        4. Standard Hugging Face cache snapshot folders.

        Args:
            ref: Dataset target spec.
            data_root: Optional data root.
            manifest_path: Optional manifest file path.
            env: Optional environment dictionary.

        Returns:
            The resolved location of the dataset.
        """
        return locate_local_dataset(
            ref,
            data_root=data_root,
            manifest_path=manifest_path,
            cache_dir=self.cache_dir,
            env=env,
        )

    def build_download_plan(
        self,
        source: Any,
        *,
        dataset_ids: Iterable[str] | None = None,
        check_local: bool = False,
        metadata_mode: str = "filtered",
    ) -> DatasetDownloadPlan:
        """Generates a complete, deduplicated download strategy for a collection of dataset sources.

        Args:
            source: Source dataset collection or manifest.
            dataset_ids: Subset of dataset repository IDs.
            check_local: Whether to run a local status scan on selected datasets.
            metadata_mode: Mode label.

        Returns:
            The cohesive download plan.
        """
        all_refs = self.parse_refs(source)
        requested_dataset_ids = tuple(
            dataset_id for dataset_id in (normalize_hf_dataset_id(item) for item in dataset_ids or ()) if dataset_id
        )
        selected_refs = filter_dataset_refs(all_refs, requested_dataset_ids)

        download_refs = _dedupe_download_refs(selected_refs)
        commands = tuple(
            self.build_download_command(ref)
            for ref in download_refs
            if ref.hf_dataset_id
        )
        local_checks = tuple(self.check_local(ref) for ref in download_refs) if check_local else ()
        access_reports = tuple(classify_dataset_access(ref) for ref in download_refs)
        metadata = _metadata_for_plan(
            source=source,
            all_refs=all_refs,
            selected_refs=download_refs,
            requested_dataset_ids=requested_dataset_ids,
            metadata_mode=metadata_mode,
        )
        return DatasetDownloadPlan(
            refs=download_refs,
            commands=commands,
            cache_dir=self.cache_dir,
            access_reports=access_reports,
            local_checks=local_checks,
            metadata=metadata,
        )

__all__ = [
    "DatasetAccessIssue",
    "DatasetAccessReport",
    "DatasetDownloadPlan",
    "DatasetLocation",
    "DatasetLocalStatus",
    "DatasetManager",
    "DatasetRef",
    "build_hf_download_command",
    "check_local_dataset",
    "classify_dataset_access",
    "dataset_location_env_var",
    "direct_dataset_dir_candidates",
    "direct_dataset_dir_candidates_for_ref",
    "direct_dataset_expected_files",
    "filter_dataset_refs",
    "find_hf_downloader",
    "hf_cache_dataset_dir",
    "is_commit_like_revision",
    "locate_local_dataset",
    "normalize_hf_dataset_id",
    "parse_dataset_refs",
]
