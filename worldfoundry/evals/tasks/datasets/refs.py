"""Dataset references: the DatasetRef record and parsing refs out of catalog and manifest payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..catalog.schema import BenchmarkDatasetRef, BenchmarkZooEntry

JsonValue = Any


_FALSEY_ACCESS_VALUES = frozenset({None, False, "false", "False", "none", "None", ""})


_HF_DATASET_KEYS = ("huggingface_datasets", "hf_datasets", "datasets")


_PAYLOAD_LIST_KEYS = ("benchmarks", "entries", "benchmark_zoo", "manifests")


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
    def from_benchmark_ref(cls, ref: BenchmarkDatasetRef, *, source: str | None = None) -> DatasetRef:
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
    def from_mapping(cls, data: Mapping[str, Any], *, source: str | None = None) -> DatasetRef:
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


def _parse_official_sources(
    data: Mapping[str, Any], refs: list[DatasetRef], *, source_prefix: str = "official_sources"
) -> None:
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
    if hasattr(source, "data") and isinstance(source.data, Mapping):
        return parse_dataset_refs(source.data)
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


__all__ = [
    "DatasetRef",
    "filter_dataset_refs",
    "is_commit_like_revision",
    "normalize_hf_dataset_id",
    "parse_dataset_refs",
]
