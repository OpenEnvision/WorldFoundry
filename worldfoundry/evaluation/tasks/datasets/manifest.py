"""Dataset sample manifests and checksum validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api.json_contract import sha256_file as file_sha256
from worldfoundry.evaluation.utils import jsonable, read_json, read_json_or_jsonl, write_json

DATASET_MANIFEST_SCHEMA_VERSION = "worldfoundry-dataset-manifest"


def _canonical_json_sha256(value: Any) -> str:
    """Compute the SHA256 hash of a JSON-serializable value under a canonical encoding.

    Args:
        value: Any JSON-serializable Python object.

    Returns:
        The SHA256 hex digest string.
    """
    payload = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_rows(payload: Any, *, source_path: Path) -> list[dict[str, Any]]:
    """Recursively extract and validate list of sample dictionary rows from JSON payloads.

    Args:
        payload: Input JSON payload (list or dict container).
        source_path: File Path of the source for error reporting.

    Returns:
        List of row dictionaries.

    Raises:
        TypeError: If the payload structure is invalid or elements are not mappings.
    """
    if isinstance(payload, Mapping):
        for key in ("samples", "items", "data", "rows"):
            value = payload.get(key)
            if value is not None:
                return _coerce_rows(value, source_path=source_path)
        raise TypeError(f"dataset sample source must contain samples/items/data/rows: {source_path}")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError(f"dataset sample source must be a sequence: {source_path}")
    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise TypeError(f"dataset sample row {index} must be an object: {source_path}")
        rows.append(dict(item))
    return rows


def read_dataset_samples(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate sample rows from a JSON or JSONL file.

    Args:
        path: Path to the JSON/JSONL dataset file.

    Returns:
        List of parsed sample dictionaries.
    """
    source_path = Path(path)
    return _coerce_rows(read_json_or_jsonl(source_path), source_path=source_path)


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    """Extract or generate a unique identifier for a sample row.

    Args:
        row: Sample dictionary row.
        index: Row position index.

    Returns:
        String identifier.
    """
    value = row.get("sample_id", row.get("id"))
    return str(value) if value not in (None, "") else f"sample-{index:04d}"


def _sample_id_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate sample identifier statistics and detect duplicates/missing values.

    Args:
        rows: Sequence of sample rows.

    Returns:
        Dictionary of sample id stats.
    """
    sample_ids = [_sample_id(row, index) for index, row in enumerate(rows)]
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen and sample_id not in duplicate_ids:
            duplicate_ids.append(sample_id)
        seen.add(sample_id)
    return {
        "sample_id_count": len(sample_ids),
        "sample_ids_sha256": _canonical_json_sha256(sample_ids),
        "missing_sample_id_count": sum(1 for row in rows if row.get("sample_id", row.get("id")) in (None, "")),
        "duplicate_sample_ids": duplicate_ids,
    }


def _relative_or_absolute(path: Path, root: Path | None) -> str:
    """Retrieve relative path string if root is provided, otherwise absolute path string.

    Args:
        path: Path to the target file.
        root: Optional root directory Path.

    Returns:
        Path string.
    """
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class DatasetManifest:
    """Represents a frozen schema descriptor for a dataset's samples and metadata.

    Attributes:
        dataset_id: Unique string identifier for the dataset.
        samples_path: Path string where the samples are stored.
        sample_count: Number of samples in the dataset.
        sha256: SHA256 checksum of the samples file.
        split: Split identifier (e.g. 'train', 'test', 'default').
        root: Optional root directory Path.
        sample_ids_sha256: Canonical SHA256 hash of all sample identifiers combined.
        sample_id_count: Number of unique sample IDs.
        missing_sample_id_count: Count of samples missing explicit IDs.
        duplicate_sample_ids: Tuple of duplicate sample identifier strings.
        source_uri: Optional original download source URL.
        license: Optional dataset license.
        access: Optional access rules/permissions metadata.
        metadata: Optional extra arbitrary metadata.
        schema_version: Schema version of the manifest format.
    """

    dataset_id: str
    samples_path: str
    sample_count: int
    sha256: str
    split: str = "default"
    root: str | None = None
    sample_ids_sha256: str | None = None
    sample_id_count: int | None = None
    missing_sample_id_count: int = 0
    duplicate_sample_ids: tuple[str, ...] = ()
    source_uri: str | None = None
    license: str | None = None
    access: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Post-initialization validation and conversion."""
        if self.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported DatasetManifest schema_version: {self.schema_version}")
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "samples_path", str(self.samples_path))
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "sha256", str(self.sha256))
        object.__setattr__(self, "split", str(self.split or "default"))
        object.__setattr__(self, "root", None if self.root is None else str(self.root))
        object.__setattr__(
            self,
            "sample_id_count",
            self.sample_count if self.sample_id_count is None else int(self.sample_id_count),
        )
        object.__setattr__(self, "missing_sample_id_count", int(self.missing_sample_id_count or 0))
        object.__setattr__(self, "duplicate_sample_ids", tuple(str(item) for item in self.duplicate_sample_ids))
        object.__setattr__(self, "source_uri", None if self.source_uri is None else str(self.source_uri))
        object.__setattr__(self, "license", None if self.license is None else str(self.license))
        object.__setattr__(self, "access", dict(self.access or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        """Reconstruct a DatasetManifest from a key-value mapping.

        Args:
            data: Input dictionary matching schema fields.

        Returns:
            Reconstructed DatasetManifest instance.
        """
        return cls(
            dataset_id=str(data["dataset_id"]),
            samples_path=str(data["samples_path"]),
            sample_count=int(data["sample_count"]),
            sha256=str(data["sha256"]),
            split=str(data.get("split") or "default"),
            root=data.get("root"),
            sample_ids_sha256=data.get("sample_ids_sha256"),
            sample_id_count=data.get("sample_id_count"),
            missing_sample_id_count=int(data.get("missing_sample_id_count") or 0),
            duplicate_sample_ids=tuple(data.get("duplicate_sample_ids") or ()),
            source_uri=data.get("source_uri"),
            license=data.get("license"),
            access=data.get("access") if isinstance(data.get("access"), Mapping) else {},
            metadata=data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {},
            schema_version=str(data.get("schema_version", DATASET_MANIFEST_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this manifest to a standard JSON-serializable dictionary.

        Returns:
            Dictionary of manifest attributes.
        """
        return jsonable(asdict(self))


def build_dataset_manifest(
    *,
    samples_path: str | Path,
    dataset_id: str | None = None,
    split: str = "default",
    root: str | Path | None = None,
    source_uri: str | None = None,
    license: str | None = None,
    access: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    """Build a complete DatasetManifest by reading and analyzing a samples file.

    Args:
        samples_path: Path of the samples file.
        dataset_id: Optional custom dataset ID.
        split: Dataset split label.
        root: Optional root path.
        source_uri: Optional source URI.
        license: Optional dataset license.
        access: Optional access info.
        metadata: Optional extra metadata.

    Returns:
        The generated DatasetManifest.
    """
    source_path = Path(samples_path)
    root_path = None if root is None else Path(root)
    if not source_path.is_absolute() and root_path is not None and not source_path.exists():
        source_path = root_path / source_path
    rows = read_dataset_samples(source_path)
    id_stats = _sample_id_stats(rows)
    return DatasetManifest(
        dataset_id=dataset_id or source_path.stem,
        split=split,
        root=None if root_path is None else str(root_path.resolve()),
        samples_path=_relative_or_absolute(source_path, root_path),
        sample_count=len(rows),
        sha256=file_sha256(source_path),
        source_uri=source_uri,
        license=license,
        access=dict(access or {}),
        metadata=dict(metadata or {}),
        **id_stats,
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load a DatasetManifest from a JSON file.

    Args:
        path: Path to the manifest file.

    Returns:
        Loaded DatasetManifest.

    Raises:
        ValueError: If the file is not a valid JSON dictionary.
    """
    source_path = Path(path)
    payload = read_json(source_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"dataset manifest must be a JSON object: {source_path}")
    return DatasetManifest.from_dict(payload)


def write_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    """Serialize and write a DatasetManifest to a JSON file.

    Args:
        manifest: Manifest object to serialize.
        path: Destination path on disk.

    Returns:
        The output file Path.
    """
    destination = Path(path)
    write_json(destination, manifest.to_dict())
    return destination


def resolve_dataset_samples_path(
    manifest: DatasetManifest | Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Resolve the absolute or relative path to the samples file described by a manifest.

    Args:
        manifest: Manifest object or equivalent mapping.
        manifest_path: Optional path of the manifest file itself to resolve relative paths against.

    Returns:
        Resolved Path of the samples file.
    """
    item = manifest if isinstance(manifest, DatasetManifest) else DatasetManifest.from_dict(manifest)
    samples_path = Path(item.samples_path)
    if samples_path.is_absolute():
        return samples_path
    if item.root:
        root = Path(item.root)
        if not root.is_absolute() and manifest_path is not None:
            root = Path(manifest_path).parent / root
        return root / samples_path
    if manifest_path is not None:
        return Path(manifest_path).parent / samples_path
    return samples_path


def validate_dataset_manifest(
    path_or_manifest: str | Path | DatasetManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a dataset manifest file/object against the actual samples file on disk.

    Verifies files presence, matching sample counts, file checksum, and ID consistency.

    Args:
        path_or_manifest: Path to manifest, or manifest object/mapping.

    Returns:
        Structured dictionary indicating verification status, issues, and statistics.
    """
    issues: list[str] = []
    warnings: list[str] = []
    manifest_path: Path | None = None
    try:
        if isinstance(path_or_manifest, DatasetManifest):
            manifest = path_or_manifest
        elif isinstance(path_or_manifest, Mapping):
            manifest = DatasetManifest.from_dict(path_or_manifest)
        else:
            manifest_path = Path(path_or_manifest)
            manifest = load_dataset_manifest(manifest_path)

        samples_path = resolve_dataset_samples_path(manifest, manifest_path=manifest_path)
        if not samples_path.exists():
            issues.append(f"samples_path not found: {samples_path}")
            sample_count = None
            sha256 = None
            id_stats: dict[str, Any] = {}
        else:
            rows = read_dataset_samples(samples_path)
            sample_count = len(rows)
            sha256 = file_sha256(samples_path)
            id_stats = _sample_id_stats(rows)
            if sample_count != manifest.sample_count:
                issues.append(f"sample_count mismatch: manifest={manifest.sample_count} actual={sample_count}")
            if sha256 != manifest.sha256:
                issues.append("sha256 mismatch")
            if manifest.sample_ids_sha256 and id_stats["sample_ids_sha256"] != manifest.sample_ids_sha256:
                issues.append("sample_ids_sha256 mismatch")
            if id_stats["duplicate_sample_ids"]:
                issues.append(f"duplicate sample ids: {', '.join(id_stats['duplicate_sample_ids'])}")
            if id_stats["missing_sample_id_count"]:
                warnings.append(f"{id_stats['missing_sample_id_count']} sample(s) rely on generated sample ids")

        return {
            "ok": not issues,
            "schema_version": manifest.schema_version,
            "dataset_id": manifest.dataset_id,
            "split": manifest.split,
            "manifest_path": None if manifest_path is None else str(manifest_path.resolve()),
            "samples_path": str(samples_path.resolve()) if samples_path.exists() else str(samples_path),
            "sample_count": sample_count,
            "manifest_sample_count": manifest.sample_count,
            "sha256": sha256,
            "manifest_sha256": manifest.sha256,
            "sample_ids_sha256": id_stats.get("sample_ids_sha256"),
            "manifest_sample_ids_sha256": manifest.sample_ids_sha256,
            "issues": issues,
            "warnings": warnings,
        }
    except Exception as exc:  # noqa: BLE001 - validation should return structured errors.
        return {
            "ok": False,
            "schema_version": None,
            "dataset_id": None,
            "split": None,
            "manifest_path": None if manifest_path is None else str(manifest_path.resolve()),
            "samples_path": None,
            "sample_count": None,
            "manifest_sample_count": None,
            "sha256": None,
            "manifest_sha256": None,
            "sample_ids_sha256": None,
            "manifest_sample_ids_sha256": None,
            "issues": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DatasetManifest",
    "build_dataset_manifest",
    "file_sha256",
    "load_dataset_manifest",
    "read_dataset_samples",
    "resolve_dataset_samples_path",
    "validate_dataset_manifest",
    "write_dataset_manifest",
]
