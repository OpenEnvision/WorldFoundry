"""Utilities for describing evaluation artifacts without embedding their data.

ArtifactRef records where an artifact lives, what kind of artifact it is, and
optional integrity metadata such as file size, MIME type, and SHA-256 hash.
Local files can be enriched with hash/size data, while remote URIs are kept as
references so callers do not accidentally treat object-store paths as files.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from worldfoundry.evaluation.api.json_contract import JsonContract, copy_mapping, sha256_bytes, sha256_file

ARTIFACT_REF_SCHEMA_VERSION = "worldfoundry-artifact-ref"
REMOTE_URI_SCHEMES = frozenset({"http", "https", "s3", "gs", "hf", "memory"})


def _mime_type_for_path(path: str | Path) -> str | None:
    return mimetypes.guess_type(str(path))[0]


def local_path_for_uri(uri: str | Path, base_dir: str | Path | None = None) -> Path | None:
    """Resolve a local artifact URI without treating remote/object-store URIs as files."""

    text = str(uri)
    parsed = urlparse(text)
    if parsed.scheme in REMOTE_URI_SCHEMES:
        return None
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


@dataclass(frozen=True, init=False)
class ArtifactRef(JsonContract):
    """Reference to an evaluation input, output, or auxiliary artifact."""

    uri: str
    kind: str
    sha256: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    media_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACT_REF_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        uri: str,
        kind: str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
        mime_type: str | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = ARTIFACT_REF_SCHEMA_VERSION,
    ) -> None:
        if not kind:
            raise ValueError("ArtifactRef requires kind.")
        if schema_version != ARTIFACT_REF_SCHEMA_VERSION:
            raise ValueError(f"Unsupported ArtifactRef schema_version: {schema_version}")
        object.__setattr__(self, "uri", str(uri))
        object.__setattr__(self, "kind", str(kind))
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "mime_type", mime_type)
        object.__setattr__(self, "media_metadata", copy_mapping(media_metadata))
        object.__setattr__(self, "metadata", copy_mapping(metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        kind: str,
        *,
        uri: str | None = None,
        mime_type: str | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        compute_hash: bool = True,
    ) -> "ArtifactRef":
        """Build a reference to a local file, hashing it unless ``compute_hash=False``.

        Set ``compute_hash=False`` to skip the full-file SHA-256 read for large
        artifacts (e.g. multi-GB videos); the cheap ``size_bytes`` stat is
        always recorded.
        """
        file_path = Path(path)
        return cls(
            uri=str(file_path) if uri is None else uri,
            kind=kind,
            sha256=sha256_file(file_path) if compute_hash else None,
            size_bytes=file_path.stat().st_size,
            mime_type=mime_type,
            media_metadata=media_metadata,
            metadata=metadata,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        uri: str,
        kind: str,
        *,
        mime_type: str | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRef":
        return cls(
            uri=uri,
            kind=kind,
            sha256=sha256_bytes(data),
            size_bytes=len(data),
            mime_type=mime_type,
            media_metadata=media_metadata,
            metadata=metadata,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str | Path,
        kind: str,
        *,
        base_dir: str | Path | None = None,
        mime_type: str | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        compute_hash: bool = True,
    ) -> "ArtifactRef":
        if not kind:
            raise ValueError("ArtifactRef.from_uri requires kind.")
        text_uri = str(uri)
        local_path = local_path_for_uri(text_uri, base_dir)
        if local_path is not None and local_path.is_file():
            return cls.from_path(
                local_path,
                kind=kind,
                uri=text_uri,
                mime_type=mime_type or _mime_type_for_path(text_uri),
                media_metadata=media_metadata,
                metadata=metadata,
                compute_hash=compute_hash,
            )
        return cls(
            uri=text_uri,
            kind=kind,
            mime_type=mime_type or _mime_type_for_path(text_uri),
            media_metadata=media_metadata,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        kind = data.get("kind")
        if not kind:
            raise ValueError("ArtifactRef requires kind.")
        return cls(
            uri=str(data["uri"]),
            kind=str(kind),
            sha256=data.get("sha256"),
            size_bytes=data.get("size_bytes"),
            mime_type=data.get("mime_type"),
            media_metadata=data.get("media_metadata"),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", ARTIFACT_REF_SCHEMA_VERSION),
        )


def restore_artifact_refs(value: Any) -> Any:
    """Restore nested ArtifactRef payloads inside JSON-compatible values."""

    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, Mapping):
        if value.get("schema_version") == ARTIFACT_REF_SCHEMA_VERSION:
            return ArtifactRef.from_dict(value)
        return {str(key): restore_artifact_refs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(restore_artifact_refs(item) for item in value)
    if isinstance(value, list):
        return [restore_artifact_refs(item) for item in value]
    return value


def coerce_artifact_refs(value: Mapping[str, Any] | None) -> dict[str, ArtifactRef]:
    """Coerce an artifact mapping into ArtifactRef objects keyed by string name.

    Raises:
        TypeError: If a value is neither an :class:`ArtifactRef` nor a mapping
            (e.g. a bare path string), with the artifact name and expected
            shape in the message.
    """

    artifacts: dict[str, ArtifactRef] = {}
    for name, artifact in (value or {}).items():
        key = str(name)
        if isinstance(artifact, ArtifactRef):
            artifacts[key] = artifact
        elif isinstance(artifact, Mapping):
            artifacts[key] = ArtifactRef.from_dict(artifact)
        else:
            raise TypeError(
                f"artifact {key!r} must be an ArtifactRef or a mapping with 'uri' and 'kind', "
                f"got {type(artifact).__name__}; wrap bare paths with ArtifactRef.from_uri(path, kind=...)"
            )
    return artifacts


def enrich_artifact_ref(
    artifact: ArtifactRef,
    base_dir: str | Path | None = None,
    *,
    compute_hash: bool = True,
) -> ArtifactRef:
    """Return an ArtifactRef with file hash/size when its URI points to a local file.

    Set ``compute_hash=False`` to only fill the cheap ``size_bytes`` stat and
    skip the full-file SHA-256 read (useful for very large artifacts).
    """

    if artifact.sha256 and artifact.size_bytes is not None:
        return artifact
    local_path = local_path_for_uri(artifact.uri, base_dir)
    if local_path is None or not local_path.is_file():
        return artifact
    if not compute_hash:
        # Fill only the cheap stat; keep any hash the reference already carries.
        if artifact.size_bytes is not None:
            return artifact
        return ArtifactRef(
            uri=artifact.uri,
            kind=artifact.kind,
            sha256=artifact.sha256,
            size_bytes=local_path.stat().st_size,
            mime_type=artifact.mime_type or _mime_type_for_path(artifact.uri),
            media_metadata=artifact.media_metadata,
            metadata=artifact.metadata,
        )
    return ArtifactRef.from_path(
        local_path,
        kind=artifact.kind,
        uri=artifact.uri,
        mime_type=artifact.mime_type or _mime_type_for_path(artifact.uri),
        media_metadata=artifact.media_metadata,
        metadata=artifact.metadata,
    )
