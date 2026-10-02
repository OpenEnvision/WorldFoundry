"""Media path classification helpers with no model semantics.

Extension groups and MIME guesses so loaders pick a decoder without
opening the file. Not a codec — see ``video.py`` / ``audio.py``.
"""

from __future__ import annotations

import mimetypes
from enum import Enum
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

# ──────────────────────────────────────────────────────────────────────────
# Suffix tables — classify without opening the file; not a codec
# ──────────────────────────────────────────────────────────────────────────


class MediaKind(str, Enum):
    """Coarse file kind inferred from URI or path suffix."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    GEOMETRY = "geometry"
    ARCHIVE = "archive"
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
AUDIO_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
GEOMETRY_EXTENSIONS = frozenset({".glb", ".gltf", ".obj", ".pcd", ".ply", ".splat", ".spz", ".stl", ".xyz"})
ARCHIVE_EXTENSIONS = frozenset({".7z", ".bz2", ".gz", ".tar", ".tgz", ".xz", ".zip", ".zst"})
JSON_EXTENSIONS = frozenset({".json", ".jsonl", ".geojson"})
TEXT_EXTENSIONS = frozenset({".csv", ".md", ".txt", ".yaml", ".yml"})
BINARY_EXTENSIONS = frozenset({".bin", ".ckpt", ".npy", ".npz", ".pkl", ".pt", ".pth", ".safetensors"})

MEDIA_EXTENSION_GROUPS: Mapping[MediaKind, frozenset[str]] = {
    MediaKind.IMAGE: IMAGE_EXTENSIONS,
    MediaKind.VIDEO: VIDEO_EXTENSIONS,
    MediaKind.AUDIO: AUDIO_EXTENSIONS,
    MediaKind.GEOMETRY: GEOMETRY_EXTENSIONS,
    MediaKind.ARCHIVE: ARCHIVE_EXTENSIONS,
    MediaKind.JSON: JSON_EXTENSIONS,
    MediaKind.TEXT: TEXT_EXTENSIONS,
    MediaKind.BINARY: BINARY_EXTENSIONS,
}
MIME_TYPES_BY_SUFFIX: Mapping[str, str] = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".jsonl": "application/x-ndjson",
    ".m4v": "video/x-m4v",
    ".ply": "model/ply",
    ".safetensors": "application/octet-stream",
    ".webp": "image/webp",
}


def suffix_for_uri(value: str | Path) -> str:
    """Return the lower-case file suffix for local paths and URL-like strings."""

    text = str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    path_text = unquote(parsed.path) if parsed.scheme else text
    name = Path(path_text).name
    suffixes = Path(name).suffixes
    if not suffixes:
        return ""
    if len(suffixes) >= 2 and suffixes[-2].casefold() == ".tar":
        return ".tar" + suffixes[-1].casefold()
    return suffixes[-1].casefold()


def infer_media_kind(value: str | Path) -> MediaKind:
    """Infer a coarse media kind from a URI or path suffix."""

    suffix = suffix_for_uri(value)
    if suffix in {".tar.gz", ".tar.xz", ".tar.bz2", ".tar.zst"}:
        return MediaKind.ARCHIVE
    for kind, extensions in MEDIA_EXTENSION_GROUPS.items():
        if suffix in extensions:
            return kind
    mime_type = guess_mime_type(value)
    if mime_type:
        prefix, _, subtype = mime_type.partition("/")
        if prefix in {MediaKind.IMAGE.value, MediaKind.VIDEO.value, MediaKind.AUDIO.value}:
            return MediaKind(prefix)
        if subtype in {"json", "x-ndjson"} or subtype.endswith("+json"):
            return MediaKind.JSON
        if prefix == "text":
            return MediaKind.TEXT
    return MediaKind.UNKNOWN


def guess_mime_type(value: str | Path) -> str | None:
    """Return the stdlib MIME type guess for a URI or path."""

    return mimetypes.guess_type(str(value))[0] or MIME_TYPES_BY_SUFFIX.get(suffix_for_uri(value))


def is_media_path(value: str | Path, *kinds: MediaKind | str) -> bool:
    """Return whether a path/URI is one of the requested media kinds.

    When no kind is provided, any known kind except ``UNKNOWN`` matches.
    """

    inferred = infer_media_kind(value)
    if not kinds:
        return inferred is not MediaKind.UNKNOWN
    requested = {kind if isinstance(kind, MediaKind) else MediaKind(str(kind)) for kind in kinds}
    return inferred in requested


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "BINARY_EXTENSIONS",
    "GEOMETRY_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "JSON_EXTENSIONS",
    "MEDIA_EXTENSION_GROUPS",
    "MIME_TYPES_BY_SUFFIX",
    "TEXT_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "MediaKind",
    "guess_mime_type",
    "infer_media_kind",
    "is_media_path",
    "suffix_for_uri",
]
