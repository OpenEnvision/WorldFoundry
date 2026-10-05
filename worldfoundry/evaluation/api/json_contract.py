"""JSON-safe dataclass helpers shared by manifest and evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar

JsonValue = Any
JsonMapping = Mapping[str, JsonValue]
T = TypeVar("T", bound="JsonContract")


def json_dumps(data: JsonValue) -> str:
    """Serialize a JSON-safe value to a string with compact separators."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_json(value: Any) -> JsonValue:
    """Return a strict JSON-compatible value with deterministic object order."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: normalize_json(getattr(value, field.name)) for field in fields(value)}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON cache payloads cannot contain NaN or Infinity.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON cache payload object keys must be strings.")
            normalized[key] = normalize_json(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    raise TypeError(f"Unsupported JSON cache payload type: {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    """Serialize a strict JSON payload with stable ordering and compact UTF-8."""

    return json.dumps(
        normalize_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-safe value to canonical, sorted-key UTF-8 bytes."""
    return canonical_json_dumps(value).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    """Compute the SHA-256 hexadecimal digest of bytes or a string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def json_sha256(value: Any) -> str:
    """Compute the SHA-256 hash of the canonical JSON representation of a value."""
    return sha256_hex(canonical_json_bytes(value))


def stable_hash_data(data: JsonValue) -> str:
    """Compute a stable SHA-256 hash of any JSON-serializable value."""
    payload = json_dumps(to_plain(data)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hexadecimal digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hash of a file on disk by reading in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_plain(value: JsonValue) -> JsonValue:
    """Recursively convert custom types and mappings into plain JSON-serializable types."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_plain(item) for item in sorted(value, key=repr)]
    return value


def copy_mapping(value: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """Safely shallow copy a mapping of strings to JSON-safe values."""
    if value is None:
        return {}
    return {str(key): item for key, item in value.items()}


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Enforce that a value is mapping-like, raising TypeError with context on failure."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def tuple_of_str(value: Any) -> tuple[str, ...]:
    """Convert a sequence of values or a single value into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


class JsonContract:
    """Base class for JSON-serializable dataclasses with stable hashing and schema versioning."""

    def to_dict(self) -> dict[str, JsonValue]:
        return to_plain(self)

    def to_json(self) -> str:
        return json_dumps(self.to_dict())

    def stable_hash(self) -> str:
        return stable_hash_data(
            {
                "contract": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "data": self.to_dict(),
            }
        )

    @classmethod
    def from_json(cls: type[T], payload: str) -> T:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError(f"{cls.__name__}.from_json expected a JSON object.")
        return cls.from_dict(data)  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return int(self.stable_hash()[:16], 16)
