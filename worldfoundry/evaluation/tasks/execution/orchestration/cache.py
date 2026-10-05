"""SQLite-backed local caching system for deterministic model generation.

This module provides a robust caching layer for Model Runner inference requests.
By indexing tasks on unique composite SHA-256 keys containing request inputs,
generation hyperparameters, and codebase/model version hashes, it prevents
redundant or expensive generation workloads from executing twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, is_generation_result_successful
from worldfoundry.evaluation.api.artifacts import ArtifactRef, local_path_for_uri
from worldfoundry.evaluation.api.json_contract import (
    canonical_json_bytes,
    canonical_json_dumps,
    json_sha256,
    normalize_json,
    sha256_hex,
)

# Standard constants for SHA256 length and file buffers
SHA256_HEX_LENGTH = 64
DEFAULT_FILE_CHUNK_SIZE = 1024 * 1024
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_IN_CHUNK_SIZE = 400
_SQLITE_BUSY_TIMEOUT_MS = 30_000

# Schema tracking versions and valid operational strings
GENERATION_RESULT_CACHE_SCHEMA_VERSION = "worldfoundry-generation-result-cache"
GENERATION_CACHE_MODES = frozenset({"off", "read", "write", "read-write", "refresh"})


def file_sha256(path: str | Path, *, chunk_size: int = DEFAULT_FILE_CHUNK_SIZE) -> str:
    """Computes the SHA-256 checksum of a file on disk by reading it in binary chunks.

    Args:
        path: Filepath of the target asset.
        chunk_size: Size of the buffer in bytes used during reading.

    Returns:
        The computed lowercase hexadecimal SHA-256 digest of the file.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256_field(name: str, value: str) -> str:
    """Validates that a string is a standard 64-character lowercase hexadecimal SHA-256 hash."""
    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character sha256 hex digest.")
    return value


@dataclass(frozen=True)
class CacheKey:
    """A globally unique composite key referencing a specific evaluation generation result.

    It combines task identity (`sample_id`), generation instruction state (`payload_hash`),
    and model/codebase versioning (`version_context_hash`) to ensure zero-collision cache lookups.

    Attributes:
        stage: The phase of execution (e.g., "generation").
        sample_id: Unique identifier for the specific evaluation sample.
        payload_hash: SHA-256 checksum representing the inputs and hyperparameter options.
        version_context_hash: SHA-256 checksum of the codebase, model, and metric configurations.
    """
    stage: str
    sample_id: str
    payload_hash: str
    version_context_hash: str

    def __post_init__(self) -> None:
        """Validates that all composite hashes strictly conform to the expected length and regex bounds."""
        for name in ("stage", "sample_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        _validate_sha256_field("payload_hash", self.payload_hash)
        _validate_sha256_field("version_context_hash", self.version_context_hash)

    @classmethod
    def from_payload(
        cls,
        *,
        stage: str,
        sample_id: str,
        payload: Any,
        version_context: Any | None = None,
    ) -> "CacheKey":
        """Generates a deterministic hash key from canonical JSON representations of inputs and context.

        Args:
            stage: Phase of evaluation.
            sample_id: Unique sample identifier.
            payload: JSON-serializable generation parameters.
            version_context: JSON-serializable codebase/model version details.

        Returns:
            A validated, frozen CacheKey instance.
        """
        return cls(
            stage=stage,
            sample_id=sample_id,
            payload_hash=json_sha256(payload),
            version_context_hash=json_sha256({} if version_context is None else version_context),
        )

    def to_dict(self) -> dict[str, str]:
        """Serializes the key's composite hash values into a standard dictionary."""
        return {
            "stage": self.stage,
            "sample_id": self.sample_id,
            "payload_hash": self.payload_hash,
            "version_context_hash": self.version_context_hash,
        }

    @cached_property
    def key_hash(self) -> str:
        """The final flattened SHA-256 string serving as the absolute SQLite primary key."""
        return json_sha256(self.to_dict())


def make_cache_key(
    *,
    stage: str,
    sample_id: str,
    payload: Any,
    version_context: Any | None = None,
) -> CacheKey:
    """Wrapper to generate an immutable evaluation CacheKey."""
    return CacheKey.from_payload(
        stage=stage,
        sample_id=sample_id,
        payload=payload,
        version_context=version_context,
    )


def _utcnow_iso() -> str:
    """Returns a zero-padded, timezone-aware ISO8601 timestamp string for record keeping."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_loads_mapping(text: str) -> dict[str, Any]:
    """Safely decodes and asserts mapping-structure of cached database blobs."""
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError("cached JSON payload must be an object")
    return dict(payload)


def normalize_generation_cache_mode(mode: str | None) -> str:
    """Standardizes CLI abbreviations into safe cache modes ('off', 'read', 'write', 'read-write', 'refresh').

    Args:
        mode: The raw cache mode string, typically passed from command line args.

    Returns:
        One of the canonical cash mode keys defined in GENERATION_CACHE_MODES.
    """
    normalized = str(mode or "off").strip().lower().replace("_", "-")
    aliases = {
        "rw": "read-write",
        "readwrite": "read-write",
        "reuse": "read-write",
        "on": "read-write",
        "true": "read-write",
        "disabled": "off",
        "false": "off",
        "none": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in GENERATION_CACHE_MODES:
        raise ValueError(f"unsupported generation cache mode: {mode!r}")
    return normalized


def _cache_mode_reads(mode: str) -> bool:
    """Indicates if the specified cache mode allows reading cached values."""
    return mode in {"read", "read-write"}


def _cache_mode_writes(mode: str) -> bool:
    """Indicates if the specified cache mode allows writing new values."""
    return mode in {"write", "read-write", "refresh"}


def generation_cache_payload(request: GenerationRequest) -> dict[str, Any]:
    """Returns the specific request fields that logically affect generation outcome.

    Filters out superficial or volatile request-level metadata (like request_id or run_id)
    to ensure matching inputs correctly trigger a cache hit regardless of the orchestration context.

    Args:
        request: Normalized generation request.
    """
    return {
        "schema_version": request.schema_version,
        "sample_id": request.sample_id,
        "task_name": request.task_name,
        "split": request.split,
        "inputs": request.inputs,
        "controls": request.controls,
        "generation_kwargs": request.generation_kwargs,
        "output_schema": request.output_schema,
    }


def make_generation_cache_key(
    request: GenerationRequest,
    version_context: Any | None = None,
    *,
    version_context_hash: str | None = None,
) -> CacheKey:
    """Constructs a deterministic cache lookup key for a model generation request."""
    if version_context_hash is not None:
        return CacheKey(
            stage="generation",
            sample_id=request.sample_id,
            payload_hash=json_sha256(generation_cache_payload(request)),
            version_context_hash=_validate_sha256_field("version_context_hash", version_context_hash),
        )
    return make_cache_key(
        stage="generation",
        sample_id=request.sample_id,
        payload=generation_cache_payload(request),
        version_context=version_context,
    )


def _truthy(value: Any) -> bool:
    """Translates common representations of boolean logic (strings, integers) to actual booleans."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_gt_zero(value: Any) -> bool:
    """Helper verifying if a hyperparameter float exists and is strictly greater than zero."""
    if value in (None, ""):
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return True


def _int_gt_one(value: Any) -> bool:
    """Helper verifying if a hyperparameter integer exists and is strictly greater than one."""
    if value in (None, ""):
        return False
    try:
        return int(value) > 1
    except (TypeError, ValueError):
        return True


def generation_request_cacheable(request: GenerationRequest) -> tuple[bool, str | None]:
    """Evaluates whether a model generation request is mathematically/logically safe to retrieve from cache.

    Analyzes sampling policies and inference hyperparameters to prevent non-deterministic tasks
    (e.g., T>0 generation, beam search, explicit cache-bypassing) from mistakenly reading stale
    or incorrect cache hits.

    Args:
        request: A normalized `GenerationRequest` holding the target payload and kwargs.

    Returns:
        tuple[bool, str | None]: `(is_cacheable, optional_reason)`.
    """
    policy = dict(request.cache_policy or {})
    policy_mode = policy.get("mode", policy.get("cache"))
    
    # Honor explicit cache directives
    if policy_mode is not None and (
        policy_mode is False or str(policy_mode).strip().lower() in {"off", "false", "disabled", "none"}
    ):
        return False, "cache_policy disabled cache"
    if _truthy(policy.get("force")) or _truthy(policy.get("deterministic")):
        return True, None

    kwargs = dict(request.generation_kwargs or {})
    
    # Non-deterministic configurations cannot be reliably cached unless forced
    if _float_gt_zero(kwargs.get("temperature")):
        return False, "temperature > 0"
    if _truthy(kwargs.get("do_sample")):
        return False, "do_sample is true"
    for key in ("n", "best_of", "num_return_sequences"):
        if _int_gt_one(kwargs.get(key)):
            return False, f"{key} > 1"
    
    return True, None


@dataclass(frozen=True)
class GenerationCacheRecord:
    """Represents a fully materialized cached evaluation task hit.

    Attributes:
        key: The CacheKey that mapped to this record.
        result: The cached GenerationResult payload.
        metadata: Arbitrary run metadata when the cache was compiled.
        created_at: ISO8601 creation timestamp.
        updated_at: ISO8601 update timestamp.
        hits: Number of times this cache entry has been retrieved.
    """
    key: CacheKey
    result: GenerationResult
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str
    hits: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes the cache record to a plain dictionary."""
        return {
            "key": self.key.to_dict(),
            "result": self.result.to_dict(),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hits": self.hits,
        }


@dataclass
class GenerationCacheStats:
    """Tracks SQLite database telemetry, access patterns, and failure rates during a benchmark run."""
    enabled: bool
    mode: str
    namespace: str
    cache_path: str | None = None
    hits: int = 0
    misses: int = 0
    writes: int = 0
    skipped: int = 0
    stale: int = 0
    errors: int = 0
    skipped_reasons: dict[str, int] | None = None
    error_messages: list[str] | None = None

    @property
    def reads_enabled(self) -> bool:
        """Indicates if reading from cache is active under current stats configuration."""
        return self.enabled and _cache_mode_reads(self.mode)

    @property
    def writes_enabled(self) -> bool:
        """Indicates if writing to cache is active under current stats configuration."""
        return self.enabled and _cache_mode_writes(self.mode)

    def skip(self, reason: str) -> None:
        """Records a skipped caching event with a specific rationale (e.g. temperature > 0)."""
        self.skipped += 1
        if self.skipped_reasons is None:
            self.skipped_reasons = {}
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def error(self, message: str) -> None:
        """Logs an unexpected cache processing failure message."""
        self.errors += 1
        if self.error_messages is None:
            self.error_messages = []
        self.error_messages.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the statistics telemetry into a standardized JSON structure."""
        return {
            "schema_version": GENERATION_RESULT_CACHE_SCHEMA_VERSION,
            "enabled": self.enabled,
            "mode": self.mode,
            "namespace": self.namespace,
            "cache_path": self.cache_path,
            "reads_enabled": self.reads_enabled,
            "writes_enabled": self.writes_enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "skipped": self.skipped,
            "stale": self.stale,
            "errors": self.errors,
            "skipped_reasons": dict(self.skipped_reasons or {}),
            "error_messages": list(self.error_messages or ()),
        }


def cache_paths_from_stats(stats: Mapping[str, Any] | GenerationCacheStats) -> dict[str, str]:
    """Resolves and returns the absolute path mapping of the SQLite database file from stats."""
    payload = stats.to_dict() if isinstance(stats, GenerationCacheStats) else stats
    cache_path = payload.get("cache_path")
    return {"generation_result_cache": str(cache_path)} if cache_path else {}


def generation_cache_hit_metadata(result: GenerationResult) -> Mapping[str, Any]:
    """Helper method to extract underlying cache-hit metadata from a GenerationResult's metadata block."""
    cache = result.metadata.get("cache") if isinstance(result.metadata, Mapping) else None
    return cache if isinstance(cache, Mapping) and cache.get("hit") is True else {}


class GenerationResultCache:
    """SQLite-backed generation-result cache for deterministic model outputs.

    Exposes atomic, threat-safe storage methods allowing distributed or asynchronous runners
    to record, scan, and retrieve past generation stages and metadata under namespace isolation.
    """

    def __init__(self, cache_dir: str | Path, *, namespace: str = "default") -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.namespace = str(namespace or "default")
        self.cache_path = self.cache_dir / "cache.db"
        self.audit_path = self.cache_dir / "audit.jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Instantiates a synchronous connection with safe transactional timeout defaults."""
        connection = sqlite3.connect(
            str(self.cache_path),
            timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_db(self) -> None:
        """Ensures the core caching tables and indices are initialized on disk."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_results (
                    namespace TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    version_context_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(namespace, key_hash)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS generation_results_lookup
                ON generation_results(namespace, sample_id, stage, version_context_hash)
                """
            )
            connection.commit()

    def _audit(self, event: str, *, key: CacheKey | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        """Appends a deterministic trace audit line of a cache event to the filesystem."""
        self._audit_many(((event, key, metadata),))

    def _audit_many(
        self,
        events: Sequence[tuple[str, CacheKey | None, Mapping[str, Any] | None]],
    ) -> None:
        """Append multiple audit records with one metadata-file open."""
        if not events:
            return
        with self.audit_path.open("a", encoding="utf-8") as handle:
            for event, key, metadata in events:
                row = {
                    "schema_version": GENERATION_RESULT_CACHE_SCHEMA_VERSION,
                    "event": event,
                    "namespace": self.namespace,
                    "time": _utcnow_iso(),
                    **({"key": key.to_dict(), "key_hash": key.key_hash} if key is not None else {}),
                    **({"metadata": dict(metadata)} if metadata else {}),
                }
                handle.write(json.dumps(normalize_json(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    def get(self, key: CacheKey) -> GenerationCacheRecord | None:
        """Searches the SQLite file for a matching key, logging audit hits and incrementing hit counters.

        Args:
            key: The target composite CacheKey.

        Returns:
            The loaded GenerationCacheRecord, or None if a cache miss occurred.
        """
        return self.get_many((key,))[0]

    def get_many(self, keys: Sequence[CacheKey]) -> list[GenerationCacheRecord | None]:
        """Resolve cache keys in bounded SQL batches and commit hit counts once."""
        if not keys:
            return []

        keyed = [(key, key.key_hash) for key in keys]
        unique_hashes = tuple(dict.fromkeys(key_hash for _, key_hash in keyed))
        rows_by_hash: dict[str, sqlite3.Row] = {}
        accessed_at = _utcnow_iso()
        with self._connect() as connection:
            for offset in range(0, len(unique_hashes), _SQLITE_IN_CHUNK_SIZE):
                chunk = unique_hashes[offset : offset + _SQLITE_IN_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT * FROM generation_results
                    WHERE namespace = ? AND key_hash IN ({placeholders})
                    """,
                    (self.namespace, *chunk),
                ).fetchall()
                rows_by_hash.update((str(row["key_hash"]), row) for row in rows)

            hit_increments: dict[str, int] = {}
            decoded: dict[str, tuple[GenerationResult, dict[str, Any]]] = {}
            records: list[GenerationCacheRecord | None] = []
            audit_events: list[tuple[str, CacheKey | None, Mapping[str, Any] | None]] = []
            for key, key_hash in keyed:
                row = rows_by_hash.get(key_hash)
                if row is None:
                    records.append(None)
                    audit_events.append(("miss", key, None))
                    continue

                increment = hit_increments.get(key_hash, 0) + 1
                hit_increments[key_hash] = increment
                if key_hash not in decoded:
                    decoded[key_hash] = (
                        GenerationResult.from_dict(_json_loads_mapping(str(row["result_json"]))),
                        _json_loads_mapping(str(row["metadata_json"])),
                    )
                result, metadata = decoded[key_hash]
                records.append(
                    GenerationCacheRecord(
                        key=key,
                        result=result,
                        metadata=metadata,
                        created_at=str(row["created_at"]),
                        updated_at=accessed_at,
                        hits=int(row["hits"]) + increment,
                    )
                )
                audit_events.append(("hit", key, None))

            try:
                connection.executemany(
                    """
                    UPDATE generation_results
                    SET hits = hits + ?, updated_at = ?
                    WHERE namespace = ? AND key_hash = ?
                    """,
                    (
                        (increment, accessed_at, self.namespace, key_hash)
                        for key_hash, increment in hit_increments.items()
                    ),
                )
                connection.commit()
            except sqlite3.OperationalError as exc:
                connection.rollback()
                if not any(reason in str(exc).lower() for reason in ("locked", "busy")):
                    raise

        self._audit_many(audit_events)
        return records

    def put(self, key: CacheKey, result: GenerationResult, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Upserts a new GenerationResult record into SQLite, indexing it by CacheKey.

        Args:
            key: The unique mapping CacheKey.
            result: The newly compiled GenerationResult payload.
            metadata: Custom key-value diagnostics captured at execution time.
        """
        self.put_many(((key, result, metadata),))

    def put_many(
        self,
        entries: Sequence[tuple[CacheKey, GenerationResult, Mapping[str, Any] | None]],
    ) -> None:
        """Upsert generation results with one SQLite transaction and audit open."""
        if not entries:
            return
        now = _utcnow_iso()
        payloads = [
            {
                "namespace": self.namespace,
                "key_hash": key.key_hash,
                "stage": key.stage,
                "sample_id": key.sample_id,
                "payload_hash": key.payload_hash,
                "version_context_hash": key.version_context_hash,
                "result_json": canonical_json_dumps(result.to_dict()),
                "metadata_json": canonical_json_dumps(metadata or {}),
                "created_at": now,
                "updated_at": now,
                "hits": 0,
            }
            for key, result, metadata in entries
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO generation_results (
                    namespace, key_hash, stage, sample_id, payload_hash, version_context_hash,
                    result_json, metadata_json, created_at, updated_at, hits
                )
                VALUES (
                    :namespace, :key_hash, :stage, :sample_id, :payload_hash, :version_context_hash,
                    :result_json, :metadata_json, :created_at, :updated_at, :hits
                )
                ON CONFLICT(namespace, key_hash) DO UPDATE SET
                    result_json = excluded.result_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                payloads,
            )
            connection.commit()
        self._audit_many(tuple(("write", key, None) for key, _, _ in entries))


def _artifact_with_base(artifact: ArtifactRef, base_dir: Path) -> ArtifactRef:
    """Re-resolves a stored media artifact file path relative to a new directory base."""
    local_path = local_path_for_uri(artifact.uri)
    if local_path is None or local_path.is_absolute():
        return artifact
    resolved = (base_dir / local_path).resolve()
    return ArtifactRef(
        uri=str(resolved),
        kind=artifact.kind,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        mime_type=artifact.mime_type,
        media_metadata=artifact.media_metadata,
        metadata=artifact.metadata,
    )


def _restore_cached_result(record: GenerationCacheRecord) -> tuple[GenerationResult | None, str | None]:
    """Validates that a hit cache record has all its physical media assets fully present on disk.

    Prevents restoring incomplete records if local directories were manually wiped or cleaned up.
    """
    base_text = record.metadata.get("artifact_base_dir")
    base_dir = Path(str(base_text)).expanduser().resolve() if base_text not in (None, "") else None
    artifacts: dict[str, ArtifactRef] = {}
    for name, artifact in record.result.artifacts.items():
        restored = _artifact_with_base(artifact, base_dir) if base_dir is not None else artifact
        local_path = local_path_for_uri(restored.uri)
        if local_path is not None and not local_path.is_file():
            return None, f"cached artifact is missing: {local_path}"
        artifacts[str(name)] = restored

    result_metadata = dict(record.result.metadata)
    result_metadata["cache"] = {
        "hit": True,
        "key_hash": record.key.key_hash,
        "namespace": record.metadata.get("namespace"),
        "cache_path": record.metadata.get("cache_path"),
        "source_run_id": record.metadata.get("run_id"),
        "source_artifact_base_dir": str(base_dir) if base_dir is not None else None,
    }
    return (
        GenerationResult(
            sample_id=record.result.sample_id,
            request_id=record.result.request_id,
            model_id=record.result.model_id,
            artifacts=artifacts,
            status=record.result.status,
            error=record.result.error,
            timings=record.result.timings,
            metadata=result_metadata,
        ),
        None,
    )


def run_generation_with_cache(
    requests: Sequence[GenerationRequest],
    generate: Callable[[Sequence[GenerationRequest]], Sequence[GenerationResult]],
    *,
    cache_dir: str | Path | None = None,
    cache_mode: str | None = "off",
    namespace: str = "default",
    version_context: Any | None = None,
    artifact_base_dir: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[list[GenerationResult], GenerationCacheStats]:
    """Wraps a generation/runner execution block with a smart caching layer.

    Instead of sending all physical requests to the GPU cluster or simulator, this function
    intercepts the batch, searches the SQLite cache for identical previously successful task
    outcomes, and filters out hits. It then passes the remaining 'miss' workload down to the
    runner callable, merging and returning the unified results.

    Args:
        requests: The batch of generation requests (task queries).
        generate: The downstream callable (e.g. `runner.generate`) doing the actual physical work.
        cache_dir: Storage path for the SQLite database.
        cache_mode: Operational mode ('off', 'read', 'write', 'read-write', 'refresh').
        namespace: Cache isolation key.
        version_context: Hashable dictionary describing model and codebase versions (prevents stale code from hitting old cache).
        artifact_base_dir: Physical disk location to resolve/validate attached media artifacts.
        run_id: Current tracing identifier attached to newly cached records.

    Returns:
        tuple containing:
        - A unified list of GenerationResults (mix of cached hits and newly computed outcomes).
        - A GenerationCacheStats object with hit/miss telemetry.
    """
    mode = normalize_generation_cache_mode(cache_mode)
    cache_enabled = cache_dir is not None and mode != "off"
    stats = GenerationCacheStats(
        enabled=cache_enabled,
        mode=mode,
        namespace=str(namespace or "default"),
        cache_path=None,
    )
    if not requests:
        return [], stats

    def generate_pending(rows: Sequence[GenerationRequest]) -> list[GenerationResult]:
        from worldfoundry.core import worldfoundry_inference_context

        with worldfoundry_inference_context():
            return list(generate(rows))

    if not cache_enabled:
        return generate_pending(requests), stats

    cache = GenerationResultCache(cache_dir, namespace=stats.namespace)
    stats.cache_path = str(cache.cache_path)
    cached: dict[str, GenerationResult] = {}
    pending: list[GenerationRequest] = []
    pending_keys: dict[str, CacheKey] = {}
    base_dir = Path(artifact_base_dir).expanduser().resolve() if artifact_base_dir is not None else None

    # Step 1: Classify requests and resolve cacheable keys in one batched read.
    classified: list[tuple[GenerationRequest, CacheKey | None]] = []
    read_positions: list[int] = []
    read_keys: list[CacheKey] = []
    version_context_hash = json_sha256({} if version_context is None else version_context)
    for request in requests:
        cacheable, reason = generation_request_cacheable(request)
        if not cacheable:
            stats.skip(reason or "not cacheable")
            classified.append((request, None))
            continue
        key = make_generation_cache_key(request, version_context_hash=version_context_hash)
        if stats.reads_enabled:
            read_positions.append(len(classified))
            read_keys.append(key)
        classified.append((request, key))

    records_by_position: dict[int, GenerationCacheRecord | None] = {}
    if read_keys:
        try:
            records = cache.get_many(read_keys)
        except Exception as exc:  # noqa: BLE001 - fall back per key after a batch-cache failure.
            stats.error(f"batch cache read failed: {type(exc).__name__}: {exc}")
            records = []
            for position, key in zip(read_positions, read_keys):
                request = classified[position][0]
                try:
                    record = cache.get(key)
                except Exception as item_exc:  # noqa: BLE001 - cache must not fail generation.
                    stats.error(
                        f"cache read failed for {request.sample_id}: "
                        f"{type(item_exc).__name__}: {item_exc}"
                    )
                    record = None
                records_by_position[position] = record
        else:
            records_by_position.update(zip(read_positions, records))

    for position, (request, key) in enumerate(classified):
        if key is None:
            pending.append(request)
            continue
        if stats.reads_enabled:
            record = records_by_position.get(position)
            if record is not None:
                # Validate that all required physical artifact files (e.g. MP4 videos) still exist on disk.
                restored, stale_reason = _restore_cached_result(record)
                if restored is not None:
                    cached[request.sample_id] = restored
                    stats.hits += 1
                    continue
                stats.stale += 1
                cache._audit("stale", key=key, metadata={"reason": stale_reason or "stale"})
            stats.misses += 1
        pending.append(request)
        pending_keys[request.sample_id] = key

    # Step 2: Execute physically required workload.
    generated = generate_pending(pending) if pending else []
    generated_by_sample_id = {result.sample_id: result for result in generated}

    # Step 3: Opportunistically write successful results in one transaction.
    if stats.writes_enabled:
        write_entries: list[tuple[CacheKey, GenerationResult, Mapping[str, Any] | None]] = []
        for request in pending:
            result = generated_by_sample_id.get(request.sample_id)
            key = pending_keys.get(request.sample_id)
            if result is None or key is None:
                continue
            if not is_generation_result_successful(result):
                stats.skip("generation failed")
                continue
            write_entries.append(
                (
                    key,
                    result,
                    {
                        "schema_version": GENERATION_RESULT_CACHE_SCHEMA_VERSION,
                        "namespace": stats.namespace,
                        "cache_path": str(cache.cache_path),
                        "run_id": run_id,
                        "artifact_base_dir": str(base_dir) if base_dir is not None else None,
                        "version_context_hash": key.version_context_hash,
                    },
                )
            )
        if write_entries:
            try:
                cache.put_many(write_entries)
            except Exception as exc:  # noqa: BLE001 - cache writes are opportunistic.
                stats.error(f"batch cache write failed: {type(exc).__name__}: {exc}")
                for key, result, metadata in write_entries:
                    try:
                        cache.put(key, result, metadata=metadata)
                    except Exception as item_exc:  # noqa: BLE001 - preserve other cache writes.
                        stats.error(
                            f"cache write failed for {key.sample_id}: "
                            f"{type(item_exc).__name__}: {item_exc}"
                        )
                    else:
                        stats.writes += 1
            else:
                stats.writes += len(write_entries)

    # Step 4: Re-align merged results with original request ordering
    results: list[GenerationResult] = []
    for request in requests:
        if request.sample_id in cached:
            results.append(cached[request.sample_id])
        elif request.sample_id in generated_by_sample_id:
            results.append(generated_by_sample_id[request.sample_id])
        else:
            results.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    status="failed",
                    error="generation cache runner did not return a result for sample",
                )
            )
    return results, stats


__all__ = [
    "CacheKey",
    "DEFAULT_FILE_CHUNK_SIZE",
    "GENERATION_CACHE_MODES",
    "GENERATION_RESULT_CACHE_SCHEMA_VERSION",
    "GenerationCacheRecord",
    "GenerationCacheStats",
    "GenerationResultCache",
    "SHA256_HEX_LENGTH",
    "cache_paths_from_stats",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "file_sha256",
    "generation_cache_hit_metadata",
    "generation_cache_payload",
    "generation_request_cacheable",
    "json_sha256",
    "make_cache_key",
    "make_generation_cache_key",
    "normalize_json",
    "normalize_generation_cache_mode",
    "run_generation_with_cache",
    "sha256_hex",
]
