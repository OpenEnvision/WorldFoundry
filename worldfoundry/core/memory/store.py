"""Bounded in-process memory records, queries, and deterministic retrieval.

Every WorldFoundry memory implementation (artifact, action, mosaic)
stores the same row shape: ``content``, ``type``, ``timestamp``,
``metadata``, optional ``score``. :class:`MemoryStore` is the list plus
capacity eviction; :class:`MemoryQuery` / :meth:`MemoryStore.select`
delegate ranking to :mod:`worldfoundry.core.memory.retrieval` so scoring
stays deterministic and testable without a vector DB.

:class:`MemoryRecord` is the typed view; the store keeps plain dicts so
subclasses can persist or JSON-dump without a custom encoder.
``capacity`` evicts from the *front* (oldest) after each append.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# ──────────────────────────────────────────────────────────────────────────
# Record / query / selection — typed view; store persists plain dicts
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MemoryRecord:
    """One bounded memory entry with normalized provenance fields."""

    content: Any
    kind: str = "other"
    timestamp: int | float = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        """Build a record from a dict; ``kind`` and ``type`` are aliases."""
        kind = value.get("kind", value.get("type", "other"))
        return cls(
            content=value.get("content"),
            kind=str(kind),
            timestamp=value.get("timestamp", 0),
            metadata=dict(value.get("metadata") or {}),
            score=None if value.get("score") is None else float(value["score"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize using the persisted ``type`` key (not ``kind``)."""
        row = {
            "content": self.content,
            "type": self.kind,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.score is not None:
            row["score"] = self.score
        return row


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Selection query shared by artifact, runtime, and action memories."""

    text: str | None = None
    prefer_type: str | None = None
    metadata: Mapping[str, Any] | None = None
    top_k: int = 1
    recency_weight: float = 1.0
    metadata_weight: float = 1.0


@dataclass(slots=True)
class MemorySelection:
    """Top-k retrieval result returned by :meth:`MemoryStore.select`."""

    records: list[MemoryRecord]

    @property
    def content(self) -> Any:
        """First selected record's content, or ``None`` when empty."""
        return self.records[0].content if self.records else None

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize every selected record."""
        return [record.to_dict() for record in self.records]


# ──────────────────────────────────────────────────────────────────────────
# Bounded list — evict from the front (oldest) after each append
# ──────────────────────────────────────────────────────────────────────────


class MemoryStore:
    """Bounded in-process memory store used by all concrete WorldFoundry memories."""

    def __init__(self, capacity: int | None = None, records: Iterable[Mapping[str, Any] | MemoryRecord] = ()) -> None:
        """Seed the store; seed rows go through :meth:`append_record` so capacity applies."""
        self.capacity = capacity
        self.records: list[dict[str, Any]] = []
        for record in records:
            self.append_record(record)

    def __len__(self) -> int:
        """Number of persisted record dicts currently held."""
        return len(self.records)

    def __iter__(self):
        """Iterate persisted dicts in append order (oldest first)."""
        return iter(self.records)

    def append(
        self,
        content: Any,
        *,
        kind: str = "other",
        timestamp: int | float | None = None,
        metadata: Mapping[str, Any] | None = None,
        score: float | None = None,
    ) -> dict[str, Any]:
        """Append a new record; *timestamp* defaults to the current length."""
        record = MemoryRecord(
            content=content,
            kind=str(kind),
            timestamp=len(self.records) if timestamp is None else timestamp,
            metadata=dict(metadata or {}),
            score=score,
        ).to_dict()
        self.records.append(record)
        self.evict()
        return record

    def append_record(self, record: Mapping[str, Any] | MemoryRecord) -> dict[str, Any]:
        """Normalize *record* to a dict, append it, and evict if over capacity."""
        row = record.to_dict() if isinstance(record, MemoryRecord) else MemoryRecord.from_dict(record).to_dict()
        self.records.append(row)
        self.evict()
        return row

    def latest(
        self, prefer_type: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Return the newest record matching optional type and metadata filters."""
        for record in reversed(self.records):
            if prefer_type is not None and record.get("type") != prefer_type:
                continue
            if metadata is not None and not _metadata_matches(record.get("metadata") or {}, metadata):
                continue
            return record
        return None

    def select(self, query: MemoryQuery | None = None, **overrides: Any) -> MemorySelection:
        """Rank stored records and return the top-*query.top_k* matches."""
        from .retrieval import select_records

        if query is None:
            query = MemoryQuery(**overrides)
        elif overrides:
            payload = {
                "text": query.text,
                "prefer_type": query.prefer_type,
                "metadata": query.metadata,
                "top_k": query.top_k,
                "recency_weight": query.recency_weight,
                "metadata_weight": query.metadata_weight,
                **overrides,
            }
            query = MemoryQuery(**payload)
        return MemorySelection(select_records(self.records, query))

    def evict(self) -> None:
        """Drop oldest records until ``len(self) <= capacity`` (no-op if unbounded)."""
        if self.capacity is not None and self.capacity >= 0 and len(self.records) > self.capacity:
            del self.records[: len(self.records) - self.capacity]

    def reset(self) -> None:
        """Remove every stored record."""
        self.records.clear()

    def replace(self, records: Iterable[Mapping[str, Any] | MemoryRecord]) -> None:
        """Clear the store and append *records* in order."""
        self.records = []
        for record in records:
            self.append_record(record)


def _metadata_matches(record_metadata: Mapping[str, Any], query_metadata: Mapping[str, Any]) -> bool:
    """Require every query key to equal the record value; extra record keys are ignored."""
    for key, value in query_metadata.items():
        if record_metadata.get(key) != value:
            return False
    return True


__all__ = [
    "MemoryQuery",
    "MemoryRecord",
    "MemorySelection",
    "MemoryStore",
]
