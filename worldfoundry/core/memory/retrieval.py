"""Deterministic memory record scoring and top-k selection.

Scoring is a weighted sum of four signals — preferred type (±1),
metadata key match ratio, bag-of-words overlap against
``type + metadata + content``, and recency ``w / (1 + age)``. Ties
break by original index so the same store + query always returns the
same top-k. This is intentionally *not* embedding retrieval; mosaic
and other spatial memories implement their own rankers.
"""

from __future__ import annotations

from typing import Any, Mapping

from .store import MemoryQuery, MemoryRecord

# ──────────────────────────────────────────────────────────────────────────
# Deterministic ranker — type / metadata / bag-of-words / recency; ties by index
# ──────────────────────────────────────────────────────────────────────────


def score_record(
    record: Mapping[str, Any], query: MemoryQuery, *, newest_timestamp: int | float | None = None
) -> float:
    """Score one memory record against *query* using type, metadata, text, and recency signals."""
    score = 0.0
    if query.prefer_type is not None:
        score += 1.0 if record.get("type") == query.prefer_type else -1.0

    if query.metadata:
        matched = sum(1 for key, value in query.metadata.items() if (record.get("metadata") or {}).get(key) == value)
        score += query.metadata_weight * (matched / max(len(query.metadata), 1))

    if query.text:
        haystack = f"{record.get('type', '')} {record.get('metadata', '')} {record.get('content', '')}".lower()
        terms = [term for term in query.text.lower().split() if term]
        if terms:
            score += sum(1 for term in terms if term in haystack) / len(terms)

    timestamp = record.get("timestamp", 0)
    if newest_timestamp is not None:
        try:
            distance = max(float(newest_timestamp) - float(timestamp), 0.0)
        except (TypeError, ValueError):
            distance = 0.0
        score += query.recency_weight / (1.0 + distance)
    return score


def select_records(records: list[Mapping[str, Any]], query: MemoryQuery) -> list[MemoryRecord]:
    """Return the top-*query.top_k* records ranked by :func:`score_record`."""
    if not records or query.top_k <= 0:
        return []

    newest_timestamp = _newest_timestamp(records)
    ranked = []
    for index, record in enumerate(records):
        scored = MemoryRecord.from_dict(record)
        scored.score = score_record(record, query, newest_timestamp=newest_timestamp)
        ranked.append((scored.score, index, scored))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in ranked[: query.top_k]]


def _newest_timestamp(records: list[Mapping[str, Any]]) -> int | float | None:
    """Newest numeric ``timestamp``, or ``None`` when every record lacks one.

    Non-numeric timestamps are ignored so a single bad row cannot disable
    recency for the whole store.
    """
    numeric = []
    for record in records:
        timestamp = record.get("timestamp")
        if isinstance(timestamp, (int, float)):
            numeric.append(timestamp)
    return max(numeric) if numeric else None


__all__ = [
    "score_record",
    "select_records",
]
