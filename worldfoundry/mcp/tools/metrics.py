"""Metric registry discovery payloads for the WorldFoundry MCP server."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics.registry import (
    default_metric_registry,
    list_metric_registry_entries,
)

from .query import matches_query


def list_metrics_payload(
    *,
    query: str | None = None,
    family: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """List registered evaluation metrics, optionally filtered."""

    rows: list[dict[str, Any]] = []
    for entry in list_metric_registry_entries():
        if family and entry.family.casefold() != family.casefold():
            continue
        if tag and tag.casefold() not in {item.casefold() for item in entry.tags}:
            continue
        payload = entry.to_dict()
        if query and not matches_query(
            query,
            entry.id,
            *entry.aliases,
            entry.description,
            entry.family,
            *entry.tags,
        ):
            continue
        rows.append(payload)
    return {"metrics": rows, "total": len(rows), "query": query, "family": family, "tag": tag}


def show_metric_payload(metric_id: str) -> dict[str, Any]:
    """Return full metadata for one metric id or alias."""

    registry = default_metric_registry()
    entry = registry.resolve_key(metric_id)
    payload = entry.to_dict()
    canonical = registry.canonical_metric_id(metric_id)
    if canonical != entry.id:
        payload["requested_metric_id"] = metric_id
        payload["canonical_metric_id"] = canonical
    return payload


__all__ = ["list_metrics_payload", "show_metric_payload"]
