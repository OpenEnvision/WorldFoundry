"""Metric registry discovery payloads for the WorldFoundry MCP server."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from worldfoundry.evals.tasks.metrics.registry import (
    default_metric_registry,
    list_metric_registry_entries,
)

from .discovery import DetailParam, Limit, Offset, Query, page, project
from .query import matches_query
from .responses import ToolNotFoundError
from .spec import mcp_tool

METRIC_SUMMARY_FIELDS = (
    "id",
    "display_name",
    "family",
    "higher_is_better",
    "requires_reference",
    "tags",
    "description",
)


@mcp_tool("list_metrics", title="List metrics", read_only=True)
def list_metrics_payload(
    *,
    query: Query = None,
    family: Annotated[str | None, Field(description="Exact metric family.")] = None,
    tag: Annotated[str | None, Field(description="Required metric tag.")] = None,
    detail: DetailParam = "summary",
    offset: Offset = 0,
    limit: Limit = 50,
) -> dict[str, Any]:
    """List registered metric ids; pass them to evaluate ``metrics`` or show_metric."""

    rows: list[dict[str, Any]] = []
    for entry in list_metric_registry_entries():
        if family and entry.family.casefold() != family.casefold():
            continue
        if tag and tag.casefold() not in {item.casefold() for item in entry.tags}:
            continue
        if query and not matches_query(query, entry.id, *entry.aliases, entry.description, entry.family, *entry.tags):
            continue
        rows.append(project(entry.to_dict(), detail, METRIC_SUMMARY_FIELDS))
    return page(rows, key="metrics", offset=offset, limit=limit, query=query, family=family, tag=tag, detail=detail)


@mcp_tool("show_metric", title="Get metric details", read_only=True)
def show_metric_payload(
    metric_id: Annotated[str, Field(description="Metric id or registered alias.")],
) -> dict[str, Any]:
    """Return the full definition of one metric: inputs, direction, aggregation, and implementation."""

    registry = default_metric_registry()
    try:
        entry = registry.resolve_key(metric_id)
    except (KeyError, ValueError) as exc:
        known = [item.id for item in list_metric_registry_entries()]
        raise ToolNotFoundError.for_id("metric", metric_id, known) from exc
    payload = entry.to_dict()
    canonical = registry.canonical_metric_id(metric_id)
    if canonical != entry.id:
        payload["requested_metric_id"] = metric_id
        payload["canonical_metric_id"] = canonical
    return payload


__all__ = ["list_metrics_payload", "show_metric_payload"]
