"""Catalog discovery payloads for the WorldFoundry MCP server.

List tools default to a compact ``summary`` view and return one page at a time
(``offset``/``limit``/``next_offset``): the full model catalog alone is roughly
200 KB of JSON. ``get_*_info`` tools return the complete entry.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field

from worldfoundry.cli.catalog import load_catalog

from .context import MCPToolContext, get_default_context
from .query import matches_query
from .responses import ToolNotFoundError
from .spec import mcp_tool

Detail = Literal["summary", "full"]
Query = Annotated[str | None, Field(description="Case-insensitive substring, or a glob such as 'wan*'.")]
DetailParam = Annotated[Detail, Field(description="summary returns key fields; full returns every catalog field.")]
Offset = Annotated[int, Field(ge=0, description="Skip this many matching entries.")]
Limit = Annotated[int, Field(ge=1, le=500, description="Maximum entries to return.")]

MODEL_SUMMARY_FIELDS = (
    "model_id",
    "name",
    "provider",
    "tasks",
    "integration_status",
    "runner_kind",
    "runnable_variants",
    "min_vram_gb",
    "requires_auth",
)
BENCHMARK_SUMMARY_FIELDS = (
    "benchmark_id",
    "name",
    "integration_status",
    "maturity",
    "domains",
    "modalities",
    "metrics",
    "leaderboard_valid",
    "requires_auth",
)
TASK_SUMMARY_FIELDS = (
    "task_type",
    "benchmark_name",
    "benchmark_zoo_id",
    "suite",
    "backend",
    "source_kind",
    "description",
)

# Discovery tools are the hottest calls in an agent session; reloading every
# manifest plus conda probing on each list/get is wasteful for a long-lived
# server (CM-30). Entries expire quickly so on-disk manifest edits are still
# picked up without a restart.
_CATALOG_CACHE_TTL_S = 30.0
_catalog_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_catalog_cache_lock = threading.Lock()


def _load_catalog(ctx: MCPToolContext) -> Any:
    """Load the unified catalog for *ctx* with a short-TTL process cache."""

    key = (str(ctx.model_manifest_dir), str(ctx.benchmark_manifest_dir))
    now = time.monotonic()
    with _catalog_cache_lock:
        cached = _catalog_cache.get(key)
        if cached is not None and now - cached[0] < _CATALOG_CACHE_TTL_S:
            return cached[1]
    catalog = load_catalog(
        model_manifest_dir=ctx.model_manifest_dir,
        benchmark_manifest_dir=ctx.benchmark_manifest_dir,
    )
    with _catalog_cache_lock:
        _catalog_cache[key] = (time.monotonic(), catalog)
    return catalog


def project(row: Mapping[str, Any], detail: str, summary_fields: Sequence[str]) -> dict[str, Any]:
    """Return a full copy of ``row`` or only its ``summary_fields``."""

    if detail == "full":
        return dict(row)
    return {key: row[key] for key in summary_fields if key in row}


def page(rows: Sequence[dict[str, Any]], *, key: str, offset: int, limit: int, **extra: Any) -> dict[str, Any]:
    """Slice ``rows`` and describe the page with ``total`` and ``next_offset``."""

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    end = offset + limit
    return {
        key: list(rows[offset:end]),
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "next_offset": end if end < len(rows) else None,
        **extra,
    }


# ── Models ──────────────────────────────────────────────────────────────


@mcp_tool("list_models", title="List models", read_only=True)
def list_models_payload(
    *,
    query: Query = None,
    runnable_only: Annotated[bool, Field(description="Only models with a runnable in-tree runner.")] = False,
    detail: DetailParam = "summary",
    offset: Offset = 0,
    limit: Limit = 50,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List catalog models matching a query; use the ``model_id`` values with evaluate and get_model_info."""

    catalog = _load_catalog(context or get_default_context())
    rows = [
        project(row.to_dict(), detail, MODEL_SUMMARY_FIELDS)
        for row in catalog.models
        if (not runnable_only or row.runner_kind == "runnable_runner")
        and (not query or matches_query(query, row.model_id, row.name, row.provider, *row.tasks))
    ]
    return page(rows, key="models", offset=offset, limit=limit, query=query, detail=detail)


@mcp_tool("get_model_info", title="Get model details", read_only=True)
def get_model_info_payload(
    model_id: Annotated[str, Field(description="Exact model id from list_models.")],
    *,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return every catalog field for one model: tasks, variants, checkpoints, runtime profile, notes."""

    catalog = _load_catalog(context or get_default_context())
    for row in catalog.models:
        if row.model_id == model_id:
            return row.to_dict()
    raise ToolNotFoundError.for_id("model", model_id, _ids(catalog.models, "model_id"))


# ── Benchmarks ──────────────────────────────────────────────────────────


@mcp_tool("list_benchmarks", title="List benchmarks", read_only=True)
def list_benchmarks_payload(
    *,
    query: Query = None,
    integrated_only: Annotated[bool, Field(description="Only benchmarks with integrated status.")] = False,
    detail: DetailParam = "summary",
    offset: Offset = 0,
    limit: Limit = 50,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List catalog benchmarks matching a query; use the ``benchmark_id`` values with evaluate."""

    catalog = _load_catalog(context or get_default_context())
    rows = [
        project(row.to_dict(), detail, BENCHMARK_SUMMARY_FIELDS)
        for row in catalog.benchmarks
        if (not integrated_only or row.integration_status == "integrated")
        and (not query or matches_query(query, row.benchmark_id, row.name, *row.domains, *row.modalities, *row.tags))
    ]
    return page(rows, key="benchmarks", offset=offset, limit=limit, query=query, detail=detail)


@mcp_tool("get_benchmark_info", title="Get benchmark details", read_only=True)
def get_benchmark_info_payload(
    benchmark_id: Annotated[str, Field(description="Exact benchmark id from list_benchmarks.")],
    *,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return every catalog field for one benchmark: datasets, metrics, maturity, verification evidence."""

    catalog = _load_catalog(context or get_default_context())
    for row in catalog.benchmarks:
        if row.benchmark_id == benchmark_id:
            return row.to_dict()
    raise ToolNotFoundError.for_id("benchmark", benchmark_id, _ids(catalog.benchmarks, "benchmark_id"))


# ── Tasks ───────────────────────────────────────────────────────────────


@mcp_tool("list_tasks", title="List evaluation tasks", read_only=True)
def list_tasks_payload(
    *,
    query: Query = None,
    suite: Annotated[str | None, Field(description="Exact suite id.")] = None,
    backend: Annotated[str | None, Field(description="Exact task backend.")] = None,
    source_kind: Annotated[str | None, Field(description="Exact task source kind.")] = None,
    detail: DetailParam = "summary",
    offset: Offset = 0,
    limit: Limit = 50,
) -> dict[str, Any]:
    """List registered benchmark tasks; a task is identified by ``task_type`` plus ``benchmark_name``."""

    from worldfoundry.evals.tasks.catalog.specs import list_benchmark_zoo_cli_tasks

    rows = [
        project(payload, detail, TASK_SUMMARY_FIELDS)
        for payload in list_benchmark_zoo_cli_tasks(suite=suite, backend=backend, source_kind=source_kind)
        if not query
        or matches_query(
            query,
            payload.get("task_type"),
            payload.get("benchmark_name"),
            payload.get("name"),
            payload.get("benchmark_zoo_id"),
            payload.get("description"),
        )
    ]
    return page(rows, key="tasks", offset=offset, limit=limit, query=query, detail=detail)


@mcp_tool("get_task_info", title="Get task details", read_only=True)
def get_task_info_payload(
    task: Annotated[str, Field(description="task_type, or 'task_type/benchmark_name'.")],
    benchmark: Annotated[str | None, Field(description="benchmark_name when the task_type is shared.")] = None,
) -> dict[str, Any]:
    """Return the full definition of one evaluation task."""

    from worldfoundry.evals.tasks.catalog.specs import get_benchmark_zoo_cli_task, list_benchmark_zoo_cli_tasks

    if benchmark is None and "/" in task:
        task, benchmark = task.split("/", 1)
    if benchmark is None:
        matches = list_benchmark_zoo_cli_tasks(task_type=task)
        if not matches:
            raise ToolNotFoundError.for_id(
                "task", task, (row.get("task_type", "") for row in list_benchmark_zoo_cli_tasks())
            )
        if len(matches) > 1:
            names = ", ".join(sorted(str(row.get("benchmark_name")) for row in matches))
            raise ValueError(f"task {task!r} exists in several benchmarks ({names}); pass benchmark")
        return matches[0]
    return get_benchmark_zoo_cli_task(task, benchmark)


def _ids(rows: Iterable[Any], attr: str) -> list[str]:
    return [str(getattr(row, attr)) for row in rows]


__all__ = [
    "get_benchmark_info_payload",
    "get_model_info_payload",
    "get_task_info_payload",
    "list_benchmarks_payload",
    "list_models_payload",
    "list_tasks_payload",
    "page",
    "project",
]
