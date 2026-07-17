"""Catalog discovery payloads for the WorldFoundry MCP server.

Provides payload-building functions that query the WorldFoundry model, benchmark,
and task catalogs and return structured dictionaries suitable for MCP tool
responses.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

from worldfoundry.cli.tui_discovery import load_tui_catalog

from .context import DEFAULT_CONTEXT, MCPToolContext

# ── Model discovery ─────────────────────────────────────────────────────


def list_models_payload(
    *,
    query: str | None = None,
    runnable_only: bool = False,
    include_notes: bool = False,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List models from the catalog, optionally filtered.

    Args:
        query: Glob or substring filter applied across model fields.
        runnable_only: Include only models with a runnable runner.
        include_notes: Keep the ``notes`` field in each payload.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Returns:
        Dictionary with ``models``, ``total``, and ``query`` keys.
    """

    ctx = context or DEFAULT_CONTEXT
    catalog = load_tui_catalog(
        model_manifest_dir=ctx.model_manifest_dir,
        benchmark_manifest_dir=ctx.benchmark_manifest_dir,
    )
    rows = []
    for row in catalog.models:
        if runnable_only and row.runner_kind != "runnable_runner":
            continue
        if query and not _matches(query, row.model_id, row.name, row.provider, *row.tasks):
            continue
        payload = row.to_dict()
        if not include_notes:
            payload.pop("notes", None)
        rows.append(payload)
    return {"models": rows, "total": len(rows), "query": query}


# ── Single-entry lookups ────────────────────────────────────────────────


def get_model_info_payload(model_id: str, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Return the full manifest payload for a single model.

    Args:
        model_id: Exact model identifier.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``model_id`` is not found in the catalog.

    Returns:
        Model manifest dictionary.
    """

    ctx = context or DEFAULT_CONTEXT
    catalog = load_tui_catalog(
        model_manifest_dir=ctx.model_manifest_dir,
        benchmark_manifest_dir=ctx.benchmark_manifest_dir,
    )
    for row in catalog.models:
        if row.model_id == model_id:
            return row.to_dict()
    raise ValueError(f"model not found: {model_id}")


# ── Benchmark discovery ────────────────────────────────────────────────


def list_benchmarks_payload(
    *,
    query: str | None = None,
    integrated_only: bool = False,
    include_notes: bool = False,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List benchmarks from the catalog, optionally filtered.

    Args:
        query: Glob or substring filter applied across benchmark fields.
        integrated_only: Include only benchmarks with ``integrated`` status.
        include_notes: Keep the ``notes`` field in each payload.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Returns:
        Dictionary with ``benchmarks``, ``total``, and ``query`` keys.
    """

    ctx = context or DEFAULT_CONTEXT
    catalog = load_tui_catalog(
        model_manifest_dir=ctx.model_manifest_dir,
        benchmark_manifest_dir=ctx.benchmark_manifest_dir,
    )
    rows = []
    for row in catalog.benchmarks:
        if integrated_only and row.integration_status != "integrated":
            continue
        if query and not _matches(query, row.benchmark_id, row.name, *row.domains, *row.modalities, *row.tags):
            continue
        payload = row.to_dict()
        if not include_notes:
            payload.pop("notes", None)
        rows.append(payload)
    return {"benchmarks": rows, "total": len(rows), "query": query}


def get_benchmark_info_payload(benchmark_id: str, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Return the full manifest payload for a single benchmark.

    Args:
        benchmark_id: Exact benchmark identifier.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``benchmark_id`` is not found in the catalog.

    Returns:
        Benchmark manifest dictionary.
    """

    ctx = context or DEFAULT_CONTEXT
    catalog = load_tui_catalog(
        model_manifest_dir=ctx.model_manifest_dir,
        benchmark_manifest_dir=ctx.benchmark_manifest_dir,
    )
    for row in catalog.benchmarks:
        if row.benchmark_id == benchmark_id:
            return row.to_dict()
    raise ValueError(f"benchmark not found: {benchmark_id}")


# ── Task discovery ─────────────────────────────────────────────────────


def list_tasks_payload(
    *,
    query: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    """List evaluation tasks from the task registry, optionally filtered.

    Args:
        query: Glob or substring filter applied across task fields.
        suite: Filter by suite identifier.
        backend: Filter by backend type.
        source_kind: Filter by task source kind.

    Returns:
        Dictionary with ``tasks``, ``total``, and ``query`` keys.
    """

    from worldfoundry.evaluation.tasks.catalog.specs import list_benchmark_zoo_cli_tasks

    rows = list_benchmark_zoo_cli_tasks(
        suite=suite,
        backend=backend,
        source_kind=source_kind,
    )
    filtered: list[dict[str, Any]] = []
    for payload in rows:
        if query and not _matches(
            query,
            payload.get("task_type"),
            payload.get("benchmark_name"),
            payload.get("name"),
            payload.get("benchmark_zoo_id"),
            payload.get("description"),
        ):
            continue
        filtered.append(payload)
    return {"tasks": filtered, "total": len(filtered), "query": query}


def get_task_info_payload(task: str, benchmark: str | None = None) -> dict[str, Any]:
    """Return the full payload for a single evaluation task.

    Args:
        task: Task type identifier. May include ``benchmark`` via ``task/benchmark`` syntax.
        benchmark: Explicit benchmark identifier to disambiguate.

    Raises:
        ValueError: If the task is ambiguous or not found.

    Returns:
        Task payload dictionary.
    """

    from worldfoundry.evaluation.tasks.catalog.specs import get_benchmark_zoo_cli_task, list_benchmark_zoo_cli_tasks

    if benchmark is None and "/" in task:
        task, benchmark = task.split("/", 1)
    if benchmark is None:
        matches = list_benchmark_zoo_cli_tasks(task_type=task)
        if len(matches) != 1:
            raise ValueError(f"task {task!r} matched {len(matches)} benchmarks; pass benchmark explicitly")
        return matches[0]
    return get_benchmark_zoo_cli_task(task, benchmark)


# ── Internal helpers ───────────────────────────────────────────────────


def _matches(query: str, *values: object) -> bool:
    """Check whether any *values* match *query* using glob or substring semantics.

    When ``query`` contains glob characters (``*``, ``?``, ``[``, ``]``),
    :func:`fnmatch.fnmatchcase` is used; otherwise a plain substring check
    is performed.  All comparisons are case-insensitive.
    """

    needle = query.casefold()
    glob_query = any(char in needle for char in "*?[]")
    return any(
        fnmatchcase(str(value).casefold(), needle) if glob_query else needle in str(value).casefold()
        for value in values
        if value
    )


__all__ = [
    "get_benchmark_info_payload",
    "get_model_info_payload",
    "get_task_info_payload",
    "list_benchmarks_payload",
    "list_models_payload",
    "list_tasks_payload",
]
