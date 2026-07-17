"""Bind WorldFoundry MCP tool payloads onto a FastMCP server.

This module defines :func:`register_tools`, which creates thin MCP tool
wrappers around the payload-building functions in :mod:`.discovery`,
:mod:`.runs`, and :mod:`.studio`, then registers them on a ``FastMCP`` server instance.
"""

from __future__ import annotations

from typing import Any

from .context import DEFAULT_CONTEXT, MCPToolContext
from .discovery import (
    get_benchmark_info_payload,
    get_model_info_payload,
    get_task_info_payload,
    list_benchmarks_payload,
    list_models_payload,
    list_tasks_payload,
)
from .metrics import list_metrics_payload, show_metric_payload
from .readiness import check_benchmark_datasets_payload
from .runs import (
    cancel_run_payload,
    get_run_result_payload,
    get_run_samples_payload,
    get_run_status_payload,
    list_runs_payload,
    preview_run_payload,
    run_evaluation_payload,
)
from .server_info import server_info_payload
from .studio import (
    get_studio_job_logs_payload,
    get_studio_job_payload,
    get_studio_manifest_payload,
    get_studio_model_info_payload,
    list_studio_artifacts_payload,
    list_studio_jobs_payload,
    list_studio_models_payload,
    stop_studio_job_payload,
    submit_studio_inference_payload,
    wait_for_studio_job_payload,
)


def register_tools(mcp: Any, context: MCPToolContext | None = None) -> None:
    """Bind all WorldFoundry MCP tool payloads onto a ``FastMCP`` server instance.

    Args:
        mcp: A ``FastMCP`` server object.
        context: Shared execution context (defaults to :data:`DEFAULT_CONTEXT`).
    """

    ctx = context or DEFAULT_CONTEXT

    @mcp.tool()
    def server_info() -> dict:
        """Return WorldFoundry MCP server metadata and configured paths."""
        return server_info_payload(context=ctx)

    # ── Discovery tools ──────────────────────────────────────────────────

    @mcp.tool()
    def list_models(query: str | None = None, runnable_only: bool = False, include_notes: bool = False) -> dict:
        """List available models, optionally filtered.

        Args:
            query: Glob or substring filter.
            runnable_only: Only include models with a runnable runner.
            include_notes: Keep the ``notes`` field in each entry.

        Returns:
            Dictionary with ``models``, ``total``, and ``query`` keys.
        """
        return list_models_payload(query=query, runnable_only=runnable_only, include_notes=include_notes, context=ctx)

    @mcp.tool()
    def get_model_info(model_id: str) -> dict:
        """Return full metadata for a single model.

        Args:
            model_id: Exact model identifier to look up.

        Returns:
            Model manifest dictionary.
        """
        return get_model_info_payload(model_id, context=ctx)

    @mcp.tool()
    def list_benchmarks(query: str | None = None, integrated_only: bool = False, include_notes: bool = False) -> dict:
        """List available benchmarks, optionally filtered.

        Args:
            query: Glob or substring filter.
            integrated_only: Only include benchmarks with ``integrated`` status.
            include_notes: Keep the ``notes`` field in each entry.

        Returns:
            Dictionary with ``benchmarks``, ``total``, and ``query`` keys.
        """
        return list_benchmarks_payload(
            query=query,
            integrated_only=integrated_only,
            include_notes=include_notes,
            context=ctx,
        )

    @mcp.tool()
    def get_benchmark_info(benchmark_id: str) -> dict:
        """Return full metadata for a single benchmark.

        Args:
            benchmark_id: Exact benchmark identifier to look up.

        Returns:
            Benchmark manifest dictionary.
        """
        return get_benchmark_info_payload(benchmark_id, context=ctx)

    @mcp.tool()
    def list_tasks(
        query: str | None = None,
        suite: str | None = None,
        backend: str | None = None,
        source_kind: str | None = None,
    ) -> dict:
        """List evaluation tasks, optionally filtered.

        Args:
            query: Glob or substring filter.
            suite: Filter by suite identifier.
            backend: Filter by backend type.
            source_kind: Filter by task source kind.

        Returns:
            Dictionary with ``tasks``, ``total``, and ``query`` keys.
        """
        return list_tasks_payload(query=query, suite=suite, backend=backend, source_kind=source_kind)

    @mcp.tool()
    def get_task_info(task: str, benchmark: str | None = None) -> dict:
        """Return full metadata for a single evaluation task.

        Args:
            task: Task type identifier (may include ``benchmark`` via ``task/benchmark`` syntax).
            benchmark: Explicit benchmark identifier to disambiguate.

        Returns:
            Task payload dictionary.
        """
        return get_task_info_payload(task, benchmark)

    @mcp.tool()
    def list_metrics(
        query: str | None = None,
        family: str | None = None,
        tag: str | None = None,
    ) -> dict:
        """List registered evaluation metrics, optionally filtered.

        Args:
            query: Glob or substring filter.
            family: Exact metric-family filter.
            tag: Required metric tag.
        """
        return list_metrics_payload(query=query, family=family, tag=tag)

    @mcp.tool()
    def show_metric(metric_id: str) -> dict:
        """Return metadata for one metric id or alias.

        Args:
            metric_id: Metric identifier or registered alias.
        """
        return show_metric_payload(metric_id)

    @mcp.tool()
    def check_benchmark_datasets(benchmark_id: str, data_root: str | None = None) -> dict:
        """Check local dataset readiness for one benchmark.

        Args:
            benchmark_id: Benchmark catalog identifier.
            data_root: Optional local dataset cache root.
        """
        return check_benchmark_datasets_payload(
            benchmark_id=benchmark_id,
            data_root=data_root,
            context=ctx,
        )

    # ── Run lifecycle tools ──────────────────────────────────────────────

    @mcp.tool()
    def preview_run(
        model: str,
        benchmark: str,
        output_dir: str | None = None,
        plan_only: bool = False,
        prepare: bool = False,
        execute_download: bool = False,
        data_root: str | None = None,
    ) -> dict:
        """Preview the command and output directory for a planned evaluation run.

        Args:
            model: Model identifier.
            benchmark: Benchmark identifier.
            output_dir: Custom output directory (optional).
            plan_only: Only produce the plan, do not execute.
            prepare: Prepare run environment before execution.
            execute_download: Allow download steps to run.
            data_root: Local dataset cache root (optional).

        Returns:
            Dictionary with ``command``, ``output_dir``, and ``plan_only`` keys.
        """
        return preview_run_payload(
            model=model,
            benchmark=benchmark,
            output_dir=output_dir,
            plan_only=plan_only,
            prepare=prepare,
            execute_download=execute_download,
            data_root=data_root,
            context=ctx,
        )

    @mcp.tool()
    async def evaluate(
        model: str,
        tasks: list[str] | None = None,
        benchmarks: list[str] | None = None,
        benchmark: str | None = None,
        output_dir: str | None = None,
        plan_only: bool = False,
        resume: bool = False,
        prepare: bool = False,
        execute_download: bool = False,
        data_root: str | None = None,
        generation_cache_dir: str | None = None,
        generation_cache_mode: str = "off",
        wait: str = "auto",
        wait_timeout_s: int = 90,
    ) -> dict:
        """Submit an evaluation run for one or more model/benchmark combinations.

        Args:
            model: Model identifier.
            tasks: Task identifiers (alternative to ``benchmarks``).
            benchmarks: Benchmark identifiers.
            benchmark: Single benchmark identifier (alternative to ``benchmarks``).
            output_dir: Custom output directory (optional).
            plan_only: Only produce the plan, do not execute.
            resume: Resume a previously started run.
            prepare: Prepare run environment before execution.
            execute_download: Allow download steps to run.
            data_root: Local dataset cache root (optional).
            generation_cache_dir: Cache directory for generated outputs.
            generation_cache_mode: Cache mode — ``"off"``, ``"read"``, or ``"write"``.
            wait: Wait strategy — ``"auto"``, ``"async"``, or ``"sync"``.
            wait_timeout_s: Seconds to wait before returning asynchronously.

        Returns:
            Run summary or result dictionary depending on ``wait`` strategy.
        """
        return await run_evaluation_payload(
            model=model,
            benchmark=benchmark,
            benchmarks=benchmarks,
            tasks=tasks,
            output_dir=output_dir,
            plan_only=plan_only,
            resume=resume,
            prepare=prepare,
            execute_download=execute_download,
            data_root=data_root,
            generation_cache_dir=generation_cache_dir,
            generation_cache_mode=generation_cache_mode,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            context=ctx,
        )

    # ── Run inspection tools ─────────────────────────────────────────────

    @mcp.tool()
    def list_runs(limit: int = 50, status: str | None = None) -> dict:
        """List MCP-managed evaluation runs, newest first.

        Args:
            limit: Maximum runs to return (1–500).
            status: Optional exact lifecycle-status filter.
        """
        return list_runs_payload(limit=limit, status=status, context=ctx)

    @mcp.tool()
    async def get_run_status(run_id: str) -> dict:
        """Return the current status of a previously submitted run.

        Args:
            run_id: Unique run identifier returned by :func:`evaluate`.

        Returns:
            Run summary dictionary with a ``log_tail`` of recent log lines.
        """
        return get_run_status_payload(run_id, context=ctx)

    @mcp.tool()
    async def get_run_result(run_id: str, include_logs: bool = False) -> dict:
        """Return the result payload of a completed run.

        Args:
            run_id: Unique run identifier.
            include_logs: Include full log output in the result.

        Returns:
            Run result dictionary.
        """
        return get_run_result_payload(run_id, include_logs=include_logs, context=ctx)

    @mcp.tool()
    async def get_run_samples(
        run_id: str,
        task_name: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        """Return sample-level results from a completed run.

        Args:
            run_id: Unique run identifier.
            task_name: Filter samples by task name (optional).
            offset: Skip the first *offset* samples.
            limit: Maximum number of samples to return (1–500).

        Returns:
            Dictionary with ``samples``, ``total``, ``offset``, and ``limit`` keys.
        """
        return get_run_samples_payload(
            run_id,
            task_name=task_name,
            offset=offset,
            limit=limit,
            context=ctx,
        )

    @mcp.tool()
    async def cancel_run(run_id: str) -> dict:
        """Cancel a running or pending evaluation run.

        Args:
            run_id: Unique run identifier.

        Returns:
            Dictionary with ``success``, ``run_id``, and ``message`` keys.
        """
        return await cancel_run_payload(run_id, context=ctx)

    # ── Studio workspace tools ───────────────────────────────────────────

    @mcp.tool()
    def list_studio_models(
        query: str | None = None,
        workload_type: str | None = None,
        supports_from_pretrained: bool | None = None,
        base_url: str | None = None,
    ) -> dict:
        """List models exposed by a local Studio workspace HTTP API.

        Args:
            query: Case-insensitive substring filter across model metadata fields.
            workload_type: Filter by exact workload type.
            supports_from_pretrained: Filter by ``supports_from_pretrained`` flag.
            base_url: Studio workspace base URL (defaults to ``WORLDFOUNDRY_STUDIO_WORKSPACE_URL``).

        Returns:
            Dictionary with ``models``, ``total``, and ``base_url`` keys.
        """
        return list_studio_models_payload(
            query=query,
            workload_type=workload_type,
            supports_from_pretrained=supports_from_pretrained,
            base_url=base_url,
        )

    @mcp.tool()
    def get_studio_model_info(model_id: str, base_url: str | None = None) -> dict:
        """Return metadata for one Studio workspace model.

        Args:
            model_id: Exact Studio model identifier.
            base_url: Studio workspace base URL (optional).

        Returns:
            Studio model dictionary.
        """
        return get_studio_model_info_payload(model_id, base_url=base_url)

    @mcp.tool()
    def submit_studio_inference(
        model_id: str,
        base_url: str | None = None,
        variant_id: str = "",
        task_profile_id: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        input_path: str = "",
        backend: str = "",
        device: str = "",
        params: dict | None = None,
        call_kwargs: dict | None = None,
        load_kwargs: dict | None = None,
        env_overrides: dict | None = None,
        wait: bool = False,
        wait_timeout_s: float = 0,
        poll_interval_s: float = 5,
    ) -> dict:
        """Submit an inference job to a Studio workspace.

        Args:
            model_id: Studio model identifier.
            base_url: Studio workspace base URL (optional).
            variant_id: Model variant selector (optional).
            task_profile_id: Task profile to apply (optional).
            prompt: Text prompt for generation.
            negative_prompt: Negative prompt for generation.
            input_path: Path to input file (optional).
            backend: Backend selector (defaults to ``auto``).
            device: Device selector (optional).
            params: Additional inference parameters.
            call_kwargs: Keyword arguments passed to the model call.
            load_kwargs: Keyword arguments passed to model loading.
            env_overrides: Environment variable overrides for the job.
            wait: Poll until the job reaches a terminal state.
            wait_timeout_s: Maximum seconds to wait (0 = unlimited).
            poll_interval_s: Seconds between status polls.

        Returns:
            Submitted job dictionary, or the final job state when ``wait`` is true.
        """
        return submit_studio_inference_payload(
            model_id=model_id,
            base_url=base_url,
            variant_id=variant_id,
            task_profile_id=task_profile_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_path=input_path,
            backend=backend,
            device=device,
            params=params,
            call_kwargs=call_kwargs,
            load_kwargs=load_kwargs,
            env_overrides=env_overrides,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool()
    def list_studio_jobs(job_type: str | None = None, base_url: str | None = None) -> dict:
        """List jobs from a Studio workspace.

        Args:
            job_type: Filter by job type (e.g. ``inference``).
            base_url: Studio workspace base URL (optional).

        Returns:
            Dictionary with ``jobs``, ``total``, and ``base_url`` keys.
        """
        return list_studio_jobs_payload(job_type=job_type, base_url=base_url)

    @mcp.tool()
    def get_studio_job(job_id: str, base_url: str | None = None) -> dict:
        """Return one Studio workspace job by id.

        Args:
            job_id: Studio job identifier.
            base_url: Studio workspace base URL (optional).

        Returns:
            Job dictionary from the Studio API.
        """
        return get_studio_job_payload(job_id, base_url=base_url)

    @mcp.tool()
    def wait_for_studio_job(
        job_id: str,
        base_url: str | None = None,
        timeout_s: float = 0,
        poll_interval_s: float = 5,
    ) -> dict:
        """Poll a Studio job until it completes, fails, is cancelled, or times out.

        Args:
            job_id: Studio job identifier.
            base_url: Studio workspace base URL (optional).
            timeout_s: Maximum seconds to wait (0 = unlimited).
            poll_interval_s: Seconds between status polls.

        Returns:
            Final or latest polled job dictionary.
        """
        return wait_for_studio_job_payload(
            job_id,
            base_url=base_url,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool()
    def stop_studio_job(job_id: str, base_url: str | None = None) -> dict:
        """Cancel a running or pending Studio job.

        Args:
            job_id: Studio job identifier.
            base_url: Studio workspace base URL (optional).

        Returns:
            Stop response dictionary with ``ok``, ``message``, and ``job`` keys.
        """
        return stop_studio_job_payload(job_id, base_url=base_url)

    @mcp.tool()
    def get_studio_job_logs(job_id: str, after: int = 0, base_url: str | None = None) -> dict:
        """Return incremental log lines for a Studio job.

        Args:
            job_id: Studio job identifier.
            after: Skip the first *after* log entries.
            base_url: Studio workspace base URL (optional).

        Returns:
            Log payload dictionary with ``offset``, ``logs``, and ``text`` keys.
        """
        return get_studio_job_logs_payload(job_id, after=after, base_url=base_url)

    @mcp.tool()
    def list_studio_artifacts(limit: int = 20, base_url: str | None = None) -> dict:
        """List artifacts registered in a Studio workspace session.

        Args:
            limit: Maximum number of artifacts to return (1–200).
            base_url: Studio workspace base URL (optional).

        Returns:
            Dictionary with ``artifacts``, ``total``, and ``base_url`` keys.
        """
        return list_studio_artifacts_payload(limit=limit, base_url=base_url)

    @mcp.tool()
    def get_studio_manifest(job_id: str, base_url: str | None = None) -> dict:
        """Return manifest metadata for a completed Studio job.

        Args:
            job_id: Studio job identifier.
            base_url: Studio workspace base URL (optional).

        Returns:
            Dictionary with ``job_id``, ``manifest_path``, ``output_dir``,
            ``artifacts``, and ``metadata`` keys.
        """
        return get_studio_manifest_payload(job_id, base_url=base_url)


__all__ = ["register_tools"]
