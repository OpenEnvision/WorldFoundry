"""WorldFoundry MCP tool payloads and FastMCP registration.

Payloads are grouped by responsibility:

- ``context``: shared :class:`MCPToolContext`.
- ``discovery``: model/benchmark/task catalog lookups.
- ``runs``: run preview/evaluate/status/result/samples/cancel lifecycle.
- ``studio``: Studio workspace model discovery, inference jobs, and artifacts.
- ``registration``: FastMCP tool binding.
"""

from __future__ import annotations

# ── Shared context ─────────────────────────────────────────────
from .context import DEFAULT_CONTEXT, DEFAULT_MCP_OUTPUT_ROOT, MCPToolContext

# ── Discovery payloads ─────────────────────────────────────────
from .discovery import (
    get_benchmark_info_payload,
    get_model_info_payload,
    get_task_info_payload,
    list_benchmarks_payload,
    list_models_payload,
    list_tasks_payload,
)

# ── Metric and readiness payloads ─────────────────────────────────
from .metrics import list_metrics_payload, show_metric_payload
from .readiness import check_benchmark_datasets_payload

# ── Registration ───────────────────────────────────────────────
from .registration import register_tools

# ── Run lifecycle payloads ─────────────────────────────────────
from .runs import (
    cancel_run_payload,
    get_run_result_payload,
    get_run_samples_payload,
    get_run_status_payload,
    list_runs_payload,
    preview_run_payload,
    run_evaluation_payload,
)

# ── Server metadata ─────────────────────────────────────────────────────
from .server_info import MCP_TOOL_NAMES, server_info_payload

# ── Studio workspace payloads ────────────────────────────────────
from .studio import (
    DEFAULT_STUDIO_WORKSPACE_URL,
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

__all__ = [
    "DEFAULT_CONTEXT",
    "DEFAULT_MCP_OUTPUT_ROOT",
    "DEFAULT_STUDIO_WORKSPACE_URL",
    "MCPToolContext",
    "MCP_TOOL_NAMES",
    "cancel_run_payload",
    "check_benchmark_datasets_payload",
    "get_benchmark_info_payload",
    "get_model_info_payload",
    "get_run_result_payload",
    "get_run_samples_payload",
    "get_run_status_payload",
    "get_studio_job_logs_payload",
    "get_studio_job_payload",
    "get_studio_manifest_payload",
    "get_studio_model_info_payload",
    "get_task_info_payload",
    "list_benchmarks_payload",
    "list_metrics_payload",
    "list_models_payload",
    "list_runs_payload",
    "list_studio_artifacts_payload",
    "list_studio_jobs_payload",
    "list_studio_models_payload",
    "list_tasks_payload",
    "preview_run_payload",
    "register_tools",
    "run_evaluation_payload",
    "server_info_payload",
    "show_metric_payload",
    "stop_studio_job_payload",
    "submit_studio_inference_payload",
    "wait_for_studio_job_payload",
]
