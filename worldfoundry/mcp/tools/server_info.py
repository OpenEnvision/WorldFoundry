"""Server metadata payloads for the WorldFoundry MCP server."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from typing import Any

from .context import DEFAULT_CONTEXT, DEFAULT_MCP_OUTPUT_ROOT, MCPToolContext
from .studio import DEFAULT_STUDIO_WORKSPACE_URL

MCP_TOOL_NAMES: tuple[str, ...] = (
    "server_info",
    "list_models",
    "get_model_info",
    "list_benchmarks",
    "get_benchmark_info",
    "list_tasks",
    "get_task_info",
    "list_metrics",
    "show_metric",
    "check_benchmark_datasets",
    "preview_run",
    "evaluate",
    "list_runs",
    "get_run_status",
    "get_run_result",
    "get_run_samples",
    "cancel_run",
    "list_studio_models",
    "get_studio_model_info",
    "submit_studio_inference",
    "list_studio_jobs",
    "get_studio_job",
    "wait_for_studio_job",
    "stop_studio_job",
    "get_studio_job_logs",
    "list_studio_artifacts",
    "get_studio_manifest",
)


def server_info_payload(*, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Summarize server configuration, tools, and documentation hints."""

    ctx = context or DEFAULT_CONTEXT
    version = _package_version()
    env_keys = sorted(name for name in os.environ if name.startswith("WORLDFOUNDRY_"))
    return {
        "server": "worldfoundry",
        "version": version,
        "python": sys.version.split()[0],
        "output_root": str(ctx.output_root),
        "model_manifest_dir": str(ctx.model_manifest_dir) if ctx.model_manifest_dir else None,
        "benchmark_manifest_dir": str(ctx.benchmark_manifest_dir),
        "studio_workspace_url": DEFAULT_STUDIO_WORKSPACE_URL,
        "tools": list(MCP_TOOL_NAMES),
        "tool_count": len(MCP_TOOL_NAMES),
        "environment": {
            "WORLDFOUNDRY_MCP_RUN_ROOT": os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", str(DEFAULT_MCP_OUTPUT_ROOT)),
            "WORLDFOUNDRY_STUDIO_WORKSPACE_URL": os.environ.get("WORLDFOUNDRY_STUDIO_WORKSPACE_URL"),
            "WORLDFOUNDRY_REPO_ROOT": os.environ.get("WORLDFOUNDRY_REPO_ROOT"),
            "WORLDFOUNDRY_UNIFIED_ENV_PREFIX": os.environ.get("WORLDFOUNDRY_UNIFIED_ENV_PREFIX"),
            "WORLDFOUNDRY_BENCHMARK_DATA_ROOT": os.environ.get("WORLDFOUNDRY_BENCHMARK_DATA_ROOT"),
            "configured_keys": env_keys,
        },
        "docs": {
            "mcp": "/docs/reference/mcp",
            "cli": "/docs/reference/cli",
            "evaluation_quickstart": "/docs/evaluation/quickstart",
            "metrics": "/docs/evaluation/metrics",
            "agent_setup": "/docs/guides/agent-setup",
            "docker": "/docs/guides/docker",
        },
        "workflow": [
            "Call server_info to confirm paths and available tools.",
            "Discover ids with list_models, list_benchmarks, list_tasks, and list_metrics.",
            "Prepare benchmark data, generated artifacts, checkpoints, and judge credentials before evaluate.",
            "Use preview_run to inspect the CLI command, then evaluate with wait=auto.",
            "Track jobs with list_runs, get_run_status, get_run_result, and get_run_samples.",
        ],
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("worldfoundry")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


__all__ = ["MCP_TOOL_NAMES", "server_info_payload"]
