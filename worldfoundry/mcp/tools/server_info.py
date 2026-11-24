"""Server metadata payload for the WorldFoundry MCP server."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from typing import Any

# Import every tool module so the registry is complete before MCP_TOOL_NAMES is derived.
from . import discovery, metrics, readiness, runs, studio  # noqa: F401
from .context import DEFAULT_MCP_OUTPUT_ROOT, MCPToolContext, get_default_context
from .spec import TOOL_SPECS, mcp_tool
from .studio import DEFAULT_STUDIO_WORKSPACE_URL

DOCS_BASE_URL = "https://openenvision.github.io/WorldFoundry/docs"

WORKFLOW = (
    "Call server_info to confirm paths and available tools.",
    "Find ids with list_models, list_benchmarks, list_tasks, and list_metrics (summary pages; use get_*_info for details).",
    "Check local data with check_benchmark_datasets; stage checkpoints, datasets, and judge credentials before evaluate.",
    "Inspect the exact command with preview_run, then submit with evaluate (wait='auto').",
    "Track runs with list_runs, get_run_status, get_run_result, and get_run_samples; cancel with cancel_run.",
    "For interactive generation, use the list_studio_* and submit_studio_inference tools against a running Studio workspace.",
)


@mcp_tool("server_info", title="Server information", read_only=True)
def server_info_payload(*, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Return server version, configured paths, the tool list with behaviour hints, and the recommended workflow."""

    ctx = context or get_default_context()
    return {
        "server": "worldfoundry",
        "version": _package_version(),
        "python": sys.version.split()[0],
        "output_root": str(ctx.output_root),
        "model_manifest_dir": str(ctx.model_manifest_dir) if ctx.model_manifest_dir else None,
        "benchmark_manifest_dir": str(ctx.benchmark_manifest_dir),
        "studio_workspace_url": DEFAULT_STUDIO_WORKSPACE_URL,
        "tools": [{"name": spec.name, "title": spec.title, **spec.annotations_dict()} for spec in TOOL_SPECS.values()],
        "tool_count": len(TOOL_SPECS),
        "environment": {
            "WORLDFOUNDRY_MCP_RUN_ROOT": os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", str(DEFAULT_MCP_OUTPUT_ROOT)),
            "WORLDFOUNDRY_STUDIO_WORKSPACE_URL": os.environ.get("WORLDFOUNDRY_STUDIO_WORKSPACE_URL"),
            "WORLDFOUNDRY_REPO_ROOT": os.environ.get("WORLDFOUNDRY_REPO_ROOT"),
            "WORLDFOUNDRY_UNIFIED_ENV_PREFIX": os.environ.get("WORLDFOUNDRY_UNIFIED_ENV_PREFIX"),
            "WORLDFOUNDRY_BENCHMARK_DATA_ROOT": os.environ.get("WORLDFOUNDRY_BENCHMARK_DATA_ROOT"),
            "configured_keys": sorted(name for name in os.environ if name.startswith("WORLDFOUNDRY_")),
        },
        "docs": {
            "cli_and_mcp": f"{DOCS_BASE_URL}/reference/cli",
            "evaluation": f"{DOCS_BASE_URL}/evaluation",
            "metrics": f"{DOCS_BASE_URL}/evaluation/metrics",
            "benchmarks": f"{DOCS_BASE_URL}/evaluation/benchmark-hub",
            "models": f"{DOCS_BASE_URL}/guides/supported-models",
            "environments": f"{DOCS_BASE_URL}/reference/environments",
            "studio": f"{DOCS_BASE_URL}/guides/studio",
        },
        "workflow": list(WORKFLOW),
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("worldfoundry")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


# Advertised tool names, derived from the registry (one entry per declared payload).
MCP_TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SPECS)

__all__ = ["MCP_TOOL_NAMES", "WORKFLOW", "server_info_payload"]
