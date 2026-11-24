"""Construct and launch the WorldFoundry MCP server."""

from __future__ import annotations

from typing import Any

from . import DEFAULT_HTTP_HOST, DEFAULT_HTTP_PORT
from .tools import MCPToolContext, register_tools, resolve_mcp_output_root, set_default_context

MCP_INSTALL_HINT = "WorldFoundry MCP requires the 'mcp' package. Install with: pip install 'worldfoundry[mcp]'"

# Surfaced to clients as the server description; guides tool choice and order.
_INSTRUCTIONS = """\
WorldFoundry runs world models (video, interactive worlds, 3D/4D, embodied action) and scores
them on benchmarks.

Typical evaluation flow:
1. server_info - paths, tool list, and workflow.
2. list_models / list_benchmarks / list_metrics - find ids (summary pages; get_*_info for details).
3. check_benchmark_datasets - confirm the benchmark data is present locally.
4. preview_run - see the exact `wf` command and output directory.
5. evaluate - submit; wait='auto' returns early for long runs.
6. get_run_status / get_run_result / get_run_samples - follow and read results; cancel_run to stop.

For interactive generation against a running Studio workspace, use list_studio_models,
submit_studio_inference, wait_for_studio_job, and get_studio_manifest.

Every tool returns {"ok": true, ...} or {"ok": false, "error", "error_type"} where error_type is
not_found, invalid_argument, unavailable, runtime, or internal.
"""


def require_fastmcp() -> Any:
    """Return the ``FastMCP`` class, or raise a clear install error."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(MCP_INSTALL_HINT) from exc
    return FastMCP


def create_mcp_server(*, host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> Any:
    """Build a ``FastMCP`` server with every WorldFoundry tool registered."""

    fast_mcp = require_fastmcp()
    # The output root is resolved to an absolute path (WORLDFOUNDRY_MCP_RUN_ROOT
    # override honoured at server start), never left CWD-relative (CM-29).
    # Installing the server's context as the module default means payload
    # functions called without an explicit ``context`` share the same job
    # store as the registered tools instead of a second, invisible one.
    context = set_default_context(MCPToolContext(output_root=resolve_mcp_output_root()))
    server = fast_mcp("worldfoundry", instructions=_INSTRUCTIONS, json_response=True, host=host, port=port)
    register_tools(server, context)
    return server


def run_server(transport: str = "stdio", *, host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> int:
    """Create the MCP server and run it on ``transport``."""

    try:
        server = create_mcp_server(host=host, port=port)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        server.run(transport=transport)
    except KeyboardInterrupt:
        # HTTP transports are usually stopped with Ctrl-C; the SDK may re-raise
        # KeyboardInterrupt after uvicorn has already shut down cleanly.
        return 0
    return 0


__all__ = ["MCP_INSTALL_HINT", "create_mcp_server", "require_fastmcp", "run_server"]
