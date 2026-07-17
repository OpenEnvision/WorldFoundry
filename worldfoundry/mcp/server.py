"""Construct and launch the WorldFoundry MCP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .tools import MCPToolContext, register_tools

# ── Constants & install hint ───────────────────────────────────

MCP_INSTALL_HINT = "WorldFoundry MCP requires the 'mcp' package. Install with: pip install 'worldfoundry[mcp]'"

# NOTE: ``_INSTRUCTIONS`` is surfaced to MCP clients as the server's
# capability description, guiding tool selection and usage order.
_INSTRUCTIONS = (
    "WorldFoundry MCP server for benchmark catalog discovery, evaluation execution, "
    "and Studio workspace inference.\n\n"
    "Use discovery tools to find model and benchmark ids, readiness tools to inspect "
    "local datasets/assets, preview_run to inspect the exact command, evaluate to "
    "submit a run, and run-management tools to inspect outputs or cancel active jobs.\n\n"
    "For local Studio workspace jobs, use list_studio_models and submit_studio_inference "
    "against the workspace HTTP API (default http://127.0.0.1:7870)."
)

# ── Server construction ────────────────────────────────────────


def require_fastmcp() -> Any:
    """Return the ``FastMCP`` class, or raise a clear install error."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(MCP_INSTALL_HINT) from exc
    return FastMCP


def create_mcp_server() -> Any:
    """Build a configured ``FastMCP`` server instance with all WorldFoundry tools registered."""

    fast_mcp = require_fastmcp()
    context = MCPToolContext(output_root=Path(os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", "runs/mcp")))
    server = fast_mcp("worldfoundry", instructions=_INSTRUCTIONS, json_response=True)
    register_tools(server, context)
    return server

# ── Entry point ────────────────────────────────────────────────


def run_server(transport: str = "stdio") -> int:
    """Create the MCP server and run it, surfacing a friendly missing-dependency error."""

    try:
        server = create_mcp_server()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        server.run(transport=transport)
    except KeyboardInterrupt:
        # Interactive HTTP transports are commonly stopped with Ctrl-C.  The
        # SDK may re-raise KeyboardInterrupt after uvicorn has already shut
        # down cleanly; treat that normal operator action as a successful stop.
        return 0
    return 0


__all__ = ["MCP_INSTALL_HINT", "create_mcp_server", "require_fastmcp", "run_server"]
