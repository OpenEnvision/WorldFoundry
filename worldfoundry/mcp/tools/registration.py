"""Bind the declared WorldFoundry MCP tools onto a FastMCP server.

Tools are declared once, next to their payloads, with
:func:`.spec.mcp_tool`; this module only installs the shared context and
registers them. See :mod:`.spec` for how schemas, behaviour hints, worker
threads, and the response envelope are applied.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .context import MCPToolContext, get_default_context, set_default_context
from .server_info import MCP_TOOL_NAMES  # noqa: F401  (imports every tool module)
from .spec import bind_tools


def register_tools(mcp: Any, context: MCPToolContext | None = None) -> dict[str, Callable[..., Any]]:
    """Register every WorldFoundry tool on ``mcp`` and return the bound callables by name.

    Args:
        mcp: A ``FastMCP`` server (or any object with a compatible ``tool(...)`` decorator).
        context: Shared execution context; installed as the process default when given.
    """

    if context is not None:
        set_default_context(context)
    return bind_tools(mcp, get_default_context())


__all__ = ["register_tools"]
