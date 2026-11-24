"""Model Context Protocol server for WorldFoundry.

Start it with ``wf mcp`` (or ``python -m worldfoundry.mcp``). Importing this
package stays lightweight: :class:`MCPClient` and the server load the optional
``mcp`` dependency only when used.
"""

from __future__ import annotations

import argparse
from typing import Any

__all__ = ["DEFAULT_HTTP_HOST", "DEFAULT_HTTP_PORT", "TRANSPORTS", "MCPClient", "add_server_arguments", "main"]

TRANSPORTS = ("stdio", "streamable-http", "sse")
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000


def __getattr__(name: str) -> Any:
    """Load ``MCPClient`` on first access so the optional ``mcp`` extra is imported lazily."""

    if name == "MCPClient":
        from .client import MCPClient

        return MCPClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the server's transport options; shared by ``wf mcp`` and ``python -m worldfoundry.mcp``."""

    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="stdio",
        help="stdio for local clients that spawn the server; streamable-http (or legacy sse) to serve over HTTP.",
    )
    parser.add_argument("--host", default=DEFAULT_HTTP_HOST, help="Bind host for HTTP transports.")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="Bind port for HTTP transports.")


def main(argv: list[str] | None = None) -> int:
    """Parse the server options and run the MCP server."""

    from worldfoundry.cli.help import WorldFoundryArgumentParser

    parser = WorldFoundryArgumentParser(prog="wf mcp", description="Start the WorldFoundry MCP server.")
    add_server_arguments(parser)
    args = parser.parse_args(argv)

    from .server import run_server

    return run_server(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
