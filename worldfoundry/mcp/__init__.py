"""Model Context Protocol interface for WorldFoundry evaluation.

Provides lazy-loaded access to :class:`MCPClient` and a CLI entry point
(:func:`main`) for launching the MCP server.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MCPClient", "main"]

# ── Lazy imports ───────────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Lazy-load ``MCPClient`` on first attribute access to avoid importing the MCP extra at package load time."""
    # NOTE: ``MCPClient`` depends on the optional ``mcp`` package; deferring
    # the import keeps the core ``worldfoundry`` import lightweight.
    if name == "MCPClient":
        from .client import MCPClient

        return MCPClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── CLI entry point ────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and launch the MCP server.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code — 0 on success, non-zero on failure.
    """
    from worldfoundry.cli.help import WorldFoundryArgumentParser

    parser = WorldFoundryArgumentParser(prog="worldfoundry-mcp", description="Start the WorldFoundry MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse"),
        default="stdio",
        help="MCP transport type.",
    )
    args = parser.parse_args(argv)

    from .server import run_server

    return run_server(args.transport)


if __name__ == "__main__":
    raise SystemExit(main())
