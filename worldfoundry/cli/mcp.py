"""worldfoundry mcp - start the MCP evaluation server."""

from __future__ import annotations

import argparse


def add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``mcp`` sub-command on the root CLI parser.

    Args:
        subparsers: The subparser collection from the root
            ``worldfoundry`` command.
    """
    parser = subparsers.add_parser(
        "mcp",
        help="Start the MCP server for agent-driven WorldFoundry evaluation",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse"),
        default="stdio",
        help="MCP transport type.",
    )
    parser.set_defaults(func=run_mcp)


def run_mcp(args: argparse.Namespace) -> int:
    """Start the MCP evaluation server with the requested transport.

    Args:
        args: Parsed CLI namespace with ``transport`` selection.

    Returns:
        Exit code from the MCP server process.
    """
    from worldfoundry.mcp.server import run_server

    return run_server(args.transport)


__all__ = ["add_mcp_parser", "run_mcp"]
