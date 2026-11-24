"""``wf mcp`` — start the WorldFoundry MCP server."""

from __future__ import annotations

import argparse


def add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``wf mcp`` on the root CLI parser."""

    from worldfoundry.mcp import add_server_arguments

    parser = subparsers.add_parser(
        "mcp",
        help="Start the MCP server for agent-driven WorldFoundry evaluation",
    )
    add_server_arguments(parser)
    parser.set_defaults(func=run_mcp)


def run_mcp(args: argparse.Namespace) -> int:
    """Run the MCP server with the parsed transport options."""

    from worldfoundry.mcp.server import run_server

    return run_server(args.transport, host=args.host, port=args.port)


__all__ = ["add_mcp_parser", "run_mcp"]
