"""Deterministic MCP registration and stdio protocol tests."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

import pytest
from jsonschema.validators import validator_for

pytest.importorskip("mcp")

from worldfoundry.mcp.client import MCPClient
from worldfoundry.mcp.server import create_mcp_server, run_server
from worldfoundry.mcp.tools.server_info import MCP_TOOL_NAMES


@pytest.mark.unit
def test_registered_tools_match_advertised_tools_and_have_valid_schemas() -> None:
    server = create_mcp_server()
    tools = server._tool_manager.list_tools()

    assert {tool.name for tool in tools} == set(MCP_TOOL_NAMES)
    assert len(tools) == len(MCP_TOOL_NAMES)
    for tool in tools:
        validator = validator_for(tool.parameters)
        validator.check_schema(tool.parameters)


@pytest.mark.unit
def test_stdio_protocol_lists_and_calls_tools() -> None:
    async def exercise() -> None:
        client = MCPClient(
            command=sys.executable,
            args=("-m", "worldfoundry.mcp"),
            env=os.environ,
        )
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == set(MCP_TOOL_NAMES)

        result = await client.call_tool("server_info")
        assert result.isError is False

        error = await client.call_tool("show_metric", {"metric_id": "definitely-not-a-metric"})
        assert error.isError is True
        assert "unknown metric id" in error.content[0].text

    asyncio.run(exercise())


@pytest.mark.unit
def test_keyboard_interrupt_is_a_clean_server_shutdown() -> None:
    class InterruptedServer:
        def run(self, *, transport: str) -> None:
            raise KeyboardInterrupt

    with patch("worldfoundry.mcp.server.create_mcp_server", return_value=InterruptedServer()):
        assert run_server("sse") == 0
