"""Deterministic MCP registration and stdio protocol tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from jsonschema.validators import validator_for

pytest.importorskip("mcp")

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
def test_keyboard_interrupt_is_a_clean_server_shutdown() -> None:
    class InterruptedServer:
        def run(self, *, transport: str) -> None:
            raise KeyboardInterrupt

    with patch("worldfoundry.mcp.server.create_mcp_server", return_value=InterruptedServer()):
        assert run_server("sse") == 0
