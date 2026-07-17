"""Stdio MCP client and OpenAI-format conversion helpers for WorldFoundry.

The :class:`MCPClient` dataclass launches a local MCP server over stdio and
exposes async/sync helpers for listing tools and invoking them. Standalone
conversion utilities map MCP tool results into OpenAI-compatible content
blocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

# ── MCPClient ──────────────────────────────────────────────────


@dataclass
class MCPClient:
    """Stdio MCP client helper for local scripts and tool export.

    Attributes:
        command: Server executable to launch.
        args: Extra arguments passed to the server command.
        env: Optional complete environment for the server subprocess.
        cwd: Optional working directory for the server subprocess.
        timeout: Maximum wait time for server responses.
    """

    command: str = "worldfoundry-mcp"
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    timeout: timedelta = timedelta(seconds=600)

    async def list_tools(self) -> list[Any]:
        """Query the MCP server for its available tools."""

        client = await self._client()
        async with client as session:
            result = await session.list_tools()
            return list(result.tools)

    async def get_function_list(self) -> list[dict[str, Any]]:
        """Return available MCP tools as OpenAI-compatible function definitions."""

        return self.tools_to_function_list(await self.list_tools())

    def get_function_list_sync(self) -> list[dict[str, Any]]:
        """Synchronous wrapper for :meth:`get_function_list`."""

        return _run_sync(self.get_function_list())

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke an MCP tool by name.

        Args:
            name: Tool identifier on the server.
            arguments: Optional keyword arguments forwarded to the tool.

        Returns:
            Raw MCP tool result object.
        """

        client = await self._client()
        async with client as session:
            return await session.call_tool(name, arguments or {})

    def call_tool_sync(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Synchronous wrapper for :meth:`call_tool`."""

        return _run_sync(self.call_tool(name, arguments))

    @staticmethod
    def tools_to_function_list(tools: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
        """Convert MCP tool metadata into OpenAI tool-call function definitions."""

        functions: list[dict[str, Any]] = []
        for tool in tools:
            # NOTE: MCP SDKs may expose the schema as ``inputSchema`` (camelCase)
            # or ``input_schema`` (snake_case); fall back to an empty object if
            # neither attribute exists.
            schema = getattr(tool, "inputSchema", None)
            if schema is None:
                schema = getattr(tool, "input_schema", None)
            if schema is None:
                schema = {"type": "object", "properties": {}}
            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(getattr(tool, "name")),
                        "description": str(getattr(tool, "description", "") or ""),
                        "parameters": schema,
                    },
                }
            )
        return functions

    def convert_result_to_openai_format(self, result: Any) -> list[dict[str, Any]]:
        """Convert MCP text/image/audio content into OpenAI message content blocks."""

        return convert_result_to_openai_format(result)

    async def _client(self) -> Any:
        """Create a stdio MCP client session context manager."""

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("Install the MCP extra first: pip install 'worldfoundry[mcp]'") from exc

        params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env=dict(self.env) if self.env is not None else None,
            cwd=self.cwd,
        )
        timeout = self.timeout

        # NOTE: _SessionContext wraps both the stdio transport and the ClientSession
        # so callers can use ``async with client as session:``.
        class _SessionContext:
            async def __aenter__(self):
                self._stack = AsyncExitStack()
                try:
                    read, write = await self._stack.enter_async_context(stdio_client(params))
                    self.session = await self._stack.enter_async_context(
                        ClientSession(read, write, read_timeout_seconds=timeout)
                    )
                    await self.session.initialize()
                    return self.session
                except BaseException:
                    await self._stack.aclose()
                    raise

            async def __aexit__(self, exc_type, exc, tb):
                return await self._stack.__aexit__(exc_type, exc, tb)

        return _SessionContext()

# ── OpenAI-format conversion ───────────────────────────────────


def convert_result_to_openai_format(result: Any) -> list[dict[str, Any]]:
    """Convert MCP tool results into OpenAI-compatible content blocks.

    Handles text, image, and audio content from MCP result objects or raw
    dictionaries, mapping them into the ``text``, ``image_url``, and
    ``audio_url`` block types expected by the OpenAI Chat API.
    """

    if result is None:
        return []
    if hasattr(result, "content"):
        return convert_result_to_openai_format(getattr(result, "content"))
    if isinstance(result, list | tuple):
        blocks: list[dict[str, Any]] = []
        for item in result:
            blocks.extend(convert_result_to_openai_format(item))
        return blocks
    # ── Dictionary-based content blocks ────────────────────────
    if isinstance(result, dict):
        if result.get("type") == "text" and "text" in result:
            return [{"type": "text", "text": str(result["text"])}]
        if result.get("type") == "image" and "data" in result:
            mime_type = str(result.get("mimeType") or result.get("mime_type") or "image/png")
            return [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{result['data']}"}}]
        if result.get("type") == "audio" and "data" in result:
            mime_type = str(result.get("mimeType") or result.get("mime_type") or "audio/wav")
            return [{"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{result['data']}"}}]
        # NOTE: Unrecognised dict content is serialised as a JSON text block.
        return [{"type": "text", "text": _json_text(result)}]

    # ── Object-based content blocks ────────────────────────────
    # NOTE: For opaque MCP result objects, ``content_type`` is derived from
    # the ``type`` attribute or the class name. Image/audio detection uses a
    # heuristic substring match ("image"/"audio" in the type string).
    content_type = str(getattr(result, "type", result.__class__.__name__)).lower()
    if hasattr(result, "text"):
        return [{"type": "text", "text": str(getattr(result, "text"))}]
    if "image" in content_type and hasattr(result, "data"):
        mime_type = str(getattr(result, "mimeType", None) or getattr(result, "mime_type", None) or "image/png")
        return [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{getattr(result, 'data')}"}}]
    if "audio" in content_type and hasattr(result, "data"):
        mime_type = str(getattr(result, "mimeType", None) or getattr(result, "mime_type", None) or "audio/wav")
        return [{"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{getattr(result, 'data')}"}}]
    # NOTE: Fallback — serialise the entire object as a text string.
    return [{"type": "text", "text": str(result)}]

# ── Utility helpers ────────────────────────────────────────────


def _json_text(value: Any) -> str:
    """Serialize a value to a compact JSON string."""

    import json

    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously, raising if already inside an event loop.

    NOTE: When a running event loop is detected, the coroutine is explicitly
    closed to avoid "coroutine was never awaited" warnings before raising.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    close = getattr(coro, "close", None)
    if callable(close):
        close()
    raise RuntimeError("Synchronous MCPClient helpers cannot run inside an active event loop.")


__all__ = ["MCPClient", "convert_result_to_openai_format"]
