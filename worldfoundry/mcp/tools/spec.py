"""Single registry of MCP tools: each payload function is declared once.

Payload modules decorate their public functions with :func:`mcp_tool`. The
decorator records the tool name, a short title, and its MCP behaviour hints;
:func:`bind_tools` turns every recorded payload into an ``async`` FastMCP tool
whose JSON schema comes from the payload's own type hints, so parameter
descriptions (``Annotated[..., Field(description=...)]``) and allowed values
(``Literal[...]``) reach the agent.

Binding rules:

- the ``context`` parameter is injected by the server and hidden from the schema;
- synchronous payloads run in a worker thread, so blocking HTTP, catalog, and
  file I/O never stalls the server event loop;
- every call is wrapped in the shared response envelope (:mod:`.responses`).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .responses import invoke_tool_async


@dataclass(frozen=True)
class ToolSpec:
    """MCP metadata for one payload function."""

    name: str
    title: str
    fn: Callable[..., Any]
    read_only: bool
    destructive: bool
    idempotent: bool
    open_world: bool

    @property
    def description(self) -> str:
        """Docstring summary, without the ``Args``/``Returns`` sections duplicated in the schema."""

        doc = inspect.cleandoc(self.fn.__doc__ or "")
        for marker in ("\nArgs:", "\nReturns:", "\nRaises:"):
            doc = doc.split(marker, 1)[0]
        return doc.strip()

    def annotations_dict(self) -> dict[str, bool]:
        return {
            "readOnlyHint": self.read_only,
            "destructiveHint": self.destructive,
            "idempotentHint": self.idempotent,
            "openWorldHint": self.open_world,
        }


TOOL_SPECS: dict[str, ToolSpec] = {}


def mcp_tool(
    name: str,
    *,
    title: str,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool | None = None,
    open_world: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a payload function as the MCP tool ``name``.

    Args:
        name: Tool name advertised to MCP clients.
        title: Short human-readable title.
        read_only: The tool does not change any state.
        destructive: The tool may cancel or remove existing work.
        idempotent: Repeating the call with the same arguments has no further effect
            (defaults to ``read_only``).
        open_world: The tool talks to a service outside this process (the Studio workspace).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in TOOL_SPECS:
            raise ValueError(f"duplicate MCP tool name: {name}")
        TOOL_SPECS[name] = ToolSpec(
            name=name,
            title=title,
            fn=fn,
            read_only=read_only,
            destructive=destructive,
            idempotent=read_only if idempotent is None else idempotent,
            open_world=open_world,
        )
        return fn

    return decorator


def _tool_callable(spec: ToolSpec, context: Any) -> Callable[..., Any]:
    """Build the async FastMCP-facing callable for one payload."""

    hints = typing.get_type_hints(spec.fn, include_extras=True)
    signature = inspect.signature(spec.fn)
    takes_context = "context" in signature.parameters
    parameters = [
        param.replace(annotation=hints.get(param.name, param.annotation), kind=inspect.Parameter.KEYWORD_ONLY)
        for param in signature.parameters.values()
        if param.name != "context"
    ]
    is_async = inspect.iscoroutinefunction(spec.fn)

    async def tool(**kwargs: Any) -> dict[str, Any]:
        if takes_context:
            kwargs["context"] = context
        if is_async:
            return await invoke_tool_async(spec.fn, **kwargs)
        return await invoke_tool_async(asyncio.to_thread, functools.partial(spec.fn, **kwargs))

    tool.__name__ = spec.name
    tool.__qualname__ = spec.name
    tool.__doc__ = spec.description
    tool.__signature__ = signature.replace(parameters=parameters, return_annotation=dict)  # ty: ignore[unresolved-attribute]
    tool.__annotations__ = {param.name: param.annotation for param in parameters} | {"return": dict}
    return tool


def bind_tools(mcp: Any, context: Any) -> dict[str, Callable[..., Any]]:
    """Register every declared tool on ``mcp`` and return the bound callables by name."""

    try:
        from mcp.types import ToolAnnotations
    except ImportError:  # payload modules stay importable without the MCP extra
        ToolAnnotations = None

    bound: dict[str, Callable[..., Any]] = {}
    for spec in TOOL_SPECS.values():
        tool = _tool_callable(spec, context)
        annotations = (
            ToolAnnotations(title=spec.title, **spec.annotations_dict()) if ToolAnnotations is not None else None
        )
        mcp.tool(name=spec.name, title=spec.title, description=spec.description, annotations=annotations)(tool)
        bound[spec.name] = tool
    return bound


__all__ = ["TOOL_SPECS", "ToolSpec", "bind_tools", "mcp_tool"]
