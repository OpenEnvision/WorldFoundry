"""Shared MCP tool response envelope and error taxonomy.

Every tool answers ``{"ok": true, ...payload}`` on success and
``{"ok": false, "error": ..., "error_type": ...}`` on failure (CM-25), so an
agent can branch on ``error_type`` instead of parsing messages:

- ``not_found``: an id, run, or file does not exist; the message names close matches when known.
- ``invalid_argument``: the arguments are inconsistent or out of range; fix them and retry.
- ``unavailable``: a dependent service (the Studio workspace) is not reachable; start it and retry.
- ``runtime``: the operation was accepted but failed while executing.
- ``internal``: an unexpected error inside the server.
"""

from __future__ import annotations

import difflib
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

_T = TypeVar("_T")


class ToolNotFoundError(LookupError):
    """Raised by payloads when a requested id, run, or file does not exist."""

    @classmethod
    def for_id(cls, kind: str, value: str, known: Iterable[str] = ()) -> ToolNotFoundError:
        """Build a not-found error that suggests the closest known ids."""

        close = difflib.get_close_matches(value, list(known), n=3, cutoff=0.6)
        hint = f"; did you mean: {', '.join(close)}" if close else ""
        return cls(f"{kind} not found: {value}{hint}")


class ToolUnavailableError(RuntimeError):
    """Raised when a service the tool depends on cannot be reached."""


def success_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ``ok=True`` when the payload does not already declare status."""

    if "ok" in payload:
        return payload
    return {"ok": True, **payload}


def error_payload(message: str, *, error_type: str = "invalid_argument") -> dict[str, Any]:
    """Return a structured MCP tool error response."""

    return {"ok": False, "error": message, "error_type": error_type}


def _normalized_error(exc: Exception) -> dict[str, Any]:
    """Map one exception onto the structured MCP error envelope."""

    if isinstance(exc, (ToolNotFoundError, FileNotFoundError)):
        return error_payload(str(exc), error_type="not_found")
    if isinstance(exc, KeyError):
        # ``str(KeyError("x"))`` is ``"'x'"``; report the bare key.
        return error_payload(f"not found: {exc.args[0] if exc.args else exc}", error_type="not_found")
    if isinstance(exc, ToolUnavailableError):
        return error_payload(str(exc), error_type="unavailable")
    if isinstance(exc, (ValueError, LookupError)):
        return error_payload(str(exc), error_type="invalid_argument")
    if isinstance(exc, RuntimeError):
        return error_payload(str(exc), error_type="runtime")
    # Anything else (OSError, TypeError, parser errors, ...) must still reach
    # the MCP client as a structured envelope rather than a bare exception.
    return error_payload(f"{type(exc).__name__}: {exc}", error_type="internal")


def _wrap(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return success_payload(result)
    return success_payload({"result": result})


def invoke_tool(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a payload builder and normalize success/error envelopes."""

    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        return _normalized_error(exc)
    return _wrap(result)


async def invoke_tool_async(
    fn: Callable[..., Awaitable[_T]],
    /,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async variant of :func:`invoke_tool`."""

    try:
        result = await fn(*args, **kwargs)
    except Exception as exc:
        return _normalized_error(exc)
    return _wrap(result)


__all__ = [
    "ToolNotFoundError",
    "ToolUnavailableError",
    "error_payload",
    "invoke_tool",
    "invoke_tool_async",
    "success_payload",
]
