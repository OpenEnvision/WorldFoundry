"""Shared query-matching helpers for MCP catalog payloads."""

from __future__ import annotations

from fnmatch import fnmatchcase


def matches_query(query: str, *values: object) -> bool:
    """Check whether any *values* match *query* using glob or substring semantics.

    When ``query`` contains glob characters (``*``, ``?``, ``[``, ``]``),
    :func:`fnmatch.fnmatchcase` is used; otherwise a plain substring check
    is performed.  All comparisons are case-insensitive.
    """

    needle = query.casefold()
    glob_query = any(char in needle for char in "*?[]")
    return any(
        fnmatchcase(str(value).casefold(), needle) if glob_query else needle in str(value).casefold()
        for value in values
        if value
    )


__all__ = ["matches_query"]
