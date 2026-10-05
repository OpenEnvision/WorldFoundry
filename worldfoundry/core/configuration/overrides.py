"""Lightweight command-line configuration formatting helpers.

Hydra overrides are strings (``model.hidden=64``, ``foo=null``).
:func:`format_override_value` turns a Python value into that spelling:
``None`` → ``null``, bools lowercased, lists as ``[a,b]``, empty
string quoted. Used when a Studio / CLI layer forwards structured
kwargs onto a Cosmos Hydra compose command without importing Hydra
itself.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Hydra override spelling — None → null; empty string stays quoted
# ──────────────────────────────────────────────────────────────────────────


def format_override_value(value) -> str:
    """Format a Python value for a Hydra command-line override."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    if value == "":
        return "''"
    return str(value)


__all__ = ["format_override_value"]
