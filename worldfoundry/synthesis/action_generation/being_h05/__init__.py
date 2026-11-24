"""Inference-only Being-H0.5 integration."""

from __future__ import annotations

__all__ = ["BeingH05Synthesis"]


def __getattr__(name: str):
    """Lazily load attributes from submodules to improve startup performance.

    This function is part of Python's module attribute access mechanism.
    It intercepts attribute lookups at the module level. If the requested
    attribute is `BeingH05Synthesis`, it imports and returns it from the
    `being_h05_synthesis` submodule only when explicitly accessed.
    For any other unknown attributes, it raises an `AttributeError`.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The requested attribute if it can be lazily loaded.

    Raises:
        AttributeError: If the requested attribute is not 'BeingH05Synthesis'.
    """
    if name == "BeingH05Synthesis":
        # Lazily import BeingH05Synthesis only when it's explicitly requested.
        # This reduces initial import time if BeingH05Synthesis is not always used.
        from .being_h05_synthesis import BeingH05Synthesis

        return BeingH05Synthesis
    raise AttributeError(name)
