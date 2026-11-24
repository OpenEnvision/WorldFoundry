"""
This package serves as the entry point for the `warp_as_history` synthesis variant,
providing access to its core components and utilities.

It re-exports key definitions such as aliases, variant enumerations, and helper functions
from its submodules. It also implements a lazy loading mechanism for the
`WarpAsHistorySynthesis` class to avoid circular imports or unnecessary initial imports.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.warp_as_history.variants import (
    WARP_AS_HISTORY_ALIASES,
    WARP_AS_HISTORY_VARIANTS,
    WarpAsHistoryVariant,
    get_warp_as_history_variant,
    runtime_root,
)

__all__ = [
    "WARP_AS_HISTORY_ALIASES",
    "WARP_AS_HISTORY_VARIANTS",
    "WarpAsHistorySynthesis",
    "WarpAsHistoryVariant",
    "get_warp_as_history_variant",
    "runtime_root",
]


def __getattr__(name: str):
    """
    Implements a lazy loading mechanism for module attributes.

    This function is called when an attribute of the module is accessed that
    has not been directly defined. It specifically handles the lazy import
    of `WarpAsHistorySynthesis` to prevent potential circular import issues
    or to defer the import cost until the class is actually used.

    Args:
        name: The name of the attribute being accessed on the module.

    Returns:
        The imported object if `name` matches a known lazily-loaded attribute.

    Raises:
        AttributeError: If the requested attribute `name` is not found
                        among the lazily-loaded attributes or the module's
                        directly defined attributes.
    """
    # Lazily import WarpAsHistorySynthesis only when it's explicitly requested.
    # This prevents potential circular dependencies and defers import costs.
    if name == "WarpAsHistorySynthesis":
        from .warp_as_history_synthesis import WarpAsHistorySynthesis

        return WarpAsHistorySynthesis
    # Raise a standard AttributeError if the requested name is not a recognized
    # lazy-loadable attribute, mimicking Python's default behavior for missing attributes.
    raise AttributeError(name)
