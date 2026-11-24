"""
This module serves as the primary entry point for the DreamZero library, providing
access to core components like the DreamZeroSynthesis class.

It implements a lazy loading mechanism for the DreamZeroSynthesis class to
optimize initial module load times and manage potential import complexities.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DreamZeroSynthesis", "runtime_root"]


def runtime_root() -> Path:
    """
    Determines and returns the absolute path to the DreamZero runtime directory.

    The inference runtime is integrated directly under this package.

    Returns:
        Path: The absolute in-tree DreamZero package directory.
    """
    return Path(__file__).resolve().parent


def __getattr__(name: str):
    """
    Implements lazy loading for the DreamZeroSynthesis class.

    This function is called when an attribute is accessed on the module that is not
    directly defined. It specifically handles the dynamic import of `DreamZeroSynthesis`
    from a submodule to prevent circular dependencies or improve initial import speed.

    Args:
        name (str): The name of the attribute being accessed.

    Returns:
        Any: The dynamically imported attribute if it matches a known lazy-loaded name.

    Raises:
        AttributeError: If the requested attribute is not found or not configured
                        for lazy loading.
    """
    if name == "DreamZeroSynthesis":
        # Dynamically import DreamZeroSynthesis only when it's explicitly requested.
        # This helps in avoiding circular imports and improves initial module load performance.
        from .dreamzero_synthesis import DreamZeroSynthesis

        return DreamZeroSynthesis
    raise AttributeError(name)
