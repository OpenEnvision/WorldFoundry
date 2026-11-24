"""
A lazy-loading entry point for the `SolarisSynthesis` class.

This module provides a `__getattr__` hook to defer the import of `SolarisSynthesis`
from its actual definition in `.solaris_synthesis` until it is first accessed.
This pattern can help optimize initial import times for packages where
`SolarisSynthesis` might have significant dependencies.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["SolarisSynthesis"]


def __getattr__(name: str):
    """
    Implements module-level lazy loading for the `SolarisSynthesis` class.

    This function is automatically called when an attribute `name` is accessed
    on this module that cannot be found through normal means. It specifically
    handles the lazy loading of `SolarisSynthesis`.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The `SolarisSynthesis` object (likely a class).

    Raises:
        AttributeError: If an attribute other than "SolarisSynthesis" is accessed.
    """
    if name != "SolarisSynthesis":
        raise AttributeError(name)
    # Dynamically import the module containing the actual SolarisSynthesis definition.
    # The leading dot indicates a relative import within the package.
    module = import_module(".solaris_synthesis", __name__)
    # Retrieve the SolarisSynthesis object (e.g., class) from the imported module.
    value = module.SolarisSynthesis
    # Cache the imported value in the module's global namespace to avoid
    # re-triggering __getattr__ on subsequent accesses, improving performance.
    globals()[name] = value
    return value
