"""
This module provides lazy loading for AC3DSynthesis and AC3DRuntime classes.

It uses the __getattr__ mechanism to import these classes only when they are first accessed,
optimizing startup time and resource usage for applications that may only use a subset
of the library's components.
"""

__all__ = ["AC3DRuntime", "AC3DSynthesis"]


def __getattr__(name):
    """
    Implements lazy loading for module attributes.

    This special module-level function is called when an attribute is accessed
    on the module that has not been explicitly loaded yet. It dynamically
    imports and returns AC3DSynthesis or AC3DRuntime upon their first access,
    avoiding unnecessary imports at module startup.

    Args:
        name (str): The name of the attribute being accessed.

    Returns:
        type: The requested class (AC3DSynthesis or AC3DRuntime).

    Raises:
        AttributeError: If the requested attribute is not found within the
                        allowed lazy-loaded names.
    """
    if name == "AC3DSynthesis":
        # Lazily import AC3DSynthesis only when it's first accessed.
        from .ac3d_synthesis import AC3DSynthesis

        return AC3DSynthesis
    if name == "AC3DRuntime":
        # Lazily import AC3DRuntime only when it's first accessed.
        from .runtime import AC3DRuntime

        return AC3DRuntime
    # Raise an AttributeError for any other requested names to indicate they don't exist.
    raise AttributeError(name)
