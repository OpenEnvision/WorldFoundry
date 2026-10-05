"""Provides a mechanism for lazy loading of the `InspatioWorldSynthesis` class.

This module primarily exposes `InspatioWorldSynthesis` via its `__getattr__` hook,
ensuring that the class is only imported when explicitly accessed. This approach
can help improve startup performance by deferring imports until they are truly
needed.
"""

__all__ = ["InspatioWorldSynthesis", "InspatioWorldV15Synthesis"]


def __getattr__(name):
    """Lazily loads the `InspatioWorldSynthesis` class when it is accessed as an attribute of this module.

    This function implements the module-level `__getattr__` hook, which is invoked
    when an attribute lookup fails through the standard methods. It's used here
    to defer the import of `InspatioWorldSynthesis` until it's actually needed,
    improving startup performance for applications that might not always use it.

    Args:
        name (str): The name of the attribute being accessed on the module.

    Returns:
        Type[InspatioWorldSynthesis]: The `InspatioWorldSynthesis` class if `name` is 'InspatioWorldSynthesis'.

    Raises:
        AttributeError: If the requested `name` is not 'InspatioWorldSynthesis'.
    """
    if name == "InspatioWorldV15Synthesis":
        from .inspatio_world_v15_synthesis import InspatioWorldV15Synthesis

        return InspatioWorldV15Synthesis
    if name == "InspatioWorldSynthesis":
        # Import the InspatioWorldSynthesis class only when it's explicitly requested
        from .inspatio_world_synthesis import InspatioWorldSynthesis

        return InspatioWorldSynthesis
    raise AttributeError(name)
