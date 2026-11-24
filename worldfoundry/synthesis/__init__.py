from __future__ import annotations

"""
This module serves as a central entry point for various synthesis components
within the WorldFoundry pipelines. It implements a lazy-loading mechanism for its
public API, allowing components like `ActionModelSynthesis` to be accessed directly
from the `synthesis` package without being eagerly imported at startup.
This approach improves application startup time and helps manage potential
circular dependencies by only importing modules when their contents are first
accessed.
"""

from importlib import import_module  # noqa: E402 - Follows a module-level setup call, as upstream.

# _EXPORTS maps names intended for direct import from this module (e.g., `from synthesis import ActionModelSynthesis`)
# to their relative module paths where they are actually defined. This dictionary drives the lazy loading.
_EXPORTS = {
    "ActionModelSynthesis": ".action_generation",
}

# __all__ defines the public API of this module when 'from synthesis import *' is used.
# It ensures that only specified names, derived from _EXPORTS, are made available and discoverable.
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """
    Implements lazy loading for top-level attributes of this module.

    When an attribute `name` is accessed that is listed in `_EXPORTS` but
    has not yet been loaded, this method dynamically imports the
    corresponding submodule and retrieves the attribute from it.
    Once loaded, the attribute is cached in the module's `globals()`
    for direct access on subsequent requests, avoiding repeated imports
    and `__getattr__` calls for that specific attribute.

    Args:
        name (str): The name of the attribute being accessed (e.g., 'ActionModelSynthesis').

    Returns:
        Any: The value of the requested attribute from the imported submodule.

    Raises:
        AttributeError: If `name` is not found in the recognized `_EXPORTS` dictionary,
                        indicating it's not a known lazily loadable attribute.
    """
    # Check if the requested attribute name is one of our lazily loadable exports.
    if name not in _EXPORTS:
        raise AttributeError(name)
    # Dynamically import the module specified by the export mapping.
    # The second argument (__name__) specifies that _EXPORTS[name] is a relative import path.
    module = import_module(_EXPORTS[name], __name__)
    # Retrieve the actual object (class, function, etc.) from the newly imported module.
    value = getattr(module, name)
    # Cache the retrieved value in the current module's globals.
    # This makes future accesses to `name` bypass __getattr__ and directly retrieve the cached value,
    # optimizing subsequent lookups.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """
    Customizes the behavior of the built-in `dir()` function for this module.

    It returns a combined list of all currently defined global names in this
    module, along with all names specified in `__all__` (which represent
    the lazily loadable components). This ensures that `dir()` correctly
    reflects all attributes that can be accessed from this module,
    including those not yet loaded via `__getattr__`.

    Returns:
        list[str]: A sorted list of attribute names available in this module.
    """
    # Combine the names already in the module's globals with the names from __all__.
    # Using a set eliminates duplicates before converting back to a list and sorting,
    # providing a comprehensive view of available attributes.
    return sorted({*globals(), *__all__})
