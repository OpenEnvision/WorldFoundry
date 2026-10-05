"""
This module serves as a lazy loader for various WAN (Wide Area Network) synthesis functionalities.

It dynamically imports specific synthesis classes or functions from submodules
only when they are first accessed, improving startup performance
by avoiding unnecessary imports. The `__all__` variable defines the public API
exported by this package, while `_EXPORTS` maps these public names to their
respective submodule paths.
"""
from __future__ import annotations

from importlib import import_module

# Defines the public API of this package, making these names accessible when
# using 'from package import *'.
__all__ = [
    "Wan2p5Synthesis",
    "Wan2p6Synthesis",
    "Wan2p7Synthesis",
]

# A mapping from the public API names (as defined in __all__) to their
# respective submodule names within this package.
_EXPORTS = {
    "Wan2p5Synthesis": "wan_2p5_synthesis",
    "Wan2p6Synthesis": "wan_2p6_synthesis",
    "Wan2p7Synthesis": "wan_2p7_synthesis",
}


def __getattr__(name: str):
    """
    Dynamically loads and returns an attribute (class or function) from a submodule.

    This function implements PEP 562 for module-level __getattr__, allowing for
    lazy loading of synthesis components. When an attribute listed in `_EXPORTS`
    is accessed, its corresponding submodule is imported, and the attribute
    is retrieved from that module.

    Args:
        name: The name of the attribute to retrieve.

    Returns:
        The dynamically imported attribute (e.g., a class or function).

    Raises:
        AttributeError: If the requested attribute is not found in `_EXPORTS`.
    """
    if name in _EXPORTS:
        # Construct the full module path using the current module's name
        # and the submodule name from _EXPORTS.
        module = import_module(f"{__name__}.{_EXPORTS[name]}")
        # Retrieve the actual object (class or function) from the imported module.
        return getattr(module, name)
    # If the requested name is not in the _EXPORTS map, it means the attribute
    # truly does not exist in this module's public interface.
    raise AttributeError(name)
