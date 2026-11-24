"""Adapters for integrating Yume runtime models with WorldFoundry.

This module provides an interface between the WorldFoundry framework and the
upstream Yume model code, which resides in the `yume_runtime` package.
It facilitates loading, distributed setup, sampling, and post-processing
of Yume models within WorldFoundry, allowing synthesis modules to remain
focused on their core logic.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def runtime_root() -> Path:
    """Returns the absolute path to the base directory of the Yume runtime modules."""
    return Path(__file__).resolve().parent / "yume_runtime"


RUNTIME_ROOT = runtime_root()
# Maps a requested runtime class name (e.g., "YumeRuntime") to its relative module path.
_RUNTIME_MODULES = {
    "YumeRuntime": ".yume_runtime.yume.worldfoundry_runtime",
    "Yume1p5Runtime": ".yume_runtime.yume_1p5.worldfoundry_runtime",
}


def __getattr__(name: str):
    """Dynamically imports Yume runtime classes from their respective modules.

    This function implements a lazy import mechanism, allowing Yume runtime classes
    like `YumeRuntime` and `Yume1p5Runtime` to be accessed directly from this
    package without explicit imports.

    Args:
        name: The name of the attribute (e.g., class name) to retrieve.

    Returns:
        The dynamically imported attribute, typically a class.

    Raises:
        AttributeError: If the requested name is not found in the predefined
                        runtime modules mapping.
    """
    module_name = _RUNTIME_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)

    # Dynamically import the specified module relative to the current package.
    module = import_module(module_name, __package__)
    # Retrieve the attribute (e.g., class) by its name from the imported module.
    value = getattr(module, name)
    # Cache the imported value in the module's globals to avoid re-importing
    # on subsequent accesses, making it behave like a standard import.
    globals()[name] = value
    return value


# Defines the public API of this module when `*` is used in an import statement.
__all__ = [
    "RUNTIME_ROOT",
    "Yume1p5Runtime",  # noqa: F822 - Provided by the module-level __getattr__.
    "YumeRuntime",  # noqa: F822 - Provided by the module-level __getattr__.
    "runtime_root",
]
