"""
This module serves as the main entry point and exposes core components of the Lyra2 project.
It dynamically loads submodules (`.runtime` and `.synthesis`) on demand using `__getattr__`
to avoid circular dependencies and improve startup performance. It also defines common
constants and the root path for the Lyra2 source package.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parent / "lyra_2"


__all__ = [
    "DEFAULT_DA3_MODEL_NAME",
    "DEFAULT_WEIGHT_DTYPE",
    "SOURCE_PACKAGE_ROOT",
    "Lyra2Runtime",
    "Lyra2Synthesis",
    "load_runtime",
]


def __getattr__(name: str) -> Any:
    """
    Implements dynamic attribute loading for the module.

    When an attribute not directly defined in this module is accessed,
    this function intercepts the request, imports the relevant submodule,
    retrieves the attribute from it, caches it in the module's globals,
    and returns it. This mechanism helps in managing module dependencies
    and lazy loading, preventing circular imports and improving startup time.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The value of the dynamically loaded attribute.

    Raises:
        AttributeError: If the requested attribute is not found in any
                        of the dynamically loaded submodules.
    """
    # Dynamically load attributes related to the runtime module.
    if name in {"DEFAULT_DA3_MODEL_NAME", "DEFAULT_WEIGHT_DTYPE", "Lyra2Runtime", "load_runtime"}:
        module = import_module(".runtime", __name__)
        value = getattr(module, name)
        # Cache the loaded attribute in the module's global namespace to avoid re-importing.
        globals()[name] = value
        return value
    # Dynamically load attributes related to the synthesis module.
    if name == "Lyra2Synthesis":
        module = import_module(".synthesis", __name__)
        value = getattr(module, name)
        # Cache the loaded attribute in the module's global namespace to avoid re-importing.
        globals()[name] = value
        return value
    raise AttributeError(name)
