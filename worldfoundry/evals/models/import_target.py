"""Import callables by ``module:attribute`` string (colon syntax, dotted attribute path).

Three resolution strategies are provided:

* :func:`import_dotted_attr` — colon-separated ``module:nested.attr`` targets
  used by dynamic runner configuration.
* :func:`import_object_path` — fully-dotted ``package.module.Class`` paths
  used by registry metadata.
* :func:`load_attr` — the shared low-level helper that walks a dot-separated
  attribute chain on an already-imported module.
"""

from __future__ import annotations

import importlib
from typing import Any

# ── Low-level attribute walker ────────────────────────────────────────


def load_attr(module_name: str, dotted_attr: str) -> Any:
    """Import ``module_name`` and resolve ``dotted_attr`` on it step by step.

    Args:
        module_name: Fully-qualified Python module name to import.
        dotted_attr: Dot-separated attribute path relative to that module
            (e.g. ``"sub.Class"``).

    Returns:
        The resolved attribute object at the end of the dotted path.
    """
    module = importlib.import_module(module_name)
    obj = module
    for part in dotted_attr.split("."):
        obj = getattr(obj, part)
    return obj


# ── Colon-syntax resolver (runner targets) ────────────────────────────


def import_dotted_attr(target: str) -> Any:
    """Load ``module:name`` or ``module:nested.attr`` exactly like dynamic runner targets.

    Args:
        target: Import path using a single colon between module and attribute
            path (e.g. ``"mymodule:MyClass"``).

    Returns:
        The resolved attribute object.

    Raises:
        ValueError: If ``target`` is missing the colon separator, module
            name, or attribute name.
    """
    module_name, separator, attr_name = target.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"runner target must use 'module:attribute' syntax: {target!r}")
    return load_attr(module_name, attr_name)


# ── Dotted-path resolver (registry metadata) ──────────────────────────


def import_object_path(dotted_path: str) -> Any:
    """Load ``package.module.attribute`` paths used by registry metadata.

    Args:
        dotted_path: Fully-qualified dotted import path ending in an
            attribute name (e.g. ``"pkg.mod.MyClass"``).

    Returns:
        The resolved attribute object.

    Raises:
        ValueError: If ``dotted_path`` lacks a dot separator, module part,
            or attribute part.
    """
    module_name, separator, attr_name = dotted_path.rpartition(".")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    return load_attr(module_name, attr_name)
