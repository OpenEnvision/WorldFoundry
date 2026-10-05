"""On-demand submodule import so optional deps stay off the import path.

Responsibility
    Stand in for a package whose heavy children (cv2, decord, …) must not load
    until an attribute is actually requested.

Boundaries
    Not a general proxy and not a replacement for :mod:`worldfoundry.core.utils`
    ``__getattr__``. This wrapper *is* a :class:`types.ModuleType` so
    ``sys.modules[name]`` can hold it. Pickle reconstructs from
    ``(name, file, import_structure)`` only — extra objects are not restored.

Public surface
    :class:`LazyModule` (alias ``_LazyModule`` for HuggingFace-style callers).
"""

from __future__ import annotations

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any


class LazyModule(ModuleType):
    """Module wrapper that imports submodules only when attributes are requested."""

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        module_spec: Any | None = None,
        extra_objects: dict[str, Any] | None = None,
    ) -> None:
        """Build the name tables without importing any child module.

        ``import_structure`` maps submodule name → exported symbols. Those
        symbols are advertised in ``__all__`` / ``__dir__`` before the child
        exists. ``extra_objects`` are returned as-is (already-materialized
        constants); they are not pickled.
        """
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> list[str]:
        """Advertise every lazy export so completion tools see names before import."""
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        """Resolve ``name`` from extras, a submodule, or a symbol on a submodule.

        The result is written back with :func:`setattr` so later lookups skip
        ``import_module``. Failure: :exc:`AttributeError` when ``name`` is not
        in the tables.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        """Import ``.<module_name>`` relative to this wrapper's package name."""
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self) -> tuple[type["LazyModule"], tuple[str, str, dict[str, list[str]]]]:
        """Reconstruct from name / file / structure; extras are dropped on pickle."""
        return self.__class__, (self._name, self.__file__, self._import_structure)


_LazyModule = LazyModule


__all__ = ["LazyModule", "_LazyModule"]
