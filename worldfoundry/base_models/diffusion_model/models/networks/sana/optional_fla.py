"""Narrow Flash Linear Attention imports for inference-only SANA-WM profiles.

Recent FLA releases expose their complete layers and models API from the
package root. Importing one convolution through that public surface therefore
registers every attention operator, even though SANA-WM only needs
``ShortConvolution``. This helper creates ordinary package namespaces and
loads the concrete convolution module directly. The actual FLA implementation
and Triton kernels remain unchanged.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _package_namespace(name: str, path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader=None,
        is_package=True,
    )
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    return module


def import_short_convolution() -> type[Any]:
    """Load FLA's ``ShortConvolution`` without its unrelated public API."""

    if "fla" in sys.modules:
        loaded = sys.modules.get("fla.modules.conv.short_conv")
        if loaded is not None:
            return loaded.ShortConvolution
        from fla.modules import ShortConvolution

        return ShortConvolution

    spec = importlib.util.find_spec("fla")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if not locations:
        raise ImportError("flash-linear-attention (fla) is required by SANA-WM.")
    root = Path(locations[0])
    package = _package_namespace("fla", root)
    try:
        package.__version__ = importlib.metadata.version("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError:
        package.__version__ = "0.0.0"
    _package_namespace("fla.modules", root / "modules")
    _package_namespace("fla.modules.conv", root / "modules" / "conv")
    # The convolution implementation imports only the context-parallel subset
    # of FLA ops. Keeping ``fla.ops`` as a namespace prevents its broad
    # convenience __init__ from registering every unrelated attention kernel.
    _package_namespace("fla.ops", root / "ops")
    module = importlib.import_module("fla.modules.conv.short_conv")
    setattr(sys.modules["fla.modules"], "ShortConvolution", module.ShortConvolution)
    return module.ShortConvolution


__all__ = ["import_short_convolution"]
