"""Load a locally rebuilt PyTorch3D extension before importing its renderers."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

_DEFAULT_OVERLAY = (
    Path(__file__).resolve().parents[4]
    / "artifacts"
    / "runtime_extensions"
    / "pytorch3d"
    / "pytorch3d"
)


def _extension_dir() -> Path:
    configured = os.environ.get("WORLDFOUNDRY_PYTORCH3D_EXTENSION_DIR")
    if not configured:
        return _DEFAULT_OVERLAY
    candidate = Path(configured).expanduser().resolve()
    return candidate.parent if candidate.is_file() else candidate


def configure_pytorch3d_extension() -> Path | None:
    """Prepend a compatible ``pytorch3d._C`` overlay when one is available.

    The unified environment can contain the Python portion of PyTorch3D while its
    compiled extension was built against a different Torch ABI.  A rebuilt
    extension is stored outside site-packages and added to the package search path
    before ``pytorch3d.renderer`` imports ``pytorch3d._C``.
    """

    extension_dir = _extension_dir()
    if not extension_dir.is_dir() or not any(extension_dir.glob("_C*.so")):
        return None

    pytorch3d = importlib.import_module("pytorch3d")
    package_path = str(extension_dir)
    if package_path not in pytorch3d.__path__:
        pytorch3d.__path__.insert(0, package_path)
        importlib.invalidate_caches()
    return extension_dir

