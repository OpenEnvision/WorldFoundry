# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Load Torch's shared libraries before importing source-built render extensions."""

from __future__ import annotations

import ctypes
from pathlib import Path

_LOADED = False


def ensure_torch_libs_loaded() -> None:
    """Expose Torch's bundled symbols to PyTorch3D without changing library search paths."""
    global _LOADED
    if _LOADED:
        return

    import torch

    torch_lib_dir = Path(torch.__file__).parent / "lib"
    for name in (
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
    ):
        path = torch_lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _LOADED = True
