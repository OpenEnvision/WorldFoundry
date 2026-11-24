from __future__ import annotations

from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent


def __getattr__(name: str):
    if name == "MatrixGame3Runtime":
        from .worldfoundry_runtime import MatrixGame3Runtime

        return MatrixGame3Runtime
    raise AttributeError(name)


__all__ = ["RUNTIME_ROOT", "MatrixGame3Runtime"]
