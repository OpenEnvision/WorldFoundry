"""Synthesis facade for the Matrix-Game-1 runtime."""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeFacadeSynthesis
from .matrix_game_1_runtime import MatrixGame1Runtime


class MatrixGame1Synthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the Matrix-Game-1 runtime."""

    RUNTIME_CLS = MatrixGame1Runtime
    MODEL_ID = MatrixGame1Runtime.MODEL_ID
    DISPLAY_NAME = MatrixGame1Runtime.DISPLAY_NAME

    def preflight(self) -> dict[str, Any]:
        """Report asset readiness under the framework-wide key names."""
        preflight = self.runtime.preflight()
        preflight["missing_assets"] = preflight["missing_checkpoint_files"]
        preflight["missing_runtime"] = preflight["missing_runtime_files"]
        return preflight


__all__ = ["MatrixGame1Synthesis"]
