"""In-tree Matrix-Game 3.5 inference runtime."""

from __future__ import annotations

from .specs import (
    MATRIX_GAME_35_MODEL_SPECS,
    MatrixGame35ModelSpec,
    get_matrix_game_35_model_spec,
)


def __getattr__(name: str):
    if name in {
        "MatrixGame35Assets",
        "MatrixGame35Runtime",
        "MatrixGame35RuntimePlan",
        "inspect_camera_npz",
    }:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)


__all__ = [
    "MATRIX_GAME_35_MODEL_SPECS",
    "MatrixGame35Assets",
    "MatrixGame35ModelSpec",
    "MatrixGame35Runtime",
    "MatrixGame35RuntimePlan",
    "get_matrix_game_35_model_spec",
    "inspect_camera_npz",
]
