"""Immutable model identities for the released Matrix-Game 3.5 base models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class MatrixGame35ModelSpec:
    """One independently addressable Matrix-Game 3.5 checkpoint recipe."""

    model_id: str
    display_name: str
    person: Literal["first", "third"]
    checkpoint_filename: str
    supports_subject_refs: bool

    @property
    def output_namespace(self) -> str:
        """Return the directory name used by the official inference entrypoint."""

        return f"{self.person}_person"


_MODEL_SPECS = {
    "matrix-game-3.5-first-person": MatrixGame35ModelSpec(
        model_id="matrix-game-3.5-first-person",
        display_name="Matrix-Game 3.5 First-Person Base",
        person="first",
        checkpoint_filename="first-person.safetensors",
        supports_subject_refs=False,
    ),
    "matrix-game-3.5-third-person": MatrixGame35ModelSpec(
        model_id="matrix-game-3.5-third-person",
        display_name="Matrix-Game 3.5 Third-Person Base",
        person="third",
        checkpoint_filename="third-person.safetensors",
        supports_subject_refs=True,
    ),
}

MATRIX_GAME_35_MODEL_SPECS: Mapping[str, MatrixGame35ModelSpec] = MappingProxyType(_MODEL_SPECS)


def get_matrix_game_35_model_spec(model_id: str) -> MatrixGame35ModelSpec:
    """Resolve a public model ID without accepting a mutable person-mode switch."""

    try:
        return MATRIX_GAME_35_MODEL_SPECS[str(model_id)]
    except KeyError as exc:
        supported = ", ".join(MATRIX_GAME_35_MODEL_SPECS)
        raise KeyError(f"Unknown Matrix-Game 3.5 model {model_id!r}; choose one of: {supported}") from exc


__all__ = [
    "MATRIX_GAME_35_MODEL_SPECS",
    "MatrixGame35ModelSpec",
    "get_matrix_game_35_model_spec",
]
