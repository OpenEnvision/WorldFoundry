"""Resolve Matrix-Game 3.5 inference recipes from WorldFoundry package data."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path

MATRIX_GAME_35_CONFIG_ROOT = resolve_data_path(
    "models",
    "runtime",
    "configs",
    "matrix_game_3p5",
)

_CONFIG_FILENAMES = {
    "common": "infer_common.yaml",
    "first": "infer_first_person.yaml",
    "first_person": "infer_first_person.yaml",
    "third": "infer_third_person.yaml",
    "third_person": "infer_third_person.yaml",
}


def matrix_game_35_infer_config_path(person: str) -> Path:
    """Return the package-data path for one immutable inference profile."""

    key = str(person).strip().lower().replace("-", "_")
    try:
        filename = _CONFIG_FILENAMES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Matrix-Game 3.5 config profile {person!r}; expected one of {sorted(_CONFIG_FILENAMES)}"
        ) from exc
    return MATRIX_GAME_35_CONFIG_ROOT / filename


__all__ = ["MATRIX_GAME_35_CONFIG_ROOT", "matrix_game_35_infer_config_path"]
