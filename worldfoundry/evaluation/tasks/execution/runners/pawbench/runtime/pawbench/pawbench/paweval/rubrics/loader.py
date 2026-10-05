"""Load PAWEval rubrics from package data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

RUBRIC_ROOT = Path(__file__).resolve().parent
RUBRIC_AXES = frozenset({"outcome", "trustworthiness"})
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def rubric_path(axis: str, scene_id: str) -> Path:
    axis = _validate_axis(axis)
    scene_id = _validate_scene_id(scene_id)
    path = (RUBRIC_ROOT / axis / f"{scene_id}.yaml").resolve()
    axis_root = (RUBRIC_ROOT / axis).resolve()
    if path.parent != axis_root:
        raise ValueError(f"PAWEval rubric path escapes rubric root: {axis}/{scene_id}")
    return path


def load_rubric(axis: str, scene_id: str) -> dict[str, Any]:
    return _load_rubric_cached(axis, scene_id)


@lru_cache(maxsize=None)
def _load_rubric_cached(axis: str, scene_id: str) -> dict[str, Any]:
    path = rubric_path(axis, scene_id)
    if not path.is_file():
        raise FileNotFoundError(f"PAWEval rubric not found for {axis}/{scene_id}: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PAWEval rubric must be a mapping: {path}")
    if payload.get("axis") != axis:
        raise ValueError(f"PAWEval rubric axis mismatch: expected {axis}, got {payload.get('axis')}")
    if payload.get("scene_id") != scene_id:
        raise ValueError(f"PAWEval rubric scene mismatch: expected {scene_id}, got {payload.get('scene_id')}")
    return payload


def _validate_axis(axis: str) -> str:
    if not isinstance(axis, str) or axis not in RUBRIC_AXES:
        raise ValueError(f"invalid PAWEval rubric axis: {axis!r}")
    return axis


def _validate_scene_id(scene_id: str) -> str:
    if not isinstance(scene_id, str) or not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"invalid PAWEval rubric scene_id: {scene_id!r}")
    return scene_id
