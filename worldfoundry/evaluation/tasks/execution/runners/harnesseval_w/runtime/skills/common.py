"""Small shared contracts for HarnessEval skills."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..aggregate import numeric


class SkillBackendIdentity(Protocol):
    backend_id: str
    version: str
    config_digest: str
    execution_mode: str


@dataclass(frozen=True)
class MetricCacheContract:
    """Declares which cached values a skill can consume safely."""

    version: str
    source_skill_ids: tuple[str, ...]
    required_metrics: tuple[str, ...]
    accepts_source_skill_score: bool
    compatibility_fields: tuple[str, ...]


def normalized_metrics(
    metrics: Mapping[str, Any],
    required: tuple[str, ...],
) -> tuple[dict[str, float], list[str], list[str]]:
    """Read required finite [0, 1] metrics without imputing missing values."""

    values: dict[str, float] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for key in required:
        raw = metrics.get(key)
        if raw is None:
            missing.append(key)
            continue
        value = None if isinstance(raw, bool) else numeric(raw)
        if value is None or not 0.0 <= value <= 1.0:
            invalid.append(key)
            continue
        values[key] = value
    return values, missing, invalid


def _video_key(video_path: Path) -> tuple[str, str, str]:
    video = video_path.resolve()
    return video.parent.parent.name, video.parent.parent.parent.name, video.parent.name


def _find_cache_root(video_path: Path) -> Path:
    configured = os.environ.get("HARNESSEVAL_CACHE_ROOT")
    if configured:
        return Path(configured).resolve()

    raise ValueError("pass cache_root or set HARNESSEVAL_CACHE_ROOT")


def skill_cache_path(
    video_path: Path,
    skill_directory: str,
    cache_root: Path | None = None,
) -> Path:
    model_id, family, case_id = _video_key(video_path)
    root = cache_root.resolve() if cache_root is not None else _find_cache_root(video_path)
    return root / "skills" / skill_directory / model_id / family / f"{case_id}.json"
