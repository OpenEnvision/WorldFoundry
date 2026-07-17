"""Discovery for checked-in model x benchmark reproduction recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.evaluation.utils import BENCHMARKS_DATA_ROOT

DEFAULT_REPRODUCTION_PROFILE_ROOT = BENCHMARKS_DATA_ROOT / "reproduction_profiles"


def _profile_summary(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"reproduction profile must be a mapping: {path}")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, Mapping) or not benchmark.get("id"):
        raise ValueError(f"reproduction profile requires benchmark.id: {path}")
    return {
        "id": str(payload.get("id") or path.stem),
        "benchmark_id": str(benchmark["id"]),
        "default_for_benchmark": bool(payload.get("default_for_benchmark", False)),
        "path": path,
    }


def reproduction_profile_summaries(
    root: str | Path = DEFAULT_REPRODUCTION_PROFILE_ROOT,
) -> tuple[dict[str, Any], ...]:
    """Return validated summaries for all checked-in reproduction profiles."""

    profile_root = Path(root).expanduser().resolve()
    if not profile_root.is_dir():
        return ()
    summaries = [_profile_summary(path) for path in sorted(profile_root.rglob("*.yaml"))]
    ids = [item["id"] for item in summaries]
    duplicates = sorted({profile_id for profile_id in ids if ids.count(profile_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate reproduction profile ids: {', '.join(duplicates)}")
    return tuple(summaries)


def resolve_reproduction_profile(
    *,
    profile_id: str | None = None,
    benchmark_id: str | None = None,
    root: str | Path = DEFAULT_REPRODUCTION_PROFILE_ROOT,
) -> Path:
    """Resolve an explicit profile or a benchmark's single declared default."""

    if bool(profile_id) == bool(benchmark_id):
        raise ValueError("select exactly one of profile_id or benchmark_id")
    summaries = reproduction_profile_summaries(root)
    if profile_id:
        matches = [item for item in summaries if item["id"] == profile_id]
        selector = f"profile {profile_id!r}"
    else:
        matches = [
            item
            for item in summaries
            if item["benchmark_id"] == benchmark_id and item["default_for_benchmark"]
        ]
        selector = f"default profile for benchmark {benchmark_id!r}"
    if not matches:
        raise KeyError(f"no reproduction {selector}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous reproduction {selector}")
    return matches[0]["path"]


__all__ = [
    "DEFAULT_REPRODUCTION_PROFILE_ROOT",
    "reproduction_profile_summaries",
    "resolve_reproduction_profile",
]
