"""Bundled benchmark asset path helpers.

Official prompts, rubrics, and manifests ship under
``worldfoundry/data/benchmarks/assets/<benchmark_id>/``. Runners resolve bundled
assets by default; ``WORLDFOUNDRY_*`` env vars and explicit kwargs override only
when the caller sets them.
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path
from worldfoundry.evaluation.utils import worldfoundry_data_path


def bundled_benchmark_assets_root(benchmark_id: str) -> Path:
    """Return the bundled asset directory for ``benchmark_id``."""
    return worldfoundry_data_path("benchmarks", "assets", benchmark_id)


def bundled_benchmark_asset(benchmark_id: str, *relative: str | Path) -> Path:
    """Return a path under the bundled asset tree for ``benchmark_id``."""
    return bundled_benchmark_assets_root(benchmark_id).joinpath(*relative)


def first_existing_dir(*candidates: Path | None) -> Path | None:
    """Return the first candidate that is an existing directory."""
    return _first_existing(candidates, require="dir")


def first_existing_file(*candidates: Path | None) -> Path | None:
    """Return the first candidate that is an existing file."""
    return _first_existing(candidates, require="file")


def _first_existing(candidates: tuple[Path | None, ...], *, require: str) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if require == "dir" and resolved.is_dir():
            return resolved
        if require == "file" and resolved.is_file():
            return resolved
    return None


__all__ = [
    "bundled_benchmark_asset",
    "bundled_benchmark_assets_root",
    "first_existing_dir",
    "first_existing_file",
    "resolve_env_path",
]
