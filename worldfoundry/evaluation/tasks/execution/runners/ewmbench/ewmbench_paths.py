"""EWMBench path resolution for bundled assets and in-tree runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    first_existing_dir,
    resolve_env_path,
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "ewmbench"
IN_TREE_EWMBENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "ewmbench"
TASK_MANIFEST_REL = Path("task_manifest.json")



def resolve_ewmbench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_EWMBENCH_ROOT"),
        IN_TREE_EWMBENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_task_manifest_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EWMBench task manifest not found: {path}")
        return path
    for env_name in ("WORLDFOUNDRY_EWMBENCH_TASK_MANIFEST", "WORLDFOUNDRY_EWMBENCH_PROMPT_MANIFEST"):
        env_manifest = resolve_env_path(env_name)
        if env_manifest is not None:
            if not env_manifest.is_file():
                raise FileNotFoundError(f"EWMBench task manifest not found: {env_manifest}")
            return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, TASK_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_ewmbench_root()
    if root is None:
        raise FileNotFoundError(
            "EWMBench task manifest is missing. Set WORLDFOUNDRY_EWMBENCH_TASK_MANIFEST "
            "or WORLDFOUNDRY_EWMBENCH_ROOT."
        )
    candidate = root / TASK_MANIFEST_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"EWMBench task manifest not found: {candidate}")
    return candidate


def load_task_manifest(*, task_manifest_path: Path | None = None) -> dict[str, Any]:
    path = resolve_task_manifest_path(explicit=task_manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"EWMBench task manifest must be a JSON object: {path}")
    return payload
