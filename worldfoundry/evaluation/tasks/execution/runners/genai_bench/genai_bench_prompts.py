"""GenAI-Bench prompt and preference-pair materialization from bundled assets."""

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
from worldfoundry.evaluation.utils import benchmark_task_sample_path

BENCHMARK_ID = "genai-bench"
METADATA_REL = Path("metadata.json")
PREFERENCE_PAIRS_REL = Path("preference_pairs.fixture.jsonl")



def resolve_genai_bench_assets_root(explicit: Path | None = None) -> Path:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_GENAI_BENCH_ASSETS_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ) or bundled_benchmark_assets_root(BENCHMARK_ID)


def resolve_metadata_path(*, assets_root: Path | None = None) -> Path:
    root = assets_root or resolve_genai_bench_assets_root()
    bundled = bundled_benchmark_asset(BENCHMARK_ID, METADATA_REL)
    for candidate in (root / METADATA_REL, bundled):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "GenAI-Bench metadata.json is missing. Expected bundled assets under "
        "worldfoundry/data/benchmarks/assets/genai-bench/."
    )


def resolve_preference_pairs_path(*, assets_root: Path | None = None) -> Path:
    root = assets_root or resolve_genai_bench_assets_root()
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PREFERENCE_PAIRS_REL)
    for candidate in (root / PREFERENCE_PAIRS_REL, bundled):
        if candidate.is_file():
            return candidate
    sample = benchmark_task_sample_path(BENCHMARK_ID)
    if sample is not None and sample.is_file():
        return sample
    raise FileNotFoundError(
        "GenAI-Bench preference fixture is missing. Provide preference_pairs.fixture.jsonl "
        "or sample_results.jsonl."
    )


def load_metadata(*, assets_root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(resolve_metadata_path(assets_root=assets_root).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GenAI-Bench metadata must be a JSON object")
    return payload


def load_preference_pair_rows(*, path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    source = path or resolve_preference_pairs_path()
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
        if limit is not None and len(rows) >= limit:
            break
    return rows
