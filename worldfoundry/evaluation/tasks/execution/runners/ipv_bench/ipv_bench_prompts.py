"""IPV-Bench prompt materialization from the official txt prompt suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    first_existing_dir,
    resolve_env_path,
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "ipv-bench"
PROMPT_SUITE_REL = Path("ipv_txt_prompt_suite.json")
JUDGEMENT_QUESTIONS_REL = Path("judgement_question.json")
JUDGEMENT_ANSWERS_REL = Path("judgement_answer.json")
MCQA_QUESTIONS_REL = Path("mcqa_question.json")
MCQA_ANSWERS_REL = Path("mcqa_answer.json")
OPENQA_QUESTIONS_REL = Path("openqa_question.json")
OPENQA_ANSWERS_REL = Path("openqa_answer.json")
CANONICAL_PROMPT_COUNT = 260




def resolve_ipv_bench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_IPV_BENCH_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_prompt_suite_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"IPV prompt suite not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_IPV_BENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"IPV prompt suite not found: {env_manifest}")
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_SUITE_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_ipv_bench_root()
    if root is None:
        raise FileNotFoundError(
            "IPV prompt suite is missing. Set WORLDFOUNDRY_IPV_BENCH_ROOT or "
            "WORLDFOUNDRY_IPV_BENCH_PROMPT_MANIFEST."
        )
    candidate = root / PROMPT_SUITE_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"IPV prompt suite not found: {candidate}")
    return candidate


def resolve_understanding_file(
    relative: Path,
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_file() else None
    bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_ipv_bench_root()
    if root is None:
        return None
    candidate = root / relative
    return candidate if candidate.is_file() else None


def _traverse_prompt_suite(data: Any, path: list[str] | None = None, result: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if path is None:
        path = []
    if result is None:
        result = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for example in value:
                    if isinstance(example, dict) and "prompt_text" in example:
                        result.append(
                            {
                                "prompt_id": str(example.get("prompt_id")),
                                "prompt": str(example.get("prompt_text") or "").strip(),
                                "prompt_taxonomy_label": example.get("prompt_taxonomy_label")
                                or " - ".join(path + [key]),
                            }
                        )
            _traverse_prompt_suite(value, path + [key], result)
    elif isinstance(data, list):
        for item in data:
            _traverse_prompt_suite(item, path, result)
    return result


def load_prompt_records(*, prompt_suite_path: Path | None = None) -> list[dict[str, Any]]:
    path = resolve_prompt_suite_path(explicit=prompt_suite_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = _traverse_prompt_suite(payload)
    records = [record for record in records if record.get("prompt_id") and record.get("prompt")]
    if not records:
        raise ValueError(f"IPV prompt records are empty after validation: {path}")
    return records


def unique_prompt_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["prompt_id"]) if str(item["prompt_id"]).isdigit() else item["prompt_id"]):
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return records


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}.mp4"


def materialize_ipv_bench_generation_requests(
    *,
    limit: int | None = None,
    prompt_suite_path: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(prompt_suite_path=prompt_suite_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="ipv-bench",
                split="standard",
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "prompt_taxonomy_label": record.get("prompt_taxonomy_label"),
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)
