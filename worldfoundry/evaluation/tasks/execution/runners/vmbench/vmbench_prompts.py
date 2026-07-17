"""VMBench prompt materialization from bundled official prompt metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)
from worldfoundry.evaluation.tasks.execution.framework.io import write_json

BENCHMARK_ID = "vmbench"
PROMPT_SUITE_REL = Path("prompts/prompts.json")
CANONICAL_PROMPT_COUNT = 1050
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def resolve_vmbench_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_prompt_suite_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VMBench prompt suite not found: {path}")
        return path
    env_manifest = _env_path("WORLDFOUNDRY_VMBENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"VMBench prompt suite not found: {env_manifest}")
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_SUITE_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_vmbench_root()
    if root is None:
        raise FileNotFoundError(
            "VMBench bundled prompt suite is missing. Restore "
            "worldfoundry/data/benchmarks/assets/vmbench/prompts/prompts.json or set "
            "WORLDFOUNDRY_VMBENCH_PROMPT_MANIFEST for an explicit override."
        )
    for relative in (PROMPT_SUITE_REL, Path("prompts.json")):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"VMBench prompt suite not found under {root}")


def _record_index(record: Mapping[str, Any], index: int) -> str:
    value = record.get("index") or record.get("prompt_id") or record.get("sample_id") or index + 1
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):04d}"
    return text


def _load_prompt_payload(path: Path) -> Sequence[Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            rows = payload.get("prompts") or payload.get("records") or payload.get("rows")
            if isinstance(rows, list):
                return rows
    elif suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".txt":
        return [{"prompt": line.strip(), "index": f"{index + 1:04d}"} for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()) if line.strip()]
    raise ValueError(f"Unsupported VMBench prompt manifest format: {path}")


def load_prompt_records(
    *,
    prompt_suite_path: Path | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    path = resolve_prompt_suite_path(explicit=prompt_suite_path, repo_root=repo_root)
    rows = _load_prompt_payload(path)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        prompt = str(row.get("prompt") or row.get("prompt_text") or row.get("text") or "").strip()
        prompt_id = _record_index(row, index)
        if not prompt or not prompt_id:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "index": prompt_id,
                "prompt": prompt,
                "subject": row.get("subject"),
                "subject_noun": row.get("subject_noun"),
                "place": row.get("place"),
                "action": row.get("action"),
                "official_video_name": f"{prompt_id}.mp4",
            }
        )
    if not records:
        raise ValueError(f"VMBench prompt records are empty after validation: {path}")
    return records


def unique_prompt_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["prompt_id"])):
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return records


def materialize_vmbench_generation_requests(
    *,
    limit: int | None = None,
    prompt_suite_path: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(prompt_suite_path=prompt_suite_path, repo_root=repo_root))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = str(record["prompt_id"])
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name=BENCHMARK_ID,
                split="standard",
                inputs={
                    "prompt": record["prompt"],
                    "generation_text": record["prompt"],
                    "prompt_id": sample_id,
                    "index": record["index"],
                    "subject": record.get("subject"),
                    "subject_noun": record.get("subject_noun"),
                    "place": record.get("place"),
                    "action": record.get("action"),
                    "official_video_name": record["official_video_name"],
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def materialize_vmbench_meta_info(
    *,
    video_dir: Path,
    output_path: Path,
    prompt_suite_path: Path | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    records = unique_prompt_records(load_prompt_records(prompt_suite_path=prompt_suite_path, repo_root=repo_root))
    video_root = video_dir.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        video_path = video_root / str(record["official_video_name"])
        if not video_path.is_file():
            alt_matches = [
                candidate
                for candidate in video_root.iterdir()
                if candidate.is_file()
                and candidate.stem == str(record["prompt_id"])
                and candidate.suffix.lower() in VIDEO_SUFFIXES
            ] if video_root.is_dir() else []
            if alt_matches:
                video_path = alt_matches[0]
        if video_path.is_file():
            rows.append({**record, "filepath": str(video_path.resolve())})
    if not rows:
        raise FileNotFoundError(
            f"No VMBench generated videos matched the official prompt ids in {video_root}. "
            "Expected names such as 0001.mp4."
        )
    write_json(output_path, rows)
    return rows
