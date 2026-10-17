"""PhyFPS-Bench-Gen prompt materialization and artifact layout helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.core.io.filesystem.file_utils import materialize_file
from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
    resolve_env_path,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
    iter_generation_video_artifacts,
    reset_generation_artifact_staging,
)
from worldfoundry.evaluation.utils import write_jsonl

BENCHMARK_ID = "phyfps-bench-gen"
CANONICAL_PROMPT_COUNT = 100
PROMPT_MANIFEST_REL = Path("prompts.txt")


def _default_prompt_manifest() -> Path | None:
    env_root = resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_ROOT")
    if env_root is not None and env_root.is_dir():
        manifest = env_root / PROMPT_MANIFEST_REL
        if manifest.is_file():
            return manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    bundled_root = bundled_benchmark_assets_root(BENCHMARK_ID)
    if bundled_root.is_dir():
        manifest = bundled_root / PROMPT_MANIFEST_REL
        if manifest.is_file():
            return manifest
    return None


def resolve_prompt_manifest_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_path = resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST")
    if env_path is not None:
        return env_path
    default = _default_prompt_manifest()
    if default is None:
        raise ValueError(
            "PhyFPS-Bench-Gen prompt manifest is missing. Set WORLDFOUNDRY_PHYFPS_BENCH_GEN_ROOT "
            "or WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST."
        )
    return default


def load_prompt_lines(explicit: Path | None = None) -> list[str]:
    path = resolve_prompt_manifest_path(explicit)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"PhyFPS-Bench-Gen prompt manifest is empty: {path}")
    return lines


def prompt_id_for_index(index: int) -> str:
    return f"{index + 1:04d}"


def video_filename_for_index(index: int) -> str:
    return f"{prompt_id_for_index(index)}.mp4"


def materialize_phyfps_generation_requests(
    *,
    limit: int | None = None,
    prompt_manifest_path: Path | None = None,
    split: str = "default",
) -> tuple[GenerationRequest, ...]:
    prompts = load_prompt_lines(prompt_manifest_path)
    if limit is not None:
        prompts = prompts[: int(limit)]
    requests: list[GenerationRequest] = []
    for index, prompt in enumerate(prompts):
        prompt_id = prompt_id_for_index(index)
        requests.append(
            GenerationRequest(
                sample_id=prompt_id,
                task_name="phyfps-bench-gen",
                split=split,
                inputs={
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "generation_text": prompt,
                    "prompt_index": index,
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_phyfps_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
) -> tuple[int, int]:
    """Copy model outputs into ``{0001..0100}.mp4`` files expected by PhyFPS-Bench-Gen."""
    rows: list[dict[str, Any]] = []
    reset_generation_artifact_staging(generation_output_dir, generated_artifact_dir)
    materialized = 0
    for result, artifact, source_path in iter_generation_video_artifacts(generation_output_dir, output_artifact):
        destination = generated_artifact_dir / f"{result.sample_id}.mp4"
        row = {
            "sample_id": result.sample_id,
            "artifact_name": output_artifact,
            "source_uri": artifact.uri,
            "destination": str(destination),
            "status": "missing",
            "placeholder": False,
        }
        if source_path is not None and source_path.is_file():
            if source_path.resolve() != destination.resolve():
                materialize_file(source_path, destination, writable=False)
            row["status"] = "copied"
            materialized += 1
        rows.append(row)
    write_jsonl(artifact_manifest_path, rows, atomic=False)
    return materialized, 0
