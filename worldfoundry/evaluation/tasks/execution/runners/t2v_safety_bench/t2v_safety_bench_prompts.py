"""Official T2VSafetyBench prompt materialization for model generation."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset


def _class_id() -> int:
    value = int(os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS", "1"))
    if value not in range(1, 15):
        raise ValueError("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS must be between 1 and 14")
    return value


def _prompt_path(class_id: int) -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_PROMPT_PATH")
    return (
        Path(explicit).expanduser()
        if explicit
        else bundled_benchmark_asset("t2v-safety-bench", "T2VSafetyBench", f"{class_id}.txt")
    )


def materialize_t2v_safety_bench_generation_requests(*, limit: int | None = None) -> tuple[GenerationRequest, ...]:
    """Materialize one official prompt class with evaluator-compatible filenames."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    class_id = _class_id()
    prompt_path = _prompt_path(class_id)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"T2VSafetyBench prompt file not found: {prompt_path}")
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    if any(not prompt for prompt in prompts):
        raise ValueError(f"T2VSafetyBench prompt file contains an empty prompt: {prompt_path}")
    if limit is not None:
        prompts = prompts[:limit]

    return tuple(
        GenerationRequest(
            sample_id=f"t2v-safety-class{class_id}-{index:04d}",
            task_name="t2v_safety_bench_standard",
            split=f"class-{class_id}",
            inputs={
                "prompt": prompt,
                "prompt_id": index,
                "safety_class": class_id,
                "official_video_name": f"{class_id}-{index}.mp4",
            },
            output_schema={"generated_video": {"kind": "generated_video"}},
        )
        for index, prompt in enumerate(prompts, start=1)
    )


__all__ = ["materialize_t2v_safety_bench_generation_requests"]
