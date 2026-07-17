#!/usr/bin/env python3
"""Run or normalize the checked-in T2VSafetyBench evaluator.

The upstream evaluator processes one prompt class at a time.  WorldFoundry
stages caller-provided videos into that class layout and refuses to run when a
prompt cannot be matched to a real video.  This keeps an incomplete or
misnamed artifact directory from being silently scored as safe.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    mean_numeric,
    normalize_unit_score,
    scalar_number,
)

CLASS_METRIC_IDS = {
    1: "pornography_nsfw_rate",
    2: "borderline_pornography_nsfw_rate",
    3: "violence_nsfw_rate",
    4: "gore_nsfw_rate",
    5: "disturbing_content_nsfw_rate",
    6: "public_figures_nsfw_rate",
    7: "discrimination_nsfw_rate",
    8: "political_sensitivity_nsfw_rate",
    9: "copyright_trademark_nsfw_rate",
    10: "illegal_activities_nsfw_rate",
    11: "misinformation_falsehoods_nsfw_rate",
    12: "temporal_risk_nsfw_rate",
    13: "temporal_risk_nsfw_rate",
    14: "temporal_risk_nsfw_rate",
}
METRIC_ORDER = (
    "pornography_nsfw_rate",
    "borderline_pornography_nsfw_rate",
    "violence_nsfw_rate",
    "gore_nsfw_rate",
    "public_figures_nsfw_rate",
    "discrimination_nsfw_rate",
    "political_sensitivity_nsfw_rate",
    "illegal_activities_nsfw_rate",
    "disturbing_content_nsfw_rate",
    "misinformation_falsehoods_nsfw_rate",
    "copyright_trademark_nsfw_rate",
    "temporal_risk_nsfw_rate",
    "nsfw_average",
)
SUPPORTED_MEDIA_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})

CONFIG = ors.BenchRunnerConfig(
    benchmark_id="t2v-safety-bench",
    display_name="T2VSafetyBench",
    root_env="WORLDFOUNDRY_T2V_SAFETY_BENCH_ROOT",
    results_path_env="WORLDFOUNDRY_T2V_SAFETY_BENCH_RESULTS_PATH",
    default_repo_subdir=("worldfoundry/evaluation/tasks/execution/runners/t2v_safety_bench/runtime/t2v_safety_bench"),
    metric_order=METRIC_ORDER,
    metric_specs={
        metric_id: {
            "name": metric_id.replace("_", " ").title(),
            "group": "safety",
            # These are violation rates: lower is safer.
            "higher_is_better": False,
        }
        for metric_id in METRIC_ORDER
    },
    metric_aliases={
        "nsfw_generation_rate": "nsfw_average",
        "nsfw_average": "nsfw_average",
        "temporal_risk": "temporal_risk_nsfw_rate",
    },
    average_metric_id="nsfw_average",
    official_entry="main.py",
    official_output_globs=("**/nsfw_results_*.xlsx", "**/nsfw_results_*.txt"),
    requires_api_env=("WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_API_KEY", "OPENAI_API_KEY"),
    usage_epilog=(
        "Run one class with --run-official, --generated-video-dir, --classes, "
        "and an OpenAI API key. Generated files must be named <class>-<1-based-index>.<video-ext>."
    ),
)
T2V_SAFETY_ASSETS_ROOT = bundled_benchmark_asset("t2v-safety-bench")


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    """Only discover results produced in this run's output directory.

    Searching the runtime checkout can accidentally reuse a stale result from a
    previous invocation after the current judge process fails.
    """

    del repo_root
    return ors.discover_by_globs([output_dir], CONFIG.official_output_globs)


def _class_from_result_name(path: Path) -> int | None:
    match = re.search(r"_class(\d+)(?:\.|$)", path.name)
    if match is None:
        return None
    value = int(match.group(1))
    return value if value in CLASS_METRIC_IDS else None


def _class_metric_row(results_path: Path, values: list[float]) -> dict[str, dict[str, Any]]:
    class_id = _class_from_result_name(results_path)
    score = mean_numeric(values)
    if class_id is None or score is None:
        return {}
    metric_id = CLASS_METRIC_IDS[class_id]
    return {
        metric_id: {
            "metric_id": metric_id,
            "raw_score": score,
            "normalized_score": normalize_unit_score(score),
            "source": str(results_path),
            "sample_count": len(values),
        }
    }


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    """Extract a class violation rate without pretending it is a full-suite mean."""

    extracted = ors.generic_extract_metrics(payload, CONFIG, str(results_path))
    if extracted:
        return extracted

    if isinstance(payload, list):
        values = [
            value for row in payload if isinstance(row, dict) if (value := scalar_number(row.get("Result"))) is not None
        ]
        return _class_metric_row(results_path, values)

    if results_path.suffix.lower() == ".txt":
        text = results_path.read_text(encoding="utf-8")
        match = re.search(r"NSFW generation rate:\s*([0-9.]+)", text)
        if match:
            return _class_metric_row(results_path, [float(match.group(1))])
    return {}


def _read_prompts(prompt_path: Path, *, limit: int | None) -> list[str]:
    if not prompt_path.is_file():
        raise ValueError(f"T2VSafetyBench prompt file not found: {prompt_path}")
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    if any(not prompt for prompt in prompts):
        raise ValueError(f"T2VSafetyBench prompt file contains an empty prompt: {prompt_path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"T2VSafetyBench prompt file contains no prompts: {prompt_path}")
    return prompts


def _source_video(generated_video_dir: Path, *, class_id: int, prompt_index: int) -> Path:
    stem = f"{class_id}-{prompt_index}"
    matches = sorted(
        path.resolve()
        for path in generated_video_dir.rglob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
    )
    if not matches:
        raise ValueError(
            f"missing generated video for prompt {prompt_index}; expected {stem}.<video-ext> "
            f"under {generated_video_dir}"
        )
    if len(matches) > 1:
        raise ValueError(f"ambiguous generated videos for prompt {prompt_index}: {matches}")
    if matches[0].stat().st_size <= 0:
        raise ValueError(f"generated video is empty: {matches[0]}")
    return matches[0]


def stage_official_inputs(
    *,
    generated_video_dir: Path,
    prompt_path: Path,
    output_dir: Path,
    class_id: int,
    limit: int | None,
) -> tuple[Path, Path, int]:
    """Create an isolated, exact prompt-to-video layout for the upstream judge."""

    if class_id not in CLASS_METRIC_IDS:
        raise ValueError("--classes must be between 1 and 14")
    if not generated_video_dir.is_dir():
        raise ValueError(f"generated video directory not found: {generated_video_dir}")

    prompts = _read_prompts(prompt_path, limit=limit)
    staging_root = output_dir / "upstream_input"
    staged_video_dir = staging_root / "video"
    staged_video_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in staged_video_dir.iterdir():
        if stale_path.is_file() or stale_path.is_symlink():
            stale_path.unlink()
    staged_prompt_path = staging_root / f"class{class_id}_prompts.txt"
    staged_prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    for prompt_index in range(1, len(prompts) + 1):
        source = _source_video(
            generated_video_dir,
            class_id=class_id,
            prompt_index=prompt_index,
        )
        destination = staged_video_dir / f"{class_id}-{prompt_index}{source.suffix.lower()}"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    return staged_video_dir, staged_prompt_path, len(prompts)


def build_official_command(
    *,
    config: ors.BenchRunnerConfig,
    repo_root: Path,
    generated_video_dir: Path,
    output_dir: Path,
    args: Any,
) -> list[str] | None:
    if not ors.first_env_value(*config.requires_api_env):
        return None

    prompt_path = args.prompt_path or env_path("WORLDFOUNDRY_T2V_SAFETY_BENCH_PROMPT_PATH")
    if prompt_path is None:
        prompt_path = T2V_SAFETY_ASSETS_ROOT / "T2VSafetyBench" / f"{args.classes}.txt"

    staged_video_dir, staged_prompt_path, _sample_count = stage_official_inputs(
        generated_video_dir=generated_video_dir,
        prompt_path=Path(prompt_path),
        output_dir=output_dir,
        class_id=args.classes,
        limit=args.limit,
    )
    model_name = args.model_name or os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_MODEL_NAME", "worldfoundry")
    save_dir = output_dir / "upstream"
    save_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(repo_root / config.official_entry),
        "--video-model",
        model_name,
        "--video-dir",
        str(staged_video_dir),
        "--prompt-path",
        str(staged_prompt_path),
        "--classes",
        str(args.classes),
        "--save-dir",
        str(save_dir),
        "--gpt-model",
        args.gpt_model,
    ]
    if args.api_base:
        command.extend(("--api-base", args.api_base))
    if args.gpt_eval_prompts:
        command.extend(("--gpt-eval-prompts", args.gpt_eval_prompts))
    return command


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in {None, ""} else int(value)


def extend_parser(parser) -> None:
    parser.add_argument("--model-name", default=None, help="label recorded in upstream result filenames")
    parser.add_argument(
        "--classes",
        type=int,
        choices=range(1, 15),
        default=int(os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS", "1")),
    )
    parser.add_argument("--prompt-path", type=Path, help="one-prompt-per-line official class file")
    parser.add_argument(
        "--limit",
        type=int,
        default=_optional_env_int("WORLDFOUNDRY_BENCHMARK_LIMIT"),
        help="bounded integration run; omit for the complete selected class",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="optional OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--gpt-model",
        default=os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_MODEL", "gpt-4o-2024-05-13"),
    )
    parser.add_argument(
        "--gpt-eval-prompts",
        default=os.environ.get("WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_EVAL_PROMPT"),
        help="optional override of the checked-in upstream evaluation prompt",
    )


def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
