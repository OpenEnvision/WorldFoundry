#!/usr/bin/env python3
"""Official runner for T2VWorldBench.

Normalize ``*_video_assessment_scores.csv`` or run ``eval.py`` on a video directory.

Environment variables
---------------------
- ``WORLDFOUNDRY_T2VWORLDBENCH_ROOT``
- ``WORLDFOUNDRY_T2VWORLDBENCH_RESULTS_PATH``
- ``WORLDFOUNDRY_T2VWORLDBENCH_PROMPT_FILE``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, mean_numeric, scalar_number

DEFAULT_PROMPT_FILE = bundled_benchmark_asset("t2vworldbench", "data", "meta_data", "meta_data.json")

CONFIG = ors.BenchRunnerConfig(
    benchmark_id="t2vworldbench",
    display_name="T2VWorldBench",
    root_env="WORLDFOUNDRY_T2VWORLDBENCH_ROOT",
    results_path_env="WORLDFOUNDRY_T2VWORLDBENCH_RESULTS_PATH",
    default_repo_subdir="worldfoundry/evaluation/tasks/execution/runners/t2vworldbench/runtime/t2vworldbench",
    metric_order=("quality", "realism", "relevance", "consistency", "final"),
    metric_specs={
        "quality": {"name": "Quality", "group": "official_assessment", "higher_is_better": True},
        "realism": {"name": "Realism", "group": "official_assessment", "higher_is_better": True},
        "relevance": {"name": "Relevance", "group": "official_assessment", "higher_is_better": True},
        "consistency": {"name": "Consistency", "group": "official_assessment", "higher_is_better": True},
        "final": {"name": "Final", "group": "aggregate", "higher_is_better": True},
    },
    metric_aliases={
        "quality": "quality",
        "quality_score": "quality",
        "min_quality_score": "quality",
        "realism": "realism",
        "realism_score": "realism",
        "min_realism_score": "realism",
        "relevance": "relevance",
        "relevance_score": "relevance",
        "min_relevance_score": "relevance",
        "consistency": "consistency",
        "consistency_score": "consistency",
        "min_consistency_score": "consistency",
        "final": "final",
        "final_score": "final",
        "final_min_score": "final",
        "total_min_score": "final",
        "overall_average_model_score": "final",
    },
    # ``final_min_score`` is the minimum group total, not a mean that can be
    # reconstructed from the four independently minimized dimensions.
    average_metric_id="__no_derived_average__",
    official_entry="eval.py",
    official_output_globs=("results/*_video_assessment_scores.csv", "*_video_assessment_scores.csv"),
    requires_api_env=(),
    usage_epilog="Examples:\n  python3 run_t2vworldbench_official_runner.py \\\n    --official-results-path my_model_video_assessment_scores.csv \\\n    --output-dir /tmp/t2vworld_out --json\n\n  python3 run_t2vworldbench_official_runner.py --run-official \\\n    --model-name my_model --prompt-file prompts.txt \\\n    --generated-video-dir /path/to/videos --output-dir /tmp/t2vworld_out --json",
)

OFFICIAL_METRIC_FIELDS = {
    "quality": (("min_quality_score", "quality_score", "quality"), 5.0),
    "realism": (("min_realism_score", "realism_score", "realism"), 5.0),
    "relevance": (("min_relevance_score", "relevance_score", "relevance"), 5.0),
    "consistency": (("min_consistency_score", "consistency_score", "consistency"), 5.0),
    "final": (
        ("final_min_score", "total_min_score", "final_score", "final", "overall_average_model_score"),
        20.0,
    ),
}


def _canonical_field_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_").rstrip(":")


def _official_result_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for row in payload:
            rows.extend(_official_result_rows(row))
        return rows
    if not isinstance(payload, dict):
        return []
    canonical = {_canonical_field_name(key) for key in payload}
    known_fields = {field for fields, _scale in OFFICIAL_METRIC_FIELDS.values() for field in fields}
    if canonical.intersection(known_fields):
        return [payload]
    rows: list[dict[str, Any]] = []
    for container in ("metrics", "scores", "results", "summary"):
        rows.extend(_official_result_rows(payload.get(container)))
    return rows


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir, output_dir / "upstream"]
    if repo_root is not None:
        search_roots.extend([repo_root, repo_root / "results"])
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in OFFICIAL_METRIC_FIELDS}
    for row in _official_result_rows(payload):
        canonical_row = {_canonical_field_name(key): value for key, value in row.items()}
        for metric_id, (source_fields, scale) in OFFICIAL_METRIC_FIELDS.items():
            score = next(
                (
                    numeric
                    for field in source_fields
                    if (numeric := scalar_number(canonical_row.get(field))) is not None
                ),
                None,
            )
            if score is not None and 0.0 <= score <= scale:
                buckets[metric_id].append(score)

    extracted: dict[str, dict[str, Any]] = {}
    for metric_id, values in buckets.items():
        avg = mean_numeric(values)
        if avg is None:
            continue
        scale = OFFICIAL_METRIC_FIELDS[metric_id][1]
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": avg,
            "normalized_score": avg / scale,
            "source": str(results_path),
            "sample_count": len(values),
        }
    return extracted


def build_official_command(
    *, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any
) -> list[str] | None:
    prompt_file = args.prompt_file or env_path("WORLDFOUNDRY_T2VWORLDBENCH_PROMPT_FILE") or DEFAULT_PROMPT_FILE
    if prompt_file is None:
        return None
    model_name = args.model_name or os.environ.get("WORLDFOUNDRY_T2VWORLDBENCH_MODEL_NAME", "model")
    upstream_output = output_dir / "upstream"
    upstream_output.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(repo_root / config.official_entry),
        "--video-path",
        str(generated_video_dir),
        "--t2v-model",
        model_name,
        "--read-prompt-file",
        str(prompt_file),
        "--output-path",
        str(upstream_output),
    ]


def extend_parser(parser) -> None:
    parser.add_argument("--model-name", default=None, help="upstream --t2v-model name")
    parser.add_argument("--prompt-file", type=Path, help="prompt file passed to eval.py")


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
