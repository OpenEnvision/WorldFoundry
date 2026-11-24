#!/usr/bin/env python3
"""4DWorldBench official runtime and result normalizer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework import official_runner as ors
from worldfoundry.evals.tasks.execution.framework.io import (
    mean_numeric,
    normalize_unit_score,
    scalar_number,
    write_json,
)

DIMENSION_METRICS = (
    "perceptual_clip_iqa_metrics",
    "perceptual_clip_aesthetic_metrics",
    "perceptual_fastvqa",
    "alignment_attribute_control",
    "alignment_relationship_control",
    "alignment_motion_control",
    "alignment_event_control",
    "alignment_scene_control",
    "alignment_camera_error_metrics",
    "physics_realism",
    "consistency_viewpoint",
    "consistency_motion_smoothness",
    "consistency_motion_qa",
    "consistency_style",
)
AGGREGATE_COMPONENTS = {
    "perceptual_quality": (
        "perceptual_clip_iqa_metrics",
        "perceptual_clip_aesthetic_metrics",
        "perceptual_fastvqa",
    ),
    "condition_4d_alignment": (
        "alignment_attribute_control",
        "alignment_relationship_control",
        "alignment_motion_control",
        "alignment_event_control",
        "alignment_scene_control",
        "alignment_camera_error_metrics",
    ),
    "physical_realism_score": ("physics_realism",),
    "four_d_consistency": (
        "consistency_viewpoint",
        "consistency_motion_smoothness",
        "consistency_motion_qa",
        "consistency_style",
    ),
}
METRIC_ORDER = (
    *DIMENSION_METRICS,
    "perceptual_quality",
    "condition_4d_alignment",
    "physical_realism_score",
    "four_d_consistency",
    "four_d_worldbench_average",
)
LOWER_IS_BETTER = frozenset({"alignment_camera_error_metrics", "consistency_viewpoint"})
DIMENSION_ALIASES = {
    "clip_iqa": "perceptual_clip_iqa_metrics",
    "clipiqa": "perceptual_clip_iqa_metrics",
    "clipiqa_metrics": "perceptual_clip_iqa_metrics",
    "clip_aesthetic": "perceptual_clip_aesthetic_metrics",
    "clipaesthetic": "perceptual_clip_aesthetic_metrics",
    "fastvqa": "perceptual_fastvqa",
    "dynamic_attribute": "alignment_attribute_control",
    "dynamic_spatial_relationship": "alignment_relationship_control",
    "motion_order_understanding": "alignment_motion_control",
    "complex_plot": "alignment_event_control",
    "complex_landscape": "alignment_scene_control",
    "camera_error": "alignment_camera_error_metrics",
    "camera_error_metrics": "alignment_camera_error_metrics",
    "motion_smoothness": "consistency_motion_smoothness",
    "motion_qa": "consistency_motion_qa",
    "style": "consistency_style",
    "viewpoint": "consistency_viewpoint",
    "overall": "four_d_worldbench_average",
    "average": "four_d_worldbench_average",
}


def _metric_name(metric_id: str) -> str:
    if metric_id == "four_d_worldbench_average":
        return "4DWorldBench Average"
    if metric_id == "condition_4d_alignment":
        return "Condition-4D Alignment"
    if metric_id == "four_d_consistency":
        return "4D Consistency"
    return metric_id.replace("_", " ").title()


CONFIG = ors.BenchRunnerConfig(
    benchmark_id="4dworldbench",
    display_name="4DWorldBench",
    root_env="WORLDFOUNDRY_4DWORLDBENCH_ROOT",
    results_path_env="WORLDFOUNDRY_4DWORLDBENCH_RESULTS_PATH",
    default_repo_subdir="worldfoundry/evals/tasks/execution/runners/four_d_worldbench/runtime/four_d_worldbench",
    metric_order=METRIC_ORDER,
    metric_specs={
        metric_id: {
            "name": _metric_name(metric_id),
            "group": "aggregate" if metric_id not in DIMENSION_METRICS else metric_id.split("_", 1)[0],
            "higher_is_better": metric_id not in LOWER_IS_BETTER,
        }
        for metric_id in METRIC_ORDER
    },
    metric_aliases={metric_id: metric_id for metric_id in METRIC_ORDER} | DIMENSION_ALIASES,
    average_metric_id="four_d_worldbench_average",
    official_entry="runner.py",
    official_output_globs=("upstream/4dworldbench_results.json", "4dworldbench_results.json", "*_results.json"),
    usage_epilog=(
        "Examples:\n"
        "  python run_four_d_worldbench_official_runner.py --run-official \\\n"
        "    --dataset-json /data/4DWorldBench/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \\\n"
        "    --model-name output_uniform_ex4d --dimension perceptual_clip_iqa_metrics \\\n"
        "    --generated-video-dir /data/4DWorldBench --output-dir /tmp/4dworldbench --json\n\n"
        "  python run_four_d_worldbench_official_runner.py \\\n"
        "    --official-results-path /tmp/4dworldbench/upstream/4dworldbench_results.json \\\n"
        "    --output-dir /tmp/4dworldbench_norm --json"
    ),
)


def _canonical_metric_id(value: Any) -> str | None:
    key = ors.canonical_key(value)
    if not key:
        return None
    return CONFIG.metric_aliases.get(key) or (key if key in METRIC_ORDER else None)


def _score_row(metric_id: str, raw_score: float, *, source: str, sample_count: int | None = None) -> dict[str, Any]:
    normalized = normalize_unit_score(raw_score)
    if metric_id in LOWER_IS_BETTER:
        normalized = None if raw_score is None else 1.0 / (1.0 + max(float(raw_score), 0.0))
    return {
        "metric_id": metric_id,
        "raw_score": raw_score,
        "normalized_score": normalized,
        "source": source,
        "sample_count": sample_count,
    }


def _add_score(
    extracted: dict[str, dict[str, Any]],
    metric_id: str,
    score: float | None,
    *,
    source: str,
    sample_count: int | None = None,
) -> None:
    if score is None:
        return
    extracted[metric_id] = _score_row(metric_id, float(score), source=source, sample_count=sample_count)


def _extract_one(payload: Any, source: str) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        metric_id = _canonical_metric_id(payload.get("dimension") or payload.get("metric_id") or payload.get("metric"))
        score = scalar_number(payload.get("score"))
        summary = payload.get("evaluation_summary")
        if score is None and isinstance(summary, dict):
            score = scalar_number(
                summary.get("average_score")
                if summary.get("average_score") is not None
                else summary.get("combined_error")
            )
        if metric_id is not None:
            sample_count = scalar_number(payload.get("generated_video_count"))
            if sample_count is None and isinstance(summary, dict):
                sample_count = scalar_number(summary.get("total_videos"))
            _add_score(
                extracted,
                metric_id,
                score,
                source=source,
                sample_count=None if sample_count is None else int(sample_count),
            )
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            for raw_key, raw_value in metrics.items():
                nested_metric_id = _canonical_metric_id(raw_key)
                if nested_metric_id is not None:
                    _add_score(extracted, nested_metric_id, scalar_number(raw_value), source=source)
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                for metric_id, row in _extract_one(item, source).items():
                    extracted.setdefault(metric_id, row)
    elif isinstance(payload, list):
        for item in payload:
            for metric_id, row in _extract_one(item, source).items():
                extracted.setdefault(metric_id, row)
    return extracted


def _apply_aggregates(extracted: dict[str, dict[str, Any]]) -> None:
    for aggregate_id, component_ids in AGGREGATE_COMPONENTS.items():
        values = [
            float(row["normalized_score"])
            for metric_id in component_ids
            if (row := extracted.get(metric_id)) and row.get("normalized_score") is not None
        ]
        avg = mean_numeric(values)
        if avg is not None:
            extracted[aggregate_id] = {
                "metric_id": aggregate_id,
                "raw_score": avg,
                "normalized_score": avg,
                "source": "component_aggregate",
                "sample_count": len(values),
            }
    aggregate_values = [
        float(row["normalized_score"])
        for metric_id in AGGREGATE_COMPONENTS
        if (row := extracted.get(metric_id)) and row.get("normalized_score") is not None
    ]
    avg = mean_numeric(aggregate_values)
    if avg is None:
        avg = mean_numeric(
            [
                float(row["normalized_score"])
                for metric_id in DIMENSION_METRICS
                if (row := extracted.get(metric_id)) and row.get("normalized_score") is not None
            ]
        )
    if avg is not None:
        extracted["four_d_worldbench_average"] = {
            "metric_id": "four_d_worldbench_average",
            "raw_score": avg,
            "normalized_score": avg,
            "source": "component_aggregate",
            "sample_count": len(aggregate_values),
        }


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    extracted = _extract_one(payload, str(results_path))
    _apply_aggregates(extracted)
    return extracted


def prepare_upstream_results(config: ors.BenchRunnerConfig, results_path: Path, args: Any, output_dir: Path) -> Path:
    if not results_path.is_dir():
        return results_path
    candidates = sorted(path for path in results_path.rglob("*.json") if path.name != "scorecard.json")
    merged = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _extract_one(payload, str(path)):
            merged.append(payload)
    if not merged:
        return results_path
    merged_path = output_dir / "merged_4dworldbench_results.json"
    write_json(merged_path, {"results": merged})
    return merged_path


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    return ors.discover_by_globs([output_dir, output_dir / "upstream"], CONFIG.official_output_globs)


def build_official_command(
    *, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any
) -> list[str] | None:
    if args.video_path is None and args.dataset_json is None:
        return None
    output_dir = output_dir.resolve()
    upstream = output_dir / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(repo_root / config.official_entry),
        "--model",
        args.model_name,
        "--dimension",
        args.dimension,
        "--device",
        args.device,
        "--output_dir",
        str(upstream),
        "--result_json",
        str(upstream / "4dworldbench_results.json"),
    ]
    if args.dataset_json is not None:
        command.extend(["--dataset_json", str(args.dataset_json.resolve())])
    if args.video_path is not None:
        command.extend(["--video_path", str(args.video_path.resolve())])
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.skip_question_generation:
        command.append("--skip_question_generation")
    return command


def extend_parser(parser) -> None:
    parser.add_argument("--dataset-json", "--dataset_json", dest="dataset_json", type=Path)
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_4DWORLDBENCH_MODEL_NAME", "model"))
    parser.add_argument(
        "--dimension", default=os.environ.get("WORLDFOUNDRY_4DWORLDBENCH_DIMENSION", "perceptual_clip_iqa_metrics")
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--skip-question-generation", action="store_true")


def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            prepare_upstream_results=prepare_upstream_results,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
