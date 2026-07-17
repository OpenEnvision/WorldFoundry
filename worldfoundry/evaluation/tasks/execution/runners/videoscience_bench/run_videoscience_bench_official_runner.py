#!/usr/bin/env python3
"""Official runner for VideoScience-Bench.

Normalize judge JSON/CSV outputs or run the in-tree VideoScience VLM judge.

Environment variables
---------------------
- ``WORLDFOUNDRY_VIDEOSCIENCE_BENCH_RESULTS_PATH``
- ``WORLDFOUNDRY_VIDEOSCIENCE_PROVIDER`` / ``WORLDFOUNDRY_VIDEOSCIENCE_MODEL``
"""

from __future__ import annotations

import os
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

CONFIG = ors.BenchRunnerConfig(
    benchmark_id='videoscience-bench',
    display_name='VideoScience-Bench',
    root_env='WORLDFOUNDRY_VIDEOSCIENCE_BENCH_ROOT',
    results_path_env='WORLDFOUNDRY_VIDEOSCIENCE_BENCH_RESULTS_PATH',
    default_repo_subdir='worldfoundry/evaluation/tasks/execution/runners/videoscience_bench/runtime/videoscience_bench',
    metric_order=('prompt_consistency', 'phenomenon_congruency', 'correct_dynamism', 'immutability', 'spatio_temporal_coherence', 'videoscience_average'),
    metric_specs={'prompt_consistency': {'name': 'Prompt Consistency', 'group': 'science', 'higher_is_better': True}, 'phenomenon_congruency': {'name': 'Phenomenon Congruency', 'group': 'science', 'higher_is_better': True}, 'correct_dynamism': {'name': 'Correct Dynamism', 'group': 'science', 'higher_is_better': True}, 'immutability': {'name': 'Immutability', 'group': 'science', 'higher_is_better': True}, 'spatio_temporal_coherence': {'name': 'Spatio-Temporal Coherence', 'group': 'science', 'higher_is_better': True}, 'videoscience_average': {'name': 'VideoScience Weighted Overall', 'group': 'aggregate', 'higher_is_better': True}},
    metric_aliases={'prompt_consistency': 'prompt_consistency', 'pcs': 'prompt_consistency', 'phenomenon_congruency': 'phenomenon_congruency', 'expected_phenomenon': 'phenomenon_congruency', 'phenomenon': 'phenomenon_congruency', 'pcg': 'phenomenon_congruency', 'correct_dynamism': 'correct_dynamism', 'dynamism': 'correct_dynamism', 'cdn': 'correct_dynamism', 'immutability': 'immutability', 'imb': 'immutability', 'spatio_temporal_coherence': 'spatio_temporal_coherence', 'coherence': 'spatio_temporal_coherence', 'spatio_temporal_continuity': 'spatio_temporal_coherence', 'stc': 'spatio_temporal_coherence', 'overall': 'videoscience_average', 'overall_weighted': 'videoscience_average', 'videoscience_average': 'videoscience_average'},
    average_metric_id='videoscience_average',
    official_entry='videoscience_batch.py',
    official_output_globs=('upstream/videoscience_judge_results.json', 'videoscience_judge_results.json', 'judge/*.json', 'judge/*.csv'),
    requires_api_env=(),
    usage_epilog='Examples:\n  python3 run_videoscience_bench_official_runner.py --official-results-path judge/results.json --output-dir /tmp/out --json',
)
VIDEOSCIENCE_EXPERIMENTS_CSV_ASSET = bundled_benchmark_asset("videoscience-bench", "database", "data.csv")

def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir, output_dir / "upstream"]
    if repo_root is not None:
        search_roots.append(repo_root / "judge")
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)

def _unit_from_1to4(value: Any) -> float | None:
    number = scalar_number(value)
    if number is None:
        return None
    number = max(1.0, min(4.0, number))
    return (number - 1.0) / 3.0

def _unit_score(value: Any, *, rubric_scale: bool = False) -> float | None:
    if rubric_scale:
        return _unit_from_1to4(value)
    number = scalar_number(value)
    if number is None:
        return None
    return normalize_unit_score(number)

def _collect_metric_buckets(container: Any, buckets: dict[str, list[float]], *, rubric_scale: bool) -> None:
    if not isinstance(container, dict):
        return
    for key, value in container.items():
        metric_id = ors.metric_id_from_key(key, CONFIG)
        if metric_id is None:
            continue
        score = _unit_score(value, rubric_scale=rubric_scale)
        if score is None:
            continue
        buckets.setdefault(metric_id, []).append(score)

def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    if isinstance(payload, dict):
        _collect_metric_buckets(payload.get("metrics"), buckets, rubric_scale=False)
        _collect_metric_buckets(payload.get("rubric"), buckets, rubric_scale=True)
        rows = payload.get("results")
    else:
        rows = payload
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            _collect_metric_buckets(row.get("metrics"), buckets, rubric_scale=False)
            _collect_metric_buckets(row.get("rubric"), buckets, rubric_scale=True)
            _collect_metric_buckets(row.get("scores"), buckets, rubric_scale=True)
    extracted: dict[str, dict[str, Any]] = {}
    for metric_id, values in buckets.items():
        score = mean_numeric(values)
        if score is None:
            continue
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": score,
            "normalized_score": score,
            "source": str(results_path),
            "sample_count": len(values),
        }
    if extracted:
        return extracted
    return ors.generic_extract_metrics(payload, CONFIG, str(results_path))

def build_official_command(*, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any) -> list[str] | None:
    experiments_csv = args.experiments_csv or env_path("WORLDFOUNDRY_VIDEOSCIENCE_EXPERIMENTS_CSV")
    if experiments_csv is None:
        experiments_csv = VIDEOSCIENCE_EXPERIMENTS_CSV_ASSET
        if not experiments_csv.is_file():
            experiments_csv = repo_root / "data" / "database" / "data.csv"
    if not Path(experiments_csv).is_file():
        return None
    provider = args.judge_provider or os.environ.get("WORLDFOUNDRY_VIDEOSCIENCE_PROVIDER", "openai")
    model = args.judge_model or os.environ.get("WORLDFOUNDRY_VIDEOSCIENCE_MODEL", "gpt-4o")
    upstream_dir = args.evaluation_results_dir or (output_dir / "upstream")
    command = [
        args.python,
        str(repo_root / str(config.official_entry)),
        "--experiments-csv",
        str(experiments_csv),
        "--generated-video-dir",
        str(args.evaluation_source_dir or generated_video_dir),
        "--output-dir",
        str(upstream_dir),
        "--provider",
        provider,
        "--model",
        model,
        "--max-frames",
        str(args.max_frames),
        "--timeout-s",
        str(args.judge_timeout_s),
    ]
    if args.authors:
        command.extend(["--authors", args.authors])
    if args.reference_videos_dir:
        command.extend(["--reference-videos-dir", str(args.reference_videos_dir)])
    if args.judge_max_runs:
        command.extend(["--max-runs", str(args.judge_max_runs)])
    if args.ready_only:
        command.append("--ready-only")
    if args.judge_fps is not None:
        command.extend(["--fps", str(args.judge_fps)])
    return command


def extend_parser(parser) -> None:
    parser.add_argument("--experiments-csv", type=Path, help="VideoScience data/database/data.csv or filtered export")
    parser.add_argument("--authors", default=None, help="author filter recorded for imported VideoScience outputs")
    parser.add_argument("--evaluation-results-dir", type=Path, help="judge output directory")
    parser.add_argument("--evaluation-source-dir", type=Path, help="generated videos root for judge")
    parser.add_argument("--reference-videos-dir", type=Path, help="reference videos root for judge")
    parser.add_argument("--judge-provider", default=None, help="VideoScience VLM judge provider")
    parser.add_argument("--judge-model", default=None, help="VideoScience VLM judge model")
    parser.add_argument("--judge-max-runs", type=int, default=0, help="limit judged videos for local validation")
    parser.add_argument("--ready-only", action="store_true", help="only judge rows marked finalized? == Done")
    parser.add_argument("--max-frames", type=int, default=24, help="max frames sent to the VLM judge")
    parser.add_argument("--judge-fps", type=float, default=None, help="frame sampling fps override")
    parser.add_argument("--judge-timeout-s", type=int, default=900, help="per-video VLM judge timeout")

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
