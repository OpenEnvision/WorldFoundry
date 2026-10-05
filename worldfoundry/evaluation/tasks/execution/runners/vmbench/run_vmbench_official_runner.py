#!/usr/bin/env python3
from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import csv
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any


from worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime import materialize_artifact_scores
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    load_json,
    mean_numeric,
    normalize_unit_score,
    scalar_number,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.runners.vmbench.vmbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    materialize_vmbench_meta_info,
    resolve_vmbench_root,
)
from worldfoundry.evaluation.tasks.execution.runners.vmbench.vmbench_official_runtime import (
    config_from_env as vmbench_official_config_from_env,
    run_official_vmbench_runtime,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_VMBENCH_ROOT = resolve_vmbench_root()
COMPONENT_METRICS = (
    "perceptible_amplitude_score",
    "object_integrity_score",
    "temporal_coherence_score",
    "commonsense_adherence_score",
    "motion_smoothness_score",
)
METRIC_ORDER = (*COMPONENT_METRICS, "vmbench_average")
METRIC_ALIASES = {
    "pas": "perceptible_amplitude_score",
    "perceptible_amplitude_score": "perceptible_amplitude_score",
    "perceptible_amplitude_socre": "perceptible_amplitude_score",
    "ois": "object_integrity_score",
    "object_integrity_score": "object_integrity_score",
    "tcs": "temporal_coherence_score",
    "temporal_coherence_score": "temporal_coherence_score",
    "cas": "commonsense_adherence_score",
    "commonsense_adherence_score": "commonsense_adherence_score",
    "mss": "motion_smoothness_score",
    "motion_smoothness_score": "motion_smoothness_score",
    "avg": "vmbench_average",
    "average": "vmbench_average",
    "total_score": "vmbench_average",
    "vmbench_average": "vmbench_average",
}


scalar = partial(scalar_number, dict_keys=("score", "raw_score", "value", "mean", "average"))
mean = mean_numeric
normalize_score = normalize_unit_score


def canonical_metric_id(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_ALIASES.get(normalized)


def extract_from_results_json(raw_results: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = raw_results
    if isinstance(raw_results, dict):
        for key in ("results", "samples", "per_sample_metrics"):
            if isinstance(raw_results.get(key), list):
                rows = raw_results[key]
                break
    if not isinstance(rows, list):
        raise ValueError("VMBench results JSON must be a list or contain a results list")

    metric_values: dict[str, list[float]] = {metric: [] for metric in COMPONENT_METRICS}
    direct_scores: dict[str, dict[str, Any]] = {}
    per_sample_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        direct_metric_id = canonical_metric_id(
            str(row.get("metric_id") or row.get("metric") or row.get("name") or "")
        )
        direct_score = normalize_score(
            scalar(row.get("score") if row.get("score") is not None else row.get("raw_score"))
        )
        if direct_metric_id in METRIC_ORDER and direct_score is not None:
            if direct_metric_id in COMPONENT_METRICS:
                metric_values[direct_metric_id].append(direct_score)
            else:
                direct_scores[direct_metric_id] = {
                    "raw_score": direct_score,
                    "source": row.get("source") or "metric_id_score_results_json",
                    "sample_count": scalar(row.get("sample_count")),
                }
            per_sample_rows.append(
                {
                    "sample_index": index,
                    "sample_id": row.get("sample_id") or row.get("video_id") or row.get("filepath"),
                    "metric_id": direct_metric_id,
                    "metrics": {direct_metric_id: direct_score},
                    "raw": row,
                }
            )
            continue
        metrics: dict[str, float | None] = {}
        for key, value in row.items():
            metric_id = canonical_metric_id(key)
            if metric_id not in COMPONENT_METRICS:
                continue
            score = normalize_score(scalar(value))
            metrics[metric_id] = score
            if score is not None:
                metric_values[metric_id].append(score)
        per_sample_rows.append(
            {
                "sample_index": index,
                "sample_id": row.get("index") or row.get("sample_id") or row.get("video_id") or row.get("filepath"),
                "prompt": row.get("prompt"),
                "filepath": row.get("filepath"),
                "metrics": metrics,
                "raw": row,
            }
        )

    extracted: dict[str, dict[str, Any]] = dict(direct_scores)
    for metric_id, values in metric_values.items():
        if values:
            extracted[metric_id] = {
                "raw_score": sum(values) / len(values),
                "source": "mean_per_sample_results_json",
                "sample_count": len(values),
            }
    return extracted, per_sample_rows


def extract_from_scores_csv(path: Path) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return extracted
        metric_key = next((key for key in reader.fieldnames if key.lower().strip() == "metric"), reader.fieldnames[0])
        score_key = next((key for key in reader.fieldnames if "score" in key.lower()), reader.fieldnames[-1])
        for row in reader:
            metric_id = canonical_metric_id(str(row.get(metric_key, "")))
            if metric_id not in METRIC_ORDER:
                continue
            raw_score = normalize_score(scalar(row.get(score_key)))
            if raw_score is not None:
                extracted[metric_id] = {
                    "raw_score": raw_score,
                    "source": "official_scores_csv",
                    "sample_count": None,
                }
    return extracted


def load_upstream_results(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Path]:
    if path.is_dir():
        json_path = path / "results.json"
        csv_path = path / "scores.csv"
        if json_path.is_file():
            path = json_path
        elif csv_path.is_file():
            path = csv_path
        else:
            raise FileNotFoundError(f"VMBench results directory must contain results.json or scores.csv: {path}")

    if path.suffix.lower() == ".csv":
        extracted = extract_from_scores_csv(path)
        return extracted, [], path

    raw_results = load_json(path)
    extracted, per_sample_rows = extract_from_results_json(raw_results)
    return extracted, per_sample_rows, path


def add_average(extracted: dict[str, dict[str, Any]]) -> None:
    component_scores = [item.get("raw_score") for metric_id, item in extracted.items() if metric_id in COMPONENT_METRICS]
    average = mean(component_scores)
    if average is not None and "vmbench_average" not in extracted:
        extracted["vmbench_average"] = {
            "raw_score": average,
            "source": "computed_from_available_vmbench_metrics",
            "sample_count": min((item.get("sample_count") or 0 for item in extracted.values()), default=0),
        }


def normalize_vmbench_results(
    extracted_scores: dict[str, dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_metrics_path = output_dir / "per_sample_metrics.jsonl"

    add_average(extracted_scores)
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalize_score(raw_score),
            "raw_score_range": [0.0, 1.0],
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_vmbench_results"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_metrics_path, per_sample_rows)
    available_count = sum(1 for row in metric_rows if row["available"])
    normalization_ok = returncode == 0 and available_count > 0
    normalizer_only = command is None
    official_verified = command is not None and normalization_ok
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": "official_verified" if official_verified else "official_results_imported" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_vmbench_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "VMBench",
            "contract_only": False,
            "requires_upstream_runtime": command is None,
            "requires_model_weights": True,
        },
        "dataset": {
            "sample_count": len(per_sample_rows),
            "expected_prompt_count": CANONICAL_PROMPT_COUNT,
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official VMBench runtime validation; full leaderboard evidence requires complete generated videos and metric checkpoint assets",
            ],
        },
        "generation": {
            "successful": len(per_sample_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "motion": ["perceptible_amplitude_score", "motion_smoothness_score", "temporal_coherence_score"],
                "semantics": ["object_integrity_score", "commonsense_adherence_score"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(per_sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": official_verified,
            "kind": "official_vmbench",
            "upstream_results": str(upstream_results_path.resolve()),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_metrics": str(per_sample_metrics_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_verified,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def run_vmbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    if args.run_fixture:
        sample_path = benchmark_task_sample_path(args.benchmark_id)
        if sample_path is None:
            raise FileNotFoundError(f"no checked-in VMBench sample results found for {args.benchmark_id}")
        args.from_upstream_results = sample_path

    if args.from_upstream_results:
        extracted, per_sample_rows, upstream_results_path = load_upstream_results(args.from_upstream_results)
        return normalize_vmbench_results(
            extracted,
            per_sample_rows,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=upstream_results_path,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    if args.run_official:
        runtime_backend = args.runtime_backend
        if runtime_backend == "auto":
            runtime_backend = "artifact" if args.artifact_score_dir is not None else "official"
        if runtime_backend == "official":
            if args.video_dir is None:
                raise ValueError("--video-dir is required for the in-tree VMBench official runtime")
            summary = run_official_vmbench_runtime(
                video_dir=args.video_dir,
                output_dir=output_dir,
                prompt_manifest=args.prompt_manifest,
                repo_root=args.vmbench_root,
                config=vmbench_official_config_from_env(python=args.python),
            )
            upstream_results_path = Path(str(summary.get("results_path") or summary.get("meta_info_path")))
            try:
                extracted, per_sample_rows, upstream_results_path = load_upstream_results(upstream_results_path)
            except (OSError, ValueError, json.JSONDecodeError):
                extracted = {}
                per_sample_rows = []
                meta_info_path = Path(str(summary.get("meta_info_path", upstream_results_path)))
                if meta_info_path.is_file():
                    try:
                        rows = json.loads(meta_info_path.read_text(encoding="utf-8"))
                        if isinstance(rows, list):
                            per_sample_rows = [
                                {
                                    "sample_index": index,
                                    "sample_id": row.get("prompt_id") or row.get("index"),
                                    "prompt": row.get("prompt"),
                                    "filepath": row.get("filepath"),
                                    "metrics": {},
                                    "raw": row,
                                }
                                for index, row in enumerate(rows)
                                if isinstance(row, dict)
                            ]
                    except json.JSONDecodeError:
                        per_sample_rows = []
            command = ["in_tree_vmbench_official_runtime", "--runtime-root", str(summary.get("runtime_root"))]
            stdout_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            return normalize_vmbench_results(
                extracted,
                per_sample_rows,
                benchmark_id=args.benchmark_id,
                output_dir=output_dir,
                upstream_results_path=upstream_results_path,
                command=command,
                duration_seconds=None,
                returncode=int(summary.get("returncode", 1)),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )

        meta_info_path = output_dir / "upstream" / "vmbench_meta_info.json"
        meta_info_rows: list[dict[str, Any]] = []
        if args.video_dir is not None:
            meta_info_rows = materialize_vmbench_meta_info(
                video_dir=args.video_dir,
                output_path=meta_info_path,
                prompt_suite_path=args.prompt_manifest,
                repo_root=args.vmbench_root,
            )
        score_dir = args.artifact_score_dir or env_path("WORLDFOUNDRY_VMBENCH_ARTIFACT_SCORE_DIR") or args.video_dir
        if score_dir is None:
            raise ValueError(
                "--artifact-score-dir, WORLDFOUNDRY_VMBENCH_ARTIFACT_SCORE_DIR, or --video-dir is required for --run-official"
            )
        upstream_results_path = output_dir / "upstream" / "vmbench_results.json"
        try:
            summary = materialize_artifact_scores(
                benchmark_id=args.benchmark_id,
                score_dir=score_dir,
                generated_video_dir=args.video_dir,
                output_path=upstream_results_path,
            )
            extracted, per_sample_rows, upstream_results_path = load_upstream_results(upstream_results_path)
            returncode = 0
        except ValueError as exc:
            summary = {
                "benchmark_id": args.benchmark_id,
                "score_dir": str(score_dir),
                "generated_video_dir": None if args.video_dir is None else str(args.video_dir),
                "meta_info_path": str(meta_info_path) if meta_info_rows else None,
                "meta_info_count": len(meta_info_rows),
                "error": str(exc),
                "blocked_reason": (
                    "VMBench in-tree integration requires metric artifact files or official results for scoring; "
                    "full CUDA metric recomputation remains checkpoint-dependent."
                ),
            }
            upstream_results_path = meta_info_path if meta_info_rows else upstream_results_path
            extracted = {}
            per_sample_rows = [
                {
                    "sample_index": index,
                    "sample_id": row.get("prompt_id") or row.get("index"),
                    "prompt": row.get("prompt"),
                    "filepath": row.get("filepath"),
                    "metrics": {},
                    "raw": row,
                }
                for index, row in enumerate(meta_info_rows)
            ]
            returncode = 1
        command = [
            "worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime",
            "--benchmark-id",
            args.benchmark_id,
            "--score-dir",
            str(score_dir),
            "--output-path",
            str(upstream_results_path),
        ]
        stdout_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return normalize_vmbench_results(
            extracted,
            per_sample_rows,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=upstream_results_path,
            command=command,
            duration_seconds=None,
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    raise ValueError("--official-results-path, WORLDFOUNDRY_VMBENCH_RESULTS_PATH, or --run-official is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize official VMBench outputs to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "vmbench"))
    parser.add_argument("--vmbench-root", type=Path, default=DEFAULT_VMBENCH_ROOT)
    parser.add_argument(
        "--official-results-path",
        dest="from_upstream_results",
        type=Path,
        default=env_path("WORLDFOUNDRY_VMBENCH_RESULTS_PATH"),
    )
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--video-dir", type=Path, default=env_path("WORLDFOUNDRY_VMBENCH_VIDEO_DIR"))
    parser.add_argument("--generated-video-dir", dest="video_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-manifest", type=Path, default=env_path("WORLDFOUNDRY_VMBENCH_PROMPT_MANIFEST"))
    parser.add_argument("--run-fixture", action="store_true", help="Normalize checked-in VMBench sample results.")
    parser.add_argument("--run-official", action="store_true", help="Run the in-tree VMBench official runtime or artifact importer.")
    parser.add_argument(
        "--runtime-backend",
        choices=("auto", "official", "artifact"),
        default=os.environ.get("WORLDFOUNDRY_VMBENCH_RUNTIME_BACKEND", "auto"),
        help="official runs vendored VMBench metric code; artifact imports precomputed metric artifacts.",
    )
    parser.add_argument("--python", default=os.environ.get("WORLDFOUNDRY_VMBENCH_PYTHON", sys.executable))
    parser.add_argument(
        "--artifact-score-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_VMBENCH_ARTIFACT_SCORE_DIR"),
        help="Directory containing VMBench metric artifacts produced by WorldFoundry base-model evaluators.",
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_vmbench(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_metrics": scorecard["artifacts"]["per_sample_metrics"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official VMBench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
