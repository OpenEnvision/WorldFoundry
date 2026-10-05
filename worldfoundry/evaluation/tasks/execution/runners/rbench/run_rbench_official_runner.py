#!/usr/bin/env python3
"""Official runner for RBench, the robotics video-generation benchmark from ReVidgen.

RBench scores generated videos with an external VLM judge (GPT or Qwen3-VL) plus a motion
operator stack (GroundingDINO / SAM 2 / CoTracker / Q-Align). Those stages stay outside
WorldFoundry; this runner reproduces the official *aggregation* in-tree, reading the
per-video artifacts the upstream scripts write:

    results/4_embodiments/<model>/<embodiment>/VQA/<vlm>/<question>/results.csv
    results/4_embodiments/<model>/<embodiment>/motion/results.json
    results/5_tasks/<model>/<task>/<vlm>/results.csv

Both official tracks are recomputed from raw per-video rows rather than trusting the
upstream summary CSVs, so a partial judge run cannot silently look complete.
"""

from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_metrics import (
    EMBODIMENT_ROW_COLUMNS,
    METRIC_ORDER,
    METRIC_SPECS,
    compute_rbench_metrics,
    track_for_metric,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    DISPLAY_NAME,
    EMBODIMENT_ORDER,
    EMBODIMENT_TRACK,
    TASK_ORDER,
    TASK_TRACK,
    VLM_BACKENDS,
    expected_video_stems,
    resolve_prompt_dir,
    split_for_id,
    video_coverage,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_runtime import (
    RBenchLayoutError,
    env_data_root,
    env_results_path,
    resolve_layout,
)
from worldfoundry.evaluation.utils import benchmark_task_sample_path

RUNNER_NAME = "benchmark_zoo_rbench_official_runner"
EVALUATION_KIND = "rbench_official_aggregation"
EVIDENCE_SCOPE = "official_result_aggregation_only"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RBench official judge and operator outputs.")
    parser.add_argument("--benchmark-id", default="rbench")
    parser.add_argument(
        "--official-results-path",
        "--results-root",
        dest="official_results_path",
        type=Path,
        help="RBench results/ root, a track directory, or one model directory.",
    )
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Accepted for CLI symmetry; RBench judging happens upstream, so this only aggregates.",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Aggregate checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
        help="Directory of generated videos for a split, used for coverage reporting.",
    )
    parser.add_argument("--benchmark-data-root", type=Path, help="RBench dataset root holding prompts/.")
    parser.add_argument("--prompt-manifest", type=Path, help="Directory holding <split>_prompts.json.")
    parser.add_argument(
        "--result-model-id",
        dest="result_model_id",
        help="Generation model directory to score when the results tree holds several.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=list(VLM_BACKENDS),
        help="VLM judge whose outputs to aggregate; auto-detected when only one is present.",
    )
    parser.add_argument(
        "--track",
        dest="tracks",
        action="append",
        choices=[EMBODIMENT_TRACK, TASK_TRACK],
        help="Restrict aggregation to one track; repeatable.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _strict_enabled(args: argparse.Namespace) -> bool:
    return args.strict or os.environ.get("WORLDFOUNDRY_RBENCH_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_results_path(args: argparse.Namespace) -> Path | None:
    for candidate in (args.official_results_path, env_results_path()):
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _metric_rows(computed: Mapping[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = metrics.get(metric_id)
        reason = None
        if score is None:
            reason = (
                "cross_track_composite_requires_both_complete_tracks"
                if metric_id == "rbench_overall"
                else "metric_not_computable_from_supplied_results"
            )
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "track": track_for_metric(metric_id),
                "worldfoundry_defined": bool(spec.get("worldfoundry_defined")),
                "source": "rbench_official_result_aggregation",
                "source_path": str(source_path),
                "evidence_scope": EVIDENCE_SCOPE,
                "components": dict(components),
                "reason": reason,
            }
        )
    return rows


def _per_split_rows(computed: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    embodiment = computed.get("embodiment_track")
    if isinstance(embodiment, Mapping):
        for split_id, summary in embodiment.get("per_split", {}).items():
            row: dict[str, Any] = {
                "track": EMBODIMENT_TRACK,
                "split_id": split_id,
                "video_count": summary.get("video_count"),
                "retained_video_count": summary.get("retained_video_count"),
                "dropped_by_amplitude_filter": summary.get("dropped_by_amplitude_filter"),
            }
            row.update({column: summary["means"].get(column) for column in EMBODIMENT_ROW_COLUMNS})
            rows.append(row)
    task = computed.get("task_track")
    if isinstance(task, Mapping):
        for split_id, summary in task.get("per_split", {}).items():
            rows.append(
                {
                    "track": TASK_TRACK,
                    "split_id": split_id,
                    "mean_score": summary.get("mean_score"),
                    "video_count": summary.get("video_count"),
                    "dropped_video_count": summary.get("dropped_video_count"),
                }
            )
    return rows


def _scored_video_counts(computed: Mapping[str, Any]) -> dict[str, int]:
    """Return the number of videos actually scored per split.

    Coverage is judged on this, not on which split directories exist: a split present
    with 20 of its 100 prompts scored is partial, and must not read as complete.
    """
    counts: dict[str, int] = {}
    embodiment = computed.get("embodiment_track")
    if isinstance(embodiment, Mapping):
        for split_id, summary in embodiment.get("per_split", {}).items():
            counts[split_id] = int(summary.get("video_count") or 0)
    task = computed.get("task_track")
    if isinstance(task, Mapping):
        for split_id, summary in task.get("per_split", {}).items():
            counts[split_id] = int(summary.get("video_count") or 0)
    return counts


def _coverage(
    computed: Mapping[str, Any],
    *,
    prompt_dir: Path | None,
    generated_artifact_dir: Path | None,
) -> dict[str, Any]:
    embodiment = computed.get("embodiment_track")
    task = computed.get("task_track")
    scored_embodiments = list(embodiment["scored_splits"]) if isinstance(embodiment, Mapping) else []
    scored_tasks = list(task["scored_splits"]) if isinstance(task, Mapping) else []
    scored_splits = [*scored_embodiments, *scored_tasks]
    scored_counts = _scored_video_counts(computed)

    per_split: dict[str, Any] = {}
    for split_id in scored_splits:
        split = split_for_id(split_id)
        scored = scored_counts.get(split_id, 0)
        entry: dict[str, Any] = {
            "scored_video_count": scored,
            "expected_prompt_count": split.prompt_count,
            "complete": scored >= split.prompt_count,
        }
        if prompt_dir is not None and generated_artifact_dir is not None:
            try:
                expected = expected_video_stems(prompt_dir, split_id)
            except (FileNotFoundError, KeyError, ValueError):
                expected = set()
            if expected:
                entry["generated_videos"] = video_coverage(
                    video_dir=generated_artifact_dir, expected=sorted(expected)
                )
        per_split[split_id] = entry

    incomplete = sorted(
        split_id for split_id, entry in per_split.items() if entry["complete"] is not True
    )
    scored_prompt_total = sum(scored_counts.get(split_id, 0) for split_id in scored_splits)
    all_splits_present = not [
        item for item in (*EMBODIMENT_ORDER, *TASK_ORDER) if item not in scored_splits
    ]

    return {
        "expected_prompt_count": CANONICAL_PROMPT_COUNT,
        "scored_prompt_count": scored_prompt_total,
        "scored_embodiments": scored_embodiments,
        "missing_embodiments": [item for item in EMBODIMENT_ORDER if item not in scored_embodiments],
        "scored_tasks": scored_tasks,
        "missing_tasks": [item for item in TASK_ORDER if item not in scored_tasks],
        "embodiment_track_complete": bool(embodiment and embodiment.get("complete")),
        "task_track_complete": bool(task and task.get("complete")),
        "incomplete_splits": incomplete,
        "complete": all_splits_present and not incomplete,
        "splits": per_split,
    }


def _scorecard(
    *,
    args: argparse.Namespace,
    layout: Mapping[str, Any],
    results_path: Path,
    metric_rows: list[dict[str, Any]],
    coverage: Mapping[str, Any],
    components: Mapping[str, Any],
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    normalization_ok = bool(available_rows)
    both_tracks = bool(components.get("both_tracks_complete"))
    full_suite = both_tracks and coverage.get("complete") is True
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": True,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalization_ok,
        "evidence_scope": EVIDENCE_SCOPE,
        "eligibility": {
            "full_suite_valid": full_suite,
            "leaderboard_valid": False,
            "embodiment_track_complete": bool(components.get("embodiment_track_complete")),
            "task_track_complete": bool(components.get("task_track_complete")),
            "both_tracks_complete": both_tracks,
        },
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0 if normalization_ok else 1,
            "official_runtime_executed": False,
            "vlm_backend": components.get("vlm_backend"),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": DISPLAY_NAME},
        "dataset": {
            "upstream_results": str(results_path.resolve()),
            "result_model_id": layout.get("model_id"),
            "available_tracks": list(layout.get("available_tracks") or []),
            "vlm_backend": components.get("vlm_backend"),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": {str(row["metric_id"]): row for row in metric_rows},
            "summary": {
                "available_metric_count": len(available_rows),
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": EVALUATION_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "importer_only": True,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "validation": {
            "normalizer_only": True,
            "official_runtime_executed": False,
            "official_runtime_succeeded": False,
            "official_results_imported": normalization_ok,
            "full_suite_complete": full_suite,
            "scope": EVIDENCE_SCOPE,
        },
        "artifacts": {
            "scorecard": str((args.output_dir / "scorecard.json").resolve()),
            "official_results_path": str(results_path.resolve()),
        },
        "coverage": dict(coverage),
        "notes": [
            "RBench VLM judging and the motion operator stack run upstream; this runner "
            "reproduces the official aggregation from their per-video outputs.",
            "Scores from the gpt and qwen judges are never mixed; one backend is aggregated per run.",
            "rbench_overall is a WorldFoundry composite of the two official track scores, "
            "not a number RBench publishes.",
        ],
    }


def aggregate_rbench_results(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _resolve_results_path(args)
    if results_path is None:
        raise ValueError(
            "--official-results-path or WORLDFOUNDRY_RBENCH_RESULTS_PATH is required"
        )
    layout = resolve_layout(
        results_path,
        model_id=args.result_model_id,
        vlm_backend=args.vlm_backend,
    )

    selected_tracks = set(args.tracks or (EMBODIMENT_TRACK, TASK_TRACK))
    embodiment_dir = layout.embodiment_track_dir if EMBODIMENT_TRACK in selected_tracks else None
    task_dir = layout.task_track_dir if TASK_TRACK in selected_tracks else None
    if embodiment_dir is None and task_dir is None:
        raise RBenchLayoutError(
            f"no RBench results available for the selected tracks: {sorted(selected_tracks)}"
        )

    computed = compute_rbench_metrics(
        embodiment_track_dir=embodiment_dir,
        task_track_dir=task_dir,
        vlm_backend=layout.vlm_backend,
    )
    prompt_dir = resolve_prompt_dir(
        explicit=args.prompt_manifest,
        data_root=args.benchmark_data_root or env_data_root(),
    )
    coverage = _coverage(
        computed,
        prompt_dir=prompt_dir,
        generated_artifact_dir=args.generated_artifact_dir,
    )
    metric_rows = _metric_rows(computed, source_path=results_path)
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    scorecard = _scorecard(
        args=args,
        layout=layout.to_dict(),
        results_path=results_path,
        metric_rows=metric_rows,
        coverage=coverage,
        components=components,
    )
    if _strict_enabled(args) and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", _per_split_rows(computed))
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "official_results_path": str(results_path),
            "metric_ids": list(METRIC_ORDER),
            "layout": layout.to_dict(),
            "scored_embodiments": coverage["scored_embodiments"],
            "scored_tasks": coverage["scored_tasks"],
            "prompt_manifest_dir": None if prompt_dir is None else str(prompt_dir),
        },
    )
    write_json(
        output_dir / "track_breakdown.json",
        {
            "embodiment_track": {
                key: value
                for key, value in (computed.get("embodiment_track") or {}).items()
                if key != "per_split"
            }
            | {
                "per_split": {
                    split_id: {key: value for key, value in summary.items() if key != "per_video"}
                    for split_id, summary in (computed.get("embodiment_track") or {}).get("per_split", {}).items()
                }
            }
            if computed.get("embodiment_track")
            else None,
            "task_track": computed.get("task_track"),
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def _failure_scorecard(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "normalizer_only": True,
        "normalization_ok": False,
        "official_results_imported": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "run": {
            "status": "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": DISPLAY_NAME},
        "dataset": {},
        "eligibility": {"full_suite_valid": False, "leaderboard_valid": False},
        "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
        "evaluation": {
            "available": False,
            "kind": EVALUATION_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "importer_only": True,
            "blocked_count": len(METRIC_ORDER),
        },
        "validation": {
            "normalizer_only": True,
            "official_runtime_executed": False,
            "official_runtime_succeeded": False,
            "official_results_imported": False,
            "full_suite_complete": False,
            "scope": EVIDENCE_SCOPE,
        },
        "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_fixture:
        sample_path = benchmark_task_sample_path("rbench")
        if sample_path is None:
            print("error: no checked-in RBench sample results found", file=sys.stderr)
            return 2
        args.official_results_path = sample_path
    try:
        scorecard = aggregate_rbench_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = _failure_scorecard(args, exc)
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), **scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"rbench: failed ({exc})", file=sys.stderr)
        return 1

    payload = {
        "ok": scorecard.get("normalization_ok") is True,
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        **scorecard,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        leaderboard = scorecard.get("metrics", {}).get("leaderboard", {})
        parts = [
            f"{key}={leaderboard[key]:.4f}"
            for key in ("embodiment_overall", "task_track_overall", "rbench_overall")
            if leaderboard.get(key) is not None
        ]
        print(f"rbench: {EVALUATION_KIND} {' '.join(parts) or 'no metric available'}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
