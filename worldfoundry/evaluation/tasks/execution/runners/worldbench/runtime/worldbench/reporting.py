"""WorldFoundry scorecard serialization for the in-tree WorldBench runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl

METRIC_ORDER = (
    "foreground_miou",
    "foreground_dice",
    "background_rmse",
    "text_based_accuracy",
    "multiple_choice_accuracy",
    "binary_accuracy",
)
METRIC_NAMES = {
    "foreground_miou": "Foreground mIoU",
    "foreground_dice": "Foreground Dice (release compatibility)",
    "background_rmse": "Background RMSE",
    "text_based_accuracy": "Text-Based Accuracy",
    "multiple_choice_accuracy": "Multiple-Choice Accuracy",
    "binary_accuracy": "Binary Accuracy",
}
METRIC_GROUPS = {
    "foreground_miou": "video_based",
    "foreground_dice": "diagnostic",
    "background_rmse": "video_based",
    "text_based_accuracy": "text_based",
    "multiple_choice_accuracy": "text_based",
    "binary_accuracy": "text_based",
}


def write_evaluation_scorecard(
    evaluation: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    command: list[str] | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    metric_path = output_dir / "raw_metric_table.jsonl"
    sample_path = output_dir / "per_sample_scores.jsonl"
    evaluation_path = output_dir / "worldbench_evaluation.json"

    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        metric = dict(evaluation.get("metrics", {}).get(metric_id) or {})
        available = bool(metric.get("available")) and isinstance(metric.get("normalized_score"), (int, float))
        row = {
            "metric_id": metric_id,
            "name": METRIC_NAMES[metric_id],
            "group": METRIC_GROUPS[metric_id],
            "available": available,
            "raw_score": metric.get("raw_score"),
            "normalized_score": metric.get("normalized_score"),
            "higher_is_better": metric.get("higher_is_better", True),
            "normalizer": "identity",
            "source": metric.get("source"),
            "sample_count": metric.get("sample_count", 0),
            "description": metric.get("description"),
        }
        if not available:
            row["reason"] = "component_not_supplied_or_no_scorable_samples"
        elif metric_id != "foreground_dice":
            leaderboard[metric_id] = float(metric["normalized_score"])
        metric_rows.append(row)
        per_metric[metric_id] = row

    samples = list(evaluation.get("per_sample_scores") or [])
    video_coverage = evaluation.get("dataset", {}).get("video_coverage", {})
    text_coverage = evaluation.get("dataset", {}).get("text_coverage", {})
    coverage_complete = bool(video_coverage.get("complete")) and bool(text_coverage.get("complete"))
    available_count = sum(row["available"] for row in metric_rows)
    execution_ok = available_count > 0 and not evaluation.get("failures")
    eligibility_reasons = [
        "Numerical parity with an official leaderboard submission has not yet been audited.",
        "The public evaluator labels a Dice implementation as mIoU; WorldFoundry reports IoU and Dice separately.",
    ]
    if not coverage_complete:
        eligibility_reasons.append("This run does not contain complete video and text task coverage.")

    write_json(evaluation_path, evaluation)
    write_jsonl(metric_path, metric_rows)
    write_jsonl(sample_path, samples)
    scorecard = {
        "schema_version": "worldfoundry-scorecard",
        "run": {
            "status": "completed" if execution_ok else "partial" if available_count else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_worldbench_in_tree_runner",
            "command": command,
            "returncode": 0 if execution_ok else 1,
            "duration_seconds": None,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "WorldBench",
            "contract_only": False,
            "requires_upstream_runtime": False,
            "official_runtime_available": True,
            "track": "IntuitivePhysics",
        },
        "dataset": evaluation.get("dataset"),
        "generation": {
            "successful": sum(
                row.get("component") == "video_based" and row.get("available") is True for row in samples
            ),
            "failed": sum(row.get("component") == "video_based" and row.get("available") is False for row in samples),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "video_based": ["foreground_miou", "background_rmse"],
                "text_based": [
                    "text_based_accuracy",
                    "multiple_choice_accuracy",
                    "binary_accuracy",
                ],
                "diagnostic": ["foreground_dice"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(samples),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": available_count > 0,
            "kind": "in_tree_worldbench_intuitive_physics",
            "num_results": len(samples),
            "leaderboard_metrics": leaderboard,
            "configuration": evaluation.get("config"),
            "segmentation_model": evaluation.get("model"),
            "failures": evaluation.get("failures", []),
        },
        "validation": {
            "normalizer_only": False,
            "official_runtime_executed": True,
            "artifact_score_imported": False,
            "generated_artifacts_evaluated": any(
                row.get("component") == "video_based" and row.get("available") is True for row in samples
            ),
            "answer_manifest_evaluated": any(
                row.get("component") == "text_based" and row.get("available") is True for row in samples
            ),
            "coverage_complete": coverage_complete,
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": eligibility_reasons,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(metric_path.resolve()),
            "per_sample_scores": str(sample_path.resolve()),
            "worldbench_evaluation": str(evaluation_path.resolve()),
            "upstream_results": None,
            "upstream_stdout": None,
            "upstream_stderr": None,
        },
        "official_benchmark_verified": False,
        "integration_evidence": execution_ok,
        "normalization_ok": execution_ok,
        "official_results_imported": False,
    }
    write_json(scorecard_path, scorecard)
    return scorecard
