"""Embodied Evaluation Scorecard and Official Metric Result Normalizer.

This module provides a robust, reusable CLI and program entry point to parse, normalize,
and standardize raw benchmark results (from JSON, JSONL, CSV, TSV, or custom result.txt logs)
into a unified, schema-compliant WorldFoundry Scorecard (`worldfoundry-scorecard`).

Core Capabilities:
1. Dynamic Parsing: Extracts multi-level per-sample/per-task metrics and metadata from nested
   or flat serialization layouts without requiring rigid schemas on input files.
2. Numeric Coercion: Standardizes varied verbal success indicators (such as "succeeded", "pass",
   "failed", percent-suffixed values like "95%", and raw booleans) into precise floating-point values.
3. Metric Scaling: Maps specific metrics using designated normalization specs (e.g., converting percentages
   to fraction units, normalizing return rewards) to ensure correct aggregation and comparability.
4. Scorecard Construction: Generates complete analytical scorecard artifacts including overall summary logs,
   filtered leaderboard statistics, per-task metric breakdowns, and reproducibility trail logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.core.io.serialization import write_json, write_jsonl
from worldfoundry.core.observability.time import utc_now_iso
from worldfoundry.evaluation.tasks.execution.framework.normalizers import apply_normalizer


# Schema version for normalized scorecard files.
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"

# Known evaluation tracks matching physical and visual simulation targets.
TRACKS = frozenset({"vla", "va", "vam", "wam"})

# Supported default metric fields denoting scalar performance attributes.
DEFAULT_NUMERIC_FIELDS = (
    "success",
    "success_rate",
    "task_success",
    "score",
    "reward",
    "completion",
    "sequence_success",
    "episode_success",
    "rollout_success",
    "goal_success",
    "normalized_return",
    "planning_success",
    "paraphrase_success_rate",
    "language_generalization_success",
    "action_accuracy",
    "world_state_consistency",
)

# Potential key mappings signifying item or episode IDs.
ID_FIELDS = ("sample_id", "episode_id", "task_id", "task", "id", "name")


def _coerce_number(value: Any) -> float | None:
    """Coerces varied inputs (booleans, verbal indicators, percentages) into a normalized float."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"true", "yes", "success", "succeeded", "pass", "passed"}:
            return 1.0
        if lowered in {"false", "no", "failure", "failed", "fail"}:
            return 0.0
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _read_json_payload(path: Path) -> Any:
    """Safely reads and deserializes a JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_json_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dynamically parses and extracts row records and summaries from unstructured JSON files.

    Handles flat lists of dicts, nested structures mapping to 'results', 'episodes', or 'samples',
    and constructs basic diagnostic items if the payload simply represents a flat flat-dictionary.
    """
    summary: dict[str, Any] = {}
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)], summary
    if not isinstance(payload, Mapping):
        return [], summary

    for key in ("summary", "overall", "metrics"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            summary.update({str(k): v for k, v in value.items()})

    rows: list[dict[str, Any]] = []
    for key in ("results", "episodes", "samples", "per_sample", "per_task", "tasks", "rollouts"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(dict(row) for row in value if isinstance(row, Mapping))
        elif isinstance(value, Mapping):
            for item_id, item in value.items():
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("task_id", item_id)
                    rows.append(row)
                else:
                    rows.append({"task_id": item_id, "score": item})

    if not rows:
        numeric = {str(key): value for key, value in payload.items() if _coerce_number(value) is not None}
        if numeric:
            rows.append({"sample_id": "summary", **numeric})
    return rows, summary


def _result_txt_row(path: Path) -> dict[str, Any] | None:
    """Parses legacy text log files (such as `_result.txt` or `result.txt`) into structured records.

    Attempts to extract sequence metrics and infer task descriptions or policy configurations
    by querying folder path structures.
    """
    values: list[float] = []
    instruction_type: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.casefold().startswith("instruction type:"):
            instruction_type = text.split(":", 1)[1].strip() or None
            continue
        number = _coerce_number(text)
        if number is not None:
            values.append(number)
    if not values:
        return None

    parts = path.parts
    task_name = path.parent.name
    sample_id = path.parent.name
    policy_name = None
    task_config = None
    ckpt_setting = None
    if "eval_result" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("eval_result")
        tail = parts[index + 1 :]
        # Upstream timestamps are only unique within a task/policy run.  Two
        # tasks can finish in the same second, so retain their relative result
        # path when assigning sample identities.
        if len(tail) > 1:
            sample_id = "/".join(tail[:-1])
        if len(tail) >= 5:
            task_name, policy_name, task_config, ckpt_setting = tail[:4]

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "task_id": task_name,
        "task": task_name,
        "success_rate": values[-1],
        "source_file": str(path),
    }
    if policy_name:
        row["policy"] = policy_name
    if task_config:
        row["task_config"] = task_config
    if ckpt_setting:
        row["checkpoint"] = ckpt_setting
    if instruction_type:
        row["instruction_type"] = instruction_type
    return row


def read_input_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Crawls and extracts structured rows, diagnostic summary maps, and source file listings.

    Supports JSON, JSONL, CSV, TSV, and result.txt log files.
    """
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    source_files: list[str] = []
    for input_path in paths:
        candidates = sorted(path for path in input_path.rglob("*") if path.is_file()) if input_path.is_dir() else [input_path]
        for path in candidates:
            suffix = path.suffix.lower()
            if suffix not in {".json", ".jsonl", ".csv", ".tsv"} and path.name not in {"_result.txt", "result.txt"}:
                continue
            source_files.append(str(path))
            if suffix == ".json":
                file_rows, file_summary = _rows_from_json_payload(_read_json_payload(path))
                summary.update(file_summary)
                rows.extend(file_rows)
            elif suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if isinstance(item, Mapping):
                        rows.append(dict(item))
            elif suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open(newline="", encoding="utf-8") as handle:
                    rows.extend(dict(row) for row in csv.DictReader(handle, delimiter=delimiter))
            else:
                row = _result_txt_row(path)
                if row is not None:
                    rows.append(row)
    return rows, summary, source_files


def sample_id_for_row(row: Mapping[str, Any], index: int) -> str:
    """Extracts or resolves a sample identifier from a parsed record row."""
    for field in ID_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return f"sample-{index:06d}"


def task_id_for_row(row: Mapping[str, Any], sample_id: str) -> str:
    """Extracts or resolves a task description identifier from a parsed record row."""
    for field in ("task_id", "task", "suite", "env", "scenario"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return sample_id


def parse_normalizer_overrides(values: Iterable[str]) -> dict[str, str]:
    """Parses CLI normalizer overrides specified as key-value pairs (metric_id=normalizer_spec)."""
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--normalizer entries must be metric_id=normalizer_spec")
        metric_id, spec = value.split("=", 1)
        metric_id = metric_id.strip()
        if not metric_id:
            raise ValueError("--normalizer metric id must not be empty")
        overrides[metric_id] = spec.strip()
    return overrides


def infer_normalizer(metric_id: str, overrides: Mapping[str, str]) -> str | None:
    """Infers the metric scaling strategy based on metric names or explicit user overrides."""
    if metric_id in overrides:
        return overrides[metric_id]
    if (
        metric_id in {
            "success",
            "success_rate",
            "task_success",
            "sequence_success",
            "episode_success",
            "rollout_success",
            "goal_success",
            "planning_success",
            "paraphrase_success_rate",
            "language_generalization_success",
            "completion",
            "action_accuracy",
        }
        or metric_id.endswith("_success")
        or metric_id.endswith("_rate")
        or metric_id.endswith("_accuracy")
    ):
        return "percent_or_fraction_to_unit"
    return None


def apply_metric_normalizer(normalizer: str | None, value: float) -> float:
    """Applies a resolved scaling spec to a floating-point metric value."""
    if not normalizer:
        return value
    return apply_normalizer(normalizer, value)


def normalized_metric_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    benchmark_id: str,
    track: str,
    normalizers: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Iterates, extracts, normalizes, and packages records into canonical metric rows."""
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = sample_id_for_row(row, index)
        task_id = task_id_for_row(row, sample_id)
        row_metric_id = row.get("metric_id") or row.get("metric") or row.get("metric_name")
        if row_metric_id and "value" in row:
            candidate_fields = (str(row_metric_id),)
            values = {str(row_metric_id): row.get("value")}
        else:
            candidate_fields = tuple(
                field
                for field in row
                if (
                    field in DEFAULT_NUMERIC_FIELDS
                    or field.endswith("_rate")
                    or field.endswith("_score")
                    or field.endswith("_success")
                    or field.endswith("_accuracy")
                )
            )
            values = row
        for metric_id in candidate_fields:
            raw_value = _coerce_number(values.get(metric_id))
            if raw_value is None:
                continue
            normalizer = infer_normalizer(metric_id, normalizers)
            normalized_value = apply_metric_normalizer(normalizer, raw_value)
            normalized.append(
                {
                    "benchmark_id": benchmark_id,
                    "track": track,
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "metric_id": metric_id,
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "normalizer": normalizer or "identity",
                    "metadata": {
                        key: value
                        for key, value in row.items()
                        if key not in values and key not in ID_FIELDS and key not in {"metric", "metric_id", "metric_name"}
                    },
                }
            )
    return normalized


def summarize(rows: list[dict[str, Any]], summary_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Computes mean performance statistics overall and per distinct task-types."""
    by_metric: dict[str, list[float]] = defaultdict(list)
    by_task_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row["normalized_value"])
        metric_id = str(row["metric_id"])
        by_metric[metric_id].append(value)
        by_task_metric[str(row["task_id"])][metric_id].append(value)

    leaderboard_metrics = {
        metric_id: sum(values) / len(values) for metric_id, values in sorted(by_metric.items()) if values
    }
    per_task = {
        task_id: {
            metric_id: sum(values) / len(values) for metric_id, values in sorted(metrics.items()) if values
        }
        for task_id, metrics in sorted(by_task_metric.items())
    }
    return {
        "available": bool(rows),
        "kind": "vla_va_wam_official_result_normalizer",
        "num_results": len(rows),
        "sample_count": len({row["sample_id"] for row in rows}),
        "task_count": len(per_task),
        "leaderboard_metrics": leaderboard_metrics,
        "per_task": per_task,
        "official_summary": dict(summary_payload),
    }


def build_scorecard(
    *,
    benchmark_id: str,
    track: str,
    input_paths: list[Path],
    source_files: list[str],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Assembles a standard, verified WorldFoundry Scorecard matching global database requirements."""
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "evaluation_available": summary["available"],
        "sample_count": summary["sample_count"],
        "run": {
            "status": "normalized_official_results",
            "started_at": utc_now_iso(),
            "runner": "worldfoundry.evaluation.tasks.embodied.normalizer",
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "track": track,
            "contract_only": False,
            "normalizer_first": True,
        },
        "dataset": {
            "input_paths": [str(path) for path in input_paths],
            "source_files": source_files,
        },
        "generation": {
            "available": False,
            "reason": "official result normalizer did not run model generation",
        },
        "evaluation": summary,
        "metrics": {
            "leaderboard": summary["leaderboard_metrics"],
            "per_task": summary["per_task"],
            "per_metric": {
                metric_id: {"available": True, "value": value}
                for metric_id, value in summary["leaderboard_metrics"].items()
            },
            "summary": {
                "sample_count": summary["sample_count"],
                "task_count": summary["task_count"],
                "normalized_result_rows": len(rows),
            },
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "normalizer-first import of official result files; upstream benchmark runtime was not executed by WorldFoundry",
            ],
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "raw_results": str((output_dir / "raw_results.jsonl").resolve()),
            "summary": str((output_dir / "evaluation" / "summary.json").resolve()),
            "per_sample_metrics": str((output_dir / "evaluation" / "per_sample_metrics.jsonl").resolve()),
        },
    }


def normalize_results(
    *,
    input_paths: list[Path],
    output_dir: Path,
    benchmark_id: str,
    track: str,
    normalizers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Orchestrates the entire crawling, normalization, summary, and serialization pipeline.

    Writes:
    1. `scorecard.json` (root directory scorecard metadata)
    2. `evaluation/summary.json` (statistical aggregation outputs)
    3. `evaluation/per_sample_metrics.jsonl` (line-by-line normalized records)
    4. `raw_results.jsonl` (raw extracted records for downstream auditing)
    """
    if track not in TRACKS:
        raise ValueError(f"track must be one of: {', '.join(sorted(TRACKS))}")
    source_rows, official_summary, source_files = read_input_rows(input_paths)
    metric_rows = normalized_metric_rows(
        source_rows,
        benchmark_id=benchmark_id,
        track=track,
        normalizers=normalizers or {},
    )
    summary = summarize(metric_rows, official_summary)
    scorecard = build_scorecard(
        benchmark_id=benchmark_id,
        track=track,
        input_paths=input_paths,
        source_files=source_files,
        rows=metric_rows,
        summary=summary,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "scorecard.json", scorecard)
    write_json(output_dir / "evaluation" / "summary.json", summary)
    write_jsonl(output_dir / "evaluation" / "per_sample_metrics.jsonl", metric_rows)
    write_jsonl(output_dir / "raw_results.jsonl", metric_rows)
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    """Builds the argparse schema representing the CLI parameters."""
    parser = argparse.ArgumentParser(
        description="Normalize official VLA / VA(VAM) / WAM JSON, JSONL, or CSV results into a WorldFoundry scorecard."
    )
    parser.add_argument("--input", action="append", type=Path, required=True, help="Official result file or directory. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--normalizer", action="append", default=[], help="Metric override as metric_id=normalizer_spec.")
    parser.add_argument("--json", action="store_true", help="Print the scorecard JSON after writing artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entry-point."""
    args = build_parser().parse_args(argv)
    try:
        scorecard = normalize_results(
            input_paths=args.input,
            output_dir=args.output_dir,
            benchmark_id=args.benchmark_id,
            track=args.track,
            normalizers=parse_normalizer_overrides(args.normalizer),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(scorecard, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"{args.benchmark_id}: normalized {scorecard['evaluation']['num_results']} metric rows")
        print(f"scorecard: {scorecard['artifacts']['scorecard']}")
    return 0


__all__ = [
    "DEFAULT_NUMERIC_FIELDS",
    "ID_FIELDS",
    "SCORECARD_SCHEMA_VERSION",
    "TRACKS",
    "apply_metric_normalizer",
    "build_parser",
    "build_scorecard",
    "infer_normalizer",
    "main",
    "normalize_results",
    "normalized_metric_rows",
    "parse_normalizer_overrides",
    "read_input_rows",
    "sample_id_for_row",
    "summarize",
    "task_id_for_row",
]
