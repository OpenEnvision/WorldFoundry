#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners._benchmark_metrics.formulas import (
    videoverse_subquestion_metrics,
)

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
PROMPT_MANIFEST_REL = Path("prompt/prompts_of_VideoVerse.json")
DECOMPOSED_PROMPT_MANIFEST_REL = Path("prompt/prompts_of_VideoVerse_decomposed.json")
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
CANONICAL_PROMPT_COUNT = 300
CANONICAL_EVENT_COUNT = 815
CANONICAL_CHECK_COUNT = 793

METRIC_ORDER = (
    "qa_accuracy",
    "event_coverage",
    "temporal_causality",
    "world_knowledge_consistency",
    "static_scene_consistency",
    "dynamic_event_consistency",
    "videoverse_average",
)
METRIC_SPECS: dict[str, dict[str, str]] = {
    "qa_accuracy": {
        "name": "QA Accuracy",
        "group": "official_result",
        "description": "Yes/No accuracy over VideoVerse verification questions.",
    },
    "event_coverage": {
        "name": "Event Coverage",
        "group": "event_following",
        "description": "Official event-following LCS score divided by total expected events.",
    },
    "temporal_causality": {
        "name": "Temporal Causality",
        "group": "event_following",
        "description": "Temporal event-order score from the official event-following response.",
    },
    "world_knowledge_consistency": {
        "name": "World Knowledge Consistency",
        "group": "world_knowledge",
        "description": "Mean pass rate over world-knowledge check types.",
    },
    "static_scene_consistency": {
        "name": "Static Scene Consistency",
        "group": "static",
        "description": "Static VideoVerse check-type pass rate.",
    },
    "dynamic_event_consistency": {
        "name": "Dynamic Event Consistency",
        "group": "dynamic",
        "description": "Dynamic VideoVerse score over event following, camera, interaction, and mechanics checks.",
    },
    "videoverse_average": {
        "name": "VideoVerse Average",
        "group": "aggregate",
        "description": "Official overall VideoVerse score normalized by the prompt-suite maximum.",
    },
}

DYNAMIC_CHECK_TYPES = frozenset({"Camera Control", "Interaction", "Mechanics"})
STATIC_CHECK_TYPES = frozenset(
    {
        "Material Properties",
        "Natural Constraints",
        "Common Sense",
        "Attribution Correctness",
        "2d_layout",
        "3d_depth",
    }
)
WORLD_KNOWLEDGE_CHECK_TYPES = frozenset({"Material Properties", "Natural Constraints", "Common Sense", "Mechanics"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize VideoVerse official evaluation outputs.")
    parser.add_argument("--benchmark-id", default="videoverse")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-official", action="store_true", help="Execute the in-tree VideoVerse judge and normalize scores.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--decomposed-prompt-manifest", type=Path)
    parser.add_argument("--limit", type=int, help="Optional prompt-count cap for in-tree judge execution.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless the canonical full suite is complete.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _default_prompt_manifest() -> Path | None:
    bundled = bundled_benchmark_asset("videoverse", PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    return None


def _default_decomposed_prompt_manifest() -> Path | None:
    bundled = bundled_benchmark_asset("videoverse", DECOMPOSED_PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _load_json(path)


def _records_by_id(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("results", "samples", "records", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return _records_by_id(value)
        if all(isinstance(value, Mapping) for value in payload.values()):
            return {str(key): dict(value) for key, value in payload.items()}
        return {}
    if isinstance(payload, list):
        rows: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                continue
            sample_id = _sample_id(item) or f"row-{index}"
            rows[sample_id] = dict(item)
        return rows
    return {}


def _sample_id(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "prompt_id", "video_id", "id", "uid"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _is_metric_row_payload(payload: Any) -> bool:
    rows = payload if isinstance(payload, list) else None
    if rows is None and isinstance(payload, Mapping):
        value = payload.get("results") or payload.get("records") or payload.get("metrics")
        rows = value if isinstance(value, list) else None
    if not rows:
        return False
    return all(isinstance(row, Mapping) and any(key in row for key in ("metric_id", "metric", "leaderboard_key")) for row in rows)


def _metric_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else None
    if rows is None and isinstance(payload, Mapping):
        value = payload.get("results") or payload.get("records") or payload.get("metrics")
        rows = value if isinstance(value, list) else None
    result: list[dict[str, Any]] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        metric_id = str(row.get("metric_id") or row.get("metric") or row.get("leaderboard_key") or "")
        if metric_id not in METRIC_ORDER:
            continue
        value = _as_float(
            row.get("normalized_score", row.get("score", row.get("value", row.get("accuracy", row.get("mean")))))
        )
        spec = METRIC_SPECS[metric_id]
        result.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": value is not None,
                "raw_score": value,
                "normalized_score": _unit(value),
                "score": _unit(value),
                "higher_is_better": True,
                "group": spec["group"],
                "source": "metric_row_import",
                "sample_count": row.get("sample_count"),
                "components": {},
                "subquestion_components": {},
                "reason": None if value is not None else "score_not_available_in_videoverse_results",
            }
        )
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                return float(text) / 100.0
            except ValueError:
                return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _unit(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 100.0:
        return value / 100.0
    return value


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return (float(numerator) / float(denominator)) if denominator else None


def _calculate_lcs(s1: str, s2: str) -> str:
    previous = [""] * (len(s2) + 1)
    for left in s1:
        current = [""]
        for index, right in enumerate(s2, start=1):
            if left == right:
                current.append(previous[index - 1] + left)
            else:
                current.append(max(previous[index], current[index - 1], key=len))
        previous = current
    return previous[-1]


def _filter_event_letters(value: Any) -> str:
    return "".join(char for char in str(value or "") if "A" <= char <= "Z")


def _yes_no(value: Any) -> str:
    text = str(value or "").strip().lower()
    match = re.search(r"yes|no", text, flags=re.IGNORECASE)
    return match.group(0).lower() if match else "wrong"


def _manifest_stats(prompt_manifest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    event_count = 0
    check_count = 0
    check_type_counts: dict[str, int] = defaultdict(int)
    for record in prompt_manifest.values():
        event_info = record.get("t2v_eval_event_info")
        if isinstance(event_info, Mapping):
            plan = event_info.get("verification_plan")
            if isinstance(plan, list):
                event_count += len(plan)
        checks = record.get("verification_checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            check_count += 1
            check_type_counts[str(check.get("check_type") or "Unknown")] += 1
    return {
        "prompt_count": len(prompt_manifest),
        "event_count": event_count,
        "check_count": check_count,
        "check_type_counts": dict(sorted(check_type_counts.items())),
        "canonical_suite": (
            len(prompt_manifest) == CANONICAL_PROMPT_COUNT
            and event_count == CANONICAL_EVENT_COUNT
            and check_count == CANONICAL_CHECK_COUNT
        ),
    }


def _bounded_prompt_manifest(
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    """Return the exact prompt subset used by a bounded judge run."""
    if limit is None:
        return {str(key): dict(value) for key, value in prompt_manifest.items()}
    if limit <= 0:
        raise ValueError("VideoVerse --limit must be a positive integer")
    return {
        prompt_id: dict(prompt_manifest[prompt_id])
        for prompt_id in sorted(prompt_manifest)[:limit]
    }


def _video_index(generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None:
        return {
            "provided": False,
            "generated_artifact_dir": None,
            "video_count": 0,
            "by_stem": {},
            "duplicate_stems": [],
        }
    by_stem: dict[str, list[str]] = defaultdict(list)
    if generated_dir.exists():
        for path in sorted(generated_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_stem[path.stem].append(str(path))
    return {
        "provided": True,
        "generated_artifact_dir": str(generated_dir),
        "video_count": sum(len(paths) for paths in by_stem.values()),
        "by_stem": dict(by_stem),
        "duplicate_stems": sorted(stem for stem, paths in by_stem.items() if len(paths) > 1),
    }


def _coverage(expected_ids: set[str], actual_ids: set[str]) -> dict[str, Any]:
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    matched = sorted(expected_ids & actual_ids)
    return {
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_ids) and not missing and not unexpected,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _result_event_score(result_record: Mapping[str, Any], manifest_record: Mapping[str, Any]) -> tuple[int, int]:
    manifest_event_info = manifest_record.get("t2v_eval_event_info")
    result_event_info = result_record.get("t2v_eval_event_info")
    event_info = result_event_info if isinstance(result_event_info, Mapping) else manifest_event_info
    if not isinstance(event_info, Mapping):
        return 0, 0
    plan = event_info.get("verification_plan")
    if not isinstance(plan, list) or not plan:
        return 0, 0
    expected = "".join(chr(ord("A") + index) for index in range(len(plan)))
    prediction = _filter_event_letters(
        event_info.get("overall_event_processed_res")
        or event_info.get("overall_event_res")
        or result_record.get("overall_event_processed_res")
        or result_record.get("overall_event_res")
    )
    return len(_calculate_lcs(expected, prediction)), len(plan)


def _result_check_scores(
    result_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    manifest_checks = manifest_record.get("verification_checks")
    result_checks = result_record.get("verification_checks")
    manifest_checks = manifest_checks if isinstance(manifest_checks, list) else []
    result_checks = result_checks if isinstance(result_checks, list) else []
    yes = 0
    no = 0
    wrong = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "wrong": 0, "total": 0})
    for index, manifest_check in enumerate(manifest_checks):
        if not isinstance(manifest_check, Mapping):
            continue
        result_check = result_checks[index] if index < len(result_checks) and isinstance(result_checks[index], Mapping) else {}
        check_type = str(result_check.get("check_type") or manifest_check.get("check_type") or "Unknown")
        response = _yes_no(result_check.get("res"))
        by_type[check_type]["total"] += 1
        if response == "yes":
            yes += 1
            by_type[check_type]["yes"] += 1
        elif response == "no":
            no += 1
            by_type[check_type]["no"] += 1
        else:
            wrong += 1
            by_type[check_type]["wrong"] += 1
    return yes, no, wrong, dict(by_type)


def _sum_type(stats: Mapping[str, Mapping[str, int]], types: frozenset[str], field: str) -> int:
    return sum(int(stats.get(check_type, {}).get(field, 0)) for check_type in types)


def compute_videoverse_scores(
    *,
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    result_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    event_score = 0
    event_total = 0
    check_yes = 0
    check_no = 0
    check_wrong = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "wrong": 0, "total": 0})

    for prompt_id, manifest_record in prompt_manifest.items():
        result_record = result_records.get(prompt_id)
        if not isinstance(result_record, Mapping):
            event_info = manifest_record.get("t2v_eval_event_info")
            if isinstance(event_info, Mapping) and isinstance(event_info.get("verification_plan"), list):
                event_total += len(event_info["verification_plan"])
            for check in manifest_record.get("verification_checks") or []:
                if isinstance(check, Mapping):
                    by_type[str(check.get("check_type") or "Unknown")]["total"] += 1
                    check_wrong += 1
                    by_type[str(check.get("check_type") or "Unknown")]["wrong"] += 1
            continue
        score, total = _result_event_score(result_record, manifest_record)
        event_score += score
        event_total += total
        yes, no, wrong, result_by_type = _result_check_scores(result_record, manifest_record)
        check_yes += yes
        check_no += no
        check_wrong += wrong
        for check_type, stats in result_by_type.items():
            for field, value in stats.items():
                by_type[check_type][field] += int(value)

    check_total = check_yes + check_no + check_wrong
    check_valid_total = check_yes + check_no
    dynamic_raw = event_score + _sum_type(by_type, DYNAMIC_CHECK_TYPES, "yes")
    dynamic_total = event_total + _sum_type(by_type, DYNAMIC_CHECK_TYPES, "total")
    static_raw = _sum_type(by_type, STATIC_CHECK_TYPES, "yes")
    static_total = _sum_type(by_type, STATIC_CHECK_TYPES, "total")
    world_raw = _sum_type(by_type, WORLD_KNOWLEDGE_CHECK_TYPES, "yes")
    world_total = _sum_type(by_type, WORLD_KNOWLEDGE_CHECK_TYPES, "total")
    overall_raw = event_score + check_yes
    overall_total = event_total + check_total

    direct_metrics = {
        "qa_accuracy": _safe_div(check_yes, check_valid_total),
        "event_coverage": _safe_div(event_score, event_total),
        "temporal_causality": _safe_div(event_score, event_total),
        "world_knowledge_consistency": _safe_div(world_raw, world_total),
        "static_scene_consistency": _safe_div(static_raw, static_total),
        "dynamic_event_consistency": _safe_div(dynamic_raw, dynamic_total),
        "videoverse_average": _safe_div(overall_raw, overall_total),
    }
    return {
        "metrics": direct_metrics,
        "components": {
            "event_following": {"score": event_score, "total": event_total},
            "verification_checks": {
                "yes": check_yes,
                "no": check_no,
                "wrong": check_wrong,
                "total": check_total,
                "valid_total": check_valid_total,
            },
            "official_leaderboard_counts": {
                "overall": overall_raw,
                "overall_total": overall_total,
                "dynamic": dynamic_raw,
                "dynamic_total": dynamic_total,
                "static": static_raw,
                "static_total": static_total,
                "event_following": event_score,
                "camera_control": _sum_type(by_type, frozenset({"Camera Control"}), "yes"),
                "interaction": _sum_type(by_type, frozenset({"Interaction"}), "yes"),
                "mechanics": _sum_type(by_type, frozenset({"Mechanics"}), "yes"),
                "material_properties": _sum_type(by_type, frozenset({"Material Properties"}), "yes"),
                "natural_constraints": _sum_type(by_type, frozenset({"Natural Constraints"}), "yes"),
                "common_sense": _sum_type(by_type, frozenset({"Common Sense"}), "yes"),
                "attribution_correctness": _sum_type(by_type, frozenset({"Attribution Correctness"}), "yes"),
                "2d_layout": _sum_type(by_type, frozenset({"2d_layout"}), "yes"),
                "3d_depth": _sum_type(by_type, frozenset({"3d_depth"}), "yes"),
            },
            "per_check_type": dict(sorted(by_type.items())),
        },
    }


def _subquestion_components(result_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = {key: value for key, value in result_records.items() if isinstance(value, Mapping)}
    return videoverse_subquestion_metrics(payload)


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    subquestion: Mapping[str, Any],
    source_path: Path,
) -> list[dict[str, Any]]:
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        score = direct_metrics.get(metric_id)
        source = "official_eval_res"
        if score is None and metric_id == "qa_accuracy" and subquestion.get("total_sub_questions"):
            score = subquestion.get("sub_question_accuracy")
            source = "official_sub_question_eval"
        if score is None and metric_id == "videoverse_average" and subquestion.get("total_videos"):
            score = subquestion.get("video_accuracy")
            source = "official_sub_question_eval"
        rows.append(
            {
                "metric_id": metric_id,
                "name": METRIC_SPECS[metric_id]["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": True,
                "group": METRIC_SPECS[metric_id]["group"],
                "source": source if score is not None else None,
                "source_path": str(source_path),
                "components": components,
                "subquestion_components": dict(subquestion),
                "reason": None if score is not None else "score_not_available_in_videoverse_results",
            }
        )
    return rows


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    prompt_manifest_path: Path,
    decomposed_prompt_manifest_path: Path | None,
    manifest_stats: Mapping[str, Any],
    result_coverage: Mapping[str, Any],
    video_coverage: Mapping[str, Any],
    metric_rows: list[dict[str, Any]],
    subquestion_components: Mapping[str, Any],
    official_runtime_executed: bool = False,
    judge_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("provided") is True
        and video_coverage.get("complete") is True
        and not video_coverage.get("duplicate_stems")
    )
    full_suite_valid = (
        manifest_stats.get("canonical_suite") is True
        and result_coverage.get("complete") is True
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    official_runtime_ok = official_runtime_executed and normalization_ok
    official_verified = official_runtime_ok and full_suite_valid
    integration_evidence = official_runtime_ok or (
        not official_runtime_executed and full_suite_valid
    )
    normalizer_only = not official_runtime_executed and not integration_evidence
    eligibility_reasons = []
    if official_runtime_executed:
        eligibility_reasons.append(
            "WorldFoundry executed the in-tree VideoVerse judge and normalized eval_res.json into the official metric surface."
        )
        if not full_suite_valid:
            eligibility_reasons.append(
                "Full-suite integration evidence requires the canonical 300-prompt manifest, complete generated videos, and complete judge outputs."
            )
    else:
        eligibility_reasons.append(
            "WorldFoundry normalized caller-provided VideoVerse official result JSON; official Gemini/API execution is external to this scorecard."
        )
    eligibility_reasons.append(
        "leaderboard_valid remains false until the upstream judge invocation, model/version, and submission protocol are independently audited."
    )
    judge_artifacts: dict[str, str] = {}
    if judge_summary is not None:
        judge_artifacts["eval_res"] = str((output_dir / "eval_res.json").resolve())
        judge_artifacts["judge_responses"] = str((output_dir / "judge_responses.jsonl").resolve())
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": normalizer_only,
        "normalization_ok": normalization_ok,
        "official_results_imported": bool(available_rows) and not official_runtime_executed,
        "run": {
            "status": (
                "official_verified"
                if official_verified and integration_evidence
                else "official_runtime_executed"
                if official_runtime_ok
                else "official_results_normalized"
                if available_rows
                else "official_results_missing_scores"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_videoverse_official_runner",
            "command": None,
            "returncode": 0 if available_rows else 1,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "VideoVerse",
            "contract_only": False,
            "requires_upstream_runtime": not official_runtime_executed,
            "official_runtime_available": official_runtime_executed,
            "official_judge": (
                (judge_summary or {}).get("judge_backend", "gemini")
                if official_runtime_executed
                else "Gemini 2.5 Pro or caller-provided equivalent official result file"
            ),
        },
        "dataset": {
            "prompt_manifest": str(prompt_manifest_path),
            "decomposed_prompt_manifest": None if decomposed_prompt_manifest_path is None else str(decomposed_prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "manifest_stats": dict(manifest_stats),
            "result_coverage": dict(result_coverage),
            "video_coverage": dict(video_coverage),
        },
        "eligibility": {
            "canonical_suite": manifest_stats.get("canonical_suite") is True,
            "full_suite_valid": full_suite_valid,
            "leaderboard_valid": False,
            "reasons": eligibility_reasons,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "event_following": ["event_coverage", "temporal_causality"],
                "dynamic": ["dynamic_event_consistency"],
                "static": ["static_scene_consistency"],
                "world_knowledge": ["world_knowledge_consistency"],
                "aggregate": ["videoverse_average"],
            },
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "blocked_metric_count": len(metric_rows) - len(available_rows),
                "result_prompt_count": result_coverage.get("actual_count"),
                "expected_prompt_count": result_coverage.get("expected_count"),
                "sub_question_accuracy": subquestion_components.get("sub_question_accuracy"),
                "sub_question_total": subquestion_components.get("total_sub_questions"),
            },
        },
        "evaluation": {
            "available": bool(available_rows),
            "kind": "videoverse_official_in_tree" if official_runtime_executed else "videoverse_official_result_normalizer",
            "evidence_level": "official_runtime_executed" if official_runtime_executed else "official_results_normalized",
            "num_results": len(metric_rows),
            "skip_count": len(metric_rows) - len(available_rows),
            "blocked_count": len(metric_rows) - len(available_rows),
            "judge_summary": dict(judge_summary or {}),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "raw_metric_table": str((output_dir / "raw_metric_table.jsonl").resolve()),
            "benchmark_contract": str((output_dir / "benchmark_contract.json").resolve()),
            "per_sample_scores": str((output_dir / "per_sample_scores.jsonl").resolve()),
            **judge_artifacts,
        },
    }


def normalize_videoverse_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    judge_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_VIDEOVERSE_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError("--official-results-path or WORLDFOUNDRY_VIDEOVERSE_RESULTS_PATH is required")
    prompt_manifest_path = (
        args.prompt_manifest
        or _env_path("WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST")
        or _default_prompt_manifest()
    )
    decomposed_prompt_manifest_path = (
        args.decomposed_prompt_manifest
        or _env_path("WORLDFOUNDRY_VIDEOVERSE_DECOMPOSED_PROMPT_MANIFEST")
        or _default_decomposed_prompt_manifest()
    )
    if prompt_manifest_path is None or not Path(prompt_manifest_path).is_file():
        raise ValueError(
            "VideoVerse prompt manifest is missing. Restore the bundled asset under "
            "worldfoundry/data/benchmarks/assets/videoverse/ or set "
            "WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST."
        )
    generated_artifact_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")

    prompt_payload = _load_json(prompt_manifest_path)
    if not isinstance(prompt_payload, Mapping):
        raise ValueError(f"VideoVerse prompt manifest must be an object keyed by prompt id: {prompt_manifest_path}")
    full_prompt_manifest = {
        str(key): dict(value)
        for key, value in prompt_payload.items()
        if isinstance(value, Mapping)
    }
    prompt_manifest = _bounded_prompt_manifest(full_prompt_manifest, args.limit)
    manifest_stats = _manifest_stats(prompt_manifest)
    results_payload = _load_json_or_jsonl(official_results_path)

    if _is_metric_row_payload(results_payload):
        rows = _metric_rows_from_payload(results_payload)
        result_records: dict[str, dict[str, Any]] = {}
        subquestion = {}
        result_coverage = _coverage(set(prompt_manifest), set())
        video_index = _video_index(generated_artifact_dir)
        video_coverage = _coverage(set(prompt_manifest), set(video_index["by_stem"]))
        video_coverage.update(
            {
                "provided": video_index["provided"],
                "generated_artifact_dir": video_index["generated_artifact_dir"],
                "video_count": video_index["video_count"],
                "duplicate_stems": video_index["duplicate_stems"],
            }
        )
    else:
        result_records = {
            prompt_id: record
            for prompt_id, record in _records_by_id(results_payload).items()
            if prompt_id in prompt_manifest
        }
        subquestion = _subquestion_components(result_records)
        result_coverage = _coverage(set(prompt_manifest), set(result_records))
        video_index = _video_index(generated_artifact_dir)
        video_coverage = _coverage(set(prompt_manifest), set(video_index["by_stem"]))
        video_coverage.update(
            {
                "provided": video_index["provided"],
                "generated_artifact_dir": video_index["generated_artifact_dir"],
                "video_count": video_index["video_count"],
                "duplicate_stems": video_index["duplicate_stems"],
            }
        )
        computed = compute_videoverse_scores(prompt_manifest=prompt_manifest, result_records=result_records)
        rows = _metric_rows(computed=computed, subquestion=subquestion, source_path=official_results_path)

    scorecard = _scorecard(
        benchmark_id=str(args.benchmark_id),
        output_dir=output_dir,
        official_results_path=official_results_path,
        prompt_manifest_path=prompt_manifest_path,
        decomposed_prompt_manifest_path=(
            decomposed_prompt_manifest_path
            if decomposed_prompt_manifest_path is not None and decomposed_prompt_manifest_path.is_file()
            else None
        ),
        manifest_stats=manifest_stats,
        result_coverage=result_coverage,
        video_coverage=video_coverage,
        metric_rows=rows,
        subquestion_components=subquestion,
        official_runtime_executed=official_runtime_executed,
        judge_summary=judge_summary,
    )
    strict = args.strict or str(os.environ.get("WORLDFOUNDRY_VIDEOVERSE_STRICT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", rows)
    write_jsonl(
        output_dir / "per_sample_scores.jsonl",
        [
            {"prompt_id": prompt_id, "available": prompt_id in result_records}
            for prompt_id in sorted(prompt_manifest)
        ],
    )
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "prompt_manifest": str(prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "canonical_suite": manifest_stats["canonical_suite"],
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_videoverse(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the in-tree VideoVerse judge and normalize the emitted eval_res.json."""
    from worldfoundry.evaluation.tasks.execution.runners.videoverse.videoverse_judge import run_videoverse_judge

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_manifest_path = (
        args.prompt_manifest
        or _env_path("WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST")
        or _default_prompt_manifest()
    )
    decomposed_prompt_manifest_path = (
        args.decomposed_prompt_manifest
        or _env_path("WORLDFOUNDRY_VIDEOVERSE_DECOMPOSED_PROMPT_MANIFEST")
        or _default_decomposed_prompt_manifest()
    )
    if prompt_manifest_path is None or not Path(prompt_manifest_path).is_file():
        raise ValueError(
            "VideoVerse prompt manifest is missing. Restore the bundled asset under "
            "worldfoundry/data/benchmarks/assets/videoverse/ or set "
            "WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST."
        )
    generated_artifact_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_artifact_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")

    prompt_payload = _load_json(prompt_manifest_path)
    if not isinstance(prompt_payload, Mapping):
        raise ValueError(f"VideoVerse prompt manifest must be an object keyed by prompt id: {prompt_manifest_path}")
    prompt_manifest = {str(key): dict(value) for key, value in prompt_payload.items() if isinstance(value, Mapping)}
    decomposed_manifest: dict[str, dict[str, Any]] | None = None
    if decomposed_prompt_manifest_path is not None and Path(decomposed_prompt_manifest_path).is_file():
        decomposed_payload = _load_json(decomposed_prompt_manifest_path)
        if isinstance(decomposed_payload, Mapping):
            decomposed_manifest = {
                str(key): dict(value) for key, value in decomposed_payload.items() if isinstance(value, Mapping)
            }

    eval_res_path = output_dir / "eval_res.json"
    judge_responses_path = output_dir / "judge_responses.jsonl"
    judge_summary = run_videoverse_judge(
        prompt_manifest=prompt_manifest,
        generated_video_dir=Path(generated_artifact_dir),
        output_path=eval_res_path,
        decomposed_manifest=decomposed_manifest,
        limit=args.limit,
        judge_responses_path=judge_responses_path,
    )

    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=eval_res_path,
        output_dir=output_dir,
        generated_artifact_dir=generated_artifact_dir,
        prompt_manifest=prompt_manifest_path,
        decomposed_prompt_manifest=decomposed_prompt_manifest_path,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_videoverse_results(
        normalize_args,
        official_runtime_executed=True,
        judge_summary=judge_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_official:
            scorecard = run_official_videoverse(args)
        else:
            scorecard = normalize_videoverse_results(args)
    except Exception as exc:  # noqa: BLE001 - CLI should always emit a scorecard-shaped failure.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_videoverse_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "VideoVerse"},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {"available": False, "kind": "videoverse_official_result_normalizer", "blocked_count": len(METRIC_ORDER)},
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
        return 1
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
    return int(scorecard.get("run", {}).get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
