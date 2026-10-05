"""Validation checks for HarnessEval outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .aggregate import numeric
from .io import read_json
from .model_order import MODEL_REGISTRY
from .protocols import FAMILIES
from .report import SCORING_POLICY, build_report


def validate_run(
    eval_root: Path,
    manifests: Iterable[Path],
    *,
    expected_models: Iterable[str] | None = None,
    resume_log: Path | None = None,
    json_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Audit a completed HarnessEval run against its exact manifest product."""

    violations: list[dict[str, Any]] = []
    case_families: dict[tuple[str, str], Path] = {}
    family_case_counts: Counter[str] = Counter()
    manifest_paths = [path.resolve() for path in manifests]
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        for case in manifest.get("cases") or ():
            case_id = str(case.get("case_id") or "")
            family = str(((case.get("taxonomy") or {}).get("probe_family") or ""))
            if not case_id or family not in FAMILIES:
                violations.append(
                    {
                        "kind": "invalid_manifest_case",
                        "manifest": str(manifest_path),
                        "case_id": case_id,
                        "family": family,
                    }
                )
                continue
            key = (family, case_id)
            if key in case_families:
                violations.append(
                    {
                        "kind": "duplicate_manifest_case",
                        "family": family,
                        "case_id": case_id,
                        "manifests": [str(case_families[key]), str(manifest_path)],
                    }
                )
                continue
            case_families[key] = manifest_path
            family_case_counts[family] += 1

    models = list(expected_models or MODEL_REGISTRY)
    expected_keys = {
        (model, family, case_id)
        for model in models
        for family, case_id in case_families
    }
    score_paths = sorted((eval_root / "rollout_scores").glob("*/*/*.scores.json"))
    scores: list[dict[str, Any]] = []
    actual_keys: list[tuple[str, str, str]] = []
    schema_counts: Counter[str] = Counter()
    coverage: Counter[tuple[str, str]] = Counter()
    parse_failures = []
    for path in score_paths:
        try:
            score = read_json(path)
        except Exception as exc:  # noqa: BLE001 - the audit must report every invalid JSON file
            parse_failures.append({"path": str(path), "error": str(exc)})
            continue
        model = str(score.get("model_id") or "")
        family = str(((score.get("taxonomy") or {}).get("probe_family") or ""))
        case_id = str(score.get("case_id") or "")
        key = (model, family, case_id)
        scores.append(score)
        actual_keys.append(key)
        schema_counts[str(score.get("schema_version"))] += 1
        coverage[(model, family)] += 1
        expected_relative = Path(model) / family / f"{case_id}.scores.json"
        if path.relative_to(eval_root / "rollout_scores") != expected_relative:
            violations.append(
                {
                    "kind": "score_path_payload_mismatch",
                    "path": str(path),
                    "payload_key": list(key),
                }
            )
        if numeric((score.get("case_score") or {}).get("score")) is None:
            violations.append({"kind": "non_numeric_case_score", "key": list(key)})

    if parse_failures:
        violations.append({"kind": "score_json_parse_failures", "items": parse_failures[:50]})
    key_counts = Counter(actual_keys)
    duplicate_keys = [list(key) for key, count in key_counts.items() if count != 1]
    if duplicate_keys:
        violations.append({"kind": "duplicate_score_keys", "keys": duplicate_keys[:50]})
    actual_key_set = set(actual_keys)
    missing_keys = sorted(expected_keys - actual_key_set)
    unexpected_keys = sorted(actual_key_set - expected_keys)
    if missing_keys:
        violations.append(
            {"kind": "missing_score_keys", "count": len(missing_keys), "keys": missing_keys[:50]}
        )
    if unexpected_keys:
        violations.append(
            {
                "kind": "unexpected_score_keys",
                "count": len(unexpected_keys),
                "keys": unexpected_keys[:50],
            }
        )
    if schema_counts != Counter({"harnesseval.skill_eval": len(expected_keys)}):
        violations.append({"kind": "schema_coverage", "counts": dict(schema_counts)})

    expected_coverage = {
        model: {family: family_case_counts[family] for family in FAMILIES}
        for model in models
    }
    actual_coverage = {
        model: {family: coverage[(model, family)] for family in FAMILIES}
        for model in models
    }
    if actual_coverage != expected_coverage:
        violations.append(
            {
                "kind": "model_family_coverage",
                "expected": expected_coverage,
                "actual": actual_coverage,
            }
        )

    report = build_report(scores)
    expected_common = {family: family_case_counts[family] for family in FAMILIES}
    if report.get("model_count") != len(models):
        violations.append(
            {
                "kind": "report_model_count",
                "expected": len(models),
                "actual": report.get("model_count"),
            }
        )
    if report.get("common_case_counts") != expected_common:
        violations.append(
            {
                "kind": "report_common_cases",
                "expected": expected_common,
                "actual": report.get("common_case_counts"),
            }
        )
    if report.get("scoring_policy") != SCORING_POLICY:
        violations.append(
            {
                "kind": "report_scoring_policy",
                "expected": SCORING_POLICY,
                "actual": report.get("scoring_policy"),
            }
        )
    observation_count_distribution = report.get("observation_count_distribution") or {}
    expected_score_count = len(models) * sum(
        family_case_counts[family] for family in FAMILIES
    )
    scores_with_observation = sum(
        int(count)
        for observation_count, count in observation_count_distribution.items()
        if int(observation_count) > 0
    )
    if scores_with_observation != expected_score_count:
        violations.append(
            {
                "kind": "observation_coverage",
                "expected": expected_score_count,
                "actual": scores_with_observation,
                "distribution": observation_count_distribution,
            }
        )
    missing_macros = [
        row.get("model_id")
        for row in report.get("leaderboard") or ()
        if numeric(row.get("overall_macro")) is None
    ]
    if missing_macros:
        violations.append({"kind": "missing_overall_macro", "models": missing_macros})

    resume = None
    if resume_log is not None:
        resume = read_json(resume_log)
        checks = {
            "preparation_written": (resume.get("preparation") or {}).get("written"),
            "preparation_blocked": (resume.get("preparation") or {}).get("blocked"),
            "preparation_failed": (resume.get("preparation") or {}).get("failed"),
            "preparation_stale": (resume.get("preparation") or {}).get(
                "stale_not_refreshed"
            ),
            "score_written": (resume.get("execution") or {}).get("written"),
            "score_blocked": (resume.get("execution") or {}).get("blocked"),
            "score_stale": (resume.get("execution") or {}).get("stale_not_refreshed"),
        }
        if any(value != 0 for value in checks.values()):
            violations.append({"kind": "resume_not_noop", "checks": checks})
        cached = ((resume.get("score_plan") or {}).get("status_counts") or {}).get("cached")
        if cached != len(expected_keys):
            violations.append(
                {"kind": "resume_score_cache_coverage", "expected": len(expected_keys), "actual": cached}
            )

    json_scan = {"file_count": 0, "failure_count": 0, "failures": []}
    seen_json_paths = set()
    for root in json_roots:
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen_json_paths:
                continue
            seen_json_paths.add(resolved)
            json_scan["file_count"] += 1
            try:
                read_json(path)
            except Exception as exc:  # noqa: BLE001 - malformed evidence must be reported
                json_scan["failure_count"] += 1
                if len(json_scan["failures"]) < 50:
                    json_scan["failures"].append({"path": str(path), "error": str(exc)})
    if json_scan["failure_count"]:
        violations.append({"kind": "json_parse_scan", **json_scan})

    return {
        "schema_version": "harnesseval.completion_audit",
        "status": "passed" if not violations else "failed",
        "manifest_count": len(manifest_paths),
        "case_count": len(case_families),
        "model_count": len(models),
        "expected_score_count": len(expected_keys),
        "actual_score_count": len(scores),
        "duplicate_key_count": len(duplicate_keys),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "schema_counts": dict(schema_counts),
        "expected_family_counts_per_model": expected_common,
        "actual_coverage": actual_coverage,
        "report": report,
        "resume": resume,
        "json_scan": json_scan,
        "violation_count": len(violations),
        "violations": violations,
    }
