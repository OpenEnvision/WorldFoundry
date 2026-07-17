#!/usr/bin/env python3
"""Official runner for PhysVidBench physical commonsense evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_captions import (
    resolve_caption_base,
    resolve_captions_dir,
)
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_judge import (
    judge_config_from_env,
    run_physvidbench_qa,
)
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_physvidbench_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.physvidbench.physvidbench_prompts import (
    load_prompt_rows,
    resolve_physvidbench_root,
    resolve_prompt_manifest_path,
    unique_prompt_records,
)

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize PhysVidBench official outputs.")
    parser.add_argument("--benchmark-id", default="physvidbench")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Execute the in-tree PhysVidBench caption-QA component; candidate-video captioning is not in-tree.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--physvidbench-root", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--captions-dir", type=Path)
    parser.add_argument("--caption-base", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _load_qa_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "rows", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
        raise ValueError(f"Unsupported PhysVidBench JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    official_component_executed: bool,
) -> list[dict[str, Any]]:
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = direct_metrics.get(metric_id)
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
                "source": (
                    "physvidbench_official_caption_qa"
                    if official_component_executed
                    else "physvidbench_results_csv"
                ),
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_physvidbench_results",
            }
        )
    return rows


def _video_index(generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None:
        return {
            "provided": False,
            "generated_artifact_dir": None,
            "video_count": 0,
            "by_stem": {},
            "duplicate_stems": [],
        }
    by_stem: dict[str, list[str]] = {}
    if generated_dir.exists():
        for path in sorted(generated_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_stem.setdefault(path.stem, []).append(str(path))
    return {
        "provided": True,
        "generated_artifact_dir": str(generated_dir),
        "video_count": sum(len(paths) for paths in by_stem.values()),
        "by_stem": dict(by_stem),
        "duplicate_stems": sorted(stem for stem, paths in by_stem.items() if len(paths) > 1),
    }


def _coverage(expected_prompt_ids: set[str], generated_dir: Path | None) -> dict[str, Any]:
    actual_names: set[str] = set()
    if generated_dir is not None and generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                actual_names.add(path.stem.lstrip("0") if path.stem.isdigit() else path.stem)
    expected_normalized = {prompt_id.lstrip("0") if prompt_id.isdigit() else prompt_id for prompt_id in expected_prompt_ids}
    actual_normalized = actual_names
    missing = sorted(expected_normalized - actual_normalized)
    unexpected = sorted(actual_normalized - expected_normalized)
    matched = sorted(expected_normalized & actual_normalized)
    return {
        "expected_count": len(expected_prompt_ids),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_prompt_ids) and not missing,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _caption_qa_component_executed(
    *,
    official_runtime_executed: bool,
    qa_summary: Mapping[str, Any] | None,
) -> bool:
    backend = str((qa_summary or {}).get("backend") or "").strip().lower()
    return official_runtime_executed and backend in {"gemini", "google"}


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    prompt_manifest_path: Path | None,
    metric_rows: list[dict[str, Any]],
    video_coverage: Mapping[str, Any],
    qa_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
    prompt_count: int,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("expected_count", 0) > 0
        and video_coverage.get("complete") is True
    )
    input_coverage_complete = (
        prompt_count > 0
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
    )
    normalization_ok = bool(available_rows)
    component_executed = _caption_qa_component_executed(
        official_runtime_executed=official_runtime_executed,
        qa_summary=qa_summary,
    )
    integration_evidence = component_executed and normalization_ok
    evidence_scope = (
        "official_caption_qa_component"
        if integration_evidence
        else "mock_or_result_import"
    )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": not integration_evidence,
        "normalization_ok": normalization_ok,
        "eligibility": {
            "full_suite_valid": False,
            "input_coverage_complete": input_coverage_complete,
            "video_coverage_complete": video_complete,
            "candidate_video_caption_provenance_verified": False,
        },
        "run": {
            "status": (
                "official_caption_qa_component_completed"
                if integration_evidence
                else "mock_or_results_normalized"
                if normalization_ok
                else "failed"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_physvidbench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "qa_summary": dict(qa_summary or {}),
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "PhysVidBench",
            "evidence_scope": evidence_scope,
        },
        "dataset": {
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "prompt_count": prompt_count,
            "caption_base": (qa_summary or {}).get("caption_base"),
            "generated_video_coverage": dict(video_coverage),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "official_component_available_count": (
                    len(available_rows) if integration_evidence else 0
                ),
                "declared_metric_count": len(METRIC_ORDER),
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": (
                "physvidbench_official_caption_qa_component"
                if integration_evidence
                else "physvidbench_mock_or_result_normalizer"
            ),
            "evidence_scope": evidence_scope,
            "candidate_video_caption_provenance_verified": False,
            "blocked_count": len(METRIC_ORDER) - len(available_rows),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "official_results_path": str(official_results_path.resolve()),
        },
        "coverage": {"videos": dict(video_coverage)},
        "notes": [
            "PhysVidBench in-tree QA consumes prompts_questions.csv and 8-caption AuroraCap tracks.",
            "The caption-QA component does not prove that supplied caption tracks came from candidate videos.",
            "The mock backend is fixture validation and never benchmark evidence.",
        ],
        "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
        "prompt_count": prompt_count,
    }


def normalize_physvidbench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    qa_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.generated_artifact_dir or _env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    official_results_path = args.official_results_path or _env_path("WORLDFOUNDRY_PHYSVIDBENCH_RESULTS_PATH")
    if official_results_path is None:
        raise ValueError(
            "--official-results-path, WORLDFOUNDRY_PHYSVIDBENCH_RESULTS_PATH, or --run-official is required"
        )
    repo_root = resolve_physvidbench_root(args.physvidbench_root)
    prompt_manifest_path = resolve_prompt_manifest_path(explicit=args.prompt_manifest, repo_root=repo_root)
    prompt_records = unique_prompt_records(load_prompt_rows(prompt_manifest_path=prompt_manifest_path))
    qa_rows = _load_qa_rows(Path(official_results_path))
    computed = compute_physvidbench_metrics(qa_rows=qa_rows)
    official_component_executed = _caption_qa_component_executed(
        official_runtime_executed=official_runtime_executed,
        qa_summary=qa_summary,
    )
    metric_rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        official_component_executed=official_component_executed,
    )
    video_coverage = _coverage({record["prompt_id"] for record in prompt_records}, generated_dir)
    scorecard = _scorecard(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        prompt_manifest_path=prompt_manifest_path,
        metric_rows=metric_rows,
        video_coverage=video_coverage,
        qa_summary=qa_summary,
        official_runtime_executed=official_runtime_executed,
        prompt_count=len(prompt_records),
    )
    strict = args.strict or os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_STRICT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    per_sample_rows = [
        {
            "prompt_id": row.get("PromptID"),
            "question": row.get("Question"),
            "types": row.get("Types"),
            "model_answer": row.get("Model_Answer"),
            "match": row.get("Match"),
        }
        for row in qa_rows
    ]
    write_jsonl(output_dir / "raw_metric_table.jsonl", metric_rows)
    write_jsonl(output_dir / "per_sample_scores.jsonl", per_sample_rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "prompt_manifest": str(prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "prompt_count": len(prompt_records),
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_physvidbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = resolve_physvidbench_root(args.physvidbench_root)
    prompt_manifest_path = resolve_prompt_manifest_path(explicit=args.prompt_manifest, repo_root=repo_root)
    caption_base = resolve_caption_base(
        explicit=args.caption_base,
        captions_dir=resolve_captions_dir(explicit=args.captions_dir, repo_root=repo_root),
        repo_root=repo_root,
    )
    output_csv = output_dir / "output.csv"
    qa_summary = run_physvidbench_qa(
        question_csv=prompt_manifest_path,
        caption_base=caption_base,
        output_csv=output_csv,
        config=judge_config_from_env(),
        limit=args.limit,
    )
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=output_csv,
        output_dir=output_dir,
        generated_artifact_dir=args.generated_artifact_dir,
        physvidbench_root=args.physvidbench_root,
        prompt_manifest=prompt_manifest_path,
        captions_dir=args.captions_dir,
        caption_base=args.caption_base,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_physvidbench_results(
        normalize_args,
        official_runtime_executed=True,
        qa_summary=qa_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_official:
            scorecard = run_official_physvidbench(args)
        else:
            scorecard = normalize_physvidbench_results(args)
    except Exception as exc:  # noqa: BLE001
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
                "runner": "benchmark_zoo_physvidbench_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "PhysVidBench"},
            "dataset": {
                "prompt_manifest": None,
                "prompt_count": 0,
                "caption_base": None,
                "generated_video_coverage": {},
            },
            "eligibility": {
                "full_suite_valid": False,
                "candidate_video_caption_provenance_verified": False,
            },
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {
                "available": False,
                "kind": "physvidbench_result_normalizer",
                "blocked_count": len(METRIC_ORDER),
            },
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
